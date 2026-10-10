#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict-combo flattening for six-attr expert training.

Original single-attr flattening is unchanged. This module emits one sample per
source row whose 4-bit GP/GN/FU/SF vector is a legal strict combo (no X),
and builds a joint prompt from the same task names used at fused generation.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_spec_mic = importlib.util.spec_from_file_location("mic_lib_strict_combo", str(_SCRIPTS / "07.multi_attr_species_mic_lib.py"))
_MIC = importlib.util.module_from_spec(_spec_mic)
_spec_mic.loader.exec_module(_MIC)
_spec_lib = importlib.util.spec_from_file_location("five_attr_lib_strict_combo", str(_SCRIPTS / "07.five_attr_classification_lora_lib.py"))
_LIB = importlib.util.module_from_spec(_spec_lib)
_spec_lib.loader.exec_module(_LIB)

COMBO_TASK_RX = re.compile(r"^COMBO:([01]{4}):([1-9][0-9]*)(?:\|(.+))?$")
FORBIDDEN_ATTR_BITS = set(_LIB.FORBIDDEN_ATTR_BITS)
COMBOS_14 = list(_LIB.COMBOS_14)


def is_combo_task(task: str) -> bool:
    return COMBO_TASK_RX.fullmatch(str(task).strip().upper()) is not None


def parse_combo_task(task: str) -> Tuple[str, int, List[str]]:
    m = COMBO_TASK_RX.fullmatch(str(task).strip().upper())
    if m is None:
        raise ValueError(f"not a strict combo task: {task}")
    bits = m.group(1)
    seq_len = int(m.group(2))
    extras: List[str] = []
    if m.group(3):
        extras = [x.strip().upper() for x in m.group(3).split("+") if x.strip()]
    return bits, seq_len, extras


def combo_constituent_tasks(task: str) -> List[str]:
    bits, seq_len, extras = parse_combo_task(task)
    names = list(_LIB.condition_to_task_names(f"{bits}:{seq_len}"))
    for extra in extras:
        if extra and extra not in names:
            names.append(extra)
    return names


def make_strict_combo_prompt(task: str) -> str:
    if is_combo_task(task):
        return "+".join(combo_constituent_tasks(task)) + ":"
    return _LIB.make_task_prompt(task)


def combo_target_length(task: str) -> int:
    if is_combo_task(task):
        return parse_combo_task(task)[1]
    if _LIB.is_length_task(task):
        return _LIB.length_from_task(task)
    return 0


def _bit(val: Any) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        v = int(float(val))
    except Exception:
        return None
    if v in (0, 1):
        return v
    return None


def _extra_sp_mic_tasks(row: pd.Series) -> List[str]:
    extras: List[str] = []
    sp_id = row.get("species_id")
    mic_val = row.get("mic_bin")
    has_sp = pd.notna(sp_id)
    has_mic = pd.notna(mic_val)
    if has_sp and has_mic:
        mic_task = f"MIC{int(mic_val)}"
        if mic_task in _MIC.MIC_TASKS:
            extras.append(_MIC.species_mic_task(int(sp_id), int(mic_val)))
            return extras
    if has_sp:
        extras.append(_MIC.species_task(int(sp_id)))
    if has_mic:
        mic_task = f"MIC{int(mic_val)}"
        if mic_task in _MIC.MIC_TASKS:
            extras.append(mic_task)
    return extras


def row_to_combo_task(row: pd.Series) -> Optional[str]:
    gp = _bit(row.get("gram_pos"))
    gn = _bit(row.get("gram_neg"))
    fu = _bit(row.get("fungus"))
    sf = _bit(row.get("safety"))
    if gp is None or gn is None or fu is None or sf is None:
        return None
    bits = f"{gp}{gn}{fu}{sf}"
    if bits in FORBIDDEN_ATTR_BITS:
        return None
    seq_len = int(row["length"])
    extras = _extra_sp_mic_tasks(row)
    extra_s = ("|" + "+".join(extras)) if extras else ""
    return f"COMBO:{bits}:{seq_len}{extra_s}"


def rows_to_strict_combo_tasks(rows: pd.DataFrame) -> pd.DataFrame:
    """One training sample per source row that has a legal strict 4-bit combo."""
    flat: List[Dict[str, Any]] = []
    n_skip_incomplete = 0
    n_skip_forbidden = 0
    for _, r in rows.iterrows():
        gp = _bit(r.get("gram_pos"))
        gn = _bit(r.get("gram_neg"))
        fu = _bit(r.get("fungus"))
        sf = _bit(r.get("safety"))
        if gp is None or gn is None or fu is None or sf is None:
            n_skip_incomplete += 1
            continue
        bits = f"{gp}{gn}{fu}{sf}"
        if bits in FORBIDDEN_ATTR_BITS:
            n_skip_forbidden += 1
            continue
        task = row_to_combo_task(r)
        if not task:
            n_skip_incomplete += 1
            continue
        seq = str(r["sequence"])
        split = str(r.get("split", "train"))
        prompt = make_strict_combo_prompt(task)
        flat.append(
            {
                "sequence": seq,
                "split": split,
                "task": task,
                "attr": "combo",
                "attr_val": int(bits, 2),
                "prefix": bits,
                "tagged_sequence": f"{prompt}{seq}",
                "combo_bits": bits,
            }
        )
    if not flat:
        raise RuntimeError("未生成任何严格组合训练样本")
    out = pd.DataFrame(flat)
    out.attrs["strict_combo_stats"] = {
        "n_combo_samples": int(len(out)),
        "n_skip_incomplete": int(n_skip_incomplete),
        "n_skip_forbidden": int(n_skip_forbidden),
        "policy": "require GP/GN/FU/SF in {0,1}; drop 0000/0001; one joint prompt per row; attach SP/MIC when present",
        "combo_bit_counts": {str(k): int(v) for k, v in out["prefix"].value_counts().sort_index().items()},
    }
    return out
