#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bipolar prefixes aligned with discriminators and SF: GP0/1, GN0/1, FU0/1, SF0/1.

Does not modify original five-attr lib. X bits are omitted (no 0/1 to emit).
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_spec_mic = importlib.util.spec_from_file_location("mic_lib_bipolar", str(_SCRIPTS / "07.multi_attr_species_mic_lib.py"))
_MIC = importlib.util.module_from_spec(_spec_mic)
_spec_mic.loader.exec_module(_MIC)
_spec_lib = importlib.util.spec_from_file_location("five_attr_lib_bipolar", str(_SCRIPTS / "07.five_attr_classification_lora_lib.py"))
_LIB = importlib.util.module_from_spec(_spec_lib)
_spec_lib.loader.exec_module(_LIB)

BASE_TASKS = ["GP0", "GP1", "GN0", "GN1", "FU0", "FU1", "SF0", "SF1"]
FIXED_TASK_RX = re.compile(r"^(GP0|GP1|GN0|GN1|FU0|FU1|SF0|SF1)$")
LENGTH_TASK_RX = _LIB.LENGTH_TASK_RX
MIC_TASK_RX = getattr(_LIB, "MIC_TASK_RX", re.compile(r"^MIC[0-4]$"))
SPECIES_TASK_RX = getattr(_LIB, "SPECIES_TASK_RX", re.compile(r"^SP\d{2}$"))
SPECIES_MIC_TASK_RX = getattr(_LIB, "SPECIES_MIC_TASK_RX", re.compile(r"^SP\d{2}_MIC[0-4]$"))


def _bit01(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        v = int(float(val))
    except Exception:
        return None
    if v in (0, 1):
        return v
    return None


def make_task_prompt(task: str) -> str:
    t = str(task).strip().upper()
    if (
        FIXED_TASK_RX.fullmatch(t)
        or LENGTH_TASK_RX.fullmatch(t)
        or MIC_TASK_RX.fullmatch(t)
        or SPECIES_TASK_RX.fullmatch(t)
        or SPECIES_MIC_TASK_RX.fullmatch(t)
    ):
        return f"{t}:"
    raise ValueError(f"unknown task: {task}")


def parse_attr_weights(text: str) -> Dict[str, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if str(x).strip()]
    if len(vals) == 4:
        vals = [vals[0], vals[1], vals[2], vals[3], 1.0]
    if len(vals) != 5:
        raise ValueError("attr_weights must contain 5 floats (GP,GN,FU,SF,LENGTH)")
    gp, gn, fu, sf, leng = vals
    return {
        "GP0": gp,
        "GP1": gp,
        "GN0": gn,
        "GN1": gn,
        "FU0": fu,
        "FU1": fu,
        "SF0": sf,
        "SF1": sf,
        "LENGTH": leng,
    }


def task_weight(task: str, weights: Dict[str, float]) -> float:
    t = str(task).strip().upper()
    if _LIB.is_length_task(t):
        return float(weights.get("LENGTH", 1.0))
    if SPECIES_MIC_TASK_RX.fullmatch(t):
        return float(weights.get("SP_MIC", weights.get("MIC", weights.get("SP", 1.0))))
    if MIC_TASK_RX.fullmatch(t):
        return float(weights.get("MIC", weights.get(t, 1.0)))
    if SPECIES_TASK_RX.fullmatch(t):
        return float(weights.get("SP", weights.get(t, 1.0)))
    return float(weights.get(t, 1.0))


def condition_to_task_names(condition: str, default_length: Optional[int] = None) -> List[str]:
    bits, target_len = _LIB.parse_condition(condition, default_length=default_length)
    gp, gn, fu, sf = _LIB.condition_to_targets(f"{bits}:{target_len}")
    tasks: List[str] = []
    if gp in (0, 1):
        tasks.append("GP0" if gp == 0 else "GP1")
    if gn in (0, 1):
        tasks.append("GN0" if gn == 0 else "GN1")
    if fu in (0, 1):
        tasks.append("FU0" if fu == 0 else "FU1")
    if sf is None:
        raise ValueError(f"safety bit cannot be X in this scheme: {condition}")
    tasks.append("SF0" if sf == 0 else "SF1")
    tasks.append(_LIB.make_length_task(target_len))
    return tasks


def rows_to_flat_tasks(rows: pd.DataFrame) -> pd.DataFrame:
    """Same as MIC flatten, but GP/GN/FU emit both 0 and 1 prefixes like SF."""
    flat: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        seq = str(r["sequence"])
        split = str(r.get("split", "train"))
        seq_len = int(r["length"])

        for col, prefix in (("gram_pos", "GP"), ("gram_neg", "GN"), ("fungus", "FU")):
            b = _bit01(r.get(col))
            if b is not None:
                task = f"{prefix}{b}"
                flat.append(_MIC._task_row(seq, split, task, col, b))

        sp_id = r.get("species_id")
        mic_val = r.get("mic_bin")
        has_sp = pd.notna(sp_id)
        has_mic = pd.notna(mic_val)
        if has_sp and has_mic:
            mic_task = f"MIC{int(mic_val)}"
            if mic_task in _MIC.MIC_TASKS:
                joint = _MIC.species_mic_task(int(sp_id), int(mic_val))
                flat.append(
                    _MIC._species_mic_task_row(
                        seq,
                        split,
                        joint,
                        species_id=int(sp_id),
                        mic_bin=int(mic_val),
                    )
                )
        else:
            if has_sp:
                task = _MIC.species_task(int(sp_id))
                flat.append(_MIC._task_row(seq, split, task, "species_id", int(sp_id)))
            if has_mic:
                task = f"MIC{int(mic_val)}"
                if task in _MIC.MIC_TASKS:
                    flat.append(_MIC._task_row(seq, split, task, "mic_bin", int(mic_val)))

        sf_val = r.get("safety")
        sf_b = _bit01(sf_val)
        if sf_b is not None:
            flat.append(_MIC._task_row(seq, split, f"SF{sf_b}", "safety", sf_b))

        flat.append(_MIC._task_row(seq, split, f"LEN{seq_len}", "length", seq_len))

    if not flat:
        raise RuntimeError("未生成任何训练样本")
    return pd.DataFrame(flat)


def flatten_sequence_level_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seq = str(r["sequence"])
        split = str(r["split"])
        seq_len = int(r["length"])
        for col, prefix in (("gram_pos", "GP"), ("gram_neg", "GN"), ("fungus", "FU")):
            b = _bit01(r.get(col))
            if b is not None:
                rows.append({"sequence": seq, "task": f"{prefix}{b}", "split": split})
        sf_b = _bit01(r.get("safety"))
        if sf_b is not None:
            rows.append({"sequence": seq, "task": f"SF{sf_b}", "split": split})
        mic_val = r.get("mic_bin")
        if pd.notna(mic_val):
            mic_task = f"MIC{int(mic_val)}"
            if mic_task in getattr(_MIC, "MIC_TASKS", {"MIC0", "MIC1", "MIC2", "MIC3", "MIC4"}):
                rows.append({"sequence": seq, "task": mic_task, "split": split})
        rows.append({"sequence": seq, "task": _LIB.make_length_task(seq_len), "split": split})
    return rows
