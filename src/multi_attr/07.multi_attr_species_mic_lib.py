#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sequence_species_mic 多属性训练数据共用工具（除匹配状态外全列纳入训练）。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

CANON_RX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

MIC_BIN_LABELS = {
    0: "极强活性 (MIC ≤ 1 μM)",
    1: "强活性 (1 < MIC ≤ 4 μM)",
    2: "中等活性 / 有效 (4 < MIC ≤ 16 μM)",
    3: "弱活性 (16 < MIC ≤ 64 μM)",
    4: "低活性 / 不敏感 (MIC > 64 μM)",
}

MIC_TASKS = ["MIC0", "MIC1", "MIC2", "MIC3", "MIC4"]
MIC_TASK_RX = re.compile(r"^MIC([0-4])$")
SPECIES_TASK_RX = re.compile(r"^SP([0-9]{2})$")
SPECIES_MIC_TASK_RX = re.compile(r"^SP([0-9]{2})_MIC([0-4])$")

COLUMN_ALIASES = {
    "sequence": ["sequence", "sequence_clean", "序列"],
    "gram_pos": ["gram_pos", "Gram+", "Gram_pos", "anti_gram_pos_final", "革兰氏阳性菌"],
    "gram_neg": ["gram_neg", "Gram-", "Gram_neg", "anti_gram_neg_final", "革兰氏阴性菌"],
    "fungus": ["fungus", "Fungus", "anti_fungi_final", "真菌"],
    "mic_um": ["MIC_uM", "mic_um", "mic_uM"],
    "safety": ["safety", "safety_label", "safe", "toxic", "安全性"],
    "length": ["length", "seq_len", "长度"],
    "species": ["species", "species_name", "菌种"],
    "species_id": ["species_id"],
    "mic_bin": ["mic_bin", "MIC_bin"],
}

FIXED_ATTR_EXPORTS = [
    ("GP1", "gram_pos", 1),
    ("GN1", "gram_neg", 1),
    ("FU1", "fungus", 1),
    ("SF0", "safety", 0),
    ("SF1", "safety", 1),
]

EXCLUDE_COLUMNS = {"match_status", "匹配状态"}

# 数据集内 20 种菌的革兰/真菌分类（训练前 gram 全 0 时用于补标）
SPECIES_GRAM_TYPE: Dict[str, str] = {
    "Acinetobacter baumannii": "gram_neg",
    "Bacillus cereus": "gram_pos",
    "Bacillus subtilis": "gram_pos",
    "Candida albicans": "fungus",
    "Candida parapsilosis ATCC 22019": "fungus",
    "Enterococcus faecalis": "gram_pos",
    "Enterococcus faecium": "gram_pos",
    "Escherichia coli": "gram_neg",
    "Klebsiella pneumoniae": "gram_neg",
    "Micrococcus luteus": "gram_pos",
    "Micrococcus luteus A270": "gram_pos",
    "Pseudomonas aeruginosa": "gram_neg",
    "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028": "gram_neg",
    "Salmonella enterica subsp. enterica serovar Typhimurium C77-31": "gram_neg",
    "Salmonella typhimurium": "gram_neg",
    "Salmonella typhimurium KCTC 1926": "gram_neg",
    "Staphylococcus aureus": "gram_pos",
    "Staphylococcus epidermidis": "gram_pos",
    "Staphylococcus epidermidis ATCC 12228": "gram_pos",
    "Staphylococcus epidermidis KCTC 1917": "gram_pos",
}

FUNGUS_GENERA = ("Candida", "Aspergillus", "Cryptococcus", "Saccharomyces", "Fusarium")
GRAM_POS_GENERA = (
    "Bacillus", "Staphylococcus", "Enterococcus", "Micrococcus", "Streptococcus",
    "Listeria", "Lactobacillus", "Clostridium",
)
GRAM_NEG_GENERA = (
    "Escherichia", "Klebsiella", "Pseudomonas", "Acinetobacter", "Salmonella",
    "Enterobacter", "Proteus", "Serratia", "Vibrio", "Haemophilus",
)


def resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mic_um_to_bin(mic_um: float) -> float:
    if pd.isna(mic_um):
        return float("nan")
    v = float(mic_um)
    if v <= 1.0:
        return 0.0
    if v <= 4.0:
        return 1.0
    if v <= 16.0:
        return 2.0
    if v <= 64.0:
        return 3.0
    return 4.0


def parse_safety_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    missing = series.isna() | raw.eq("") | raw.str.upper().isin(["NA", "NAN", "NONE", "NULL", "N/A"])
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(~missing, other=float("nan"))
    out = out.where(out.isin([0.0, 1.0]), other=float("nan"))
    return out


def species_task(species_id: int) -> str:
    return f"SP{int(species_id):02d}"


def species_mic_task(species_id: int, mic_bin: int) -> str:
    mic_task = f"MIC{int(mic_bin)}"
    if mic_task not in MIC_TASKS:
        raise ValueError(f"invalid mic_bin for joint task: {mic_bin}")
    return f"{species_task(int(species_id))}_{mic_task}"


def is_species_mic_task(task: str) -> bool:
    return SPECIES_MIC_TASK_RX.fullmatch(str(task).strip().upper()) is not None


def parse_species_mic_task(task: str) -> Tuple[int, int]:
    m = SPECIES_MIC_TASK_RX.fullmatch(str(task).strip().upper())
    if m is None:
        raise ValueError(f"not a species+mic joint task: {task}")
    return int(m.group(1)), int(m.group(2))


def expand_species_mic_joint_tasks(tasks: Sequence[str]) -> List[str]:
    """将分开的 MICk + SPxx 展开为 SPxx_MICk；已是联合或仅单侧任务则保持。"""
    joint: List[str] = []
    mic_only: List[str] = []
    sp_only: List[str] = []
    other: List[str] = []
    for raw in tasks or []:
        t = str(raw).strip().upper()
        if not t:
            continue
        if is_species_mic_task(t):
            joint.append(t)
        elif MIC_TASK_RX.fullmatch(t):
            mic_only.append(t)
        elif SPECIES_TASK_RX.fullmatch(t):
            sp_only.append(t)
        else:
            other.append(t)
    if mic_only and sp_only:
        expanded: List[str] = []
        for sp in sp_only:
            sp_id = int(sp[2:])
            for mic in mic_only:
                expanded.append(species_mic_task(sp_id, int(mic[3:])))
        out = list(other)
        for item in expanded:
            if item not in out:
                out.append(item)
        return out
    out = list(other)
    for item in joint + mic_only + sp_only:
        if item not in out:
            out.append(item)
    return out


def build_species_mapping(species_names: Sequence[str]) -> Dict[str, Any]:
    ordered = sorted(str(s).strip() for s in species_names if str(s).strip() and str(s).upper() != "NAN")
    name_to_id = {name: idx for idx, name in enumerate(ordered)}
    id_to_name = {str(idx): name for name, idx in name_to_id.items()}
    id_to_task = {str(idx): species_task(idx) for idx in range(len(ordered))}
    return {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "id_to_task": id_to_task,
        "n_species": len(ordered),
    }


def infer_gram_type_from_species(species_name: str) -> Optional[str]:
    """根据菌种名推断 gram_pos / gram_neg / fungus；无法推断返回 None。"""
    name = str(species_name).strip()
    if not name or name.upper() in ("NA", "NAN", "NONE", "NULL"):
        return None
    if name in SPECIES_GRAM_TYPE:
        return SPECIES_GRAM_TYPE[name]
    genus = name.split()[0]
    if genus in FUNGUS_GENERA:
        return "fungus"
    if genus in GRAM_POS_GENERA:
        return "gram_pos"
    if genus in GRAM_NEG_GENERA:
        return "gram_neg"
    return None


def supplement_gram_from_species(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """gram 全 0 且同时有菌种+MIC 的行：按菌种补 GP/GN/FU；无法归类则删行。"""
    out = df.copy()
    for col in ["gram_pos", "gram_neg", "fungus"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    gram_sum = out["gram_pos"] + out["gram_neg"] + out["fungus"]
    has_species = out["species"].notna()
    has_mic = out["mic_um"].notna()
    need_infer = (gram_sum == 0) & has_species & has_mic

    n_need = int(need_infer.sum())
    n_inferred_gp = 0
    n_inferred_gn = 0
    n_inferred_fu = 0
    drop_indices: List[Any] = []
    dropped_species: List[str] = []

    for idx in out.index[need_infer]:
        sp = out.at[idx, "species"]
        gtype = infer_gram_type_from_species(str(sp))
        if gtype is None:
            drop_indices.append(idx)
            dropped_species.append(str(sp))
        elif gtype == "gram_pos":
            out.at[idx, "gram_pos"] = 1
            n_inferred_gp += 1
        elif gtype == "gram_neg":
            out.at[idx, "gram_neg"] = 1
            n_inferred_gn += 1
        elif gtype == "fungus":
            out.at[idx, "fungus"] = 1
            n_inferred_fu += 1

    if drop_indices:
        out = out.drop(index=drop_indices).reset_index(drop=True)

    stats = {
        "gram_zero_with_species_mic_rows": n_need,
        "inferred_gram_pos": n_inferred_gp,
        "inferred_gram_neg": n_inferred_gn,
        "inferred_fungus": n_inferred_fu,
        "dropped_unclassified_rows": len(drop_indices),
        "dropped_unclassified_species": sorted(set(dropped_species)),
        "policy": "gram全0且有菌种+MIC -> 按菌种补标；无法归类则删行",
    }
    return out, stats


def load_rowlevel_table(csv_path: str, min_seq_len: int = 1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(str(csv_path))
    for col in list(df.columns):
        if col in EXCLUDE_COLUMNS:
            df = df.drop(columns=[col])

    seq_col = resolve_column(df, COLUMN_ALIASES["sequence"])
    if seq_col is None:
        raise RuntimeError("CSV 缺少序列列 (sequence / 序列)")

    df["sequence"] = df[seq_col].astype(str).str.strip().str.upper()

    for attr in ["gram_pos", "gram_neg", "fungus"]:
        col = resolve_column(df, COLUMN_ALIASES[attr])
        if col is None:
            raise RuntimeError(f"CSV 缺少列: {attr}")
        df[attr] = pd.to_numeric(df[col], errors="coerce")

    species_col = resolve_column(df, COLUMN_ALIASES["species"])
    if species_col is not None:
        df["species"] = df[species_col].astype(str).str.strip()
        df.loc[df["species"].isin(["", "nan", "NaN", "NA", "None"]), "species"] = pd.NA
    else:
        df["species"] = pd.NA

    mic_col = resolve_column(df, COLUMN_ALIASES["mic_um"])
    if mic_col is not None:
        df["mic_um"] = pd.to_numeric(df[mic_col], errors="coerce")
    elif "mic_um" not in df.columns:
        df["mic_um"] = float("nan")

    if "mic_bin" in df.columns:
        df["mic_bin"] = pd.to_numeric(df["mic_bin"], errors="coerce")
    else:
        df["mic_bin"] = df["mic_um"].apply(mic_um_to_bin)

    safety_col = resolve_column(df, COLUMN_ALIASES["safety"])
    if safety_col is not None:
        df["safety"] = parse_safety_series(df[safety_col])
    elif "safety_label" in df.columns:
        df["safety"] = parse_safety_series(df["safety_label"])
    else:
        df["safety"] = float("nan")

    length_col = resolve_column(df, COLUMN_ALIASES["length"])
    if length_col is not None:
        df["length"] = pd.to_numeric(df[length_col], errors="coerce")
    else:
        df["length"] = float("nan")

    species_mapping = build_species_mapping(df["species"].dropna().unique().tolist())
    name_to_id = species_mapping["name_to_id"]
    df["species_id"] = df["species"].map(name_to_id)

    df = df[df["sequence"].str.len() > 0].copy()
    df = df[df["sequence"].apply(lambda s: CANON_RX.fullmatch(s or "") is not None)].copy()
    df = df[df["sequence"].str.len() >= int(min_seq_len)].copy()

    for col in ["gram_pos", "gram_neg", "fungus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df, gram_stats = supplement_gram_from_species(df)

    if "length" not in df.columns or df["length"].isna().any():
        df["length"] = df["length"].fillna(df["sequence"].str.len())
    df["length"] = df["length"].astype(int)
    for col in ["gram_pos", "gram_neg", "fungus"]:
        df[col] = df[col].astype(int)

    keep_cols = [
        "sequence",
        "gram_pos",
        "gram_neg",
        "fungus",
        "species",
        "species_id",
        "mic_um",
        "mic_bin",
        "safety",
        "length",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    species_mapping["gram_inference_stats"] = gram_stats
    return out, species_mapping


def rows_to_flat_tasks(rows: pd.DataFrame) -> pd.DataFrame:
    """每行按可用属性生成训练前缀；缺失属性跳过，不丢弃整行。"""
    flat: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        seq = str(r["sequence"])
        split = str(r.get("split", "train"))
        seq_len = int(r["length"])
        n_tasks = 0

        if int(r.get("gram_pos", 0) or 0) == 1:
            flat.append(_task_row(seq, split, "GP1", "gram_pos", 1))
            n_tasks += 1
        if int(r.get("gram_neg", 0) or 0) == 1:
            flat.append(_task_row(seq, split, "GN1", "gram_neg", 1))
            n_tasks += 1
        if int(r.get("fungus", 0) or 0) == 1:
            flat.append(_task_row(seq, split, "FU1", "fungus", 1))
            n_tasks += 1

        sp_id = r.get("species_id")
        mic_val = r.get("mic_bin")
        has_sp = pd.notna(sp_id)
        has_mic = pd.notna(mic_val)
        if has_sp and has_mic:
            mic_task = f"MIC{int(mic_val)}"
            if mic_task in MIC_TASKS:
                joint = species_mic_task(int(sp_id), int(mic_val))
                flat.append(
                    _species_mic_task_row(
                        seq,
                        split,
                        joint,
                        species_id=int(sp_id),
                        mic_bin=int(mic_val),
                    )
                )
                n_tasks += 1
        else:
            if has_sp:
                task = species_task(int(sp_id))
                flat.append(_task_row(seq, split, task, "species_id", int(sp_id)))
                n_tasks += 1
            if has_mic:
                task = f"MIC{int(mic_val)}"
                if task in MIC_TASKS:
                    flat.append(_task_row(seq, split, task, "mic_bin", int(mic_val)))
                    n_tasks += 1

        sf_val = r.get("safety")
        if pd.notna(sf_val) and int(sf_val) in (0, 1):
            task = "SF0" if int(sf_val) == 0 else "SF1"
            flat.append(_task_row(seq, split, task, "safety", int(sf_val)))
            n_tasks += 1

        len_task = f"LEN{seq_len}"
        flat.append(_task_row(seq, split, len_task, "length", seq_len))
        n_tasks += 1

        if n_tasks <= 1:
            # 仅 LEN 也保留（序列长度控制仍有效）
            pass

    if not flat:
        raise RuntimeError("未生成任何训练样本")
    return pd.DataFrame(flat)


def _task_row(seq: str, split: str, task: str, attr: str, attr_val: int) -> Dict[str, Any]:
    prefix = str(task)
    return {
        "sequence": seq,
        "split": split,
        "task": prefix,
        "attr": attr,
        "attr_val": int(attr_val),
        "prefix": prefix,
        "tagged_sequence": f"{prefix}:{seq}",
    }


def _species_mic_task_row(
    seq: str,
    split: str,
    task: str,
    *,
    species_id: int,
    mic_bin: int,
) -> Dict[str, Any]:
    prefix = str(task)
    return {
        "sequence": seq,
        "split": split,
        "task": prefix,
        "attr": "species_mic",
        "attr_val": int(species_id) * 10 + int(mic_bin),
        "species_id": int(species_id),
        "mic_bin": int(mic_bin),
        "prefix": prefix,
        "tagged_sequence": f"{prefix}:{seq}",
    }


def save_species_mapping(mapping: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
