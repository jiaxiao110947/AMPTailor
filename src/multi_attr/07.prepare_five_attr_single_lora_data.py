#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

_LIB_PATH = Path(__file__).resolve().parent / "07.multi_attr_species_mic_lib.py"
_spec = importlib.util.spec_from_file_location("multi_attr_species_mic_lib_07_prep", str(_LIB_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_LIB_PATH}")
_MIC_LIB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_MIC_LIB)

CANON_RX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
FIXED_ATTR_EXPORTS = _MIC_LIB.FIXED_ATTR_EXPORTS
MIC_ATTR_EXPORTS = [(f"MIC{i}", "mic_bin", i) for i in range(5)]
ATTR_ALIASES = _MIC_LIB.COLUMN_ALIASES


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return _MIC_LIB.resolve_column(df, candidates)


def _parse_safety_series(series: pd.Series) -> pd.Series:
    return _MIC_LIB.parse_safety_series(series)


def _mic_um_to_bin(mic_um: float) -> float:
    return _MIC_LIB.mic_um_to_bin(mic_um)


def _has_rowlevel_species(df: pd.DataFrame) -> bool:
    return _resolve_column(df, ATTR_ALIASES["species"]) is not None or "species_id" in df.columns


def load_and_clean_rowlevel(
    csv_path: str,
    min_seq_len: int,
    split_seed: str,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    row_df, species_mapping = _MIC_LIB.load_rowlevel_table(str(csv_path), int(min_seq_len))
    row_df["split"] = row_df["sequence"].apply(lambda s: assign_split(split_seed, s, val_frac, test_frac))
    row_df["safety_labeled"] = row_df["safety"].notna().astype(int)
    row_df.attrs["species_mapping"] = species_mapping
    return row_df


def _hash_to_unit_interval(s: str, seed: str) -> float:
    import hashlib

    h = hashlib.md5((str(seed) + "|" + str(s)).encode("utf-8")).hexdigest()
    return float(int(h[:12], 16)) / float(1 << 48)


def _agg_safety(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.max())


def _agg_mic_bin(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.min())


def assign_split(seed: str, seq: str, val_frac: float, test_frac: float) -> str:
    u = _hash_to_unit_interval(seq, seed)
    train_cut = 1.0 - float(val_frac) - float(test_frac)
    val_cut = 1.0 - float(test_frac)
    if u < train_cut:
        return "train"
    if u < val_cut:
        return "val"
    return "test"


def load_and_clean(
    csv_path: str,
    min_seq_len: int,
    split_seed: str,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path))
    seq_col = _resolve_column(df, ["sequence", "sequence_clean", "序列"])
    if seq_col is None:
        raise RuntimeError("CSV must contain sequence or sequence_clean")
    df["sequence"] = df[seq_col].astype(str).str.strip().str.upper()
    optional_attrs = {"mic_bin"}
    for attr, aliases in ATTR_ALIASES.items():
        if attr == "length":
            continue
        col = _resolve_column(df, aliases)
        if col is None:
            if attr in optional_attrs:
                continue
            raise RuntimeError(f"CSV missing attribute column for {attr}")
        if attr == "safety":
            df[attr] = _parse_safety_series(df[col])
        else:
            df[attr] = pd.to_numeric(df[col], errors="coerce")
    mic_um_col = _resolve_column(df, ["MIC_uM", "mic_um", "mic_uM"])
    if "mic_bin" not in df.columns and mic_um_col is not None:
        mic_um = pd.to_numeric(df[mic_um_col], errors="coerce")
        df["mic_bin"] = mic_um.apply(_mic_um_to_bin)
    length_col = _resolve_column(df, ATTR_ALIASES["length"])
    if length_col is not None:
        df["length"] = pd.to_numeric(df[length_col], errors="coerce")
    df = df[df["sequence"].str.len() > 0].copy()
    df = df[df["sequence"].apply(lambda s: CANON_RX.fullmatch(s or "") is not None)].copy()
    df = df[df["sequence"].str.len() >= int(min_seq_len)].copy()
    # 不因 gram 全 0 丢弃行
    group_cols = ["gram_pos", "gram_neg", "fungus"]
    if "length" in df.columns:
        group_cols.append("length")
    if "safety" in df.columns:
        group_cols.append("safety")
    if "mic_bin" in df.columns:
        group_cols.append("mic_bin")
    agg_map = {c: "max" for c in group_cols if c not in {"safety", "mic_bin"}}
    if "safety" in group_cols:
        agg_map["safety"] = _agg_safety
    if "mic_bin" in group_cols:
        agg_map["mic_bin"] = _agg_mic_bin
    df = df.groupby("sequence", as_index=False).agg(agg_map)
    if "length" not in df.columns:
        df["length"] = df["sequence"].str.len()
    df["length"] = df["length"].astype(int)
    for attr in ["gram_pos", "gram_neg", "fungus", "length"]:
        df[attr] = df[attr].astype(int)
    df["split"] = df["sequence"].apply(lambda s: assign_split(split_seed, s, val_frac, test_frac))
    if "safety" in df.columns:
        df["safety_labeled"] = df["safety"].notna().astype(int)
    return df


def build_flat_five_attr(df: pd.DataFrame) -> pd.DataFrame:
    if "species_id" in df.columns or "species" in df.columns:
        return _MIC_LIB.rows_to_flat_tasks(df)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seq = str(r["sequence"])
        split = str(r["split"])
        seq_len = int(r["length"])
        for prefix, attr, target in FIXED_ATTR_EXPORTS:
            if attr == "safety":
                if pd.isna(r.get("safety")) or int(r[attr]) not in (0, 1):
                    continue
            if int(r[attr]) != int(target):
                continue
            rows.append(
                {
                    "sequence": seq,
                    "split": split,
                    "task": prefix,
                    "attr": attr,
                    "attr_val": int(target),
                    "prefix": prefix,
                    "tagged_sequence": f"{prefix}{seq}",
                }
            )
        if "mic_bin" in r.index and pd.notna(r.get("mic_bin")):
            mic_val = int(r["mic_bin"])
            for prefix, attr, target in MIC_ATTR_EXPORTS:
                if mic_val != int(target):
                    continue
                rows.append(
                    {
                        "sequence": seq,
                        "split": split,
                        "task": prefix,
                        "attr": attr,
                        "attr_val": int(target),
                        "prefix": prefix,
                        "tagged_sequence": f"{prefix}{seq}",
                    }
                )
        len_task = f"LEN{seq_len}"
        rows.append(
            {
                "sequence": seq,
                "split": split,
                "task": len_task,
                "attr": "length",
                "attr_val": seq_len,
                "prefix": len_task,
                "tagged_sequence": f"{len_task}{seq}",
            }
        )
    if not rows:
        raise RuntimeError("No five-attribute rows were produced")
    return pd.DataFrame(rows)


def _length_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "length_min": int(df["length"].min()),
        "length_max": int(df["length"].max()),
        "length_median": float(df["length"].median()),
        "length_mean": float(df["length"].mean()),
        "unique_lengths": int(df["length"].nunique()),
    }


def write_prefixed_texts(flat_df: pd.DataFrame, out_dir: Path) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    text_root = out_dir / "texts"
    text_root.mkdir(parents=True, exist_ok=True)
    for prefix in sorted(flat_df["prefix"].unique()):
        sub = flat_df[flat_df["prefix"] == prefix].copy()
        stats[str(prefix)] = {"all": int(len(sub))}
        safe_name = str(prefix).replace("/", "_")
        (text_root / f"{safe_name}.txt").write_text(
            "\n".join(sub["tagged_sequence"].tolist()) + ("\n" if len(sub) > 0 else ""),
            encoding="utf-8",
        )
        for split in ["train", "val", "test"]:
            split_dir = text_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            split_sub = sub[sub["split"] == split]
            stats[str(prefix)][split] = int(len(split_sub))
            (split_dir / f"{safe_name}.txt").write_text(
                "\n".join(split_sub["tagged_sequence"].tolist()) + ("\n" if len(split_sub) > 0 else ""),
                encoding="utf-8",
            )
    return stats


def main() -> None:
    p = argparse.ArgumentParser("07.prepare_five_attr_single_lora_data.py")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_seq_len", type=int, default=1)
    p.add_argument("--split_seed", default="split_v1")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument(
        "--all_train",
        action="store_true",
        help="全部序列划入 train，不划分 val/test",
    )
    a = p.parse_args()

    out_dir = Path(str(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    val_frac = 0.0 if bool(a.all_train) else float(a.val_frac)
    test_frac = 0.0 if bool(a.all_train) else float(a.test_frac)

    probe = pd.read_csv(str(a.csv_path), nrows=1)
    rowlevel = _has_rowlevel_species(probe) or "species_id" in probe.columns
    if rowlevel:
        clean_df = load_and_clean_rowlevel(
            csv_path=str(a.csv_path),
            min_seq_len=int(a.min_seq_len),
            split_seed=str(a.split_seed),
            val_frac=val_frac,
            test_frac=test_frac,
        )
    else:
        clean_df = load_and_clean(
            csv_path=str(a.csv_path),
            min_seq_len=int(a.min_seq_len),
            split_seed=str(a.split_seed),
            val_frac=val_frac,
            test_frac=test_frac,
        )
    flat_df = build_flat_five_attr(clean_df)

    clean_csv = out_dir / "amp_five_attr_source.csv"
    flat_csv = out_dir / "amp_five_attr_flat.csv"
    clean_df.to_csv(clean_csv, index=False)
    flat_df.to_csv(flat_csv, index=False)
    prefix_stats = write_prefixed_texts(flat_df, out_dir)

    species_stats: Dict[str, Any] = {}
    species_mapping = clean_df.attrs.get("species_mapping") if rowlevel else None
    if species_mapping is None and "species_id" in clean_df.columns:
        species_mapping = _MIC_LIB.build_species_mapping(clean_df["species"].dropna().unique().tolist())
    if species_mapping is not None:
        map_path = out_dir / "species_map.json"
        _MIC_LIB.save_species_mapping(species_mapping, map_path)
        species_stats = {
            "species_map_json": str(map_path),
            "n_species": int(species_mapping["n_species"]),
            "species_missing_rows": int(clean_df["species_id"].isna().sum()) if "species_id" in clean_df.columns else 0,
        }

    safety_stats: Dict[str, Any] = {}
    if "safety" in clean_df.columns:
        n_na = int(clean_df["safety"].isna().sum())
        n_labeled = int(clean_df["safety"].notna().sum())
        safety_stats = {
            "safety_column": "safety_label",
            "safety_na_sequences": n_na,
            "safety_labeled_sequences": n_labeled,
            "safety_label_0": int((clean_df["safety"] == 0).sum()),
            "safety_label_1": int((clean_df["safety"] == 1).sum()),
            "policy": "NA -> skip SF0/SF1 rows; still train GP/GN/FU/LEN/MIC",
        }

    mic_stats: Dict[str, Any] = {}
    if "mic_bin" in clean_df.columns:
        mic_stats = {
            "mic_bin_missing_sequences": int(clean_df["mic_bin"].isna().sum()),
            "mic_bin_counts": {
                str(int(k)): int(v)
                for k, v in clean_df["mic_bin"].dropna().value_counts().sort_index().items()
            },
            "policy": "NA -> skip MIC0-MIC4 rows; row-level mic_bin per sequence×species",
        }

    unit = "rows" if rowlevel else "sequences"
    payload = {
        "input_csv": str(a.csv_path),
        "clean_source_csv": str(clean_csv),
        "flat_five_attr_csv": str(flat_csv),
        "data_mode": "rowlevel_species_mic" if rowlevel else "sequence_level",
        f"n_{unit}": int(len(clean_df)),
        "n_unique_sequences": int(clean_df["sequence"].nunique()),
        "split_counts": {str(k): int(v) for k, v in clean_df["split"].value_counts().to_dict().items()},
        "length_stats": _length_stats(clean_df),
        "safety_stats": safety_stats,
        "mic_stats": mic_stats,
        "species_stats": species_stats,
        "five_attr_counts": prefix_stats,
    }
    (out_dir / "prepare_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
