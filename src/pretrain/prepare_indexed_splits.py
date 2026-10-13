#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pass byte-offset index for train/val/test (same hash split as train_model.py)."""

from __future__ import annotations

import sys as _eos_sys
from pathlib import Path as _EosPath
_eos_src = str(_EosPath(__file__).resolve().parents[1])
if _eos_src not in _eos_sys.path:
    _eos_sys.path.insert(0, _eos_src)
from eos_compat import EOS_TEXT, normalize_eos_text

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np



def _hash_to_unit_interval(s: str, seed: int) -> float:
    h = hashlib.md5((str(seed) + "|" + s).encode("utf-8")).hexdigest()
    v = int(h[:12], 16)
    return v / float(1 << 48)


def _assign_split(line: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    u = _hash_to_unit_interval(line, seed)
    if u < test_ratio:
        return "test"
    if u < (test_ratio + val_ratio):
        return "val"
    return "train"


def _maybe_add_eos(s: str, ensure_eos: bool) -> str:
    x = normalize_eos_text(s).strip()
    if not x:
        return ""
    if ensure_eos and (not x.endswith(EOS_TEXT)):
        x = x + EOS_TEXT
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.001)
    p.add_argument("--test_ratio", type=float, default=0.001)
    p.add_argument("--eval_max_samples", type=int, default=8000)
    p.add_argument("--ensure_eos", action="store_true")
    a = p.parse_args()

    data_file = Path(a.data_file).resolve()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_offs: list[int] = []
    val_offs: list[int] = []
    test_offs: list[int] = []
    max_eval = int(a.eval_max_samples)

    with data_file.open("rb") as f:
        while True:
            off = f.tell()
            raw = f.readline()
            if not raw:
                break
            try:
                ln = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            s = _maybe_add_eos(ln, bool(a.ensure_eos))
            if not s:
                continue
            sp = _assign_split(s, int(a.seed), float(a.val_ratio), float(a.test_ratio))
            if sp == "train":
                train_offs.append(off)
            elif sp == "val":
                if len(val_offs) < max_eval:
                    val_offs.append(off)
            elif sp == "test":
                if len(test_offs) < max_eval:
                    test_offs.append(off)

    np.save(out_dir / "train_offsets.npy", np.array(train_offs, dtype=np.uint64))
    np.save(out_dir / "val_offsets.npy", np.array(val_offs, dtype=np.uint64))
    np.save(out_dir / "test_offsets.npy", np.array(test_offs, dtype=np.uint64))

    meta = {
        "data_file": str(data_file),
        "seed": int(a.seed),
        "val_ratio": float(a.val_ratio),
        "test_ratio": float(a.test_ratio),
        "eval_max_samples": max_eval,
        "ensure_eos": bool(a.ensure_eos),
        "train_count": len(train_offs),
        "val_count": len(val_offs),
        "test_count": len(test_offs),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[OK] indexed splits -> {out_dir} "
        f"train={len(train_offs)} val={len(val_offs)} test={len(test_offs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
