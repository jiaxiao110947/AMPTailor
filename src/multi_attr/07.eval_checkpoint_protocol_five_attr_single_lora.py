#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 系列统一评估入口：
- 无 --in_fasta：按 checkpoint 做 regular6 + strict14 生成与打分（协议模式）。
- 有 --in_fasta：对单份 FASTA 打分并写 eval_payload.json。

属性分来自仓库里的五个专家骨干判别器（GP、GN、FU、SF、MIC）。
GP/GN/FU/SF 加长度参与条件满足率；MIC 只写入分布，不进入 4-bit 组合条件。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Inlined from 06.eval_control_expert_attr_promptemb.py (FASTA 打分)
# ---------------------------------------------------------------------------

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
CANON_RX_EVAL = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
COMBO_RX = re.compile(r"combo=([01X]{4}:\d+)", re.IGNORECASE)
CONDITION_RX = re.compile(r"combo=([01X]{4}:\d+)", re.IGNORECASE)
# regular6 生成头可能只有 4-bit（无 :length），仍需参与属性 adherence 统计
COMBO_BITS_RX = re.compile(r"combo=([01X]{4})(?:\||$)", re.IGNORECASE)
LEN_RX = re.compile(r"LEN=(\d+)", re.IGNORECASE)


def _eval_run_cmd(cmd: List[str], timeout: int = 3600, cwd: Optional[str] = None) -> None:
    kw: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": int(timeout)}
    if cwd is not None and str(cwd).strip():
        kw["cwd"] = str(cwd)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{(r.stderr or '')[-2000:]}")


def _eval_norm_id(x: str) -> str:
    s = str(x).strip()
    if s.startswith(">"):
        s = s[1:].strip()
    s = s.split()[0]
    return s.split("|", 1)[0]


def _eval_count_fasta_seqs(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith(">"):
                n += 1
    return int(n)


def _eval_read_fasta(path: str) -> List[Tuple[str, str]]:
    items, pid, seq = [], "", ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s.startswith(">"):
                if pid and seq:
                    items.append((pid, seq))
                pid, seq = s[1:].strip(), ""
            else:
                seq += s
        if pid and seq:
            items.append((pid, seq))
    return items


def _eval_sanitize_fasta(in_fa: str, out_fa: str, min_len: int, max_len: int) -> Dict[str, Any]:
    items = _eval_read_fasta(in_fa)
    kept = []
    for pid, seq in items:
        s = "".join([c for c in str(seq).strip().upper() if c in AA20])
        if not s:
            continue
        if len(s) < int(min_len) or len(s) > int(max_len):
            continue
        if CANON_RX_EVAL.fullmatch(s) is None:
            continue
        kept.append((pid, s))
    Path(out_fa).parent.mkdir(parents=True, exist_ok=True)
    with open(out_fa, "w", encoding="utf-8") as w:
        for pid, seq in kept:
            w.write(f">{pid}\n{seq}\n")
    return {"total": len(items), "kept": len(kept), "dropped": len(items) - len(kept)}


def _eval_read_tsv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f, delimiter="\t")
        try:
            header = [h.strip() for h in next(rd)]
        except StopIteration:
            return [], []
        rows = []
        for parts in rd:
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append({header[i]: str(parts[i]).strip() for i in range(len(header))})
    return header, rows


def _eval_pick_col(header: List[str], cands: List[str]) -> str:
    low = {h.lower(): h for h in header}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return ""


def _eval_parse_binary_label(x: str) -> int:
    s = str(x).strip().lower()
    if s in {"1", "yes", "y", "true", "t", "pos", "positive", "amp"}:
        return 1
    if s in {"0", "no", "n", "false", "f", "neg", "negative", "non-amp"}:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except Exception:
        return 0


def _eval_read_ampsorter_summary(tsv: str, prob_thresh: float = 0.5) -> Dict[str, float]:
    h, rows = _eval_read_tsv(tsv)
    id_col = _eval_pick_col(h, ["id", "seq_id", "seqid", "name"])
    p_col = _eval_pick_col(h, ["prob", "probability", "score", "PredictedProb", "predictedprob"])
    lab_col = _eval_pick_col(h, ["label", "predictedlabel", "PredictedLabel", "predicted", "class"])
    probs: List[float] = []
    labels: List[int] = []
    keep_n = 0
    for r in rows:
        _ = _eval_norm_id(r.get(id_col, "")) if id_col else ""
        try:
            p = float(r.get(p_col, "0")) if p_col else 0.0
        except Exception:
            p = 0.0
        if lab_col:
            lab = _eval_parse_binary_label(r.get(lab_col, "0"))
        else:
            lab = 1 if p >= float(prob_thresh) else 0
        probs.append(float(p))
        labels.append(int(lab))
        if int(lab) == 1 and float(p) >= float(prob_thresh):
            keep_n += 1
    n = len(rows)
    n_pos = sum(labels)
    return {
        "n": float(n),
        "n_pos": float(n_pos),
        "pos_rate": float(n_pos) / float(max(1, n)),
        "mean_prob": float(sum(probs) / max(1, len(probs))) if probs else 0.0,
        "top1_prob": float(probs[0]) if probs else 0.0,
        "keep_n": float(keep_n),
        "keep_rate": float(keep_n) / float(max(1, n)),
    }


def _eval_read_tox_map(tsv: str) -> Dict[str, float]:
    h, rows = _eval_read_tsv(tsv)
    id_col = _eval_pick_col(h, ["id", "seq_id", "seqid", "name"])
    p_col = _eval_pick_col(h, ["prob", "probability", "score", "tox_prob", "PredictedProb", "predictedprob"])
    out: Dict[str, float] = {}
    for r in rows:
        rid = _eval_norm_id(r.get(id_col, "")) if id_col else ""
        if not rid:
            continue
        try:
            out[rid] = float(r.get(p_col, "0"))
        except Exception:
            out[rid] = 0.0
    return out


def _eval_summarize_prob_map(mp: Dict[str, float], thr: float) -> Dict[str, float]:
    vals = [float(v) for v in mp.values()]
    n = len(vals)
    pos_n = sum(1 for v in vals if float(v) >= float(thr))
    return {
        "n": float(n),
        "pos_n": float(pos_n),
        "pos_rate": float(pos_n) / float(max(1, n)),
        "mean_prob": float(sum(vals) / max(1, len(vals))) if vals else 0.0,
    }


def _eval_to_prob(x: Any) -> float:
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "t", "pos", "positive", "1"}:
        return 1.0
    if s in {"no", "n", "false", "f", "neg", "negative", "0"}:
        return 0.0
    try:
        return float(x)
    except Exception:
        return float("nan")


def _eval_read_iampcn_map(csv_path: str) -> Dict[str, Dict[str, float]]:
    import pandas as pd

    p = Path(csv_path)
    if not p.exists() or p.stat().st_size <= 0:
        return {}
    df = pd.read_csv(str(p))
    if df.shape[0] == 0:
        return {}
    cols = {str(c).strip().lower(): c for c in df.columns}
    id_col = cols.get("seq_id", None) or cols.get("name", None) or list(df.columns)[0]
    gp_col = (
        cols.get("anti-gram-positive", None)
        or cols.get("antigram-positive", None)
        or cols.get("anti_gram_positive", None)
    )
    gn_col = (
        cols.get("anti-gram-negative", None)
        or cols.get("antigram-negative", None)
        or cols.get("anti_gram_negative", None)
    )
    fu_col = cols.get("antifungal", None) or cols.get("anti-fungal", None)
    out: Dict[str, Dict[str, float]] = {}
    if gp_col is None or gn_col is None or fu_col is None:
        return out
    for _, r in df.iterrows():
        sid = _eval_norm_id(r.get(id_col, ""))
        if not sid:
            continue
        out[sid] = {
            "gp": _eval_to_prob(r.get(gp_col, float("nan"))),
            "gn": _eval_to_prob(r.get(gn_col, float("nan"))),
            "fu": _eval_to_prob(r.get(fu_col, float("nan"))),
        }
    return out


def _eval_summarize_iampcn_map(mp: Dict[str, Dict[str, float]], thr: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in ["gp", "gn", "fu"]:
        vals = [float(rec[key]) for rec in mp.values() if key in rec and not math.isnan(float(rec[key]))]
        pos_n = sum(1 for v in vals if float(v) >= float(thr))
        out[key] = {
            "n_valid": float(len(vals)),
            "pos_n": float(pos_n),
            "pos_rate": float(pos_n) / float(max(1, len(vals))),
            "mean_prob": float(sum(vals) / max(1, len(vals))) if vals else 0.0,
        }
    return out


def _eval_parse_combo(pid: str) -> str:
    m = COMBO_RX.search(str(pid))
    if m:
        return str(m.group(1)).upper()
    m2 = CONDITION_RX.search(str(pid))
    if m2:
        return str(m2.group(1)).upper()
    m3 = COMBO_BITS_RX.search(str(pid))
    if m3:
        return str(m3.group(1)).upper()
    return ""


def _eval_parse_target_length(pid: str) -> Optional[int]:
    m = LEN_RX.search(str(pid))
    if m:
        return int(m.group(1))
    combo = _eval_parse_combo(pid)
    if ":" in combo:
        return int(combo.split(":", 1)[1])
    return None


def _eval_attr_bits(combo: str) -> str:
    c = str(combo).strip().upper()
    return c.split(":", 1)[0] if ":" in c else c


def _eval_sat(prob: float, want: int, thr: float) -> bool:
    return float(prob) >= float(thr) if int(want) == 1 else float(prob) < float(thr)


def _release_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / "paths.env").is_file():
            return parent
    return here.parents[2]


def _prob_summary(values: List[float], thr: float) -> Dict[str, float]:
    n = len(values)
    pos_n = sum(1 for value in values if float(value) >= float(thr))
    return {
        "n": float(n),
        "n_valid": float(n),
        "pos_n": float(pos_n),
        "pos_rate": float(pos_n) / float(max(1, n)),
        "mean_prob": float(sum(values) / n) if n else 0.0,
    }


def _score_fasta_with_expert_discs(
    fasta: str,
    out_dir: Path,
    *,
    thr: float,
    disc_root: str,
    species_id: int,
) -> Dict[str, Any]:
    """Score a FASTA with models/discriminators GP, GN, FU, SF and MIC."""
    root = _release_root()
    score_py = root / "src" / "generation" / "score_expert_discriminators.py"
    spec = importlib.util.spec_from_file_location("score_expert_discriminators_protocol", str(score_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {score_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(score_py),
        "--fasta",
        str(fasta),
        "--out_dir",
        str(out_dir),
        "--thr",
        str(float(thr)),
        "--species_id",
        str(int(species_id)),
    ]
    if str(disc_root).strip():
        argv.extend(["--disc_root", str(disc_root)])
    old_argv = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    return json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))


def run_control_expert_fasta_eval(
    in_fasta: str,
    out_dir: str,
    *,
    ampsorter_script: str = "",
    ampsorter_backbone: str = "",
    ampsorter_tokenizer: str = "",
    ampsorter_classifier_pt: str = "",
    tox_script: str = "",
    tox_backbone: str = "",
    tox_tokenizer: str = "",
    tox_classifier_pt: str = "",
    iampcn_repo: str = "",
    iampcn_py: str = "",
    tox_thresh: float = 0.5,
    iampcn_thresh: float = 0.5,
    min_len: int = 1,
    max_len: int = 128,
    timeout: int = 3600,
    length_tolerance: int = 0,
    disc_root: str = "",
    disc_thr: float = 0.5,
    species_id: int = 0,
) -> Dict[str, Any]:
    """用五个专家骨干判别器打分，并在 out_dir 写入 eval_payload.json。"""
    del (
        ampsorter_script,
        ampsorter_backbone,
        ampsorter_tokenizer,
        ampsorter_classifier_pt,
        tox_script,
        tox_backbone,
        tox_tokenizer,
        tox_classifier_pt,
        iampcn_repo,
        iampcn_py,
        timeout,
    )
    thr = float(disc_thr)
    del tox_thresh, iampcn_thresh
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_fa = str(out / "generated.clean.fa")
    san = _eval_sanitize_fasta(in_fasta, clean_fa, int(min_len), int(max_len))

    scored = _score_fasta_with_expert_discs(
        clean_fa,
        out / "expert_disc",
        thr=thr,
        disc_root=disc_root,
        species_id=int(species_id),
    )
    disc_rows = list(scored.get("rows") or [])
    by_id = {str(row.get("id", "")): row for row in disc_rows}
    gp_summary = _prob_summary([float(row["p_gp"]) for row in disc_rows], thr)
    gn_summary = _prob_summary([float(row["p_gn"]) for row in disc_rows], thr)
    fu_summary = _prob_summary([float(row["p_fu"]) for row in disc_rows], thr)
    sf_summary = _prob_summary([float(row["p_sf"]) for row in disc_rows], thr)
    mic_bins = [int(row["mic_bin"]) for row in disc_rows if "mic_bin" in row]
    mic_summary = {
        "n": float(len(mic_bins)),
        "mean_bin": float(sum(mic_bins) / len(mic_bins)) if mic_bins else 0.0,
        "bin_counts": [float(mic_bins.count(b)) for b in range(5)],
    }

    clean_items = _eval_read_fasta(clean_fa)
    length_pairs_all, length_pairs_by_combo = _collect_length_pairs_from_items(clean_items)
    length_control = _length_control_stats(length_pairs_all)
    length_control_by_target = _length_control_by_target(length_pairs_all)
    length_control_by_tier = _length_control_by_tier(length_pairs_all)

    combo_stats: Dict[str, Dict[str, float]] = {}
    attr_stats: Dict[str, Dict[str, float]] = {
        "gram_pos": {"n_common": 0.0, "n_ok": 0.0},
        "gram_neg": {"n_common": 0.0, "n_ok": 0.0},
        "fungus": {"n_common": 0.0, "n_ok": 0.0},
        "safety": {"n_common": 0.0, "n_ok": 0.0},
        "length": {"n_common": 0.0, "n_ok": 0.0},
    }
    for pid, seq in clean_items:
        combo_full = _eval_parse_combo(pid)
        combo = _eval_attr_bits(combo_full)
        target_len = _eval_parse_target_length(pid)
        sid = _eval_norm_id(pid)
        if len(combo) != 4:
            continue
        rec = by_id.get(sid)
        if rec is None:
            continue
        if any(math.isnan(float(rec[k])) for k in ("p_gp", "p_gn", "p_fu", "p_sf")):
            continue

        def _parse_bit(ch: str, idx: int) -> Tuple[bool, Optional[int]]:
            c = str(combo[idx]).upper()
            if c == "X":
                return True, None
            return False, int(c)

        _, gv = _parse_bit(combo, 0)
        ok_gp = _eval_sat(float(rec["p_gp"]), gv, thr) if gv is not None else True
        _, gv = _parse_bit(combo, 1)
        ok_gn = _eval_sat(float(rec["p_gn"]), gv, thr) if gv is not None else True
        _, gv = _parse_bit(combo, 2)
        ok_fu = _eval_sat(float(rec["p_fu"]), gv, thr) if gv is not None else True
        _, tv = _parse_bit(combo, 3)
        ok_sf = _eval_sat(float(rec["p_sf"]), tv, thr) if tv is not None else True
        if target_len is not None:
            actual_len = len(str(seq))
            tol = max(0, int(length_tolerance))
            ok_len = (
                int(actual_len) == int(target_len)
                if tol <= 0
                else abs(int(actual_len) - int(target_len)) <= tol
            )
        else:
            ok_len = True
        st = combo_stats.setdefault(combo_full or combo, {"n_common": 0.0, "n_all_ok": 0.0})
        st["n_common"] += 1.0
        st["n_all_ok"] += 1.0 if (ok_gp and ok_gn and ok_fu and ok_sf and ok_len) else 0.0
        attr_stats["gram_pos"]["n_common"] += 1.0
        attr_stats["gram_pos"]["n_ok"] += 1.0 if ok_gp else 0.0
        attr_stats["gram_neg"]["n_common"] += 1.0
        attr_stats["gram_neg"]["n_ok"] += 1.0 if ok_gn else 0.0
        attr_stats["fungus"]["n_common"] += 1.0
        attr_stats["fungus"]["n_ok"] += 1.0 if ok_fu else 0.0
        attr_stats["safety"]["n_common"] += 1.0
        attr_stats["safety"]["n_ok"] += 1.0 if ok_sf else 0.0
        attr_stats["length"]["n_common"] += 1.0
        attr_stats["length"]["n_ok"] += 1.0 if ok_len else 0.0

    per_combo: Dict[str, Any] = {}
    vals: List[float] = []
    for c, st in combo_stats.items():
        adh = float(st["n_all_ok"]) / float(max(1.0, st["n_common"]))
        combo_len_stats = _length_control_stats(length_pairs_by_combo.get(c, []))
        per_combo[c] = {
            "adherence": adh,
            "n_common": st["n_common"],
            "length_control_variance": combo_len_stats["length_control_variance"],
            "length_control_std": combo_len_stats["length_control_std"],
            "length_control_mae": combo_len_stats["length_control_mae"],
            "length_control_exact_rate": combo_len_stats["length_control_exact_rate"],
        }
        vals.append(adh)

    per_attr_satisfaction: Dict[str, Dict[str, float]] = {}
    attr_sats: List[float] = []
    for attr, st in attr_stats.items():
        sat = float(st["n_ok"]) / float(max(1.0, st["n_common"]))
        per_attr_satisfaction[attr] = {"satisfaction": sat, "n_common": float(st["n_common"])}
        attr_sats.append(sat)
    attr_mean_satisfaction = float(sum(attr_sats) / max(1, len(attr_sats))) if attr_sats else 0.0

    strict_adherence_mean = float(sum(vals) / max(1, len(vals))) if vals else 0.0
    payload: Dict[str, Any] = {
        "clean_total": float(san["total"]),
        "clean_kept": float(san["kept"]),
        "clean_dropped": float(san["dropped"]),
        "scorer": "expert_discriminators",
        "disc_n": float(len(disc_rows)),
        "discriminators": {
            "gp": gp_summary,
            "gn": gn_summary,
            "fu": fu_summary,
            "sf": sf_summary,
            "mic": mic_summary,
        },
        "combo_adherence_mean": strict_adherence_mean,
        "combo_adherence_min": float(min(vals)) if vals else 0.0,
        "strict_adherence_mean": strict_adherence_mean,
        "attr_mean_satisfaction": attr_mean_satisfaction,
        "length_control": length_control,
        "length_control_variance": length_control["length_control_variance"],
        "length_control_std": length_control["length_control_std"],
        "length_control_mae": length_control["length_control_mae"],
        "length_control_rmse": length_control["length_control_rmse"],
        "length_control_mean_dev": length_control["length_control_mean_dev"],
        "length_control_exact_rate": length_control["length_control_exact_rate"],
        "length_control_by_target": length_control_by_target,
        "length_control_by_tier": length_control_by_tier,
        "per_combo": per_combo,
        "per_attr_satisfaction": per_attr_satisfaction,
    }
    (out / "eval_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def _maybe_run_in_fasta_cli() -> bool:
    """若 argv 含 --in_fasta，则只跑打分并退出。"""
    if "--in_fasta" not in sys.argv:
        return False
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--in_fasta", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--disc_root", default="")
    p.add_argument("--disc_thr", type=float, default=0.5)
    p.add_argument("--species_id", type=int, default=0, help="MIC 的 SPxx，0 对应 SP00")
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--length_tolerance", type=int, default=0, help="自然生成时长度容差（AA）；0=严格 exact")
    args, _unk = p.parse_known_args()
    run_control_expert_fasta_eval(
        args.in_fasta,
        args.out_dir,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        length_tolerance=int(args.length_tolerance),
        disc_root=str(args.disc_root),
        disc_thr=float(args.disc_thr),
        species_id=int(args.species_id),
    )
    return True


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LIB = _load_module_from_path(
    "five_attr_classification_lora_lib_eval_protocol_07",
    Path(__file__).resolve().parent / "07.five_attr_classification_lora_lib.py",
)
COMBOS_14 = _LIB.COMBOS_14
REGULAR_CONDITIONS_6 = _LIB.REGULAR_CONDITIONS_6
DEFAULT_EVAL_LENGTHS = _LIB.DEFAULT_EVAL_LENGTHS
LENGTH_TIERS = _LIB.LENGTH_TIERS
combos_to_jsonable = _LIB.combos_to_jsonable
expand_conditions_with_lengths = _LIB.expand_conditions_with_lengths
generate_sequences_for_combos = _LIB.generate_sequences_for_combos
_FT = _load_module_from_path(
    "five_attr_finetune_lib_eval_protocol",
    Path(__file__).resolve().parent / "07.five_attr_finetune_lib.py",
)


def load_single_lora_model(model_dir: str, tokenizer_dir: str = "", device_str: str = "cuda"):
    """协议评估同时覆盖 LoRA 与全参微调 checkpoint。"""
    tok_dir = str(tokenizer_dir).strip()
    if not tok_dir:
        tok_dir = str(Path(__file__).resolve().parents[2] / "models" / "AMPGuide")
    return _FT.load_five_attr_model(model_dir, tokenizer_dir=tok_dir, device_str=device_str)


length_tier_for_target = _LIB.length_tier_for_target
parse_attr_weights = _LIB.parse_attr_weights
parse_combo_text = _LIB.parse_combo_text
parse_length_list = _LIB.parse_length_list


def _expand_eval_targets(targets: Sequence[str], lengths: Sequence[int]) -> List[str]:
    out: List[str] = []
    for t in targets:
        s = str(t).strip().upper()
        if ":" in s:
            out.append(s)
        elif len(s) == 4:
            out.extend(expand_conditions_with_lengths([s], [int(x) for x in lengths if int(x) > 0]))
        else:
            out.append(s)
    return sorted(dict.fromkeys(out))


def _weight_list_from_args(args: argparse.Namespace) -> List[float]:
    w = parse_attr_weights(str(args.attr_weights))
    return [w["GP1"], w["GN1"], w["FU1"], w["SF0"], w["LENGTH"]]


def _discover_checkpoints(root: Path, pattern: str) -> List[Path]:
    ckpts = [p for p in root.glob(pattern) if p.is_dir()]

    def _sort_key(p: Path) -> Tuple[int, str]:
        name = p.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-", 1)[1]), name
            except Exception:
                pass
        return 10**18, name

    return sorted(ckpts, key=_sort_key)


def _length_summary(lengths: Sequence[int]) -> Dict[str, float]:
    vals = [int(x) for x in lengths]
    if not vals:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    svals = sorted(vals)
    p90_idx = min(len(svals) - 1, max(0, int(round(0.9 * (len(svals) - 1)))))
    return {
        "min": float(svals[0]),
        "median": float(statistics.median(svals)),
        "mean": float(sum(svals) / len(svals)),
        "p90": float(svals[p90_idx]),
        "max": float(svals[-1]),
    }


def _length_control_stats(pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """pairs: (actual_len, target_len). 返回长度控制偏差统计，含方差。"""
    if not pairs:
        return {
            "n": 0.0,
            "length_control_mean_dev": 0.0,
            "length_control_variance": 0.0,
            "length_control_std": 0.0,
            "length_control_mae": 0.0,
            "length_control_rmse": 0.0,
            "length_control_exact_rate": 0.0,
        }
    devs = [float(int(a) - int(t)) for a, t in pairs]
    n = float(len(devs))
    mean_dev = float(sum(devs) / n)
    var = float(sum((d - mean_dev) ** 2 for d in devs) / n)
    abs_devs = [abs(d) for d in devs]
    return {
        "n": n,
        "length_control_mean_dev": mean_dev,
        "length_control_variance": var,
        "length_control_std": float(math.sqrt(var)),
        "length_control_mae": float(sum(abs_devs) / n),
        "length_control_rmse": float(math.sqrt(sum(d * d for d in devs) / n)),
        "length_control_exact_rate": float(sum(1 for d in abs_devs if d == 0.0) / n),
    }


def _length_control_by_target(pairs: Sequence[Tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
    by_target: Dict[int, List[Tuple[int, int]]] = {}
    for actual, target in pairs:
        by_target.setdefault(int(target), []).append((int(actual), int(target)))
    out: Dict[str, Dict[str, Any]] = {}
    for target in sorted(by_target):
        tier_id, tier_label = length_tier_for_target(target)
        stats = _length_control_stats(by_target[target])
        out[str(target)] = {
            **stats,
            "target_length": float(target),
            "tier_id": tier_id,
            "tier_label": tier_label,
        }
    return out


def _length_control_by_tier(pairs: Sequence[Tuple[int, int]]) -> Dict[str, Dict[str, Any]]:
    tier_pairs: Dict[str, List[Tuple[int, int]]] = {tier_id: [] for tier_id, _, _, _ in LENGTH_TIERS}
    for actual, target in pairs:
        tier_id, _ = length_tier_for_target(int(target))
        if tier_id in tier_pairs:
            tier_pairs[tier_id].append((int(actual), int(target)))
    out: Dict[str, Dict[str, Any]] = {}
    for tier_id, tier_label, lo, hi in LENGTH_TIERS:
        stats = _length_control_stats(tier_pairs.get(tier_id, []))
        out[tier_id] = {
            "tier_label": tier_label,
            "target_range": [lo, hi],
            **stats,
        }
    return out


def _collect_length_pairs_from_items(
    items: Sequence[Tuple[str, str]],
) -> Tuple[List[Tuple[int, int]], Dict[str, List[Tuple[int, int]]]]:
    all_pairs: List[Tuple[int, int]] = []
    per_combo: Dict[str, List[Tuple[int, int]]] = {}
    for pid, seq in items:
        target_len = _eval_parse_target_length(pid)
        if target_len is None:
            continue
        actual = int(len(str(seq)))
        combo_full = _eval_parse_combo(pid) or "unknown"
        pair = (actual, int(target_len))
        all_pairs.append(pair)
        per_combo.setdefault(combo_full, []).append(pair)
    return all_pairs, per_combo


def _scalar_triplet_from_eval_payload(d: Dict[str, Any], nested_key: str) -> Tuple[float, float, float]:
    nested = d.get(nested_key) if isinstance(d.get(nested_key), dict) else None
    if nested:
        if nested_key == "regular6":
            mean = float(
                nested.get(
                    "condition_adherence_mean",
                    nested.get("combo_adherence_mean", nested.get("strict_adherence_mean", 0.0)),
                )
            )
            mn = float(
                nested.get(
                    "condition_adherence_min",
                    nested.get("combo_adherence_min", 0.0),
                )
            )
        else:
            mean = float(
                nested.get(
                    "combo_adherence_mean",
                    nested.get("strict_adherence_mean", 0.0),
                )
            )
            mn = float(nested.get("combo_adherence_min", 0.0))
        attr = float(
            nested.get(
                "five_attr_mean_satisfaction",
                nested.get("attr_mean_satisfaction", 0.0),
            )
        )
        return mean, mn, attr
    mean = float(d.get("combo_adherence_mean", d.get("strict_adherence_mean", 0.0)))
    mn = float(d.get("combo_adherence_min", 0.0))
    attr = float(d.get("attr_mean_satisfaction", d.get("five_attr_mean_satisfaction", 0.0)))
    return mean, mn, attr


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint_step(name: str) -> int:
    if str(name).startswith("checkpoint-"):
        try:
            return int(str(name).split("-", 1)[1])
        except Exception:
            return -1
    return -1


def _run_generation(
    checkpoint_dir: Path,
    tokenizer_dir: str,
    out_dir: Path,
    targets: Sequence[str],
    n_per_target: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model, tok, device = load_single_lora_model(str(checkpoint_dir), str(tokenizer_dir), device_str=str(args.device))
    fasta_path = out_dir / "generated.fa"
    stats = generate_sequences_for_combos(
        model=model,
        tok=tok,
        device=device,
        out_fasta=str(fasta_path),
        combos=targets,
        n_per_combo=int(n_per_target),
        min_seq_len=int(args.min_seq_len),
        max_seq_len=int(args.max_seq_len),
        min_new_tokens=int(args.min_new_tokens),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        restrict_aa_tokens=(int(args.restrict_aa_tokens) == 1),
        do_sample=(int(args.do_sample) == 1),
        shuffle_combos=(int(args.shuffle_combos) == 1),
        seed=int(args.seed),
        max_attempts_multiplier=int(args.max_attempts_multiplier),
        attr_weights=_weight_list_from_args(args),
        enforce_exact_length=(int(getattr(args, "enforce_exact_length", 0)) == 1),
        default_length=int(getattr(args, "default_length", 24)),
        return_stats=True,
    )
    length_stats = _length_summary(stats.get("lengths", []))
    gen_pairs, _ = _collect_length_pairs_from_items(_eval_read_fasta(str(fasta_path)))
    gen_length_control = _length_control_stats(gen_pairs)
    return {
        "fasta_path": str(fasta_path),
        "attempts": int(stats.get("attempts", 0)),
        "written": int(stats.get("written", 0)),
        "target": int(stats.get("target", 0)),
        "accept_rate": float(stats.get("accept_rate", 0.0)),
        "unique_sequences": int(stats.get("unique_sequences", 0)),
        "unique_ratio": float(stats.get("unique_ratio", 0.0)),
        "reject_too_short": int(stats.get("reject_too_short", 0)),
        "reject_invalid": int(stats.get("reject_invalid", 0)),
        "reject_len_out_of_range": int(stats.get("reject_len_out_of_range", 0)),
        "reject_len_mismatch": int(stats.get("reject_len_mismatch", 0)),
        "length_min": length_stats["min"],
        "length_median": length_stats["median"],
        "length_mean": length_stats["mean"],
        "length_p90": length_stats["p90"],
        "length_max": length_stats["max"],
        "length_control_variance": gen_length_control["length_control_variance"],
        "length_control_std": gen_length_control["length_control_std"],
        "length_control_mae": gen_length_control["length_control_mae"],
        "length_control_rmse": gen_length_control["length_control_rmse"],
        "length_control_mean_dev": gen_length_control["length_control_mean_dev"],
        "length_control_exact_rate": gen_length_control["length_control_exact_rate"],
        "written_per_target": stats.get("written_per_combo", {}),
        "attempts_per_target": stats.get("attempts_per_combo", {}),
    }


def _run_downstream_eval(fasta_path: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """与训练期 ComboEvalCallback 一致：eval_payload.json 写在 out_dir 根目录（非 downstream_eval 子目录）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_control_expert_fasta_eval(
        str(fasta_path),
        str(out_dir),
        min_len=int(args.eval_min_len),
        max_len=int(args.eval_max_len),
        disc_root=str(getattr(args, "disc_root", "")),
        disc_thr=float(getattr(args, "disc_thr", 0.5)),
        species_id=int(getattr(args, "species_id", 0)),
    )


def _protocol_payload(args: argparse.Namespace, ckpts: Sequence[Path], tokenizer_dir: str) -> Dict[str, Any]:
    return {
        "checkpoints_root": str(args.checkpoints_root),
        "tokenizer_dir": str(tokenizer_dir),
        "checkpoint_pattern": str(args.checkpoint_pattern),
        "checkpoint_names": [p.name for p in ckpts],
        "regular_conditions_6": REGULAR_CONDITIONS_6,
        "strict_combos_14": COMBOS_14,
        "eval_lengths": [int(x) for x in str(getattr(args, "eval_lengths", str(args.default_length))).split(",") if str(x).strip()],
        "default_length": int(args.default_length),
        "enforce_exact_length": int(args.enforce_exact_length),
        "regular_n_per_target": int(args.regular_n_per_target),
        "strict_n_per_target": int(args.strict_n_per_target),
        "generation": {
            "min_seq_len": int(args.min_seq_len),
            "max_seq_len": int(args.max_seq_len),
            "min_new_tokens": int(args.min_new_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "repetition_penalty": float(args.repetition_penalty),
            "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
            "restrict_aa_tokens": int(args.restrict_aa_tokens),
            "do_sample": int(args.do_sample),
            "shuffle_combos": int(args.shuffle_combos),
            "seed": int(args.seed),
            "max_attempts_multiplier": int(args.max_attempts_multiplier),
            "attr_weights": parse_attr_weights(str(args.attr_weights)),
            "device": str(args.device),
            "disc_root": str(getattr(args, "disc_root", "")),
            "disc_thr": float(getattr(args, "disc_thr", 0.5)),
            "species_id": int(getattr(args, "species_id", 0)),
        },
    }


def _row_from_protocol(
    checkpoint_dir: Path,
    generation: Dict[str, Any],
    downstream: Dict[str, Any],
) -> Dict[str, Any]:
    gen = generation if isinstance(generation, dict) else {}
    row = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_name": checkpoint_dir.name,
        "step": _checkpoint_step(checkpoint_dir.name),
        "fasta_path": gen.get("fasta_path", ""),
        "attempts": int(gen.get("attempts", 0)),
        "written": int(gen.get("written", 0)),
        "target": int(gen.get("target", 0)),
        "accept_rate": float(gen.get("accept_rate", 0.0)),
        "unique_sequences": int(gen.get("unique_sequences", 0)),
        "unique_ratio": float(gen.get("unique_ratio", 0.0)),
        "reject_too_short": int(gen.get("reject_too_short", 0)),
        "reject_invalid": int(gen.get("reject_invalid", 0)),
        "reject_len_out_of_range": int(gen.get("reject_len_out_of_range", 0)),
        "length_min": float(gen.get("length_min", 0.0)),
        "length_median": float(gen.get("length_median", 0.0)),
        "length_mean": float(gen.get("length_mean", 0.0)),
        "length_p90": float(gen.get("length_p90", 0.0)),
        "length_max": float(gen.get("length_max", 0.0)),
        "length_control_variance": float(downstream.get("length_control_variance", gen.get("length_control_variance", 0.0))),
        "length_control_std": float(downstream.get("length_control_std", gen.get("length_control_std", 0.0))),
        "length_control_mae": float(downstream.get("length_control_mae", gen.get("length_control_mae", 0.0))),
        "length_control_rmse": float(downstream.get("length_control_rmse", gen.get("length_control_rmse", 0.0))),
        "length_control_exact_rate": float(downstream.get("length_control_exact_rate", gen.get("length_control_exact_rate", 0.0))),
        "clean_total": float(downstream.get("clean_total", 0.0)),
        "clean_kept": float(downstream.get("clean_kept", 0.0)),
        "disc_gp_pos_rate": float((((downstream.get("discriminators") or {}).get("gp") or {}).get("pos_rate", 0.0))),
        "disc_gn_pos_rate": float((((downstream.get("discriminators") or {}).get("gn") or {}).get("pos_rate", 0.0))),
        "disc_fu_pos_rate": float((((downstream.get("discriminators") or {}).get("fu") or {}).get("pos_rate", 0.0))),
        "disc_sf_pos_rate": float((((downstream.get("discriminators") or {}).get("sf") or {}).get("pos_rate", 0.0))),
        "disc_mic_mean_bin": float((((downstream.get("discriminators") or {}).get("mic") or {}).get("mean_bin", 0.0))),
    }
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "checkpoint_name",
        "step",
        "regular6_condition_adherence_mean",
        "regular6_condition_adherence_min",
        "regular6_five_attr_mean_satisfaction",
        "regular6_length_control_variance",
        "strict14_combo_adherence_mean",
        "strict14_combo_adherence_min",
        "strict14_five_attr_mean_satisfaction",
        "strict14_length_control_variance",
        "clean_kept_regular6",
        "clean_kept_strict14",
        "attempts_regular6",
        "attempts_strict14",
        "accept_rate_regular6",
        "accept_rate_strict14",
        "checkpoint_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    if _maybe_run_in_fasta_cli():
        return

    p = argparse.ArgumentParser("07.eval_checkpoint_protocol_five_attr_single_lora.py")
    p.add_argument("--checkpoints_root", required=True)
    p.add_argument("--tokenizer_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint_pattern", default="checkpoint-*")
    p.add_argument("--regular_n_per_target", type=int, default=100)
    p.add_argument("--strict_n_per_target", type=int, default=100)
    p.add_argument("--min_seq_len", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--min_new_tokens", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=2)
    p.add_argument("--restrict_aa_tokens", type=int, default=1)
    p.add_argument("--do_sample", type=int, default=0)
    p.add_argument("--shuffle_combos", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_attempts_multiplier", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attr_weights", type=str, default="1,1,1,1,1")
    p.add_argument("--default_length", type=int, default=24)
    p.add_argument("--eval_lengths", type=str, default="", help="逗号分隔评估长度；空则使用 default_length")
    p.add_argument("--enforce_exact_length", type=int, choices=[0], default=0)
    p.add_argument(
        "--combo_eval_script",
        default="",
        help="保留兼容 run 脚本传参；打分已内联本文件，此参数不生效。",
    )
    p.add_argument("--disc_root", default="", help="五个专家骨干判别器目录，默认 models/discriminators")
    p.add_argument("--disc_thr", type=float, default=0.5)
    p.add_argument("--species_id", type=int, default=0, help="MIC 的 SPxx，0 对应 SP00")
    p.add_argument("--eval_min_len", type=int, default=1)
    p.add_argument("--eval_max_len", type=int, default=256)
    p.add_argument("--eval_timeout", type=int, default=3600)
    p.add_argument("--eval_regular6_only", action="store_true", help="协议评估仅跑 regular6，跳过 strict14")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    ckpt_root = Path(str(a.checkpoints_root))
    out_root = Path(str(a.out_dir))
    tok_dir = str(a.tokenizer_dir).strip() or str(ckpt_root)
    ckpts = _discover_checkpoints(ckpt_root, str(a.checkpoint_pattern))
    if int(a.limit) > 0:
        ckpts = ckpts[: int(a.limit)]
    if not ckpts:
        raise RuntimeError(f"no checkpoints matched under {ckpt_root} with pattern {a.checkpoint_pattern}")

    eval_lengths = parse_length_list(str(a.eval_lengths), single_fallback=int(a.default_length))
    regular_targets = _expand_eval_targets(REGULAR_CONDITIONS_6, eval_lengths)
    strict_targets = [] if bool(a.eval_regular6_only) else _expand_eval_targets(COMBOS_14, eval_lengths)
    print(
        f"[INFO] eval_lengths={eval_lengths} -> regular6={len(regular_targets)}"
        + ("" if bool(a.eval_regular6_only) else f" strict14={len(strict_targets)}")
        + " targets",
        flush=True,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "protocol.json", _protocol_payload(a, ckpts, tok_dir))

    rows: List[Dict[str, Any]] = []
    for idx, ckpt in enumerate(ckpts, start=1):
        ckpt_out = out_root / ckpt.name
        ckpt_out.mkdir(parents=True, exist_ok=True)
        summary_path = ckpt_out / "protocol_summary.json"
        if summary_path.exists() and not bool(a.overwrite):
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print(f"[SKIP] {idx}/{len(ckpts)} {ckpt.name} -> existing summary reused", flush=True)
            continue

        print(f"[EVAL] {idx}/{len(ckpts)} {ckpt.name} regular6", flush=True)
        regular_dir = ckpt_out / "regular6"
        regular_dir.mkdir(parents=True, exist_ok=True)
        regular_gen = _run_generation(ckpt, tok_dir, regular_dir, regular_targets, int(a.regular_n_per_target), a)
        regular_eval = _run_downstream_eval(Path(regular_gen["fasta_path"]), regular_dir, a)

        strict_gen: Dict[str, Any] = {}
        strict_eval: Dict[str, Any] = {}
        if not bool(a.eval_regular6_only):
            print(f"[EVAL] {idx}/{len(ckpts)} {ckpt.name} strict14", flush=True)
            strict_dir = ckpt_out / "strict14"
            strict_dir.mkdir(parents=True, exist_ok=True)
            strict_gen = _run_generation(ckpt, tok_dir, strict_dir, strict_targets, int(a.strict_n_per_target), a)
            strict_eval = _run_downstream_eval(Path(strict_gen["fasta_path"]), strict_dir, a)
        else:
            print(f"[EVAL] {idx}/{len(ckpts)} {ckpt.name} skip strict14 (eval_regular6_only)", flush=True)

        r_mean, r_min, r_attr = _scalar_triplet_from_eval_payload(regular_eval, "regular6")
        if strict_eval:
            s_mean, s_min, s_attr = _scalar_triplet_from_eval_payload(strict_eval, "strict14")
        else:
            s_mean, s_min, s_attr = 0.0, 0.0, 0.0

        row = {
            "checkpoint_dir": str(ckpt),
            "checkpoint_name": ckpt.name,
            "step": _checkpoint_step(ckpt.name),
            "regular6": {
                "generation": regular_gen,
                "evaluation": regular_eval,
            },
            "strict14": {
                "generation": strict_gen,
                "evaluation": strict_eval,
            },
            "regular6_condition_adherence_mean": r_mean,
            "regular6_condition_adherence_min": r_min,
            "regular6_five_attr_mean_satisfaction": r_attr,
            "regular6_length_control_variance": float(regular_eval.get("length_control_variance", regular_gen.get("length_control_variance", 0.0))),
            "strict14_combo_adherence_mean": s_mean,
            "strict14_combo_adherence_min": s_min,
            "strict14_five_attr_mean_satisfaction": s_attr,
            "strict14_length_control_variance": float(strict_eval.get("length_control_variance", strict_gen.get("length_control_variance", 0.0))),
            "clean_kept_regular6": float(regular_eval.get("clean_kept", 0.0)),
            "clean_kept_strict14": float(strict_eval.get("clean_kept", 0.0)),
            "attempts_regular6": int(regular_gen.get("attempts", 0)),
            "attempts_strict14": int(strict_gen.get("attempts", 0)),
            "accept_rate_regular6": float(regular_gen.get("accept_rate", 0.0)),
            "accept_rate_strict14": float(strict_gen.get("accept_rate", 0.0)),
            "regular6_summary": _row_from_protocol(ckpt, regular_gen, regular_eval),
            "strict14_summary": _row_from_protocol(ckpt, strict_gen, strict_eval),
        }
        _write_json(summary_path, row)
        rows.append(row)
        print(
            f"[DONE] {ckpt.name} regular_mean={row['regular6_condition_adherence_mean']:.4f} "
            f"strict_mean={row['strict14_combo_adherence_mean']:.4f} "
            f"len_var_r6={row['regular6_length_control_variance']:.4f} "
            f"len_var_s14={row['strict14_length_control_variance']:.4f}",
            flush=True,
        )
        for tier_id, tier_label, _, _ in LENGTH_TIERS:
            r_tier = (regular_eval.get("length_control_by_tier") or {}).get(tier_id, {})
            s_tier = (strict_eval.get("length_control_by_tier") or {}).get(tier_id, {})
            if float(r_tier.get("n", 0.0)) <= 0 and float(s_tier.get("n", 0.0)) <= 0:
                continue
            print(
                f"       [{tier_label}] r6: mae={float(r_tier.get('length_control_mae', 0.0)):.2f} "
                f"exact={float(r_tier.get('length_control_exact_rate', 0.0)):.3f} | "
                f"s14: mae={float(s_tier.get('length_control_mae', 0.0)):.2f} "
                f"exact={float(s_tier.get('length_control_exact_rate', 0.0)):.3f}",
                flush=True,
            )

    rows = sorted(rows, key=lambda x: (int(x.get("step", -1)), str(x.get("checkpoint_name", ""))))
    _write_json(out_root / "summary.json", {"rows": rows})
    _write_csv(out_root / "summary.csv", rows)
    print(f"[OK] summary_json={out_root / 'summary.json'}", flush=True)
    print(f"[OK] summary_csv={out_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
