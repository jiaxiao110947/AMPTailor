#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Length-conditioned generation eval: per-target-length precision and variance."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers.trainer_callback import TrainerCallback

from amp_generate_lib import extract_peptide_after_length_prompt, generate_fasta_with_length_targets
from eval_throttle import should_run_heavy_eval

LEN_HEADER_RX = re.compile(r"LEN=(\d+)", re.IGNORECASE)
DEFAULT_EVAL_LENGTHS = [6, 8, 10, 12, 16, 20, 24, 28, 32, 40]


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default))


def _parse_lengths(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _try_wandb_log(payload: Dict[str, Any], step: int) -> None:
    try:
        import wandb

        payload = dict(payload)
        payload["trainer/global_step"] = int(step)
        wandb.log(payload, commit=True)
    except Exception:
        pass


def _length_metrics(actual: List[int], target: int) -> Dict[str, float]:
    n = len(actual)
    if n <= 0:
        return {
            "n": 0.0,
            "target": float(target),
            "mean_len": 0.0,
            "variance": 0.0,
            "precision": 0.0,
            "mae": 0.0,
        }
    mean_len = sum(actual) / float(n)
    var = sum((x - mean_len) ** 2 for x in actual) / float(n)
    exact = sum(1 for x in actual if int(x) == int(target))
    mae = sum(abs(int(x) - int(target)) for x in actual) / float(n)
    return {
        "n": float(n),
        "target": float(target),
        "mean_len": float(mean_len),
        "variance": float(var),
        "precision": float(exact / n),
        "mae": float(mae),
    }


def _read_lengths_from_fasta(fasta_path: str) -> List[Tuple[int, int]]:
    """Return list of (target_len, actual_len)."""
    rows: List[Tuple[int, int]] = []
    target: Optional[int] = None
    with open(fasta_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s.startswith(">"):
                target = None
                m = LEN_HEADER_RX.search(s)
                if m:
                    target = int(m.group(1))
                continue
            prompt = f"LEN{target}:" if target is not None else ""
            if target is None:
                m2 = re.match(r"^LEN(\d+):", s.upper())
                if m2:
                    target = int(m2.group(1))
                    prompt = m2.group(0)
            seq = extract_peptide_after_length_prompt(s, prompt)
            if target is not None and seq:
                rows.append((int(target), len(seq)))
    return rows


class LengthEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, enabled: bool = True):
        self.enabled = bool(enabled)
        self.tok = tokenizer
        self.last_step = -1

    def on_evaluate(self, args, state, control, metrics: Optional[Dict[str, float]] = None, **kwargs):
        if not self.enabled:
            return
        step = int(getattr(state, "global_step", 0))
        if step <= 0 or step == self.last_step:
            return
        self.last_step = step
        eval_steps = int(getattr(args, "eval_steps", 0) or 0)
        if not should_run_heavy_eval(step, eval_steps):
            print(
                f"[LENGTH_EVAL] skip step={step} (heavy eval throttle, "
                f"HEAVY_EVAL_EVERY_N={os.environ.get('HEAVY_EVAL_EVERY_N', '1')})",
                flush=True,
            )
            return

        model = kwargs.get("model", None)
        tok = kwargs.get("tokenizer", None) or self.tok
        if model is None or tok is None:
            print("[LENGTH_EVAL] missing model/tokenizer; skip", flush=True)
            return

        target_lengths = _parse_lengths(
            _env("LENGTH_EVAL_LENGTHS", ",".join(str(x) for x in DEFAULT_EVAL_LENGTHS))
        )
        n_per = int(_env("LENGTH_EVAL_N_PER_LEN", "10"))
        seed = int(_env("LENGTH_EVAL_SEED", _env("AMPEP_EVAL_SEED", "42")))
        max_new = int(_env("LENGTH_EVAL_MAX_NEW", _env("AMPEP_GEN_MAX_NEW", "128")))
        temp = float(_env("LENGTH_EVAL_TEMP", _env("AMPEP_GEN_TEMP", "1.0")))
        top_p = float(_env("LENGTH_EVAL_TOP_P", _env("AMPEP_GEN_TOP_P", "0.95")))
        top_k = int(_env("LENGTH_EVAL_TOP_K", _env("AMPEP_GEN_TOP_K", "0")))

        out_dir = Path(str(args.output_dir))
        eval_dir = out_dir / "eval_stage" / f"step_{step:08d}" / "05_length_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = str(eval_dir / "gen.length_cond.fasta")

        generate_fasta_with_length_targets(
            model=model,
            tok=tok,
            out_fasta=fasta_path,
            target_lengths=target_lengths,
            n_per_length=n_per,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            seed=seed + step,
        )

        pairs = _read_lengths_from_fasta(fasta_path)
        by_target: Dict[int, List[int]] = {L: [] for L in target_lengths}
        for tgt, act in pairs:
            if tgt in by_target:
                by_target[tgt].append(int(act))

        per_len: Dict[str, Dict[str, float]] = {}
        all_actual: List[int] = []
        all_target: List[int] = []
        payload: Dict[str, Any] = {}

        for L in target_lengths:
            actual = by_target.get(L, [])
            m = _length_metrics(actual, L)
            per_len[str(L)] = m
            all_actual.extend(actual)
            all_target.extend([L] * len(actual))
            payload[f"length/L{L}/precision"] = m["precision"]
            payload[f"length/L{L}/variance"] = m["variance"]
            payload[f"length/L{L}/mean_len"] = m["mean_len"]
            payload[f"length/L{L}/mae"] = m["mae"]

        if all_actual:
            overall_mean = sum(all_actual) / float(len(all_actual))
            overall_var = sum((x - overall_mean) ** 2 for x in all_actual) / float(len(all_actual))
            overall_exact = sum(
                1 for a, t in zip(all_actual, all_target) if int(a) == int(t)
            ) / float(len(all_actual))
            overall_mae = sum(abs(int(a) - int(t)) for a, t in zip(all_actual, all_target)) / float(
                len(all_actual)
            )
        else:
            overall_mean = overall_var = overall_exact = overall_mae = 0.0

        summary = {
            "step": int(step),
            "target_lengths": target_lengths,
            "n_per_length": int(n_per),
            "n_generated": int(len(pairs)),
            "per_length": per_len,
            "overall": {
                "mean_len": float(overall_mean),
                "variance": float(overall_var),
                "precision": float(overall_exact),
                "mae": float(overall_mae),
            },
        }
        summary_path = eval_dir / "length_eval.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        payload["length/overall/precision"] = float(overall_exact)
        payload["length/overall/variance"] = float(overall_var)
        payload["length/overall/mae"] = float(overall_mae)
        payload["length/overall/mean_len"] = float(overall_mean)
        payload["length/n_generated"] = float(len(pairs))

        print(
            f"[LENGTH_EVAL] step={step} n={len(pairs)} "
            f"precision={overall_exact:.3f} variance={overall_var:.3f} mae={overall_mae:.3f} "
            f"-> {summary_path}",
            flush=True,
        )
        _try_wandb_log(payload, step)
