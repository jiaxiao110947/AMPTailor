#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate from a six-attribute finetune checkpoint and score it.

The checkpoint may be LoRA r16, LoRA r64, full fine-tuning, freeze-top2, or BitFit.
Attribute scores come from the five expert-backbone discriminators shipped in
models/discriminators. Length control is measured on LEN{n}: prompts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

def _repo_root() -> Path:
    import os

    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / "paths.env").is_file():
            return parent
    return here.parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _length_summary(fasta: Path, lengths: list[int]) -> dict:
    header_rx = re.compile(r"LEN=(\d+)", re.IGNORECASE)
    by = {int(x): [] for x in lengths}
    with fasta.open(encoding="utf-8") as handle:
        target = None
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                match = header_rx.search(text)
                target = int(match.group(1)) if match else None
                continue
            if target in by:
                by[target].append(len(re.sub(r"[^A-Za-z]", "", text)))
    per = {}
    actual_all = []
    target_all = []
    for length in lengths:
        got = by.get(int(length), [])
        n = len(got)
        if n == 0:
            per[str(length)] = {"n": 0, "precision": 0.0, "mae": 0.0}
            continue
        exact = sum(1 for x in got if int(x) == int(length))
        mae = sum(abs(int(x) - int(length)) for x in got) / float(n)
        per[str(length)] = {"n": n, "precision": exact / float(n), "mae": mae}
        actual_all.extend(got)
        target_all.extend([int(length)] * n)
    if actual_all:
        overall_exact = sum(1 for a, t in zip(actual_all, target_all) if int(a) == int(t)) / float(len(actual_all))
        overall_mae = sum(abs(int(a) - int(t)) for a, t in zip(actual_all, target_all)) / float(len(actual_all))
    else:
        overall_exact = overall_mae = 0.0
    return {
        "n": len(actual_all),
        "precision": overall_exact,
        "mae": overall_mae,
        "per_length": per,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and score a six-attr finetune checkpoint")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_dir", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lengths", default="6,8,10,12,16,20,24,28,32,40")
    parser.add_argument("--n_per_combo", type=int, default=10)
    parser.add_argument("--n_per_length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_regular6", action="store_true")
    parser.add_argument("--skip_length", action="store_true")
    parser.add_argument("--skip_score", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    multi = root / "src" / "multi_attr"
    lib = _load("five_attr_lib_eval", multi / "07.five_attr_classification_lora_lib.py")
    ft = _load("five_attr_finetune_eval", multi / "07.five_attr_finetune_lib.py")
    amp = _load("amp_generate_lib_eval", root / "src" / "pretrain" / "amp_generate_lib.py")
    protocol = _load(
        "finetune_eval_protocol",
        multi / "07.eval_checkpoint_protocol_five_attr_single_lora.py",
    )

    tok_dir = args.tokenizer_dir.strip() or str(root / "models" / "AMPGuide")
    model, tok, device = ft.load_five_attr_model(args.model_dir, tokenizer_dir=tok_dir, device_str=args.device)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model_dir": args.model_dir, "lengths": lengths}

    if not args.skip_regular6:
        combos = lib.expand_conditions_with_lengths(lib.REGULAR_CONDITIONS_6, lengths)
        fasta = out / "regular6.fa"
        lib.generate_sequences_for_combos(
            model=model,
            tok=tok,
            device=device,
            out_fasta=str(fasta),
            combos=combos,
            n_per_combo=int(args.n_per_combo),
            min_seq_len=1,
            max_seq_len=256,
            min_new_tokens=1,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=0,
            restrict_aa_tokens=True,
            seed=int(args.seed),
            max_attempts_multiplier=20,
            attr_weights=[1, 1, 1, 1, 1],
            enforce_exact_length=True,
            default_length=lengths[0],
            prompt_mode="fused",
        )
        report["regular6_fasta"] = str(fasta)
        if not args.skip_score:
            payload = protocol.run_control_expert_fasta_eval(str(fasta), str(out / "regular6_score"))
            report["regular6_score"] = {
                "combo_adherence_mean": payload["combo_adherence_mean"],
                "combo_adherence_min": payload["combo_adherence_min"],
                "attr_mean_satisfaction": payload["attr_mean_satisfaction"],
                "per_attr_satisfaction": payload["per_attr_satisfaction"],
                "discriminators": payload["discriminators"],
            }

    if not args.skip_length:
        fasta = out / "length.fa"
        amp.generate_fasta_with_length_targets(
            model=model,
            tok=tok,
            out_fasta=str(fasta),
            target_lengths=lengths,
            n_per_length=int(args.n_per_length),
            max_new_tokens=128,
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            seed=int(args.seed),
        )
        report["length"] = _length_summary(fasta, lengths)

    (out / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "regular6_score"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
