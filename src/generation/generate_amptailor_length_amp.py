#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMPTailor tri-model fusion with length-conditioned prompts (LEN{n}:)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPTS = Path(__file__).resolve().parent.parent / "pretrain"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from amp_generate_lib import extract_peptide_after_length_prompt, make_length_prompt  # noqa: E402

from generate_amptailor_amp import (  # noqa: E402
    AMPTailorGeneratorAAOnly,
    set_seed,
    summarize_seqs,
)

DEFAULT_EVAL_LENGTHS = [6, 8, 10, 12, 16, 20, 24, 28, 32, 40]


def _parse_lengths(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _length_metrics_multi(
    targets: List[int], actuals: List[int]
) -> Dict[str, object]:
    n = len(actuals)
    if n <= 0:
        return {
            "n": 0,
            "overall_mae": 0.0,
            "overall_precision": 0.0,
            "per_length": {},
        }
    mae = sum(abs(int(a) - int(t)) for a, t in zip(actuals, targets)) / float(n)
    exact = sum(1 for a, t in zip(actuals, targets) if int(a) == int(t)) / float(n)
    per: Dict[str, Dict[str, float]] = {}
    by_tgt: Dict[int, List[int]] = {}
    for t, a in zip(targets, actuals):
        by_tgt.setdefault(int(t), []).append(int(a))
    for L, acts in sorted(by_tgt.items()):
        nn = len(acts)
        per[str(L)] = {
            "n": float(nn),
            "target": float(L),
            "mean_len": float(sum(acts) / nn),
            "mae": float(sum(abs(x - L) for x in acts) / nn),
            "precision": float(sum(1 for x in acts if x == L) / nn),
        }
    return {
        "n": int(n),
        "overall_mae": float(mae),
        "overall_precision": float(exact),
        "per_length": per,
    }


def generate_length_cond(
    gen: AMPTailorGeneratorAAOnly,
    *,
    target_length: int,
    nsamples: int,
    max_new_tokens: int,
    alpha: float,
    alpha_eos: Optional[float],
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
    keep_eos_after_filter: bool,
    repetition_penalty: float,
    rep_window: int,
    max_homopolymer_run: int,
    no_repeat_ngram_size: int,
) -> Tuple[List[str], List[int]]:
    """Return (seqs, target_lens per seq). Length via LEN{n}: + EOS; min_new_tokens=0."""
    L = int(target_length)
    prompt = make_length_prompt(L)
    max_new = int(max_new_tokens)

    seqs: List[str] = []
    targets: List[int] = []
    got = 0
    while got < int(nsamples):
        cur = min(int(batch_size), int(nsamples) - got)
        outs = gen.generate_batch(
            prompts=[prompt] * int(cur),
            max_new_tokens=max_new,
            min_new_tokens=0,
            alpha=float(alpha),
            alpha_eos=alpha_eos,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            keep_eos_after_filter=bool(keep_eos_after_filter),
            repetition_penalty=float(repetition_penalty),
            rep_window=int(rep_window),
            max_homopolymer_run=int(max_homopolymer_run),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
        for raw in outs:
            pep = extract_peptide_after_length_prompt(raw, prompt)
            if not pep:
                continue
            seqs.append(pep)
            targets.append(L)
            got += 1
            if got >= int(nsamples):
                break
    print(f"[len] target={L} collected={got}/{nsamples}", flush=True)
    return seqs, targets


def generate_multi_lengths(
    gen: AMPTailorGeneratorAAOnly,
    *,
    target_lengths: List[int],
    n_per_length: int,
    max_new_tokens: int,
    **kwargs,
) -> Tuple[List[str], List[int]]:
    all_seqs: List[str] = []
    all_targets: List[int] = []
    for L in target_lengths:
        seqs, tgts = generate_length_cond(
            gen, target_length=L, nsamples=int(n_per_length), max_new_tokens=max_new_tokens, **kwargs
        )
        all_seqs.extend(seqs)
        all_targets.extend(tgts)
    return all_seqs, all_targets


def write_length_fasta(
    path: Path, seqs: List[str], targets: List[int], header_prefix: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (s, t) in enumerate(zip(seqs, targets), 1):
            f.write(f">{header_prefix}_{i}|LEN={int(t)}|prompt=LEN{int(t)}:\n{s}\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser("AMPTailor length-conditioned AMP generator")
    ap.add_argument("--base_model_dir", required=True, type=str)
    ap.add_argument("--expert_model_dir", required=True, type=str)
    ap.add_argument("--anti_model_dir", required=True, type=str)
    ap.add_argument("--tokenizer_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--header_prefix", default="dexp_len", type=str)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target_length", type=int, help="single target length")
    g.add_argument(
        "--target_lengths",
        type=str,
        help=f"comma-separated lengths (eval default: {','.join(map(str, DEFAULT_EVAL_LENGTHS))})",
    )
    ap.add_argument("--nsamples", type=int, default=100, help="per length when using --target_lengths")
    ap.add_argument("--n_per_length", type=int, default=None, help="alias of nsamples for multi-length")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--keep_eos_after_filter", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alpha_eos", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--rep_window", type=int, default=64)
    ap.add_argument("--max_homopolymer_run", type=int, default=0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--out_fasta_name",
        type=str,
        default="",
        help="output fasta basename (default auto)",
    )
    args = ap.parse_args()

    n_per = int(args.n_per_length) if args.n_per_length is not None else int(args.nsamples)
    if args.target_lengths:
        lengths = _parse_lengths(args.target_lengths)
        multi = True
    else:
        lengths = [int(args.target_length)]
        multi = len(lengths) > 1 or False
        if not multi:
            n_per = int(args.nsamples)

    if not lengths or any(L <= 0 for L in lengths):
        raise ValueError("target length(s) must be positive")

    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = AMPTailorGeneratorAAOnly(
        base_dir=args.base_model_dir,
        expert_dir=args.expert_model_dir,
        anti_dir=args.anti_model_dir,
        tok_dir=args.tokenizer_dir,
        seed=int(args.seed),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
    )

    gen_kw = dict(
        max_new_tokens=int(args.max_new_tokens),
        alpha=float(args.alpha),
        alpha_eos=float(args.alpha_eos) if args.alpha_eos is not None else None,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        batch_size=int(args.batch_size),
        keep_eos_after_filter=bool(args.keep_eos_after_filter),
        repetition_penalty=float(args.repetition_penalty),
        rep_window=int(args.rep_window),
        max_homopolymer_run=int(args.max_homopolymer_run),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
    )

    if args.target_lengths or len(lengths) > 1:
        seqs, targets = generate_multi_lengths(
            gen, target_lengths=lengths, n_per_length=n_per, **gen_kw
        )
    else:
        seqs, targets = generate_length_cond(
            gen, target_length=lengths[0], nsamples=n_per, **gen_kw
        )

    actuals = [len(s) for s in seqs]
    n_total = len(seqs)
    ae = float(args.alpha_eos) if args.alpha_eos is not None else float(args.alpha)

    if args.out_fasta_name:
        out_fa = out_dir / args.out_fasta_name
    elif args.target_lengths:
        len_tag = "-".join(str(x) for x in lengths)
        tag = f"L{len_tag}_npl{n_per}_nt{n_total}_a{args.alpha}_t{args.temperature}_k{args.top_k}"
        out_fa = out_dir / f"amptailor_len_{tag}.fasta"
    else:
        L = lengths[0]
        tag = f"L{L}_n{n_per}_a{args.alpha}_ae{ae}_t{args.temperature}_p{args.top_p}_k{args.top_k}"
        out_fa = out_dir / f"amptailor_len_{tag}.fasta"

    write_length_fasta(out_fa, seqs, targets, str(args.header_prefix))

    st = summarize_seqs(seqs)
    lm = _length_metrics_multi(targets, actuals)
    st["length_control"] = {
        "target_lengths": lengths,
        "n_per_length": int(n_per),
        "min_new_tokens": 0,
        "max_new_tokens": int(args.max_new_tokens),
        **lm,
    }

    out_json = out_dir / f"{out_fa.stem}.summary.json"
    out_json.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] out_fasta={out_fa}", flush=True)
    print(f"[OK] out_summary_json={out_json}", flush=True)
    print(
        f"[OK] n={n_total} length_mae={lm['overall_mae']:.3f} "
        f"precision={lm['overall_precision']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
