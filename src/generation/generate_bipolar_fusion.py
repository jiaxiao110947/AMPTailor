#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate with the bipolar six-attribute adapter by fusing 0/1 prefixes.

contrast: z = z_LEN + mean_a w_a (z_target - alpha * z_opposite)
mix: equal-weight mean of wanted prefixes, plus alpha times the contrast residual

This is the fusion used in the September 2026 bipolar-versus-combo runs.
The released adapter is checkpoint-5000.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo_root() -> Path:
    import os

    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "paths.env").is_file():
            return p
    return here.parents[2]


def _fwd(model, inp, use_bf16: bool):
    if use_bf16 and inp.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(input_ids=inp).logits[:, -1, :].float()
    return model(input_ids=inp).logits[:, -1, :].float()


def _unique_tasks(bits: str, length: int):
    gp, gn, fu, sf = [int(x) for x in str(bits).zfill(4)]
    wanted = [f"GP{gp}", f"GN{gn}", f"FU{fu}", f"SF{sf}", f"LEN{int(length)}"]
    opposite = [f"GP{1 - gp}", f"GN{1 - gn}", f"FU{1 - fu}", f"SF{1 - sf}"]
    ordered = []
    for task in wanted + opposite:
        if task not in ordered:
            ordered.append(task)
    return wanted, opposite, ordered


@torch.no_grad()
def generate_batch(model, tok, device, combo, n, *, kind, alpha, attr_weights, temperature, top_p, repetition_penalty, lib, bip, use_bf16, max_new_tokens=256):
    bits, length = lib.parse_condition(combo, default_length=None)
    wanted, opposite, tasks = _unique_tasks(bits, length)
    weights = bip.parse_attr_weights(attr_weights)
    weight_map = {task: float(bip.task_weight(task, weights)) for task in tasks}
    prompt_ids = {task: tok(bip.make_task_prompt(task), add_special_tokens=False)["input_ids"] for task in tasks}
    batch = int(n)
    generated = torch.empty((batch, 0), dtype=torch.long, device=device)
    len_task = f"LEN{int(length)}"
    content = [task for task in wanted if not str(task).startswith("LEN")]
    id_to_aa, aa_ids = lib._build_aa_char_maps(tok)
    aa_ids = aa_ids.to(device)
    if int(max_new_tokens) <= 0 or batch <= 0:
        raise ValueError("max_new_tokens and n must be positive")
    eos_id = int(tok.eos_token_id)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(int(max_new_tokens)):
        logits = {}
        for task, token_ids in prompt_ids.items():
            prefix = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
            inp = torch.cat([prefix, generated], dim=1) if generated.numel() else prefix
            logits[task] = _fwd(model, inp, use_bf16)
        z_len = logits[len_task]
        if kind == "contrast":
            acc = None
            wsum = 0.0
            for task, opp in zip(content, opposite):
                w = weight_map[task]
                delta = logits[task] - float(alpha) * logits[opp]
                acc = delta * w if acc is None else acc + delta * w
                wsum += w
            fused = z_len + acc / max(wsum, 1e-6)
        elif kind == "mix":
            acc_m = z_len * weight_map[len_task]
            wsum_m = weight_map[len_task]
            acc_c = None
            wsum_c = 0.0
            for task, opp in zip(content, opposite):
                w = weight_map[task]
                acc_m = acc_m + logits[task] * w
                wsum_m += w
                delta = logits[task] - logits[opp]
                acc_c = delta * w if acc_c is None else acc_c + delta * w
                wsum_c += w
            fused = acc_m / max(wsum_m, 1e-6) + float(alpha) * (acc_c / max(wsum_c, 1e-6))
        else:
            raise ValueError(f"unknown kind: {kind}")
        z = fused / max(1e-6, float(temperature))
        if float(repetition_penalty) != 1.0 and generated.numel() > 0:
            penalty = float(repetition_penalty)
            for i in range(batch):
                uniq = torch.unique(generated[i])
                pos = z[i, uniq] > 0
                z[i, uniq] = torch.where(pos, z[i, uniq] / penalty, z[i, uniq] * penalty)
        masked = torch.full_like(z, -float("inf"))
        masked[:, aa_ids] = z[:, aa_ids]
        z = lib._top_k_top_p_filtering(masked, top_k=0, top_p=float(top_p))
        z[finished] = -float("inf")
        z[finished, eos_id] = 0.0
        nxt = torch.multinomial(torch.softmax(z, dim=-1), num_samples=1)
        finished |= nxt.squeeze(-1).eq(eos_id)
        generated = torch.cat([generated, nxt], dim=1)
        if bool(finished.all()):
            break
    if not bool(finished.all()):
        print(f"[WARN] {int((~finished).sum().item())} sequences reached max_new_tokens without EOS", flush=True)
    seqs = []
    for i in range(batch):
        ids = generated[i].tolist()
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        seqs.append("".join(id_to_aa[int(x)] for x in ids if int(x) in id_to_aa))
    return bits, int(length), seqs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bipolar prefix fusion generation")
    parser.add_argument("--bits", default="1110", help="GP GN FU SF, each 0 or 1")
    parser.add_argument("--length", type=int, default=18)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--kind", choices=("contrast", "mix"), default="contrast")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--attr_weights", default="1,1,1,1,1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Safety cap; target length is controlled by the prompt and EOS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--out_fasta", default="")
    args = parser.parse_args()

    root = _repo_root()
    lib_dir = Path(__file__).resolve().parents[1] / "multi_attr"
    lib = _load("five_attr_lib_gen", lib_dir / "07.five_attr_classification_lora_lib.py")
    bip = _load("bipolar_lib_gen", lib_dir / "07.six_attr_bipolar_prefix_lib.py")
    adapter = args.adapter or str(root / "models" / "six_attr_bipolar")
    tokenizer = args.tokenizer or str(root / "models" / "AMPGuide")
    out_fasta = Path(args.out_fasta) if args.out_fasta else root / "outputs" / "generate" / f"bipolar_{args.kind}_L{args.length}.fasta"
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    model, tok, device = lib.load_single_lora_model(adapter, tokenizer, "cuda" if torch.cuda.is_available() else "cpu")
    bits, length, seqs = generate_batch(
        model,
        tok,
        device,
        f"{args.bits}:{int(args.length)}",
        int(args.n),
        kind=args.kind,
        alpha=float(args.alpha),
        attr_weights=args.attr_weights,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        lib=lib,
        bip=bip,
        use_bf16=bool(args.bf16),
        max_new_tokens=int(args.max_new_tokens),
    )
    with out_fasta.open("w", encoding="utf-8") as handle:
        for i, seq in enumerate(seqs):
            handle.write(
                f">seq_{i:04d}|combo={bits}:{length}|scheme=bipolar_{args.kind}|alpha={args.alpha}"
                f"|GP={bits[0]}|GN={bits[1]}|FU={bits[2]}|SF={bits[3]}|LEN={length}\n{seq}\n"
            )
    print(f"[OK] n={len(seqs)} out={out_fasta}", flush=True)


if __name__ == "__main__":
    main()
