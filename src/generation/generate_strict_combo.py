#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate with the strict-combo adapter using one joint training prefix.

Example prompt for GP=1, GN=1, FU=1, SF=0, length 18:
    GP1+GN1+FU1+SF0+LEN18:
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


@torch.no_grad()
def generate_until_eos(model, tok, device, prompt: str, target_len: int, n: int, *, temperature, top_p, repetition_penalty, lib, use_bf16, max_new_tokens=256):
    token_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    batch = int(n)
    prefix = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
    generated = torch.empty((batch, 0), dtype=torch.long, device=device)
    id_to_aa, aa_ids = lib._build_aa_char_maps(tok)
    aa_ids = aa_ids.to(device)
    if int(max_new_tokens) <= 0 or batch <= 0:
        raise ValueError("max_new_tokens and n must be positive")
    eos_id = int(tok.eos_token_id)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(int(max_new_tokens)):
        inp = torch.cat([prefix, generated], dim=1) if generated.numel() else prefix
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inp).logits[:, -1, :].float()
        else:
            logits = model(input_ids=inp).logits[:, -1, :].float()
        logits = logits / max(1e-6, float(temperature))
        if float(repetition_penalty) != 1.0 and generated.numel() > 0:
            penalty = float(repetition_penalty)
            for i in range(batch):
                uniq = torch.unique(generated[i])
                pos = logits[i, uniq] > 0
                logits[i, uniq] = torch.where(pos, logits[i, uniq] / penalty, logits[i, uniq] * penalty)
        masked = torch.full_like(logits, -float("inf"))
        masked[:, aa_ids] = logits[:, aa_ids]
        logits = lib._top_k_top_p_filtering(masked, top_k=0, top_p=float(top_p))
        logits[finished] = -float("inf")
        logits[finished, eos_id] = 0.0
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
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
    return seqs


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-combo joint-prompt generation")
    parser.add_argument("--bits", default="1110")
    parser.add_argument("--length", type=int, default=18)
    parser.add_argument("--n", type=int, default=8)
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
    bits = str(args.bits).strip()
    if len(bits) != 4 or any(ch not in "01" for ch in bits):
        raise SystemExit("--bits must be 4 characters of 0/1, for example 1110")

    root = _repo_root()
    lib_dir = Path(__file__).resolve().parents[1] / "multi_attr"
    lib = _load("five_attr_lib_combo", lib_dir / "07.five_attr_classification_lora_lib.py")
    combo = _load("strict_combo_lib_gen", lib_dir / "07.six_attr_strict_combo_lib.py")
    adapter = args.adapter or str(root / "models" / "six_attr_strict_combo")
    tokenizer = args.tokenizer or str(root / "models" / "AMPGuide")
    prompt = combo.make_strict_combo_prompt(f"COMBO:{bits}:{int(args.length)}")
    out_fasta = Path(args.out_fasta) if args.out_fasta else root / "outputs" / "generate" / f"strict_combo_L{args.length}.fasta"
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, device = lib.load_single_lora_model(adapter, tokenizer, device_name)
    seqs = generate_until_eos(
        model,
        tok,
        device,
        prompt,
        int(args.length),
        int(args.n),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        lib=lib,
        use_bf16=bool(args.bf16),
        max_new_tokens=int(args.max_new_tokens),
    )
    with out_fasta.open("w", encoding="utf-8") as handle:
        for i, seq in enumerate(seqs):
            handle.write(
                f">seq_{i:04d}|combo={bits}:{int(args.length)}|prompt={prompt}|scheme=strict_combo_joint_prompt"
                f"|GP={bits[0]}|GN={bits[1]}|FU={bits[2]}|SF={bits[3]}|LEN={int(args.length)}\n{seq}\n"
            )
    print(f"[OK] prompt={prompt} n={len(seqs)} out={out_fasta}", flush=True)


if __name__ == "__main__":
    main()
