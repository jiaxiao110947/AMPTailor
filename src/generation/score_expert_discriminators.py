#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score peptides with the five expert-backbone discriminators.

Heads: gram-positive, gram-negative, antifungal, safety, and MIC.
The AMP vs non-AMP head is not part of this release; its backbone was the
length base model, not the expert.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
COMBO_RX = re.compile(r"combo=([01X]{4})(?::(\d+))?", re.IGNORECASE)
LEN_RX = re.compile(r"LEN=(\d+)", re.IGNORECASE)
HEADS = {
    "gp": "gram_positive",
    "gn": "gram_negative",
    "fu": "antifungal",
    "sf": "safety",
}


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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_fasta(path: str) -> List[Tuple[str, str]]:
    items, pid, seq = [], "", ""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for ln in handle:
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


def parse_combo(pid: str) -> Tuple[str, Optional[int]]:
    match = COMBO_RX.search(pid)
    if match:
        bits = match.group(1).upper()
        tlen = int(match.group(2)) if match.group(2) else None
        if tlen is None:
            length_match = LEN_RX.search(pid)
            tlen = int(length_match.group(1)) if length_match else None
        return bits, tlen
    length_match = LEN_RX.search(pid)
    return "1110", int(length_match.group(1)) if length_match else None


def sat(prob: float, want: int, thr: float) -> bool:
    return float(prob) >= float(thr) if int(want) == 1 else float(prob) < float(thr)


def encode_aa(tok: GPT2TokenizerFast, seq: str) -> List[int]:
    ids = []
    unk = tok.unk_token_id
    for ch in seq:
        if ch not in AA20:
            continue
        i = tok.convert_tokens_to_ids(ch)
        if i is not None and int(i) >= 0 and i != unk:
            ids.append(int(i))
    return ids


class SeqDS(Dataset):
    def __init__(self, seqs: List[str], tok: GPT2TokenizerFast, species_id: Optional[int] = None):
        self.ids = []
        for seq in seqs:
            ids = encode_aa(tok, seq)
            if species_id is not None:
                sp = tok.convert_tokens_to_ids(f"SP{int(species_id):02d}")
                ids = [int(sp)] + ids
            self.ids.append(ids or [int(tok.pad_token_id)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"ids": self.ids[i], "label": 0}


class LeftPad:
    def __init__(self, pad_id: int, max_length: int):
        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, batch):
        raw = [x["ids"][: self.max_length] or [self.pad_id] for x in batch]
        width = max(len(s) for s in raw)
        input_ids, attn = [], []
        for s in raw:
            pad = width - len(s)
            input_ids.append([self.pad_id] * pad + s)
            attn.append([0] * pad + [1] * len(s))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        }


def make_loader(seqs, tok, pad_id, max_length, bs, workers, species_id: Optional[int] = None):
    return DataLoader(
        SeqDS(seqs, tok, species_id=species_id),
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=LeftPad(pad_id, max_length),
    )


def load_binary(backbone: str, ckpt: str, device):
    disc_dir = Path(__file__).resolve().parents[1] / "discriminators"
    v2 = _load("disc_v2_score", disc_dir / "train_six_attr_discriminators_v2.py")
    model = v2.LoraDiscriminator(backbone, 2, 0.0, 32, 64, 0.0)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def load_mic(backbone: str, tok_dir: str, ckpt: str, device):
    disc_dir = Path(__file__).resolve().parents[1] / "discriminators"
    hybrid = _load("disc_hybrid_score", disc_dir / "train_six_attr_discriminators_hybrid.py")
    tok = GPT2TokenizerFast.from_pretrained(tok_dir, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    encoder = hybrid.build_mic_encoder(backbone, tok)
    sp_start = int(tok.convert_tokens_to_ids("SP00"))
    model = hybrid.MicLoraDisc(encoder, 0.0, 32, 64, 0.0, sp_start)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.eval()
    return model.to(device), tok


@torch.no_grad()
def predict_p1(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def predict_mic(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 5), dtype=np.float32)


def adherence_from_rows(rows: List[dict], thr: float) -> dict:
    combo_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "ok": 0.0})
    attr = {k: {"n": 0.0, "ok": 0.0} for k in ("gp", "gn", "fu", "sf", "length")}
    n_all = 0
    for row in rows:
        bits, tlen = row["bits"], row["target_len"]
        if not bits or len(bits) != 4:
            continue
        ok_gp = True if bits[0] == "X" else sat(row["p_gp"], int(bits[0]), thr)
        ok_gn = True if bits[1] == "X" else sat(row["p_gn"], int(bits[1]), thr)
        ok_fu = True if bits[2] == "X" else sat(row["p_fu"], int(bits[2]), thr)
        ok_sf = True if bits[3] == "X" else sat(row["p_sf"], int(bits[3]), thr)
        ok_len = True if tlen is None else int(row["length"]) == int(tlen)
        key = f"{bits}:{tlen}" if tlen is not None else bits
        combo_stats[key]["n"] += 1
        combo_stats[key]["ok"] += 1.0 if (ok_gp and ok_gn and ok_fu and ok_sf and ok_len) else 0.0
        for name, requested, ok in (
            ("gp", bits[0] != "X", ok_gp), ("gn", bits[1] != "X", ok_gn),
            ("fu", bits[2] != "X", ok_fu), ("sf", bits[3] != "X", ok_sf),
            ("length", tlen is not None, ok_len),
        ):
            if requested:
                attr[name]["n"] += 1
                attr[name]["ok"] += 1.0 if ok else 0.0
        n_all += 1
        row["ok_attr"] = bool(ok_gp and ok_gn and ok_fu and ok_sf)
        row["ok_combo"] = bool(ok_gp and ok_gn and ok_fu and ok_sf and ok_len)
    adhs = [st["ok"] / max(1.0, st["n"]) for st in combo_stats.values()]
    per_attr = {k: v["ok"] / v["n"] for k, v in attr.items() if v["n"] > 0}
    per_attr_n = {k: int(v["n"]) for k, v in attr.items()}
    return {
        "n": n_all,
        "combo_adherence_mean": float(sum(adhs) / max(1, len(adhs))) if adhs else 0.0,
        "combo_adherence_min": float(min(adhs)) if adhs else 0.0,
        "attr_mean_satisfaction": float(sum(per_attr.values()) / max(1, len(per_attr))),
        "per_attr": per_attr,
        "per_attr_n": per_attr_n,
        "amp_pos_rate": 0.0,
        "precise_amp_attr_rate": 0.0,
        "per_combo": {k: {"n": v["n"], "adherence": v["ok"] / max(1.0, v["n"])} for k, v in combo_stats.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a FASTA with expert-backbone discriminators")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--disc_root", default="")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--species_id", type=int, default=0, help="MIC atomic species token SP00-SP19")
    parser.add_argument("--skip_mic", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    disc_root = Path(args.disc_root) if args.disc_root else root / "models" / "discriminators"
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = read_fasta(args.fasta)
    seqs = [seq for _, seq in items]
    if not seqs:
        raise SystemExit(f"no sequences in {args.fasta}")

    probs: Dict[str, np.ndarray] = {}
    for name, folder in HEADS.items():
        head = disc_root / folder
        tok = GPT2TokenizerFast.from_pretrained(str(head / "tokenizer"), local_files_only=True)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        model = load_binary(str(head / "backbone"), str(head / "weights" / "best.pt"), device)
        loader = make_loader(seqs, tok, int(tok.pad_token_id), args.max_length, args.bs, args.num_workers)
        probs[name] = predict_p1(model, loader, device)
        print(f"[score] {name} mean={float(probs[name].mean()):.4f}", flush=True)
        del model

    mic_prob = None
    if not args.skip_mic:
        mic_head = disc_root / "mic"
        model, tok = load_mic(
            str(mic_head / "backbone"),
            str(mic_head / "tokenizer"),
            str(mic_head / "weights" / "best.pt"),
            device,
        )
        loader = make_loader(
            seqs,
            tok,
            int(tok.pad_token_id),
            args.max_length,
            args.bs,
            args.num_workers,
            species_id=int(args.species_id),
        )
        mic_prob = predict_mic(model, loader, device)
        print(f"[score] mic argmax_mean={float(mic_prob.argmax(1).mean()):.3f}", flush=True)

    rows = []
    for i, (pid, seq) in enumerate(items):
        bits, tlen = parse_combo(pid)
        row = {
            "id": pid.split("|", 1)[0],
            "header": pid,
            "sequence": seq,
            "length": len(seq),
            "bits": bits,
            "target_len": tlen,
            "p_gp": float(probs["gp"][i]),
            "p_gn": float(probs["gn"][i]),
            "p_fu": float(probs["fu"][i]),
            "p_sf": float(probs["sf"][i]),
        }
        if mic_prob is not None:
            row["mic_bin"] = int(mic_prob[i].argmax())
            for b in range(5):
                row[f"p_mic{b}"] = float(mic_prob[i, b])
        rows.append(row)
    stats = adherence_from_rows(rows, float(args.thr))
    (out_dir / "scores.json").write_text(json.dumps({"rows": rows, "summary": stats}, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] n={stats['n']} attr_mean={stats['attr_mean_satisfaction']:.3f} out={out_dir / 'scores.json'}", flush=True)


if __name__ == "__main__":
    main()
