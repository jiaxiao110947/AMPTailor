#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain V2/V3 with generator vocab, one amino-acid = one token (no BPE merges)."""
from __future__ import annotations

import sys as _eos_sys
from pathlib import Path as _EosPath
_eos_src = str(_EosPath(__file__).resolve().parents[1])
if _eos_src not in _eos_sys.path:
    _eos_sys.path.insert(0, _eos_src)
from eos_compat import load_eos_tokenizer


def _repo_root():
    import os
    from pathlib import Path
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "paths.env").is_file():
            return p
    return here.parents[2]


_ROOT = _repo_root()

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import GPT2TokenizerFast, get_cosine_schedule_with_warmup

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


v1 = _load("disc_v1", SCRIPTS / "train_six_attr_discriminators.py")
v2 = _load("disc_v2", SCRIPTS / "train_six_attr_discriminators_v2.py")
v3 = _load("disc_v3", SCRIPTS / "train_six_attr_discriminators_v3.py")

ALL_TASKS = v1.ALL_TASKS


def encode_gen_chars(tok: GPT2TokenizerFast, text: str) -> List[int]:
    """Map each character to generator vocab id. No BPE merge."""
    ids: List[int] = []
    unk = tok.unk_token_id
    for ch in text:
        i = tok.convert_tokens_to_ids(ch)
        if i is None or i < 0 or i == unk:
            continue
        ids.append(int(i))
    return ids


def peptide_text(seq: str, species_id, use_sp_prefix: bool) -> str:
    if use_sp_prefix:
        sid = v2.valid_label(species_id)
        if sid is not None:
            return f"SP{int(sid):02d}:{seq}"
    return seq


class IdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok, use_sp_prefix: bool, with_v3_feats: bool, max_aa: int):
        self.labels = df["label"].astype(int).tolist()
        self.seqs = df["sequence"].tolist()
        self.ids = [encode_gen_chars(tok, peptide_text(seq, df.iloc[i].get("species_id", None) if "species_id" in df.columns else None, use_sp_prefix)) for i, seq in enumerate(self.seqs)]
        self.with_v3_feats = with_v3_feats
        self.max_aa = max_aa
        if with_v3_feats:
            if "species_id" in df.columns:
                self.sids = [20 if v2.valid_label(v) is None else int(v2.valid_label(v)) for v in df["species_id"].tolist()]
            else:
                self.sids = [20] * len(self.seqs)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {"ids": self.ids[i], "label": self.labels[i], "seq": self.seqs[i]}
        if self.with_v3_feats:
            item["aa"] = v3.aa_ids(self.seqs[i], self.max_aa)
            item["phys"] = v3.physchem(self.seqs[i])
            item["sid"] = self.sids[i]
        return item


class LeftPadCollator:
    def __init__(self, pad_id: int, max_length: int, with_v3_feats: bool, max_aa: int):
        self.pad_id = pad_id
        self.max_length = max_length
        self.with_v3_feats = with_v3_feats
        self.max_aa = max_aa

    def __call__(self, batch: list) -> dict:
        raw = [x["ids"][: self.max_length] or [self.pad_id] for x in batch]
        L = max(len(s) for s in raw)
        input_ids, attn = [], []
        for s in raw:
            pad = L - len(s)
            input_ids.append([self.pad_id] * pad + s)
            attn.append([0] * pad + [1] * len(s))
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        }
        if self.with_v3_feats:
            aa = torch.full((len(batch), self.max_aa), v3.PAD_AA, dtype=torch.long)
            aa_mask = torch.zeros((len(batch), self.max_aa), dtype=torch.float32)
            for i, x in enumerate(batch):
                n = min(len(x["aa"]), self.max_aa)
                if n:
                    aa[i, :n] = torch.tensor(x["aa"][:n], dtype=torch.long)
                    aa_mask[i, :n] = 1.0
            out["aa_ids"] = aa
            out["aa_mask"] = aa_mask
            out["phys"] = torch.tensor(np.stack([x["phys"] for x in batch]), dtype=torch.float32)
            out["species_ids"] = torch.tensor([x["sid"] for x in batch], dtype=torch.long)
        return out


def make_loader(df, tok, pad_id, max_length, max_aa, bs, shuffle, workers, use_sp_prefix, with_v3_feats, balance: bool):
    ds = IdDataset(df, tok, use_sp_prefix, with_v3_feats, max_aa)
    sampler = None
    if balance and shuffle:
        y = np.array(ds.labels)
        freq = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
        freq = np.maximum(freq, 1.0)
        w = 1.0 / freq[y]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=LeftPadCollator(pad_id, max_length, with_v3_feats, max_aa),
    )


@torch.no_grad()
def eval_v2(model, loader, device, task):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    return v1.metrics_from_store({task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}, (task,))


@torch.no_grad()
def eval_v3(model, loader, device, task):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            aa_ids=batch["aa_ids"].to(device),
            aa_mask=batch["aa_mask"].to(device),
            phys=batch["phys"].to(device),
            species_ids=batch["species_ids"].to(device),
        )
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    return v1.metrics_from_store({task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}, (task,))


def train_v2_task(task, train_df, val_df, test_df, tok, pad_id, backbone, out_dir, device, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    model = v2.LoraDiscriminator(backbone, n_labels, args.v2_dropout, args.v2_lora_r, args.v2_lora_alpha, args.v2_lora_dropout).to(device)
    w = v2.class_weight(train_df["label"].astype(int).tolist(), n_labels, device)
    use_sp = task == "MIC"
    kw = dict(tok=tok, pad_id=pad_id, max_length=args.max_length, max_aa=args.max_aa, workers=args.num_workers, use_sp_prefix=use_sp, with_v3_feats=False, balance=False)
    train_loader = make_loader(train_df, bs=args.v2_train_bs, shuffle=True, **kw)
    val_loader = make_loader(val_df, bs=args.eval_bs, shuffle=False, **{**kw, "balance": False})
    test_loader = make_loader(test_df, bs=args.eval_bs, shuffle=False, **{**kw, "balance": False})
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.v2_lora_lr}, {"params": head_params, "lr": args.v2_head_lr}], weight_decay=args.weight_decay)
    total = max(1, len(train_loader)) * args.v2_epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best, patience, hist, t0 = -1e9, args.v2_patience, [], time.time()
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.v2_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                    loss = F.cross_entropy(logits, y, weight=w)
            else:
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            n += 1
        val = eval_v2(model, val_loader, device, task)
        score = float(val["_mean_primary"])
        print(f"[aa-v2 {task}] epoch {epoch}/{args.v2_epochs} loss={running/max(1,n):.4f} val={score:.4f}", flush=True)
        hist.append({"epoch": epoch, "train_loss": running / max(1, n), "val_mean_primary": score})
        if score > best + 1e-4:
            best = score
            patience = args.v2_patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "scheme": "aa_v2"}, best_path)
        else:
            patience -= 1
            if patience <= 0:
                print(f"[aa-v2 {task}] early stop", flush=True)
                break
    model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    summary = {"scheme": "aa_v2", "task": task, "seconds": time.time() - t0, "best_epoch": torch.load(best_path, map_location="cpu")["epoch"], "val": eval_v2(model, val_loader, device, task), "test": eval_v2(model, test_loader, device, task), "history": hist}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def train_v3_task(task, train_df, val_df, test_df, tok, pad_id, backbone, out_dir, device, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    use_species = task == "MIC"
    model = v3.V3Discriminator(backbone, n_labels, use_species, args.v3_dropout, args.v3_lora_r, args.v3_lora_alpha, args.v3_lora_dropout, args.v3_unfreeze_last).to(device)
    ytr = train_df["label"].astype(int).tolist()
    counts = np.bincount(np.array(ytr), minlength=n_labels).astype(np.float64)
    w = torch.tensor(counts.sum() / (n_labels * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    use_sp = False  # peptide-only on GPT-2; species via embedding
    balance = task in ("GP", "GN")
    base = dict(tok=tok, pad_id=pad_id, max_length=args.max_length, max_aa=args.max_aa, workers=args.num_workers, use_sp_prefix=use_sp, with_v3_feats=True)
    train_loader = make_loader(train_df, bs=args.v3_train_bs, shuffle=True, balance=balance, **base)
    val_loader = make_loader(val_df, bs=args.eval_bs, shuffle=False, balance=False, **base)
    test_loader = make_loader(test_df, bs=args.eval_bs, shuffle=False, balance=False, **base)
    lora_params, full_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder.") and "lora_" in n:
            lora_params.append(p)
        elif n.startswith("encoder."):
            full_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.v3_lora_lr})
    if full_params:
        groups.append({"params": full_params, "lr": args.v3_full_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.v3_head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.v3_weight_decay)
    total = max(1, len(train_loader)) * args.v3_epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best, patience, hist, t0 = -1e9, args.v3_patience, [], time.time()
    best_path = out_dir / "best.pt"
    gamma = 2.0 if task in ("GP", "GN") else 1.0
    for epoch in range(1, args.v3_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            kwargs = dict(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                aa_ids=batch["aa_ids"].to(device),
                aa_mask=batch["aa_mask"].to(device),
                phys=batch["phys"].to(device),
                species_ids=batch["species_ids"].to(device),
            )
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**kwargs)
                    loss = v3.mic_loss(logits, y, w) if task == "MIC" else v3.focal_ce(logits, y, w, gamma)
            else:
                logits = model(**kwargs)
                loss = v3.mic_loss(logits, y, w) if task == "MIC" else v3.focal_ce(logits, y, w, gamma)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            n += 1
        val = eval_v3(model, val_loader, device, task)
        score = float(val["_mean_primary"])
        print(f"[aa-v3 {task}] epoch {epoch}/{args.v3_epochs} loss={running/max(1,n):.4f} val={score:.4f}", flush=True)
        hist.append({"epoch": epoch, "train_loss": running / max(1, n), "val_mean_primary": score})
        if score > best + 1e-4:
            best = score
            patience = args.v3_patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "scheme": "aa_v3"}, best_path)
        else:
            patience -= 1
            if patience <= 0:
                print(f"[aa-v3 {task}] early stop", flush=True)
                break
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    summary = {"scheme": "aa_v3", "task": task, "seconds": time.time() - t0, "best_epoch": ckpt["epoch"], "val": eval_v3(model, val_loader, device, task), "test": eval_v3(model, test_loader, device, task), "history": hist}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def frames_v2(train_df, val_df, test_df, task):
    return v2.task_frame(train_df, task), v2.task_frame(val_df, task), v2.task_frame(test_df, task)


def frames_v3(train_df, val_df, test_df, task):
    if task == "MIC":
        return v3.mic_frame(train_df), v3.mic_frame(val_df), v3.mic_frame(test_df)
    tr = v3.collapse_binary(train_df, task)
    va = val_df[val_df[task].map(v3.valid_label).notna()].copy()
    va["text"] = va["sequence"]
    va["label"] = va[task].map(lambda x: int(v3.valid_label(x)))
    te = test_df[test_df[task].map(v3.valid_label).notna()].copy()
    te["text"] = te["sequence"]
    te["label"] = te[task].map(lambda x: int(v3.valid_label(x)))
    return tr, va, te


def write_report(out_root: Path, v2s, v3s):
    rows = []
    for s in v2s + v3s:
        rec = dict(s["test"][s["task"]])
        rec.update(scheme=s["scheme"], split="test", task=s["task"])
        rows.append(rec)
        rec2 = dict(s["val"][s["task"]])
        rec2.update(scheme=s["scheme"], split="val", task=s["task"])
        rows.append(rec2)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "metrics.csv", index=False)
    lines = [
        "# 生成器词表（氨基酸级、无 BPE）下的 V2 vs V3",
        "",
        "- 编码：`convert_tokens_to_ids(字符)`，一个氨基酸一个 token，不跑 merges",
        "- 词表：生成器 `models/AMPGuide`",
        "- V2 MIC 前缀 `SPxx:` 同样按字符查词表（S,P,0,0,:）",
        "- V3 MIC 肽走氨基酸 id，菌种仍用 embedding",
        "",
        "| 任务 | AA-V2 | AA-V3 | 差值(V3-V2) |",
        "|------|------:|------:|------------:|",
    ]
    a2, a3 = [], []
    for t in ALL_TASKS:
        p2 = float(table[(table.scheme == "aa_v2") & (table.split == "test") & (table.task == t)].iloc[0]["primary"])
        p3 = float(table[(table.scheme == "aa_v3") & (table.split == "test") & (table.task == t)].iloc[0]["primary"])
        a2.append(p2)
        a3.append(p3)
        lines.append(f"| {t} | {p2:.4f} | {p3:.4f} | {p3-p2:+.4f} |")
    lines.append(f"| **均值** | **{np.mean(a2):.4f}** | **{np.mean(a3):.4f}** | **{np.mean(a3)-np.mean(a2):+.4f}** |")
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(ALL_TASKS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x - w / 2, a2, w, label="V2 last-token + LoRA")
    ax.bar(x + w / 2, a3, w, label="V3 mean-pool + AA/phys")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ALL_TASKS))
    ax.set_ylabel("Test primary metric")
    ax.set_title("V2 vs V3 with generator char vocab (no BPE)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "compare_aa_v2_v3.png", dpi=160)
    plt.close(fig)
    return a2, a3


def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--split_root", default=str(_ROOT / "data/splits"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_aa_vocab"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_aa", type=int, default=80)
    ap.add_argument("--eval_bs", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--v2_epochs", type=int, default=8)
    ap.add_argument("--v2_patience", type=int, default=3)
    ap.add_argument("--v2_train_bs", type=int, default=64)
    ap.add_argument("--v2_lora_r", type=int, default=16)
    ap.add_argument("--v2_lora_alpha", type=int, default=32)
    ap.add_argument("--v2_lora_dropout", type=float, default=0.05)
    ap.add_argument("--v2_dropout", type=float, default=0.1)
    ap.add_argument("--v2_lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--v2_head_lr", type=float, default=1e-3)
    ap.add_argument("--v3_epochs", type=int, default=12)
    ap.add_argument("--v3_patience", type=int, default=4)
    ap.add_argument("--v3_train_bs", type=int, default=48)
    ap.add_argument("--v3_lora_r", type=int, default=32)
    ap.add_argument("--v3_lora_alpha", type=int, default=64)
    ap.add_argument("--v3_lora_dropout", type=float, default=0.08)
    ap.add_argument("--v3_dropout", type=float, default=0.15)
    ap.add_argument("--v3_lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--v3_full_lr", type=float, default=2e-5)
    ap.add_argument("--v3_head_lr", type=float, default=1e-3)
    ap.add_argument("--v3_weight_decay", type=float, default=0.03)
    ap.add_argument("--v3_unfreeze_last", type=int, default=2)
    args = ap.parse_args()

    v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tok = load_eos_tokenizer(args.backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    full = v2.load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = v2.apply_seq_split(full, Path(args.split_root))
    sample = encode_gen_chars(tok, "AAGKVLKLLKKLL")
    prep = {
        "tokenizer": args.backbone,
        "encoding": "per-character convert_tokens_to_ids, no BPE merges",
        "example_AAGKVLKLLKKLL": sample,
        "pad_id": pad_id,
        "split_root": args.split_root,
        "n_seq_train": int(train_df["sequence"].nunique()),
    }
    (out_root / "prep.json").write_text(json.dumps(prep, indent=2) + "\n")
    print("[aa prep]", json.dumps(prep), flush=True)

    v2_summ, v3_summ = [], []
    for task in ALL_TASKS:
        tr, va, te = frames_v2(train_df, val_df, test_df, task)
        print(f"[aa-v2 {task}] n={len(tr)}/{len(va)}/{len(te)}", flush=True)
        v2_summ.append(train_v2_task(task, tr, va, te, tok, pad_id, args.backbone, out_root / "v2" / f"per_attr_{task}", device, args))
    for task in ALL_TASKS:
        tr, va, te = frames_v3(train_df, val_df, test_df, task)
        print(f"[aa-v3 {task}] n={len(tr)}/{len(va)}/{len(te)}", flush=True)
        v3_summ.append(train_v3_task(task, tr, va, te, tok, pad_id, args.backbone, out_root / "v3" / f"per_attr_{task}", device, args))
    write_report(out_root, v2_summ, v3_summ)
    print("[done] aa-vocab report", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
