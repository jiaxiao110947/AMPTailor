#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-FT discriminators: generator vocab, one AA per token, atomic SPxx species tags."""
from __future__ import annotations


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
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import GPT2Model, GPT2TokenizerFast, get_cosine_schedule_with_warmup

SCRIPTS = Path(__file__).resolve().parent
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
SP_TOKENS = [f"SP{i:02d}" for i in range(20)]


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


def build_tokenizer_and_encoder(backbone: str, out_tok_dir: Path):
    tok = GPT2TokenizerFast.from_pretrained(backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    piece_ids = {sp: tok.encode(sp, add_special_tokens=False) for sp in SP_TOKENS}
    n_added = tok.add_special_tokens({"additional_special_tokens": SP_TOKENS})
    encoder = GPT2Model.from_pretrained(backbone, local_files_only=True)
    encoder.resize_token_embeddings(len(tok))
    with torch.no_grad():
        w = encoder.wte.weight
        for sp in SP_TOKENS:
            new_id = tok.convert_tokens_to_ids(sp)
            old = piece_ids[sp]
            if old:
                w[new_id] = w[old].mean(dim=0)
    encoder.config.vocab_size = len(tok)
    encoder.config.pad_token_id = tok.pad_token_id
    out_tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_tok_dir)
    atomic = {sp: {"id": tok.convert_tokens_to_ids(sp), "n_tokens": 1, "old_bpe_pieces": piece_ids[sp]} for sp in SP_TOKENS}
    assert all(tok.convert_tokens_to_ids(sp) != tok.unk_token_id for sp in SP_TOKENS)
    assert all(len(tok.encode(sp, add_special_tokens=False)) == 1 for sp in SP_TOKENS)
    return tok, encoder, n_added, atomic


def encode_aa(tok, seq: str) -> List[int]:
    ids = []
    for ch in seq:
        if ch not in AA20:
            continue
        i = tok.convert_tokens_to_ids(ch)
        if i is not None and i >= 0 and i != tok.unk_token_id:
            ids.append(int(i))
    return ids


def encode_row(tok, seq: str, species_id, with_sp: bool) -> List[int]:
    aa = encode_aa(tok, seq)
    if with_sp:
        sid = v2.valid_label(species_id)
        if sid is None:
            return aa
        sp_id = tok.convert_tokens_to_ids(f"SP{int(sid):02d}")
        return [int(sp_id)] + aa
    return aa


class IdDS(Dataset):
    def __init__(self, df: pd.DataFrame, tok, with_sp: bool):
        self.labels = df["label"].astype(int).tolist()
        seqs = df["sequence"].tolist()
        sids = df["species_id"].tolist() if "species_id" in df.columns else [None] * len(df)
        self.ids = [encode_row(tok, seqs[i], sids[i], with_sp) for i in range(len(df))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"ids": self.ids[i], "label": self.labels[i]}


class LeftPad:
    def __init__(self, pad_id: int, max_length: int):
        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, batch):
        raw = [x["ids"][: self.max_length] or [self.pad_id] for x in batch]
        L = max(len(s) for s in raw)
        input_ids, attn = [], []
        for s in raw:
            p = L - len(s)
            input_ids.append([self.pad_id] * p + s)
            attn.append([0] * p + [1] * len(s))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        }


class FullFTDisc(nn.Module):
    def __init__(self, encoder: GPT2Model, n_labels: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        h = int(encoder.config.n_embd)
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


def mic_loss(logits, y, weight):
    ce = F.cross_entropy(logits, y, weight=weight, label_smoothing=0.05)
    prob = torch.softmax(logits.float(), dim=-1)
    bins = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
    exp = (prob * bins).sum(-1)
    return ce + 0.5 * F.smooth_l1_loss(exp, y.float())


def make_loader(df, tok, pad_id, max_length, bs, shuffle, workers, with_sp, balance):
    ds = IdDS(df, tok, with_sp)
    sampler = None
    if balance and shuffle:
        y = np.array(ds.labels)
        freq = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
        w = 1.0 / np.maximum(freq, 1.0)[y]
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=LeftPad(pad_id, max_length),
    )


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    return v1.metrics_from_store({task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}, (task,))


def clone_encoder(template: GPT2Model) -> GPT2Model:
    cloned = GPT2Model(template.config)
    cloned.load_state_dict(template.state_dict())
    return cloned


def train_one(task, train_df, val_df, test_df, tok, template_encoder, out_dir, device, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    with_sp = task == "MIC"
    model = FullFTDisc(clone_encoder(template_encoder), n_labels, args.dropout).to(device)
    best_path = out_dir / "best.pt"
    start_epoch = 0
    if args.resume and best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[fullft {task}] resume from epoch {start_epoch} val={ckpt.get('val_mean_primary', 'na')}", flush=True)
    ytr = train_df["label"].astype(int).tolist()
    counts = np.bincount(np.array(ytr), minlength=n_labels).astype(np.float64)
    w = torch.tensor(counts.sum() / (n_labels * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    balance = (task in ("GP", "GN")) and (not args.resume)
    kw = dict(tok=tok, pad_id=int(tok.pad_token_id), max_length=args.max_length, workers=args.num_workers, with_sp=with_sp)
    train_loader = make_loader(train_df, bs=args.train_bs, shuffle=True, balance=balance, **kw)
    val_loader = make_loader(val_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    test_loader = make_loader(test_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    enc_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.enc_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    total = max(1, len(train_loader)) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best, patience, hist, t0 = -1e9, args.patience, [], time.time()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.resume and best_path.exists():
        best = float(evaluate(model, val_loader, device, task)["_mean_primary"])
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                    loss = mic_loss(logits, y, w) if task == "MIC" else F.cross_entropy(logits, y, weight=w)
            else:
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                loss = mic_loss(logits, y, w) if task == "MIC" else F.cross_entropy(logits, y, weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            n += 1
        val = evaluate(model, val_loader, device, task)
        score = float(val["_mean_primary"])
        print(f"[fullft {task}] epoch {epoch}/{end_epoch} loss={running/max(1,n):.4f} val={score:.4f}", flush=True)
        hist.append({"epoch": epoch, "train_loss": running / max(1, n), "val_mean_primary": score, "val": val[task]})
        if score > best + 1e-4:
            best = score
            patience = args.patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "n_trainable": n_train, "val_mean_primary": score, "config": model.encoder.config.to_dict()}, best_path)
        else:
            patience -= 1
            if patience <= 0:
                print(f"[fullft {task}] early stop", flush=True)
                break
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    summary = {
        "scheme": "fullft_atomic_sp",
        "task": task,
        "seconds": time.time() - t0,
        "best_epoch": ckpt["epoch"],
        "n_trainable": n_train,
        "n_train": int(len(train_df)),
        "val": evaluate(model, val_loader, device, task),
        "test": evaluate(model, test_loader, device, task),
        "history": hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def frames(train_df, val_df, test_df, task):
    if task == "MIC":
        return v3.mic_frame(train_df), v3.mic_frame(val_df), v3.mic_frame(test_df)
    tr = v3.collapse_binary(train_df, task)
    va = val_df[val_df[task].map(v3.valid_label).notna()].copy()
    va["label"] = va[task].map(lambda x: int(v3.valid_label(x)))
    te = test_df[test_df[task].map(v3.valid_label).notna()].copy()
    te["label"] = te[task].map(lambda x: int(v3.valid_label(x)))
    return tr, va, te


def write_report(out_root: Path, summaries):
    rows = []
    for s in summaries:
        for split in ("val", "test"):
            rec = dict(s[split][s["task"]])
            rec.update(scheme="fullft_atomic_sp", split=split, task=s["task"])
            rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "metrics.csv", index=False)
    prev = {
        "BPE-V2": {"GP": 0.7810, "GN": 0.8359, "FU": 0.8132, "SF": 0.7937, "MIC": 0.3909},
        "AA-V2": {"GP": 0.8362, "GN": 0.8464, "FU": 0.7961, "SF": 0.8046, "MIC": 0.3543},
        "BPE-V3": {"GP": 0.8378, "GN": 0.8159, "FU": 0.7877, "SF": 0.7896, "MIC": 0.4252},
    }
    lines = [
        "# 全参微调判别器（原子 SPxx + 氨基酸级生成器词表）",
        "",
        "- 肽：生成器词表，一氨基酸一 token，不跑 BPE",
        "- 菌种：`SP00`–`SP19` 各 1 个 special token，MIC 输入为 `[SPxx] + 肽`",
        "- 无 LoRA，GPT-2 encoder + 头全部训练",
        "",
        "| 任务 | BPE-V2 | AA-V2 | BPE-V3 | **全参+原子SP** |",
        "|------|-------:|------:|-------:|---------------:|",
    ]
    cur = []
    for t in ALL_TASKS:
        p = float(table[(table.split == "test") & (table.task == t)].iloc[0]["primary"])
        cur.append(p)
        lines.append(f"| {t} | {prev['BPE-V2'][t]:.4f} | {prev['AA-V2'][t]:.4f} | {prev['BPE-V3'][t]:.4f} | **{p:.4f}** |")
    def mean(d):
        return np.mean([d[t] for t in ALL_TASKS])
    lines.append(
        f"| **均值** | {mean(prev['BPE-V2']):.4f} | {mean(prev['AA-V2']):.4f} | {mean(prev['BPE-V3']):.4f} | **{np.mean(cur):.4f}** |"
    )
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n")


def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--split_root", default=str(_ROOT / "data/splits"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_fullft_atomic"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--enc_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tok_dir = out_root / "tokenizer"
    if args.resume and tok_dir.exists() and (tok_dir / "tokenizer_config.json").exists():
        tok = GPT2TokenizerFast.from_pretrained(str(tok_dir), local_files_only=True)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        encoder = GPT2Model.from_pretrained(args.backbone, local_files_only=True)
        encoder.resize_token_embeddings(len(tok))
        encoder.to("cpu")
        n_added, atomic = 20, {}
        print(f"[fullft] loaded tokenizer vocab={len(tok)} from {tok_dir}", flush=True)
    else:
        tok, encoder, n_added, atomic = build_tokenizer_and_encoder(args.backbone, tok_dir)
        encoder.to("cpu")
    full = v2.load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = v2.apply_seq_split(full, Path(args.split_root))
    ex_aa = encode_aa(tok, "AAGKVLKLLKKLL")
    ex_mic = encode_row(tok, "AAGKVLKLLKKLL", 0, True)
    prep = {
        "backbone": args.backbone,
        "n_added_sp_tokens": n_added,
        "sp00_id": tok.convert_tokens_to_ids("SP00"),
        "sp00_encode_len": len(tok.encode("SP00", add_special_tokens=False)),
        "example_peptide_ids": ex_aa,
        "example_mic_ids_SP00_plus_peptide": ex_mic,
        "atomic_sp": atomic,
        "full_ft": True,
        "lora": False,
        "n_seq_train": int(train_df["sequence"].nunique()),
        "vocab_size": len(tok),
    }
    (out_root / "prep.json").write_text(json.dumps(prep, indent=2) + "\n")
    print("[fullft prep]", json.dumps({k: prep[k] for k in ["sp00_id", "sp00_encode_len", "example_peptide_ids", "example_mic_ids_SP00_plus_peptide", "vocab_size", "n_added_sp_tokens"]}), flush=True)

    summaries = []
    for task in ALL_TASKS:
        tr, va, te = frames(train_df, val_df, test_df, task)
        print(f"[fullft {task}] n={len(tr)}/{len(va)}/{len(te)}", flush=True)
        summaries.append(train_one(task, tr, va, te, tok, encoder, out_root / f"per_attr_{task}", device, args))
    write_report(out_root, summaries)
    print("[done] fullft report", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
