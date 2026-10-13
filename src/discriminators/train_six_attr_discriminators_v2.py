#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stronger per-attribute discriminators: LoRA on full encoder, MIC = seq+species."""
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
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2TokenizerFast, get_cosine_schedule_with_warmup

V1_PATH = Path(__file__).resolve().parent / "train_six_attr_discriminators.py"
_spec = importlib.util.spec_from_file_location("disc_v1", str(V1_PATH))
v1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(v1)

ALL_TASKS = v1.ALL_TASKS
BINARY_TASKS = v1.BINARY_TASKS


class SingleTaskDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"text": self.texts[i], "label": int(self.labels[i])}


class SingleCollator:
    def __init__(self, tok: GPT2TokenizerFast, max_length: int):
        self.tok = tok
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        enc = self.tok(
            [x["text"] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return enc


class LoraDiscriminator(nn.Module):
    def __init__(self, backbone_dir: str, n_labels: int, dropout: float, lora_r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        encoder = GPT2Model.from_pretrained(backbone_dir, local_files_only=True)
        hidden = int(encoder.config.n_embd)
        cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True,
            bias="none",
        )
        self.encoder = get_peft_model(encoder, cfg)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        pooled = self.dropout(out.last_hidden_state[:, -1])
        return self.head(pooled)


def valid_label(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return int(v)


def load_with_species(csv_path: Path) -> pd.DataFrame:
    src = pd.read_csv(csv_path)
    src["sequence"] = src["sequence"].map(v1.clean_seq)
    src = src[src["sequence"].str.len().between(5, 190)].copy()
    rows = []
    for rec in src.itertuples(index=False):
        seq = rec.sequence
        sid = v1.to_optional_int(getattr(rec, "species_id", None))
        item = {
            "sequence": seq,
            "species_id": sid,
            "text_seq": seq,
            "text_mic": None if sid is None else f"SP{int(sid):02d}:{seq}",
            "GP": v1.to_optional_int(getattr(rec, "gram_pos", None)),
            "GN": v1.to_optional_int(getattr(rec, "gram_neg", None)),
            "FU": v1.to_optional_int(getattr(rec, "fungus", None)),
        }
        sf = v1.to_optional_int(getattr(rec, "safety", None))
        item["SF"] = sf if sf in (0, 1) else None
        mic = v1.to_optional_int(getattr(rec, "mic_bin", None))
        item["MIC"] = mic if mic in (0, 1, 2, 3, 4) else None
        rows.append(item)
    return pd.DataFrame(rows)


def apply_seq_split(full: pd.DataFrame, split_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = set(pd.read_csv(split_root / "split_train.csv")["sequence"].astype(str))
    va = set(pd.read_csv(split_root / "split_val.csv")["sequence"].astype(str))
    te = set(pd.read_csv(split_root / "split_test.csv")["sequence"].astype(str))
    overlap = (tr & va) | (tr & te) | (va & te)
    if overlap:
        raise RuntimeError(f"sequence leak across splits: {len(overlap)}")
    train = full[full["sequence"].isin(tr)].reset_index(drop=True)
    val = full[full["sequence"].isin(va)].reset_index(drop=True)
    test = full[full["sequence"].isin(te)].reset_index(drop=True)
    return train, val, test


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "MIC":
        m = df["MIC"].map(valid_label).notna() & df["text_mic"].notna()
        out = df.loc[m].copy()
        out["text"] = out["text_mic"]
        out["label"] = out["MIC"].map(lambda x: int(x))
        return out
    m = df[task].map(valid_label).notna()
    out = df.loc[m].copy()
    out["text"] = out["text_seq"]
    out["label"] = out[task].map(lambda x: int(x))
    return out


def make_loader(df: pd.DataFrame, tok, max_length: int, bs: int, shuffle: bool, workers: int) -> DataLoader:
    ds = SingleTaskDataset(df["text"].tolist(), df["label"].astype(int).tolist())
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=SingleCollator(tok, max_length),
    )


def class_weight(labels: List[int], n_class: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=int), minlength=n_class).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_class * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_task(model: LoraDiscriminator, loader: DataLoader, device: torch.device, task: str) -> dict:
    model.eval()
    logits_all = []
    y_all = []
    for batch in loader:
        y = batch.pop("labels").to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    store = {task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}
    return v1.metrics_from_store(store, (task,))


def train_task(
    *,
    task: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tok,
    backbone: str,
    out_dir: Path,
    device: torch.device,
    args,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    model = LoraDiscriminator(
        backbone, n_labels, args.dropout, args.lora_r, args.lora_alpha, args.lora_dropout
    ).to(device)
    w = class_weight(train_df["label"].astype(int).tolist(), n_labels, device)
    train_loader = make_loader(train_df, tok, args.max_length, args.train_bs, True, args.num_workers)
    val_loader = make_loader(val_df, tok, args.max_length, args.eval_bs, False, args.num_workers)
    test_loader = make_loader(test_df, tok, args.max_length, args.eval_bs, False, args.num_workers)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_score = -1e9
    best_path = out_dir / "best.pt"
    history = []
    patience_left = args.patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(logits, y, weight=w)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n_batches += 1
        val_metrics = eval_task(model, val_loader, device, task)
        score = float(val_metrics["_mean_primary"])
        print(
            f"[v2 {task}] epoch {epoch}/{args.epochs} loss={running / max(1, n_batches):.4f} val_primary={score:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch, "train_loss": running / max(1, n_batches), "val_mean_primary": score, "val": val_metrics[task]})
        if score > best_score + 1e-4:
            best_score = score
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": task,
                    "epoch": epoch,
                    "val_mean_primary": score,
                    "n_trainable": n_train,
                    "n_all": n_all,
                    "n_labels": n_labels,
                    "lora_r": args.lora_r,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[v2 {task}] early stop at epoch {epoch}", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    summary = {
        "name": f"v2_per_attr_{task}",
        "task": task,
        "seconds": time.time() - t0,
        "n_trainable": n_train,
        "n_all": n_all,
        "best_epoch": ckpt["epoch"],
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "val": eval_task(model, val_loader, device, task),
        "test": eval_task(model, test_loader, device, task),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def write_outputs(out_root: Path, prep: dict, summaries: List[dict], v1_csv: Path) -> None:
    rows = []
    for s in summaries:
        t = s["task"]
        for split in ("val", "test"):
            rec = dict(s[split][t])
            rec.update({"scheme": "v2_lora_per_attr", "model": s["name"], "split": split, "task": t})
            rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "metrics.csv", index=False)

    v1 = pd.read_csv(v1_csv)
    v1_test = v1[(v1.scheme == "per_attr") & (v1.split == "test")].set_index("task")
    lines = [
        "# 六属性判别器 v2（强化）",
        "",
        "- 方案：每属性独立模型 + 全层 LoRA（r=16）+ 可训练分类头",
        "- MIC 输入：`SP{id}:{sequence}`（有菌种才训练/评估）",
        "- GP/GN/FU/SF 输入：裸序列（避免菌种泄漏抗菌谱）",
        f"- 沿用 v1 按序列划分：`{prep['split_root']}`",
        f"- backbone: `{prep['backbone']}`",
        "",
        "## 测试集 vs v1 独立模型",
        "",
        "| 任务 | v1 主指标 | v2 主指标 | 差值(v2-v1) | v2 n |",
        "|------|----------:|----------:|------------:|-----:|",
    ]
    v2_ps, v1_ps = [], []
    for t in ALL_TASKS:
        srow = table[(table.split == "test") & (table.task == t)].iloc[0]
        v2p = float(srow["primary"])
        v1p = float(v1_test.loc[t, "primary"])
        v2_ps.append(v2p)
        v1_ps.append(v1p)
        lines.append(f"| {t} | {v1p:.4f} | {v2p:.4f} | {v2p - v1p:+.4f} | {int(srow['n'])} |")
    lines.append(f"| **均值** | **{np.mean(v1_ps):.4f}** | **{np.mean(v2_ps):.4f}** | **{np.mean(v2_ps) - np.mean(v1_ps):+.4f}** | |")
    lines += [
        "",
        "## 强化点",
        "",
        "- Encoder 不再冻 8 层，改为全部 Transformer 上 LoRA。",
        "- MIC 条件化菌种，与谱分类输入刻意分开。",
        "- 头学习率高于 LoRA；8 epoch + patience=3。",
        "",
    ]
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ALL_TASKS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x - w / 2, v1_ps, w, label="v1 per-attr (freeze-8)")
    ax.bar(x + w / 2, v2_ps, w, label="v2 LoRA + MIC(seq+species)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ALL_TASKS))
    ax.set_ylabel("Test primary metric")
    ax.set_title("Discriminator v1 vs v2 (same sequence split)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "compare_v1_v2.png", dpi=160)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--split_root", default=str(_ROOT / "data/splits"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_v2"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    split_root = Path(args.split_root)

    full = load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = apply_seq_split(full, split_root)
    train_df.to_csv(out_root / "split_train.csv", index=False)
    val_df.to_csv(out_root / "split_val.csv", index=False)
    test_df.to_csv(out_root / "split_test.csv", index=False)

    tok = load_eos_tokenizer(args.backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    prep = {
        "csv_path": args.csv_path,
        "backbone": args.backbone,
        "split_root": str(split_root),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_seq_train": int(train_df["sequence"].nunique()),
        "n_seq_val": int(val_df["sequence"].nunique()),
        "n_seq_test": int(test_df["sequence"].nunique()),
        "lora_r": args.lora_r,
        "lora_lr": args.lora_lr,
        "head_lr": args.head_lr,
        "epochs": args.epochs,
        "mic_input": "SP{id}:{sequence}",
        "binary_input": "raw sequence",
    }
    (out_root / "prep.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
    print("[v2 prep]", json.dumps(prep, ensure_ascii=False), flush=True)

    summaries = []
    for task in ALL_TASKS:
        tr = task_frame(train_df, task)
        va = task_frame(val_df, task)
        te = task_frame(test_df, task)
        print(f"[v2 {task}] n_train={len(tr)} n_val={len(va)} n_test={len(te)}", flush=True)
        summaries.append(
            train_task(
                task=task,
                train_df=tr,
                val_df=va,
                test_df=te,
                tok=tok,
                backbone=args.backbone,
                out_dir=out_root / f"per_attr_{task}",
                device=device,
                args=args,
            )
        )
    write_outputs(out_root, prep, summaries, split_root / "compare_metrics.csv")
    print("[done] v2 report:", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
