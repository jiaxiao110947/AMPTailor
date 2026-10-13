#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train six-attr discriminators: per-attribute models vs shared-encoder multi-head.

Same backbone, same sequence-grouped split, same peptide-only input.
"""
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
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2TokenizerFast, get_cosine_schedule_with_warmup

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
BINARY_TASKS = ("GP", "GN", "FU", "SF")
ALL_TASKS = BINARY_TASKS + ("MIC",)
TASK_COLS = {
    "GP": "gram_pos",
    "GN": "gram_neg",
    "FU": "fungus",
    "SF": "safety",
    "MIC": "mic_bin",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_seq(raw: object) -> str:
    s = str(raw).strip().upper().replace(" ", "")
    return "".join(c for c in s if c in AA20)


def to_optional_int(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    x = int(float(v))
    return x


class RowDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].tolist()
        self.groups = df["sequence"].tolist()
        labels = {}
        for t in ALL_TASKS:
            labels[t] = df[t].tolist()
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        item = {"text": self.texts[i]}
        for t in ALL_TASKS:
            v = self.labels[t][i]
            item[t] = -100 if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)
        return item


class Collator:
    def __init__(self, tok: GPT2TokenizerFast, max_length: int):
        self.tok = tok
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        texts = [x["text"] for x in batch]
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        for t in ALL_TASKS:
            enc[t] = torch.tensor([x[t] for x in batch], dtype=torch.long)
        return enc


class MultiHeadDiscriminator(nn.Module):
    def __init__(self, backbone_dir: str, tasks: Tuple[str, ...], dropout: float = 0.1):
        super().__init__()
        self.tasks = tuple(tasks)
        self.encoder = GPT2Model.from_pretrained(backbone_dir, local_files_only=True)
        hidden = int(self.encoder.config.n_embd)
        self.dropout = nn.Dropout(dropout)
        heads = {}
        for t in self.tasks:
            n = 5 if t == "MIC" else 2
            heads[t] = nn.Linear(hidden, n)
        self.heads = nn.ModuleDict(heads)

    def freeze_bottom_layers(self, n_freeze: int) -> None:
        if n_freeze <= 0:
            return
        for p in self.encoder.wte.parameters():
            p.requires_grad = False
        for p in self.encoder.wpe.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.encoder.h):
            if i < n_freeze:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        hidden = out.last_hidden_state
        pooled = hidden[:, -1]
        pooled = self.dropout(pooled)
        return {t: self.heads[t](pooled) for t in self.tasks}


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    valid = labels.ne(-100)
    if not bool(valid.any()):
        return logits.new_zeros(())
    return F.cross_entropy(logits[valid], labels[valid], weight=weight)


def compute_class_weights(df: pd.DataFrame, task: str, device: torch.device) -> Optional[torch.Tensor]:
    vals = []
    for v in df[task].tolist():
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        vals.append(int(v))
    if not vals:
        return None
    if task == "MIC":
        counts = np.bincount(np.array(vals, dtype=int), minlength=5).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (len(counts) * counts)
        return torch.tensor(w, dtype=torch.float32, device=device)
    counts = np.bincount(np.array(vals, dtype=int), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_logits(model: MultiHeadDiscriminator, loader: DataLoader, device: torch.device, tasks: Tuple[str, ...]) -> dict:
    model.eval()
    store = {t: {"logit": [], "y": []} for t in tasks}
    for batch in loader:
        labels = {t: batch.pop(t).to(device) for t in ALL_TASKS}
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        for t in tasks:
            store[t]["logit"].append(logits[t].float().cpu())
            store[t]["y"].append(labels[t].cpu())
    for t in tasks:
        store[t]["logit"] = torch.cat(store[t]["logit"], dim=0).numpy()
        store[t]["y"] = torch.cat(store[t]["y"], dim=0).numpy()
    return store


def metrics_from_store(store: dict, tasks: Tuple[str, ...]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in tasks:
        logits = store[t]["logit"]
        y = store[t]["y"]
        m = y != -100
        yv = y[m]
        lv = logits[m]
        rec: dict = {"n": int(m.sum()), "task": t}
        if rec["n"] < 8:
            rec["skip"] = True
            out[t] = rec
            continue
        if t == "MIC":
            pred = lv.argmax(axis=1)
            rec.update(
                {
                    "accuracy": float(accuracy_score(yv, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(yv, pred)),
                    "macro_f1": float(f1_score(yv, pred, average="macro", zero_division=0)),
                    "mae": float(mean_absolute_error(yv, pred)),
                    "qwk": float(cohen_kappa_score(yv, pred, weights="quadratic")),
                    "primary": float(balanced_accuracy_score(yv, pred)),
                }
            )
        else:
            if lv.shape[1] == 2:
                prob = torch.softmax(torch.tensor(lv), dim=1).numpy()[:, 1]
            else:
                prob = lv.ravel()
            pred = (prob >= 0.5).astype(int)
            rec.update(
                {
                    "accuracy": float(accuracy_score(yv, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(yv, pred)),
                    "f1": float(f1_score(yv, pred, zero_division=0)),
                    "primary": None,
                }
            )
            try:
                rec["auroc"] = float(roc_auc_score(yv, prob))
            except ValueError:
                rec["auroc"] = float("nan")
            try:
                rec["auprc"] = float(average_precision_score(yv, prob))
            except ValueError:
                rec["auprc"] = float("nan")
            rec["primary"] = rec["auroc"] if rec["auroc"] == rec["auroc"] else rec["balanced_accuracy"]
        out[t] = rec
    primaries = [v["primary"] for v in out.values() if v.get("primary") is not None and v["primary"] == v["primary"]]
    out["_mean_primary"] = float(np.mean(primaries)) if primaries else float("nan")
    return out


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sequence"] = df["sequence"].map(clean_seq)
    df = df[df["sequence"].str.len() >= 5].copy()
    df = df[df["sequence"].str.len() <= 190].copy()
    rows = []
    for rec in df.itertuples(index=False):
        seq = rec.sequence
        item = {"sequence": seq, "text": seq}
        item["GP"] = to_optional_int(getattr(rec, "gram_pos", None))
        item["GN"] = to_optional_int(getattr(rec, "gram_neg", None))
        item["FU"] = to_optional_int(getattr(rec, "fungus", None))
        sf = to_optional_int(getattr(rec, "safety", None))
        item["SF"] = sf if sf in (0, 1) else None
        mic = to_optional_int(getattr(rec, "mic_bin", None))
        item["MIC"] = mic if mic in (0, 1, 2, 3, 4) else None
        if all(item[t] is None for t in ALL_TASKS):
            continue
        rows.append(item)
    return pd.DataFrame(rows)


def split_by_sequence(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["sequence"].to_numpy()
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=groups))
    rel = val_frac / max(1e-8, (1.0 - test_frac))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed + 1)
    tr_rel, va_rel = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def make_loader(df: pd.DataFrame, tok, max_length: int, bs: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        RowDataset(df),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=Collator(tok, max_length),
    )


def train_one(
    *,
    name: str,
    tasks: Tuple[str, ...],
    backbone: str,
    tok,
    train_df,
    val_df,
    test_df,
    out_dir: Path,
    device: torch.device,
    args,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MultiHeadDiscriminator(backbone, tasks, dropout=args.dropout).to(device)
    model.freeze_bottom_layers(args.freeze_bottom_layers)
    weights = {t: compute_class_weights(train_df, t, device) for t in tasks}

    train_loader = make_loader(train_df, tok, args.max_length, args.train_bs, True, args.num_workers)
    val_loader = make_loader(val_df, tok, args.max_length, args.eval_bs, False, args.num_workers)
    test_loader = make_loader(test_df, tok, args.max_length, args.eval_bs, False, args.num_workers)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
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
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = {t: batch.pop(t).to(device) for t in ALL_TASKS}
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    losses = [masked_ce(logits[t], labels[t], weights[t]) for t in tasks]
                    loss = torch.stack(losses).mean() / args.grad_accum
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                losses = [masked_ce(logits[t], labels[t], weights[t]) for t in tasks]
                loss = torch.stack(losses).mean() / args.grad_accum
            loss.backward()
            running += float(loss.detach().item()) * args.grad_accum
            n_batches += 1
            if step % args.grad_accum == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        val_store = collect_logits(model, val_loader, device, tasks)
        val_metrics = metrics_from_store(val_store, tasks)
        score = val_metrics["_mean_primary"]
        row = {"epoch": epoch, "train_loss": running / max(1, n_batches), "val_mean_primary": score, "val": {k: v for k, v in val_metrics.items() if k != "_mean_primary"}}
        history.append(row)
        print(f"[{name}] epoch {epoch}/{args.epochs} loss={row['train_loss']:.4f} val_primary={score:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score = score
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "tasks": list(tasks),
                    "epoch": epoch,
                    "val_mean_primary": score,
                    "n_trainable": n_train,
                    "n_all": n_all,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{name}] early stop at epoch {epoch}", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    val_metrics = metrics_from_store(collect_logits(model, val_loader, device, tasks), tasks)
    test_metrics = metrics_from_store(collect_logits(model, test_loader, device, tasks), tasks)
    summary = {
        "name": name,
        "tasks": list(tasks),
        "seconds": time.time() - t0,
        "n_trainable": n_train,
        "n_all": n_all,
        "best_epoch": ckpt["epoch"],
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def flatten_metrics(scheme: str, model_name: str, split: str, metrics: dict) -> List[dict]:
    rows = []
    for task, rec in metrics.items():
        if task.startswith("_") or not isinstance(rec, dict):
            continue
        row = {"scheme": scheme, "model": model_name, "split": split, "task": task}
        for k, v in rec.items():
            if k == "task":
                continue
            row[k] = v
        rows.append(row)
    return rows


def write_report(out_root: Path, prep: dict, shared: dict, singles: List[dict], table: pd.DataFrame) -> None:
    lines = []
    lines.append("# 六属性判别器：每属性一模型 vs 共享 encoder 多头")
    lines.append("")
    lines.append(f"- backbone: `{prep['backbone']}`")
    lines.append(f"- 输入: 仅肽序列（与 AMPSorter 相同，无条件前缀）")
    lines.append(f"- 划分: 按唯一序列 GroupShuffleSplit，seed={prep['seed']}，train/val/test={prep['n_train']}/{prep['n_val']}/{prep['n_test']} 行；唯一序列 {prep['n_seq_train']}/{prep['n_seq_val']}/{prep['n_seq_test']}")
    lines.append(f"- freeze_bottom_layers={prep['freeze_bottom_layers']}，max_length={prep['max_length']}，epochs={prep['epochs']}，lr={prep['lr']}")
    lines.append("")
    lines.append("## 测试集主指标（越高越好，MIC 为 balanced accuracy，二分类为 AUROC）")
    lines.append("")
    lines.append("| 任务 | 共享多头 | 独立模型 | 差值(共享-独立) |")
    lines.append("|------|----------:|----------:|----------------:|")
    for t in ALL_TASKS:
        s = table[(table.scheme == "shared") & (table.split == "test") & (table.task == t)]
        p = table[(table.scheme == "per_attr") & (table.split == "test") & (table.task == t)]
        if s.empty or p.empty:
            continue
        sv = float(s.iloc[0]["primary"])
        pv = float(p.iloc[0]["primary"])
        lines.append(f"| {t} | {sv:.4f} | {pv:.4f} | {sv - pv:+.4f} |")
    smean = float(table[(table.scheme == "shared") & (table.split == "test")]["primary"].mean())
    pmean = float(table[(table.scheme == "per_attr") & (table.split == "test")]["primary"].mean())
    lines.append(f"| **均值** | **{smean:.4f}** | **{pmean:.4f}** | **{smean - pmean:+.4f}** |")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 共享多头：一个 GPT-2 encoder，五个分类头，联合训练。")
    lines.append("- 独立模型：五个模型，各一个头，超参与数据完全相同。")
    lines.append("- Safety / MIC 缺失标签在 loss 中 mask（-100），不把 NA 当 0。")
    lines.append("- MIC 在无菌种输入下是序列级分箱预测，难度更高，两方案同等吃亏。")
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_compare(table: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[(table.split == "test") & (table.task.isin(ALL_TASKS))].copy()
    tasks = list(ALL_TASKS)
    shared = [float(sub[(sub.scheme == "shared") & (sub.task == t)]["primary"].iloc[0]) for t in tasks]
    per = [float(sub[(sub.scheme == "per_attr") & (sub.task == t)]["primary"].iloc[0]) for t in tasks]
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w / 2, shared, w, label="shared multi-head")
    ax.bar(x + w / 2, per, w, label="per-attribute")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Test primary metric")
    ax.set_title("Six-attr discriminators (same backbone / split / input)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_v1"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_bottom_layers", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(Path(args.csv_path))
    train_df, val_df, test_df = split_by_sequence(df, args.seed, args.val_frac, args.test_frac)
    train_df.to_csv(out_root / "split_train.csv", index=False)
    val_df.to_csv(out_root / "split_val.csv", index=False)
    test_df.to_csv(out_root / "split_test.csv", index=False)

    tok = load_eos_tokenizer(args.backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    prep = {
        "csv_path": args.csv_path,
        "backbone": args.backbone,
        "seed": args.seed,
        "n_rows": int(len(df)),
        "n_unique_seq": int(df["sequence"].nunique()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_seq_train": int(train_df["sequence"].nunique()),
        "n_seq_val": int(val_df["sequence"].nunique()),
        "n_seq_test": int(test_df["sequence"].nunique()),
        "freeze_bottom_layers": args.freeze_bottom_layers,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "labeled_counts_train": {t: int(train_df[t].notna().sum()) for t in ALL_TASKS},
    }
    (out_root / "prep.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
    print("[prep]", json.dumps(prep, ensure_ascii=False), flush=True)

    shared = train_one(
        name="shared_multihead",
        tasks=ALL_TASKS,
        backbone=args.backbone,
        tok=tok,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        out_dir=out_root / "shared_multihead",
        device=device,
        args=args,
    )
    singles = []
    for t in ALL_TASKS:
        singles.append(
            train_one(
                name=f"per_attr_{t}",
                tasks=(t,),
                backbone=args.backbone,
                tok=tok,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                out_dir=out_root / f"per_attr_{t}",
                device=device,
                args=args,
            )
        )

    rows = []
    rows += flatten_metrics("shared", "shared_multihead", "val", shared["val"])
    rows += flatten_metrics("shared", "shared_multihead", "test", shared["test"])
    for s in singles:
        t = s["tasks"][0]
        rows += flatten_metrics("per_attr", s["name"], "val", s["val"])
        rows += flatten_metrics("per_attr", s["name"], "test", s["test"])
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "compare_metrics.csv", index=False)
    write_report(out_root, prep, shared, singles, table)
    try:
        plot_compare(table, out_root / "compare_test_primary.png")
    except Exception as e:
        print("[warn] plot failed:", e, flush=True)
    print("[done] report:", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
