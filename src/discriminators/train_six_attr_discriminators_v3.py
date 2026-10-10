#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 discriminators: mean-pool + AA embedding + physchem + species emb (MIC) + ordinal/focal."""
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import GPT2Model, GPT2TokenizerFast, get_cosine_schedule_with_warmup

V1_PATH = Path(__file__).resolve().parent / "train_six_attr_discriminators.py"
_spec = importlib.util.spec_from_file_location("disc_v1", str(V1_PATH))
v1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(v1)

ALL_TASKS = v1.ALL_TASKS
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2ID = {a: i for i, a in enumerate(AA20)}
PAD_AA = 20
HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}


def valid_label(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return int(v)


def physchem(seq: str) -> np.ndarray:
    n = max(len(seq), 1)
    ch = sum(CHARGE.get(a, 0.0) for a in seq)
    gy = sum(HYDRO.get(a, 0.0) for a in seq) / n
    return np.array(
        [
            min(n / 50.0, 4.0),
            ch / 10.0,
            gy / 5.0,
            sum(a in "FWY" for a in seq) / n,
            sum(a in "AILMFVW" for a in seq) / n,
            sum(a in "KRH" for a in seq) / n,
            sum(a in "DE" for a in seq) / n,
            sum(a == "C" for a in seq) / n,
        ],
        dtype=np.float32,
    )


def aa_ids(seq: str, max_len: int) -> List[int]:
    ids = [AA2ID.get(a, PAD_AA) for a in seq[:max_len]]
    return ids


class FeatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_aa: int, use_species: bool):
        self.texts = df["text"].tolist()
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_aa = max_aa
        self.use_species = use_species
        if use_species:
            sids = []
            for v in df["species_id"].tolist():
                x = valid_label(v)
                sids.append(20 if x is None else int(x))
            self.sids = sids
        else:
            self.sids = [20] * len(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        seq = self.seqs[i]
        return {
            "text": self.texts[i],
            "aa": aa_ids(seq, self.max_aa),
            "phys": physchem(seq),
            "sid": self.sids[i],
            "label": self.labels[i],
        }


class FeatCollator:
    def __init__(self, tok: GPT2TokenizerFast, max_length: int, max_aa: int):
        self.tok = tok
        self.max_length = max_length
        self.max_aa = max_aa

    def __call__(self, batch: List[dict]) -> dict:
        enc = self.tok(
            [x["text"] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        aa = torch.full((len(batch), self.max_aa), PAD_AA, dtype=torch.long)
        aa_mask = torch.zeros((len(batch), self.max_aa), dtype=torch.float32)
        for i, x in enumerate(batch):
            n = min(len(x["aa"]), self.max_aa)
            aa[i, :n] = torch.tensor(x["aa"][:n], dtype=torch.long)
            aa_mask[i, :n] = 1.0
        enc["aa_ids"] = aa
        enc["aa_mask"] = aa_mask
        enc["phys"] = torch.tensor(np.stack([x["phys"] for x in batch]), dtype=torch.float32)
        enc["species_ids"] = torch.tensor([x["sid"] for x in batch], dtype=torch.long)
        enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return enc


class V3Discriminator(nn.Module):
    def __init__(
        self,
        backbone_dir: str,
        n_labels: int,
        use_species: bool,
        dropout: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        unfreeze_last: int,
    ):
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
        base = self.encoder.get_base_model()
        n_layer = len(base.h)
        for block in base.h[max(0, n_layer - unfreeze_last) :]:
            for p in block.parameters():
                p.requires_grad = True
        for p in base.ln_f.parameters():
            p.requires_grad = True
        self.aa_emb = nn.Embedding(21, 64, padding_idx=PAD_AA)
        self.use_species = use_species
        self.species_emb = nn.Embedding(21, 48, padding_idx=20) if use_species else None
        in_dim = hidden + 64 + 8 + (48 if use_species else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_labels),
        )

    def forward(self, input_ids, attention_mask, aa_ids, aa_mask, phys, species_ids) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        aa_h = self.aa_emb(aa_ids)
        am = aa_mask.unsqueeze(-1)
        aa_pooled = (aa_h * am).sum(1) / am.sum(1).clamp(min=1.0)
        parts = [pooled, aa_pooled, phys]
        if self.species_emb is not None:
            parts.append(self.species_emb(species_ids))
        return self.head(torch.cat(parts, dim=-1))


def collapse_binary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for seq, g in df.groupby("sequence", sort=False):
        labs = [valid_label(v) for v in g[task].tolist()]
        labs = [x for x in labs if x is not None]
        if not labs:
            continue
        ones = sum(x == 1 for x in labs)
        zeros = sum(x == 0 for x in labs)
        label = 1 if ones >= zeros else 0
        rows.append({"sequence": seq, "text": seq, "label": label, "species_id": 20})
    return pd.DataFrame(rows)


def mic_frame(df: pd.DataFrame) -> pd.DataFrame:
    m = df["MIC"].map(valid_label).notna() & df["species_id"].map(valid_label).notna()
    out = df.loc[m].copy()
    out["text"] = out["sequence"]
    out["label"] = out["MIC"].map(lambda x: int(x))
    return out.reset_index(drop=True)


def make_loader(df, tok, max_length, max_aa, bs, shuffle, workers, use_species, balance: bool) -> DataLoader:
    ds = FeatDataset(df, max_aa, use_species)
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
        collate_fn=FeatCollator(tok, max_length, max_aa),
    )


def focal_ce(logits, y, weight, gamma: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def mic_loss(logits, y, weight) -> torch.Tensor:
    ce = F.cross_entropy(logits, y, weight=weight, label_smoothing=0.05)
    prob = torch.softmax(logits.float(), dim=-1)
    bins = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
    exp = (prob * bins).sum(-1)
    ord_loss = F.smooth_l1_loss(exp, y.float())
    return ce + 0.5 * ord_loss


@torch.no_grad()
def eval_task(model, loader, device, task: str) -> dict:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "aa_ids": batch["aa_ids"].to(device),
            "aa_mask": batch["aa_mask"].to(device),
            "phys": batch["phys"].to(device),
            "species_ids": batch["species_ids"].to(device),
        }
        logits_all.append(model(**kwargs).float().cpu())
        y_all.append(y.cpu())
    store = {task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}
    return v1.metrics_from_store(store, (task,))


def train_task(*, task, train_df, val_df, test_df, tok, backbone, out_dir, device, args) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    use_species = task == "MIC"
    model = V3Discriminator(
        backbone, n_labels, use_species, args.dropout, args.lora_r, args.lora_alpha, args.lora_dropout, args.unfreeze_last
    ).to(device)
    ytr = train_df["label"].astype(int).tolist()
    counts = np.bincount(np.array(ytr), minlength=n_labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = torch.tensor(counts.sum() / (n_labels * counts), dtype=torch.float32, device=device)
    balance = task in ("GP", "GN")
    train_loader = make_loader(train_df, tok, args.max_length, args.max_aa, args.train_bs, True, args.num_workers, use_species, balance)
    val_loader = make_loader(val_df, tok, args.max_length, args.max_aa, args.eval_bs, False, args.num_workers, use_species, False)
    test_loader = make_loader(test_df, tok, args.max_length, args.max_aa, args.eval_bs, False, args.num_workers, use_species, False)

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
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_lr})
    if full_params:
        groups.append({"params": full_params, "lr": args.full_lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_score, patience_left = -1e9, args.patience
    best_path = out_dir / "best.pt"
    history = []
    t0 = time.time()
    gamma = 2.0 if task in ("GP", "GN") else 1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "aa_ids": batch["aa_ids"].to(device),
                "aa_mask": batch["aa_mask"].to(device),
                "phys": batch["phys"].to(device),
                "species_ids": batch["species_ids"].to(device),
            }
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**kwargs)
                    loss = mic_loss(logits, y, w) if task == "MIC" else focal_ce(logits, y, w, gamma)
            else:
                logits = model(**kwargs)
                loss = mic_loss(logits, y, w) if task == "MIC" else focal_ce(logits, y, w, gamma)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n_batches += 1
        val_metrics = eval_task(model, val_loader, device, task)
        score = float(val_metrics["_mean_primary"])
        print(f"[v3 {task}] epoch {epoch}/{args.epochs} loss={running / max(1, n_batches):.4f} val_primary={score:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": running / max(1, n_batches), "val_mean_primary": score, "val": val_metrics[task]})
        if score > best_score + 1e-4:
            best_score = score
            patience_left = args.patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "val_mean_primary": score, "n_trainable": n_train}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[v3 {task}] early stop at epoch {epoch}", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    summary = {
        "name": f"v3_per_attr_{task}",
        "task": task,
        "seconds": time.time() - t0,
        "n_trainable": n_train,
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


def write_outputs(out_root: Path, prep: dict, summaries: List[dict], v2_csv: Path) -> None:
    rows = []
    for s in summaries:
        t = s["task"]
        for split in ("val", "test"):
            rec = dict(s[split][t])
            rec.update({"scheme": "v3", "model": s["name"], "split": split, "task": t})
            rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "metrics.csv", index=False)
    v2 = pd.read_csv(v2_csv)
    v2_test = v2[(v2.split == "test")].set_index("task")
    lines = [
        "# 六属性判别器 v3",
        "",
        "- 均值池化 + 残基 embedding + 物化特征；MIC 用菌种 embedding（不再靠 SP 字符串）",
        "- GP/GN：序列级多数票 + 类别平衡采样 + focal loss；解冻最后 2 层 + LoRA r=32",
        f"- 同一序列划分：`{prep['split_root']}`",
        "",
        "| 任务 | v2 | v3 | 差值 |",
        "|------|---:|---:|-----:|",
    ]
    v2s, v3s = [], []
    for t in ALL_TASKS:
        v3p = float(table[(table.split == "test") & (table.task == t)].iloc[0]["primary"])
        v2p = float(v2_test.loc[t, "primary"])
        v2s.append(v2p)
        v3s.append(v3p)
        lines.append(f"| {t} | {v2p:.4f} | {v3p:.4f} | {v3p - v2p:+.4f} |")
    lines.append(f"| **均值** | **{np.mean(v2s):.4f}** | **{np.mean(v3s):.4f}** | **{np.mean(v3s) - np.mean(v2s):+.4f}** |")
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(ALL_TASKS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x - w / 2, v2s, w, label="v2 LoRA")
    ax.bar(x + w / 2, v3s, w, label="v3 mean-pool+AA+phys(+species)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ALL_TASKS))
    ax.set_ylabel("Test primary metric")
    ax.set_title("Discriminator v2 vs v3 (same sequence split)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "compare_v2_v3.png", dpi=160)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--split_root", default=str(_ROOT / "data/splits"))
    ap.add_argument("--v2_metrics", default=str(_ROOT / "outputs/discriminators_v2/metrics.csv"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_v3"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_aa", type=int, default=80)
    ap.add_argument("--train_bs", type=int, default=48)
    ap.add_argument("--eval_bs", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--full_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.03)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.08)
    ap.add_argument("--unfreeze_last", type=int, default=2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("disc_v2", str(Path(__file__).resolve().parent / "train_six_attr_discriminators_v2.py"))
    v2m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2m)
    full = v2m.load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = v2m.apply_seq_split(full, Path(args.split_root))

    tok = GPT2TokenizerFast.from_pretrained(args.backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    prep = {
        "backbone": args.backbone,
        "split_root": args.split_root,
        "lora_r": args.lora_r,
        "unfreeze_last": args.unfreeze_last,
        "binary_train": "unique-sequence majority vote",
        "mic_condition": "learned species embedding + ordinal loss",
        "n_seq_train": int(train_df["sequence"].nunique()),
    }
    (out_root / "prep.json").write_text(json.dumps(prep, indent=2) + "\n")
    print("[v3 prep]", json.dumps(prep), flush=True)

    summaries = []
    for task in ALL_TASKS:
        if task == "MIC":
            tr, va, te = mic_frame(train_df), mic_frame(val_df), mic_frame(test_df)
        else:
            tr = collapse_binary(train_df, task)
            va = val_df[val_df[task].map(valid_label).notna()].copy()
            va["text"] = va["sequence"]
            va["label"] = va[task].map(lambda x: int(valid_label(x)))
            te = test_df[test_df[task].map(valid_label).notna()].copy()
            te["text"] = te["sequence"]
            te["label"] = te[task].map(lambda x: int(valid_label(x)))
        pos = float(np.mean(tr["label"])) if task != "MIC" else None
        print(f"[v3 {task}] n_train={len(tr)} n_val={len(va)} n_test={len(te)} pos={pos}", flush=True)
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
    write_outputs(out_root, prep, summaries, Path(args.v2_metrics))
    print("[done] v3 report:", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
