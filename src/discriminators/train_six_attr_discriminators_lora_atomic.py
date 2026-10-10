#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA discriminators on atomic SPxx tokenizer + one-AA-per-token peptide ids."""
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
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Model, GPT2TokenizerFast, get_cosine_schedule_with_warmup

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ft = _load("fullft_atomic", SCRIPTS / "train_six_attr_discriminators_fullft.py")
v2 = ft.v2
ALL_TASKS = ft.ALL_TASKS
SP_TOKENS = ft.SP_TOKENS


def build_resized_encoder(backbone: str, tok: GPT2TokenizerFast) -> GPT2Model:
    base_tok = GPT2TokenizerFast.from_pretrained(backbone, local_files_only=True)
    encoder = GPT2Model.from_pretrained(backbone, local_files_only=True)
    encoder.resize_token_embeddings(len(tok))
    with torch.no_grad():
        w = encoder.wte.weight
        for sp in SP_TOKENS:
            old = base_tok.encode(sp, add_special_tokens=False)
            new_id = tok.convert_tokens_to_ids(sp)
            if old and new_id is not None and int(new_id) >= 0:
                w[int(new_id)] = w[old].mean(dim=0)
    encoder.config.vocab_size = len(tok)
    encoder.config.pad_token_id = tok.pad_token_id
    return encoder


class AtomicLoraDisc(nn.Module):
    def __init__(
        self,
        encoder: GPT2Model,
        n_labels: int,
        dropout: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        sp_start_id: int,
    ):
        super().__init__()
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
        # Only the 20 atomic SP rows in wte get gradients; AA embeddings stay frozen.
        wte = self.encoder.get_base_model().wte
        wte.weight.requires_grad = True
        start = int(sp_start_id)

        def _sp_only_grad(grad):
            g = grad.clone()
            g[:start] = 0
            return g

        wte.weight.register_hook(_sp_only_grad)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, -1])
        return self.head(pooled)


def param_groups(model: AtomicLoraDisc, lora_lr: float, embed_lr: float, head_lr: float):
    lora, embed, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head."):
            head.append(p)
        elif "wte" in n:
            embed.append(p)
        else:
            lora.append(p)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": lora_lr})
    if embed:
        groups.append({"params": embed, "lr": embed_lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    return groups


@torch.no_grad()
def evaluate_loss_metrics(model, loader, device, task: str):
    model.eval()
    total, n = 0.0, 0
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        total += float(F.cross_entropy(logits, y, reduction="sum").item())
        n += int(y.numel())
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    metrics = ft.v1.metrics_from_store(
        {task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}},
        (task,),
    )
    metrics["_eval_loss"] = total / max(n, 1)
    return metrics


def clone_encoder(src: GPT2Model) -> GPT2Model:
    dst = GPT2Model(src.config)
    dst.load_state_dict(src.state_dict())
    return dst


def train_task(task, train_df, val_df, test_df, tok, template, device, args, wandb_mod):
    out_dir = Path(args.out_root) / f"per_attr_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 5 if task == "MIC" else 2
    with_sp = task == "MIC"
    sp_start = int(tok.convert_tokens_to_ids("SP00"))
    model = AtomicLoraDisc(
        clone_encoder(template),
        n_labels,
        args.dropout,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        sp_start,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{task}] trainable {n_train}/{n_all} LoRA r={args.lora_r} SP-only wte from id={sp_start}", flush=True)

    pad_id = int(tok.pad_token_id)
    kw = dict(tok=tok, pad_id=pad_id, max_length=args.max_length, workers=args.num_workers, with_sp=with_sp)
    train_loader = ft.make_loader(train_df, bs=args.train_bs, shuffle=True, balance=False, **kw)
    val_loader = ft.make_loader(val_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    test_loader = ft.make_loader(test_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)

    if with_sp:
        n_sp = sum(1 for ids in train_loader.dataset.ids if ids and ids[0] >= 50257)
        print(f"[{task}] atomic SP prefix rows={n_sp}/{len(train_loader.dataset)} first_ids_example={train_loader.dataset.ids[0][:6]}", flush=True)
    else:
        print(f"[{task}] peptide-only example={train_loader.dataset.ids[0][:8]}", flush=True)

    w = v2.class_weight(train_df["label"].astype(int).tolist(), n_labels, device)
    opt = torch.optim.AdamW(param_groups(model, args.lora_lr, args.embed_lr, args.head_lr), weight_decay=args.weight_decay)
    total = max(1, len(train_loader)) * args.max_epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    run = None
    if wandb_mod is not None:
        try:
            run = wandb_mod.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=f"{args.wandb_run_prefix}_{task}",
                config={
                    "task": task,
                    "lora_r": args.lora_r,
                    "lora_lr": args.lora_lr,
                    "embed_lr": args.embed_lr,
                    "head_lr": args.head_lr,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "stop_on": "eval_loss",
                    "mic_input": "[SPxx]+AA" if with_sp else "AA",
                    "n_trainable": n_train,
                },
            )
            wandb_mod.define_metric("epoch")
            wandb_mod.define_metric("train/*", step_metric="epoch")
            wandb_mod.define_metric("eval/*", step_metric="epoch")
        except Exception as e:
            print(f"[{task}] wandb.init failed: {e}", flush=True)
            run = None

    val0 = evaluate_loss_metrics(model, val_loader, device, task)
    best_loss = float(val0["_eval_loss"])
    best_path = out_dir / "best.pt"
    torch.save({"model": model.state_dict(), "task": task, "epoch": 0, "eval_loss": best_loss, "n_trainable": n_train}, best_path)
    patience_left = args.patience
    hist = []
    last_epoch = 0
    t0 = time.time()
    print(f"[{task}] start eval_loss={best_loss:.4f} primary={val0['_mean_primary']:.4f}", flush=True)
    if run is not None:
        wandb_mod.log({"epoch": 0, "eval/loss": best_loss, "eval/primary": val0["_mean_primary"]}, step=0)

    for epoch in range(1, args.max_epochs + 1):
        last_epoch = epoch
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                    if task == "MIC":
                        loss = ft.mic_loss(logits, y, w)
                    else:
                        loss = F.cross_entropy(logits, y, weight=w)
            else:
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                if task == "MIC":
                    loss = ft.mic_loss(logits, y, w)
                else:
                    loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n_batches += 1
        val = evaluate_loss_metrics(model, val_loader, device, task)
        el = float(val["_eval_loss"])
        prim = float(val["_mean_primary"])
        tr_loss = running / max(1, n_batches)
        hist.append({"epoch": epoch, "train_loss": tr_loss, "eval_loss": el, "eval_primary": prim})
        print(f"[{task}] epoch {epoch} train_loss={tr_loss:.4f} eval_loss={el:.4f} primary={prim:.4f}", flush=True)
        if run is not None:
            wandb_mod.log({"epoch": epoch, "train/loss": tr_loss, "eval/loss": el, "eval/primary": prim}, step=epoch)
        if el < best_loss - args.min_delta:
            best_loss = el
            patience_left = args.patience
            torch.save(
                {"model": model.state_dict(), "task": task, "epoch": epoch, "eval_loss": el, "eval_primary": prim, "n_trainable": n_train},
                best_path,
            )
            print(f"[{task}] eval_loss improved -> {el:.4f} (keep training)", flush=True)
        else:
            patience_left -= 1
            print(f"[{task}] eval_loss not decreasing ({patience_left} patience left)", flush=True)
            if patience_left <= 0:
                print(f"[{task}] stop: eval_loss no longer decreasing", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test = evaluate_loss_metrics(model, test_loader, device, task)
    summary = {
        "task": task,
        "best_epoch": ckpt.get("epoch"),
        "best_eval_loss": ckpt.get("eval_loss"),
        "end_epoch": last_epoch,
        "seconds": time.time() - t0,
        "n_trainable": n_train,
        "n_all": n_all,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "val": evaluate_loss_metrics(model, val_loader, device, task),
        "test": test,
        "history": hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if run is not None:
        wandb_mod.summary["test_primary"] = test["_mean_primary"]
        wandb_mod.summary["test_eval_loss"] = test["_eval_loss"]
        wandb_mod.summary["best_eval_loss"] = ckpt.get("eval_loss")
        wandb_mod.finish()
    return summary


def write_report(out_root: Path, summaries):
    rows = []
    for s in summaries:
        for split in ("val", "test"):
            rec = dict(s[split][s["task"]])
            rec.update(scheme="lora_atomic_sp", split=split, task=s["task"], eval_loss=s[split]["_eval_loss"])
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out_root / "metrics.csv", index=False)
    prev = {
        "AA-V2": {"GP": 0.8362, "GN": 0.8464, "FU": 0.7961, "SF": 0.8046, "MIC": 0.3543},
        "BPE-V3": {"GP": 0.8378, "GN": 0.8159, "FU": 0.7877, "SF": 0.7896, "MIC": 0.4252},
        "fullft18": {"GP": 0.7895, "GN": 0.8061, "FU": 0.7342, "SF": 0.7747, "MIC": 0.4002},
    }
    lines = [
        "# LoRA + 原子 SPxx + 氨基酸级词表",
        "",
        "- 肽：生成器词表，一氨基酸一 token，不跑 BPE",
        "- 菌种：`SP00`–`SP19` 各 1 个 special token；MIC 输入 `[SPxx] + 肽`",
        "- 全层 LoRA r=16，同时训 `wte`（让原子 SP embedding 能更新）+ 分类头",
        "- 早停：验证集 unweighted CE（eval_loss）不再下降",
        "",
        "| 任务 | AA-V2 | BPE-V3 | fullft-18 | **LoRA+原子SP** | test eval_loss |",
        "|------|------:|-------:|----------:|----------------:|---------------:|",
    ]
    ps = []
    for t in ALL_TASKS:
        row = [s for s in summaries if s["task"] == t][0]
        p = float(row["test"]["_mean_primary"])
        el = float(row["test"]["_eval_loss"])
        ps.append(p)
        lines.append(
            f"| {t} | {prev['AA-V2'][t]:.4f} | {prev['BPE-V3'][t]:.4f} | {prev['fullft18'][t]:.4f} | **{p:.4f}** | {el:.4f} |"
        )
    lines.append(f"| **均值** | 0.7275 | 0.7312 | 0.7010 | **{np.mean(ps):.4f}** | |")
    (out_root / "compare_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("OMP_NUM_THREADS", None)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(_ROOT / "data/corpus/sequence_species_mic_binned_for_six_attr.csv"))
    ap.add_argument("--backbone", default=str(_ROOT / "models/AMPGuide"))
    ap.add_argument("--split_root", default=str(_ROOT / "data/splits"))
    ap.add_argument("--tok_dir", default=str(_ROOT / "models/discriminators/mic_tokenizer"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_lora_atomic"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--max_epochs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--embed_lr", type=float, default=1e-4)
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
    ap.add_argument("--wandb_project", default="AMP_Model_Training")
    ap.add_argument("--wandb_group", default="six_attr_disc_lora_atomic")
    ap.add_argument("--wandb_run_prefix", default="disc_lora_atomic")
    ap.add_argument("--wandb_mode", default="online")
    args = ap.parse_args()

    ft.v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tok_src = Path(args.tok_dir)
    tok_dst = out_root / "tokenizer"
    if not (tok_dst / "tokenizer_config.json").exists():
        shutil.copytree(tok_src, tok_dst, dirs_exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained(str(tok_dst), local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    for sp in SP_TOKENS:
        ids = tok.encode(sp, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{sp} is not atomic: {ids}")
    print(f"[prep] vocab={len(tok)} SP00_id={tok.convert_tokens_to_ids('SP00')}", flush=True)

    template = build_resized_encoder(args.backbone, tok).to("cpu")
    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb_mod = None
    try:
        import wandb

        wandb_mod = wandb
    except Exception as e:
        print("[warn] wandb import failed", e, flush=True)

    full = v2.load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = v2.apply_seq_split(full, Path(args.split_root))
    prep = {
        "backbone": args.backbone,
        "tok_dir": str(tok_dst),
        "split_root": args.split_root,
        "vocab_size": len(tok),
        "mic_input": "[SPxx] + amino-acid ids",
        "binary_input": "amino-acid ids",
        "lora_r": args.lora_r,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
    }
    (out_root / "prep.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
    print("[prep]", json.dumps(prep, ensure_ascii=False), flush=True)

    summaries = []
    for task in ALL_TASKS:
        tr, va, te = ft.frames(train_df, val_df, test_df, task)
        print(f"[{task}] n_train={len(tr)} n_val={len(va)} n_test={len(te)}", flush=True)
        summaries.append(train_task(task, tr, va, te, tok, template, device, args, wandb_mod))
    write_report(out_root, summaries)
    print("[done] lora atomic report", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
