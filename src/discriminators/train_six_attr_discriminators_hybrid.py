#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid discriminators: AA-V2 for GP/GN/FU/SF (row-level, AUROC ckpt); atomic SPxx for MIC."""
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
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ft = _load("fullft_atomic", SCRIPTS / "train_six_attr_discriminators_fullft.py")
v2mod = _load("disc_v2", SCRIPTS / "train_six_attr_discriminators_v2.py")
ALL_TASKS = ft.ALL_TASKS
SP_TOKENS = ft.SP_TOKENS
v1 = ft.v1
v2 = ft.v2
make_id_loader = ft.make_loader
mic_loss = ft.mic_loss
LoraDiscriminator = v2mod.LoraDiscriminator
class_weight = v2mod.class_weight


def build_mic_encoder(backbone: str, tok: GPT2TokenizerFast) -> GPT2Model:
    base_tok = load_eos_tokenizer(backbone, local_files_only=True)
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


class MicLoraDisc(nn.Module):
    def __init__(self, encoder: GPT2Model, dropout: float, lora_r: int, lora_alpha: int, lora_dropout: float, sp_start_id: int):
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
        wte = self.encoder.get_base_model().wte
        wte.weight.requires_grad = True
        start = int(sp_start_id)

        def _sp_only_grad(grad):
            g = grad.clone()
            g[:start] = 0
            return g

        wte.weight.register_hook(_sp_only_grad)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 5)

    def forward(self, input_ids, attention_mask):
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.head(self.dropout(out.last_hidden_state[:, -1]))


def clone_encoder(src: GPT2Model) -> GPT2Model:
    dst = GPT2Model(src.config)
    dst.load_state_dict(src.state_dict())
    return dst


def metric_pack(rec: dict, task: str) -> dict:
    if task == "MIC":
        return {
            "primary": rec.get("balanced_accuracy"),
            "m1_name": "balanced_accuracy",
            "m1": rec.get("balanced_accuracy"),
            "m2_name": "macro_f1",
            "m2": rec.get("macro_f1"),
            "m3_name": "qwk",
            "m3": rec.get("qwk"),
        }
    return {
        "primary": rec.get("auroc"),
        "m1_name": "auroc",
        "m1": rec.get("auroc"),
        "m2_name": "auprc",
        "m2": rec.get("auprc"),
        "m3_name": "balanced_accuracy",
        "m3": rec.get("balanced_accuracy"),
    }


@torch.no_grad()
def evaluate(model, loader, device, task: str):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        y = batch.pop("labels").to(device)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        logits_all.append(logits.float().cpu())
        y_all.append(y.cpu())
    return v1.metrics_from_store({task: {"logit": torch.cat(logits_all).numpy(), "y": torch.cat(y_all).numpy()}}, (task,))


def wandb_init(wandb_mod, args, task, extra):
    if wandb_mod is None:
        return None
    try:
        run = wandb_mod.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"{args.wandb_run_prefix}_{task}",
            config={"task": task, **extra},
        )
        wandb_mod.define_metric("epoch")
        wandb_mod.define_metric("train/*", step_metric="epoch")
        wandb_mod.define_metric("eval/*", step_metric="epoch")
        return run
    except Exception as e:
        print(f"[{task}] wandb.init failed: {e}", flush=True)
        return None


def train_binary(task, train_df, val_df, test_df, backbone, device, args, wandb_mod):
    out_dir = Path(args.out_root) / f"per_attr_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = LoraDiscriminator(backbone, 2, args.dropout, args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{task}] AA-V2-style row-level LoRA r={args.lora_r} trainable={n_train} n={len(train_df)}", flush=True)

    tok = load_eos_tokenizer(backbone, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    kw = dict(tok=tok, pad_id=pad_id, max_length=args.max_length, workers=args.num_workers, with_sp=False)
    train_loader = make_id_loader(train_df, bs=args.train_bs, shuffle=True, balance=False, **kw)
    val_loader = make_id_loader(val_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    test_loader = make_id_loader(test_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    print(f"[{task}] peptide-only example={train_loader.dataset.ids[0][:8]}", flush=True)

    w = class_weight(train_df["label"].astype(int).tolist(), 2, device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lora_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    total = max(1, len(train_loader)) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run = wandb_init(
        wandb_mod,
        args,
        task,
        {"lora_r": args.lora_r, "select": "val_auroc", "n_train": len(train_df), "input": "AA ids only"},
    )

    best, patience, hist, t0 = -1e9, args.patience, [], time.time()
    best_path = out_dir / "best.pt"
    last_epoch = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
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
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n += 1
        val = evaluate(model, val_loader, device, task)
        rec = val[task]
        pack = metric_pack(rec, task)
        score = float(pack["m1"])
        tr_loss = running / max(1, n)
        hist.append({"epoch": epoch, "train_loss": tr_loss, **{k: pack[k] for k in ("m1", "m2", "m3")}})
        print(
            f"[{task}] epoch {epoch} train={tr_loss:.4f} AUROC={pack['m1']:.4f} AUPRC={pack['m2']:.4f} bal_acc={pack['m3']:.4f}",
            flush=True,
        )
        if run is not None:
            wandb_mod.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "eval/auroc": pack["m1"],
                    "eval/auprc": pack["m2"],
                    "eval/balanced_accuracy": pack["m3"],
                },
                step=epoch,
            )
        if score > best + 1e-4:
            best = score
            patience = args.patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "val_auroc": score, "val": rec}, best_path)
            print(f"[{task}] val AUROC improved -> {score:.4f}", flush=True)
        else:
            patience -= 1
            print(f"[{task}] AUROC not improving ({patience} patience left)", flush=True)
            if patience <= 0:
                print(f"[{task}] stop on AUROC", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, task)
    val = evaluate(model, val_loader, device, task)
    summary = {
        "task": task,
        "scheme": "aa_v2_tuned",
        "best_epoch": ckpt.get("epoch"),
        "end_epoch": last_epoch,
        "seconds": time.time() - t0,
        "n_trainable": n_train,
        "n_train": int(len(train_df)),
        "val": val,
        "test": test,
        "history": hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if run is not None:
        tp = metric_pack(test[task], task)
        wandb_mod.summary["test_auroc"] = tp["m1"]
        wandb_mod.summary["test_auprc"] = tp["m2"]
        wandb_mod.summary["test_balanced_accuracy"] = tp["m3"]
        wandb_mod.finish()
    return summary


def train_mic(train_df, val_df, test_df, tok, template, device, args, wandb_mod):
    task = "MIC"
    out_dir = Path(args.out_root) / "per_attr_MIC"
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_start = int(tok.convert_tokens_to_ids("SP00"))
    model = MicLoraDisc(clone_encoder(template), args.dropout, args.lora_r, args.lora_alpha, args.lora_dropout, sp_start).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MIC] atomic SP + LoRA r={args.lora_r} trainable={n_train} n={len(train_df)}", flush=True)

    kw = dict(tok=tok, pad_id=int(tok.pad_token_id), max_length=args.max_length, workers=args.num_workers, with_sp=True)
    train_loader = make_id_loader(train_df, bs=args.train_bs, shuffle=True, balance=False, **kw)
    val_loader = make_id_loader(val_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    test_loader = make_id_loader(test_df, bs=args.eval_bs, shuffle=False, balance=False, **kw)
    print(f"[MIC] example={train_loader.dataset.ids[0][:6]}", flush=True)

    w = class_weight(train_df["label"].astype(int).tolist(), 5, device)
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
    opt = torch.optim.AdamW(
        [
            {"params": lora, "lr": args.lora_lr},
            {"params": embed, "lr": args.embed_lr},
            {"params": head, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total = max(1, len(train_loader)) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    run = wandb_init(
        wandb_mod, args, task, {"lora_r": args.lora_r, "select": "val_balanced_accuracy", "input": "[SPxx]+AA"}
    )

    best, patience, hist, t0 = -1e9, args.patience, [], time.time()
    best_path = out_dir / "best.pt"
    last_epoch = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            y = batch.pop("labels").to(device)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                    loss = mic_loss(logits, y, w)
            else:
                logits = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                loss = mic_loss(logits, y, w)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            opt.step()
            sched.step()
            running += float(loss.detach().item())
            n += 1
        val = evaluate(model, val_loader, device, task)
        rec = val[task]
        pack = metric_pack(rec, task)
        score = float(pack["m1"])
        tr_loss = running / max(1, n)
        hist.append({"epoch": epoch, "train_loss": tr_loss, **{k: pack[k] for k in ("m1", "m2", "m3")}})
        print(
            f"[MIC] epoch {epoch} train={tr_loss:.4f} bal_acc={pack['m1']:.4f} macro_f1={pack['m2']:.4f} qwk={pack['m3']:.4f}",
            flush=True,
        )
        if run is not None:
            wandb_mod.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "eval/balanced_accuracy": pack["m1"],
                    "eval/macro_f1": pack["m2"],
                    "eval/qwk": pack["m3"],
                },
                step=epoch,
            )
        if score > best + 1e-4:
            best = score
            patience = args.patience
            torch.save({"model": model.state_dict(), "task": task, "epoch": epoch, "val_balanced_accuracy": score, "val": rec}, best_path)
            print(f"[MIC] val bal_acc improved -> {score:.4f}", flush=True)
        else:
            patience -= 1
            print(f"[MIC] bal_acc not improving ({patience} patience left)", flush=True)
            if patience <= 0:
                print("[MIC] stop on balanced_accuracy", flush=True)
                break

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, task)
    val = evaluate(model, val_loader, device, task)
    summary = {
        "task": task,
        "scheme": "atomic_sp_mic",
        "best_epoch": ckpt.get("epoch"),
        "end_epoch": last_epoch,
        "seconds": time.time() - t0,
        "n_trainable": n_train,
        "n_train": int(len(train_df)),
        "val": val,
        "test": test,
        "history": hist,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if run is not None:
        tp = metric_pack(test[task], task)
        wandb_mod.summary["test_balanced_accuracy"] = tp["m1"]
        wandb_mod.summary["test_macro_f1"] = tp["m2"]
        wandb_mod.summary["test_qwk"] = tp["m3"]
        wandb_mod.finish()
    return summary


def write_report(out_root: Path, summaries):
    rows = []
    for s in summaries:
        for split in ("val", "test"):
            rec = dict(s[split][s["task"]])
            rec.update(scheme=s["scheme"], split=split, task=s["task"])
            rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "metrics.csv", index=False)
    prev = {
        "AA-V2": {"GP": 0.8362, "GN": 0.8464, "FU": 0.7961, "SF": 0.8046, "MIC": 0.3543},
        "BPE-V3": {"GP": 0.8378, "GN": 0.8159, "FU": 0.7877, "SF": 0.7896, "MIC": 0.4252},
    }
    lines = [
        "# 混合判别器：谱/安全 = 调参 AA-V2；MIC = 原子 SPxx",
        "",
        "- GP/GN/FU/SF：行级氨基酸 id，LoRA r=32，按 **验证 AUROC** 存 ckpt",
        "- 二分类同时看：**AUROC、AUPRC、balanced accuracy**",
        "- MIC：`[SPxx]+肽` 原子菌种 token，按 **验证 balanced accuracy** 存 ckpt",
        "- MIC 同时看：**balanced accuracy、macro-F1、QWK**",
        "",
        "## 测试集",
        "",
        "| 任务 | 指标1 | 本方案 | AA-V2对照 | 指标2 | 本方案 | 指标3 | 本方案 |",
        "|------|--------|-------:|----------:|--------|-------:|--------|-------:|",
    ]
    for t in ALL_TASKS:
        s = [x for x in summaries if x["task"] == t][0]
        rec = s["test"][t]
        pack = metric_pack(rec, t)
        ref = prev["AA-V2"][t]
        lines.append(
            f"| {t} | {pack['m1_name']} | **{pack['m1']:.4f}** | {ref:.4f} | {pack['m2_name']} | {pack['m2']:.4f} | {pack['m3_name']} | {pack['m3']:.4f} |"
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
    ap.add_argument("--tok_dir", default=str(_ROOT / "models/discriminators/mic_tokenizer"))
    ap.add_argument("--out_root", default=str(_ROOT / "outputs/discriminators_hybrid"))
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_bs", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lora_lr", type=float, default=1.5e-4)
    ap.add_argument("--embed_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default="AMP_Model_Training")
    ap.add_argument("--wandb_group", default="six_attr_disc_hybrid")
    ap.add_argument("--wandb_run_prefix", default="disc_hybrid")
    ap.add_argument("--wandb_mode", default="disabled")
    args = ap.parse_args()

    v1.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tok_src = Path(args.tok_dir)
    tok_dst = out_root / "tokenizer"
    if not (tok_dst / "tokenizer_config.json").exists():
        shutil.copytree(tok_src, tok_dst, dirs_exist_ok=True)
    mic_tok = load_eos_tokenizer(str(tok_dst), local_files_only=True)
    mic_tok.padding_side = "left"
    mic_tok.pad_token = mic_tok.eos_token
    for sp in SP_TOKENS:
        if len(mic_tok.encode(sp, add_special_tokens=False)) != 1:
            raise RuntimeError(f"{sp} not atomic")
    mic_template = build_mic_encoder(args.backbone, mic_tok).cpu()

    os.environ["WANDB_MODE"] = args.wandb_mode
    try:
        import wandb

        wandb_mod = wandb
    except Exception as e:
        print("[warn] wandb", e, flush=True)
        wandb_mod = None

    full = v2.load_with_species(Path(args.csv_path))
    train_df, val_df, test_df = v2.apply_seq_split(full, Path(args.split_root))
    prep = {
        "binary": "AA-V2 row-level AA ids, select val AUROC, r=32",
        "mic": "[atomic SPxx] + AA ids, select val balanced_accuracy",
        "metrics_binary": ["auroc", "auprc", "balanced_accuracy"],
        "metrics_mic": ["balanced_accuracy", "macro_f1", "qwk"],
        "n_train_rows": int(len(train_df)),
    }
    (out_root / "prep.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
    print("[prep]", json.dumps(prep, ensure_ascii=False), flush=True)

    summaries = []
    for task in ("GP", "GN", "FU", "SF"):
        tr = v2.task_frame(train_df, task)
        va = v2.task_frame(val_df, task)
        te = v2.task_frame(test_df, task)
        summaries.append(train_binary(task, tr, va, te, args.backbone, device, args, wandb_mod))
    tr, va, te = v2.task_frame(train_df, "MIC"), v2.task_frame(val_df, "MIC"), v2.task_frame(test_df, "MIC")
    summaries.append(train_mic(tr, va, te, mic_tok, mic_template, device, args, wandb_mod))
    write_report(out_root, summaries)
    print("[done] hybrid report", out_root / "compare_report.md", flush=True)


if __name__ == "__main__":
    main()
