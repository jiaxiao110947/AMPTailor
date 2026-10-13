#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train per-attribute LoRA experts with polarity-margin (contrastive) loss.

Adapters: gp, gn, fu, sf, len. FU also sees Candida MIC-bin prefixes FU_MIC0–4.
Does not modify the original shared-LoRA bipolar trainer.
"""
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
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_cosine_schedule_with_warmup

EOS_TEXT = "<|endoftext|>"
LEN_RX = re.compile(r"^LEN(\d+)$")
ADAPTERS = ["gp", "gn", "fu", "sf", "len"]
TASK_TO_ADAPTER = {
    "GP0": "gp",
    "GP1": "gp",
    "GN0": "gn",
    "GN1": "gn",
    "FU0": "fu",
    "FU1": "fu",
    "SF0": "sf",
    "SF1": "sf",
}
for i in range(5):
    TASK_TO_ADAPTER[f"FU_MIC{i}"] = "fu"
OPPOSITE = {
    "GP0": "GP1",
    "GP1": "GP0",
    "GN0": "GN1",
    "GN1": "GN0",
    "FU0": "FU1",
    "FU1": "FU0",
    "SF0": "SF1",
    "SF1": "SF0",
    "FU_MIC0": "FU0",
    "FU_MIC1": "FU0",
    "FU_MIC2": "FU0",
    "FU_MIC3": "FU0",
    "FU_MIC4": "FU0",
}


def status(path: Optional[Path], msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as w:
            w.write(line + "\n")


def is_fungus_species(name: Any) -> bool:
    s = str(name).lower()
    return any(k in s for k in ("candida", "aspergillus", "cryptococcus", "saccharomyces"))


def _bit01(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        v = int(float(val))
    except Exception:
        return None
    return v if v in (0, 1) else None


def make_prompt(task: str) -> str:
    return f"{str(task).strip().upper()}:"


def lora_config(r: int, alpha: int, dropout: float, save_lm_head: bool) -> LoraConfig:
    kw = dict(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=["c_attn", "c_proj", "c_fc"],
        modules_to_save=["lm_head"] if save_lm_head else None,
        bias="none",
        fan_in_fan_out=True,
        use_rslora=True,
        init_lora_weights=True,
    )
    return LoraConfig(**kw)


def build_pools(csv_path: str, min_len: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    pools: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ADAPTERS}
    for _, r in df.iterrows():
        seq = str(r["sequence"]).strip().upper()
        if not seq or len(seq) < int(min_len):
            continue
        L = int(r["length"]) if "length" in r and pd.notna(r["length"]) else len(seq)
        for col, prefix in (("gram_pos", "GP"), ("gram_neg", "GN"), ("fungus", "FU")):
            b = _bit01(r.get(col))
            if b is None:
                continue
            task = f"{prefix}{b}"
            pools[TASK_TO_ADAPTER[task]].append(
                {"task": task, "prompt": make_prompt(task), "sequence": seq, "opp": make_prompt(OPPOSITE[task])}
            )
        bsf = _bit01(r.get("safety"))
        if bsf is not None:
            task = f"SF{bsf}"
            pools["sf"].append(
                {"task": task, "prompt": make_prompt(task), "sequence": seq, "opp": make_prompt(OPPOSITE[task])}
            )
        pools["len"].append(
            {"task": f"LEN{L}", "prompt": make_prompt(f"LEN{L}"), "sequence": seq, "opp": None}
        )
        if is_fungus_species(r.get("species")) and pd.notna(r.get("mic_bin")):
            mic = int(float(r["mic_bin"]))
            if 0 <= mic <= 4:
                task = f"FU_MIC{mic}"
                pools["fu"].append(
                    {
                        "task": task,
                        "prompt": make_prompt(task),
                        "sequence": seq,
                        "opp": make_prompt("FU0"),
                    }
                )
    return pools


def sample_weights(items: List[Dict[str, Any]]) -> np.ndarray:
    counts: Dict[str, int] = defaultdict(int)
    for it in items:
        counts[str(it["task"])] += 1
    w = []
    for it in items:
        c = max(1, counts[str(it["task"])])
        w.append(1.0 / c)
    arr = np.asarray(w, dtype=np.float64)
    arr /= arr.sum()
    return arr


def collate(tok: GPT2TokenizerFast, batch: List[Dict[str, Any]], max_length: int = 256) -> Dict[str, torch.Tensor]:
    pad_id = int(tok.pad_token_id)
    seqs, prompt_lens = [], []
    for it in batch:
        p_ids = tok(str(it["prompt"]), add_special_tokens=False)["input_ids"]
        s_ids = tok(str(it["sequence"]) + EOS_TEXT, add_special_tokens=False)["input_ids"]
        ids = (list(p_ids) + list(s_ids))[:max_length]
        seqs.append(ids)
        prompt_lens.append(min(len(p_ids), len(ids)))
    max_len = max(len(x) for x in seqs)
    max_len = ((max_len + 7) // 8) * 8
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    for i, ids in enumerate(seqs):
        pad = max_len - len(ids)
        input_ids[i, pad:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, pad:] = 1
        labels[i, pad:] = torch.tensor(ids, dtype=torch.long)
        labels[i, pad : pad + prompt_lens[i]] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": (attention_mask.cumsum(-1) - 1).clamp(min=0),
    }


def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    tok_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100).float()
    return (tok_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def draw_batch(items: List[Dict[str, Any]], probs: np.ndarray, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    idx = rng.choices(range(len(items)), weights=probs, k=int(n))
    return [items[i] for i in idx]


def _flatten_adapter_dir(sub: Path, name: str) -> None:
    nested = sub / name
    if (sub / "adapter_config.json").is_file() or not (nested / "adapter_config.json").is_file():
        return
    for item in nested.iterdir():
        dest = sub / item.name
        if dest.exists():
            dest.unlink()
        item.rename(dest)
    nested.rmdir()
    for extra in list(sub.iterdir()):
        if extra.is_dir() and extra.name != name:
            for child in extra.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(extra.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            extra.rmdir()


def save_all_adapters(model: PeftModel, out: Path, base: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ADAPTERS:
        sub = out / name
        sub.mkdir(parents=True, exist_ok=True)
        model.set_adapter(name)
        try:
            model.save_pretrained(str(sub), selected_adapters=[name])
        except TypeError:
            model.save_pretrained(str(sub))
        _flatten_adapter_dir(sub, name)
    index = {
        "base": base,
        "adapters": list(ADAPTERS),
        "task_to_adapter": dict(TASK_TO_ADAPTER),
        "len_adapter": "len",
    }
    (out / "adapter_index.json").write_text(json.dumps(index, indent=2) + "\n")


def load_multi_adapters(adapter_root: str, tokenizer_dir: str, device_str: str = "cuda"):
    root = Path(adapter_root)
    index = json.loads((root / "adapter_index.json").read_text())
    names = list(index["adapters"])
    tok = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base_path = Path(str(index["base"]))
    if not base_path.is_dir():
        rel = _ROOT / str(index["base"])
        base_path = rel if rel.is_dir() else (_ROOT / "models" / "AMPGuide")
    base = GPT2LMHeadModel.from_pretrained(str(base_path))
    base.config.eos_token_id = tok.eos_token_id
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(root / names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(str(root / n), adapter_name=n)
    device = torch.device(device_str)
    model.to(device).eval()
    return model, tok, device, index


def _params_for_all_adapters(model, adapter_names=ADAPTERS):
    """Collect the union of trainable adapter parameters without unfreezing the base."""
    original = model.active_adapter
    params = {}
    try:
        for name in adapter_names:
            model.set_adapter(name)
            for param in model.parameters():
                if param.requires_grad:
                    params[id(param)] = param
    finally:
        model.set_adapter(original)
    return list(params.values())


def _step_adapter_optimizer(params, opt, sched, pending, grad_accum):
    """Flush accumulated gradients, including a shorter final window."""
    if pending <= 0:
        return
    active = [p for p in params if p.grad is not None]
    if pending < grad_accum:
        for param in active:
            param.grad.mul_(float(grad_accum) / float(pending))
    torch.nn.utils.clip_grad_norm_(active, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


def train(args) -> Path:
    if min(int(args.max_steps), int(args.train_bs), int(args.grad_accum), int(args.save_steps), int(args.logging_steps)) <= 0:
        raise ValueError("step, batch, accumulation, save, and logging counts must be positive")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    status_path = Path(args.status_path) if args.status_path else None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    pools = build_pools(args.csv_path, args.min_seq_len)
    sizes = {k: len(v) for k, v in pools.items()}
    (out / "pool_sizes.json").write_text(json.dumps(sizes, indent=2) + "\n")
    status(status_path, f"train pools {sizes}")
    probs = {k: sample_weights(v) for k, v in pools.items() if v}

    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer_dir)
    if tok.eos_token is None:
        tok.eos_token = EOS_TEXT
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = GPT2LMHeadModel.from_pretrained(args.init_model_dir)
    base.config.pad_token_id = tok.pad_token_id
    base.config.eos_token_id = tok.eos_token_id
    base.config.use_cache = False
    save_lm = not bool(args.lora_no_lm_head)
    cfg0 = lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, save_lm)
    try:
        model = get_peft_model(base, cfg0, adapter_name="gp")
    except TypeError:
        model = get_peft_model(base, cfg0)
        if "gp" not in (getattr(model, "peft_config", {}) or {}):
            # default adapter name is "default"
            model.add_adapter("gp", lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, save_lm))
            try:
                model.delete_adapter("default")
            except Exception:
                pass
    for name in ADAPTERS:
        if name == "gp":
            continue
        model.add_adapter(name, lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, save_lm))
    model.set_adapter("gp")
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    status(status_path, f"trainable={n_tr:,} device={device} peft_adapters={list(model.peft_config)}")

    params = _params_for_all_adapters(model)
    opt = AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    valid_steps = sum(bool(pools[ADAPTERS[(step - 1) % len(ADAPTERS)]]) for step in range(1, int(args.max_steps) + 1))
    if not valid_steps:
        raise ValueError("No samples available for the requested adapter training steps")
    total_updates = (valid_steps + int(args.grad_accum) - 1) // int(args.grad_accum)
    warmup = max(1, int(total_updates * float(args.warmup_ratio)))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_updates)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()
    running = 0.0
    n_seen = 0
    opt.zero_grad(set_to_none=True)
    order = list(ADAPTERS)
    pending = 0
    for step in range(1, int(args.max_steps) + 1):
        adapter = order[(step - 1) % len(order)]
        items = pools[adapter]
        if not items:
            continue
        batch = draw_batch(items, probs[adapter], int(args.train_bs), rng)
        model.set_adapter(adapter)
        enc = collate(tok, batch)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outp = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                position_ids=enc["position_ids"],
            )
            nll = nll_from_logits(outp.logits.float(), enc["labels"])
            loss_lm = nll.mean()
            loss_m = torch.zeros((), device=device)
            if float(args.lambda_contrastive) > 0:
                opp_batch = []
                keep = []
                for it in batch:
                    if it.get("opp"):
                        opp_batch.append({"prompt": it["opp"], "sequence": it["sequence"]})
                        keep.append(True)
                    else:
                        keep.append(False)
                if opp_batch:
                    enc_opp = collate(tok, opp_batch)
                    enc_opp = {k: v.to(device) for k, v in enc_opp.items()}
                    out_opp = model(
                        input_ids=enc_opp["input_ids"],
                        attention_mask=enc_opp["attention_mask"],
                        position_ids=enc_opp["position_ids"],
                    )
                    nll_opp = nll_from_logits(out_opp.logits.float(), enc_opp["labels"])
                    nll_true = nll[torch.tensor(keep, device=device)]
                    loss_m = F.relu(float(args.margin) + nll_true - nll_opp).mean()
            loss = (loss_lm + float(args.lambda_contrastive) * loss_m) / max(1, int(args.grad_accum))
        loss.backward()
        pending += 1
        running += float(loss_lm.detach())
        n_seen += 1
        if pending == int(args.grad_accum) or step == int(args.max_steps):
            _step_adapter_optimizer(params, opt, sched, pending, int(args.grad_accum))
            pending = 0
        if step % int(args.logging_steps) == 0 or step == 1:
            lr = sched.get_last_lr()[0]
            status(
                status_path,
                f"step {step}/{args.max_steps} adapter={adapter} lm={running/max(1,n_seen):.4f} "
                f"margin={float(loss_m):.4f} lr={lr:.2e} {time.time()-t0:.0f}s",
            )
            running = 0.0
            n_seen = 0
        if step % int(args.save_steps) == 0 or step == int(args.max_steps):
            ckpt = out / f"checkpoint-{step}"
            save_all_adapters(model, ckpt, args.init_model_dir)
            (ckpt / "train_state.json").write_text(
                json.dumps({"step": step, "elapsed_sec": time.time() - t0}, indent=2) + "\n"
            )
            status(status_path, f"saved {ckpt}")
    if pending:
        _step_adapter_optimizer(params, opt, sched, pending, int(args.grad_accum))
    # The final raw step can be skipped when its adapter pool is empty.
    final_ckpt = out / f"checkpoint-{int(args.max_steps)}"
    if not final_ckpt.is_dir():
        save_all_adapters(model, final_ckpt, args.init_model_dir)
        (final_ckpt / "train_state.json").write_text(
            json.dumps({"step": int(args.max_steps), "elapsed_sec": time.time() - t0}) + "\n"
        )
    save_all_adapters(model, out / "adapters", args.init_model_dir)
    status(status_path, f"train done {time.time()-t0:.0f}s")
    return out / f"checkpoint-{int(args.max_steps)}"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--init_model_dir", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--status_path", default="")
    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--train_bs", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_seq_len", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_no_lm_head", action="store_true")
    p.add_argument("--lambda_contrastive", type=float, default=0.4)
    p.add_argument("--margin", type=float, default=0.2)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
