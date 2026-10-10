#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六属性微调对比辅助库（full / freeze_top / bitfit / LoRA 统一加载）。"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments

try:
    from peft import PeftModel

    _PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    _PEFT_AVAILABLE = False


def _gpt2_block_idx(param_name: str) -> Optional[int]:
    m = re.search(r"transformer\.h\.(\d+)\.", str(param_name))
    return int(m.group(1)) if m else None


def apply_finetune_mode(model: GPT2LMHeadModel, mode: str, freeze_bottom_layers: int = 10) -> None:
    mode = str(mode).strip().lower()
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "freeze_top":
        n_freeze = int(freeze_bottom_layers)
        n_layer = int(model.config.n_layer)
        train_blocks = set(range(n_freeze, n_layer))
        for name, p in model.named_parameters():
            blk = _gpt2_block_idx(name)
            if blk is not None:
                p.requires_grad = blk in train_blocks
            elif name.startswith("lm_head") or name.startswith("transformer.ln_f"):
                p.requires_grad = True
            elif name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
                p.requires_grad = False
            else:
                p.requires_grad = True
    elif mode == "bitfit":
        for name, p in model.named_parameters():
            p.requires_grad = "bias" in name or name.startswith("lm_head")
    else:
        raise ValueError(f"unknown finetune_mode: {mode}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[Finetune] {mode} trainable={trainable:,}/{total:,} ({100 * trainable / max(1, total):.2f}%)",
        flush=True,
    )


def make_get_peft_replacement(pre_args: Any) -> Callable:
    mode = str(pre_args.finetune_mode).strip().lower()

    def _fake_get_peft_model(model: GPT2LMHeadModel, _config: Any) -> GPT2LMHeadModel:
        if int(getattr(pre_args, "gradient_checkpointing", 0)) == 1:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            print("[Finetune] gradient_checkpointing enabled", flush=True)
        apply_finetune_mode(model, mode, int(getattr(pre_args, "freeze_bottom_layers", 10)))
        return model

    return _fake_get_peft_model


def wrap_build_training_args(orig_fn: Callable, pre_args: Any) -> Callable:
    def _wrapped(a, outdir, use_auc_for_best: bool, *, do_eval: bool = True) -> TrainingArguments:
        ta = orig_fn(a, outdir, use_auc_for_best, do_eval=do_eval)
        sig = inspect.signature(TrainingArguments.__init__)
        supported = set(sig.parameters.keys())
        patches: dict = {}
        if int(getattr(pre_args, "bf16", 0)) == 1 and "bf16" in supported:
            patches["bf16"] = True
        if int(getattr(pre_args, "gradient_checkpointing", 0)) == 1 and "gradient_checkpointing" in supported:
            patches["gradient_checkpointing"] = True
        if not patches:
            return ta
        kwargs = {k: getattr(ta, k) for k in ta.__dict__ if not k.startswith("_")}
        kwargs.update(patches)
        filtered = {k: v for k, v in kwargs.items() if k in supported}
        return TrainingArguments(**filtered)

    return _wrapped


def make_trainer_subclass(base_cls: Type, five_attr_model_cls: Type) -> Type:
    class TrainerWithFinetuneResume(base_cls):
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if model is None:
                model = self.model
            if isinstance(model, five_attr_model_cls):
                ckpt = str(resume_from_checkpoint)
                lm = model.lm
                if hasattr(lm, "load_adapter") and (Path(ckpt) / "adapter_config.json").exists():
                    if hasattr(lm, "active_adapters"):
                        active = lm.active_adapters[0]
                    else:
                        active = lm.active_adapter
                    lm.load_adapter(ckpt, active, is_trainable=True)
                    print(f"[Resume] PEFT adapter loaded from {ckpt}", flush=True)
                elif (Path(ckpt) / "config.json").exists():
                    loaded = GPT2LMHeadModel.from_pretrained(ckpt)
                    lm.load_state_dict(loaded.state_dict())
                    print(f"[Resume] full model loaded from {ckpt}", flush=True)
                sh = Path(ckpt) / "stop_head.pt"
                if sh.exists():
                    model.stop_head.load_state_dict(torch.load(str(sh), map_location="cpu", weights_only=True))
                    print(f"[Resume] StopHead loaded from {sh}", flush=True)
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    return TrainerWithFinetuneResume


def load_tokenizer(tokenizer_dir: str) -> GPT2TokenizerFast:
    from importlib.util import spec_from_file_location, module_from_spec

    lib_path = Path(__file__).resolve().parent / "07.five_attr_classification_lora_lib.py"
    spec = spec_from_file_location("five_attr_lora_lib", str(lib_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_tokenizer(tokenizer_dir)


def load_single_lora_model(model_dir: str, tokenizer_dir: str = "", device_str: str = "cuda"):
    from importlib.util import spec_from_file_location, module_from_spec

    lib_path = Path(__file__).resolve().parent / "07.five_attr_classification_lora_lib.py"
    spec = spec_from_file_location("five_attr_lora_lib", str(lib_path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_single_lora_model(model_dir, tokenizer_dir=tokenizer_dir, device_str=device_str)


def load_five_attr_model(
    model_dir: str,
    tokenizer_dir: str = "",
    device_str: str = "cuda",
) -> Tuple[Any, GPT2TokenizerFast, torch.device]:
    """自动识别 LoRA adapter 或全参 checkpoint。"""
    model_root = Path(model_dir)
    tok_root = Path(tokenizer_dir) if tokenizer_dir else model_root
    adapter_cfg = model_root / "adapter_config.json"
    lora_sub = model_root / "lora_adapter"
    if adapter_cfg.exists() or (lora_sub / "adapter_config.json").exists():
        return load_single_lora_model(model_dir, tokenizer_dir=tokenizer_dir, device_str=device_str)
    tok = load_tokenizer(str(tok_root))
    model = GPT2LMHeadModel.from_pretrained(str(model_root))
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    ds = str(device_str).strip() or "cuda"
    device = torch.device(ds)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA 但当前环境不可用 CUDA GPU")
    model.to(device).eval()
    return model, tok, device


def is_lora_checkpoint(model_dir: str) -> bool:
    root = Path(model_dir)
    return (root / "adapter_config.json").exists() or (root / "lora_adapter" / "adapter_config.json").exists()
