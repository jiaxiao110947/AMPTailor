#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared five-attribute LoRA generation/eval helpers.

This file is intentionally self-contained because training and evaluation scripts
load it by absolute path at runtime.
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

import hashlib
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    from peft import PeftModel

    _PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    _PEFT_AVAILABLE = False


ATTRS = ["gram_pos", "gram_neg", "fungus", "safety", "length"]
BASE_TASKS = ["GP1", "GN1", "FU1", "SF0", "SF1"]
TASKS = BASE_TASKS
TASK_TO_WEIGHT_INDEX = {"GP1": 0, "GN1": 1, "FU1": 2, "SF0": 3, "SF1": 3, "LENGTH": 4}
FORBIDDEN_ATTR_BITS = {"0000", "0001"}
COMBOS_14 = ["0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]
REGULAR_CONDITIONS_6 = ["1XX0", "1XX1", "X1X0", "X1X1", "XX10", "XX11"]
DEFAULT_EVAL_LENGTHS = [6, 8, 10, 12, 16, 20, 24, 28, 32, 40]
LENGTH_TIERS = [
    ("ultra_short", "ultra_short", 1, 10),
    ("short", "short", 11, 16),
    ("medium", "medium", 17, 24),
    ("medium_long", "medium_long", 25, 10**9),
]
AA20_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA20_SET = set(AA20_LIST)
EOS_TEXT = "<|endoftext|>"
GENERATION_SCHEME_HEADER = "five_attr_single_lora_prompt_fusion"
CANON_RX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
FIXED_TASK_RX = re.compile(r"^(GP1|GN1|FU1|SF0|SF1)$")
LENGTH_TASK_RX = re.compile(r"^LEN([1-9][0-9]*)$")
MIC_TASK_RX = re.compile(r"^MIC([0-4])$")
SPECIES_TASK_RX = re.compile(r"^SP([0-9]{2})$")
SPECIES_MIC_TASK_RX = re.compile(r"^SP([0-9]{2})_MIC([0-4])$")
CONDITION_RX = re.compile(r"^([01X]{4})(?::([1-9][0-9]*))?$")


def make_length_task(n: int) -> str:
    return f"LEN{int(n)}"


def is_length_task(task: str) -> bool:
    return LENGTH_TASK_RX.fullmatch(str(task).strip().upper()) is not None


def length_from_task(task: str) -> int:
    m = LENGTH_TASK_RX.fullmatch(str(task).strip().upper())
    if m is None:
        raise ValueError(f"not a length task: {task}")
    return int(m.group(1))


def make_task_prompt(task: str) -> str:
    t = str(task).strip().upper()
    if (
        FIXED_TASK_RX.fullmatch(t)
        or LENGTH_TASK_RX.fullmatch(t)
        or MIC_TASK_RX.fullmatch(t)
        or SPECIES_TASK_RX.fullmatch(t)
        or SPECIES_MIC_TASK_RX.fullmatch(t)
    ):
        return f"{t}:"
    raise ValueError(f"unknown task: {task}")


def make_single_attr_prompt(attr: str, val: int) -> str:
    attr_s = str(attr).strip().lower()
    val_i = int(val)
    mapping = {
        ("gram_pos", 1): "GP1",
        ("gram_neg", 1): "GN1",
        ("fungus", 1): "FU1",
        ("safety", 0): "SF0",
        ("safety", 1): "SF1",
        ("safe", 0): "SF0",
        ("safe", 1): "SF1",
        ("toxic", 0): "SF0",
        ("toxic", 1): "SF1",
    }
    if attr_s == "length":
        return make_task_prompt(make_length_task(val_i))
    task = mapping.get((attr_s, val_i))
    if task is None:
        raise ValueError(f"unsupported attr/value prompt: {attr}={val}")
    return make_task_prompt(task)


def clean_seq(seq: str) -> str:
    s = re.sub(r"[^A-Z]", "", str(seq).strip().upper())
    return "".join(ch for ch in s if ch in AA20_SET)


def parse_condition(condition: str, default_length: Optional[int] = None) -> Tuple[str, int]:
    s = str(condition).strip().upper()
    m = CONDITION_RX.fullmatch(s)
    if m is None:
        raise ValueError(f"invalid condition: {condition} (expected e.g. 1010:24 or 1XX0:32)")
    bits = m.group(1)
    if bits in FORBIDDEN_ATTR_BITS:
        raise ValueError(f"forbidden condition: {condition}")
    if m.group(2) is not None:
        target_len = int(m.group(2))
    elif default_length is not None:
        target_len = int(default_length)
    else:
        raise ValueError(f"condition missing explicit length: {condition}")
    if target_len <= 0:
        raise ValueError(f"invalid target length: {target_len}")
    return bits, target_len


def condition_to_targets(condition: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    bits, _ = parse_condition(condition, default_length=24)
    out: List[Optional[int]] = []
    for ch in bits:
        out.append(None if ch == "X" else int(ch))
    return out[0], out[1], out[2], out[3]


def condition_to_task_names(condition: str, default_length: Optional[int] = None) -> List[str]:
    bits, target_len = parse_condition(condition, default_length=default_length)
    gp, gn, fu, sf = condition_to_targets(f"{bits}:{target_len}")
    tasks: List[str] = []
    if gp == 1:
        tasks.append("GP1")
    if gn == 1:
        tasks.append("GN1")
    if fu == 1:
        tasks.append("FU1")
    if sf is None:
        raise ValueError(f"safety bit cannot be X in this scheme: {condition}")
    tasks.append("SF0" if sf == 0 else "SF1")
    tasks.append(make_length_task(target_len))
    return tasks


def condition_to_content_task_names(condition: str, default_length: Optional[int] = None) -> List[str]:
    bits, target_len = parse_condition(condition, default_length=default_length)
    gp, gn, fu, sf = condition_to_targets(f"{bits}:{target_len}")
    tasks: List[str] = []
    if gp == 1:
        tasks.append("GP1")
    if gn == 1:
        tasks.append("GN1")
    if fu == 1:
        tasks.append("FU1")
    if sf is None:
        raise ValueError(f"safety bit cannot be X in this scheme: {condition}")
    tasks.append("SF0" if sf == 0 else "SF1")
    return tasks


def condition_to_length_task_names(condition: str, default_length: Optional[int] = None) -> List[str]:
    _, target_len = parse_condition(condition, default_length=default_length)
    return [make_length_task(target_len)]


def parse_combo_text(text: str, default_length: Optional[int] = None) -> List[str]:
    combos: List[str] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        bits, target_len = parse_condition(item, default_length=default_length)
        if bits in FORBIDDEN_ATTR_BITS:
            continue
        combos.append(f"{bits}:{target_len}")
    if not combos and str(text).strip():
        raise ValueError(f"no valid combos parsed from: {text}")
    return sorted(dict.fromkeys(combos))


def normalize_condition(condition: str, default_length: Optional[int] = None) -> str:
    bits, target_len = parse_condition(condition, default_length=default_length)
    return f"{bits}:{target_len}"


def select_combos(
    gram_pos: str = "all",
    gram_neg: str = "all",
    fungus: str = "all",
    safety: str = "all",
    toxic: str = "all",
    target_length: Optional[int] = None,
    combos: Optional[Sequence[str]] = None,
) -> List[str]:
    if combos:
        return parse_combo_text(",".join(str(x) for x in combos), default_length=target_length)
    vals: List[List[str]] = []
    for x in [gram_pos, gram_neg, fungus]:
        s = str(x).strip().lower()
        if s == "all":
            vals.append(["0", "1"])
        elif s in {"0", "1", "x"}:
            vals.append(["X"] if s == "x" else [s])
        else:
            raise ValueError(f"selector must be 0/1/all/X, got {x}")
    sf_s = str(safety if str(safety).lower() != "all" else toxic).strip().lower()
    if sf_s not in {"0", "1", "all", "x"}:
        raise ValueError(f"selector must be 0/1/all/X, got {safety}")
    sf_vals = ["0", "1"] if sf_s == "all" else (["X"] if sf_s == "x" else [sf_s])
    out: List[str] = []
    for gp in vals[0]:
        for gn in vals[1]:
            for fu in vals[2]:
                for sf in sf_vals:
                    bits = f"{gp}{gn}{fu}{sf}".upper()
                    if bits not in FORBIDDEN_ATTR_BITS:
                        out.append(f"{bits}:{int(target_length)}" if target_length else bits)
    return sorted(dict.fromkeys(out))


def combos_to_jsonable(combos: Iterable[str]) -> List[str]:
    return [str(x) for x in combos]


def expand_conditions_with_lengths(conditions: Sequence[str], lengths: Sequence[int]) -> List[str]:
    out: List[str] = []
    for c in conditions:
        s = str(c).strip().upper()
        if ":" in s:
            out.append(normalize_condition(s))
        else:
            for n in lengths:
                if int(n) > 0:
                    out.append(normalize_condition(s, default_length=int(n)))
    return sorted(dict.fromkeys(out))


def expand_lenonly_targets(lengths: Sequence[int], attr_bits: str = "XX10") -> List[str]:
    bits = str(attr_bits).strip().upper()
    return expand_conditions_with_lengths([bits], lengths)


def parse_length_list(text: str, single_fallback: Optional[int] = None) -> List[int]:
    vals: List[int] = []
    for raw in str(text or "").split(","):
        item = raw.strip()
        if item:
            vals.append(int(item))
    if not vals and single_fallback is not None:
        vals = [int(single_fallback)]
    return sorted(dict.fromkeys(int(x) for x in vals if int(x) > 0))


def parse_attr_weights(text: str) -> Dict[str, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if str(x).strip()]
    if len(vals) == 4:
        vals = [vals[0], vals[1], vals[2], vals[3], 1.0]
    if len(vals) != 5:
        raise ValueError("attr_weights must contain 5 floats (GP,GN,FU,SF,LENGTH)")
    return {"GP1": vals[0], "GN1": vals[1], "FU1": vals[2], "SF0": vals[3], "SF1": vals[3], "LENGTH": vals[4]}


def task_weight(task: str, weights: Dict[str, float]) -> float:
    t = str(task).strip().upper()
    if is_length_task(t):
        return float(weights.get("LENGTH", 1.0))
    if SPECIES_MIC_TASK_RX.fullmatch(t):
        return float(weights.get("SP_MIC", weights.get("MIC", weights.get("SP", 1.0))))
    if MIC_TASK_RX.fullmatch(t):
        return float(weights.get("MIC", weights.get(t, 1.0)))
    if SPECIES_TASK_RX.fullmatch(t):
        return float(weights.get("SP", weights.get(t, 1.0)))
    return float(weights.get(t, 1.0))


def length_tier_for_target(target_len: int) -> Tuple[str, str]:
    n = int(target_len)
    for tier_id, tier_label, lo, hi in LENGTH_TIERS:
        if int(lo) <= n <= int(hi):
            return str(tier_id), str(tier_label)
    return LENGTH_TIERS[-1][0], LENGTH_TIERS[-1][1]


def compute_length_stop_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    target_seq_lens: torch.Tensor,
    stop_head: torch.nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Binary CE over sequence token positions: 1 at target stop token, 0 before it."""
    logits = stop_head(hidden_states).squeeze(-1)
    losses: List[torch.Tensor] = []
    n_terms = 0
    for i in range(int(labels.size(0))):
        tgt = int(target_seq_lens[i].item())
        if tgt <= 0:
            continue
        valid = labels[i].ne(-100).nonzero(as_tuple=False).view(-1)
        if valid.numel() <= 0:
            continue
        seq_pos = valid[: min(tgt, int(valid.numel()))]
        if seq_pos.numel() <= 0:
            continue
        y = torch.zeros((int(seq_pos.numel()),), dtype=logits.dtype, device=logits.device)
        y[-1] = 1.0
        losses.append(F.binary_cross_entropy_with_logits(logits[i, seq_pos], y))
        n_terms += int(seq_pos.numel())
    if not losses:
        return logits.sum() * 0.0, {"stop_loss_n_terms": 0.0}
    return torch.stack(losses).mean(), {"stop_loss_n_terms": float(n_terms)}


class StopHead(nn.Module):
    """Predict whether the current sequence position should be the stop point."""

    def __init__(self, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        h = int(hidden_size)
        mid = max(32, h // 4)
        self.net = nn.Sequential(
            nn.Linear(h, mid),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mid, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def load_tokenizer(tokenizer_dir: str) -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))
    if tok.eos_token is None:
        tok.eos_token = EOS_TEXT
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _build_aa_char_maps(tok: GPT2TokenizerFast) -> Tuple[Dict[int, str], torch.Tensor]:
    id_to_aa: Dict[int, str] = {}
    eos_id = int(tok.eos_token_id)
    # len(tokenizer) includes added tokens; vocab_size covers only the base vocabulary.
    for i in range(len(tok)):
        txt = tok.decode([i], clean_up_tokenization_spaces=False)
        t = "".join(ch for ch in str(txt).upper() if not ch.isspace())
        if len(t) == 1 and t in AA20_SET and i not in id_to_aa:
            id_to_aa[int(i)] = t
    covered = set(id_to_aa.values())
    miss = [x for x in AA20_LIST if x not in covered]
    if miss:
        raise RuntimeError(f"tokenizer missing single-AA tokens: {miss}")
    allowed = sorted(set(list(id_to_aa.keys()) + [eos_id]))
    return id_to_aa, torch.tensor(allowed, dtype=torch.long)


def load_single_lora_model(model_dir: str, tokenizer_dir: str = "", device_str: str = "cuda") -> Tuple[Any, GPT2TokenizerFast, torch.device]:
    if not _PEFT_AVAILABLE:
        raise RuntimeError("peft is required for LoRA generation, but is not installed")
    import json

    model_root = Path(model_dir)
    tok_root = Path(tokenizer_dir) if tokenizer_dir else model_root
    tok = load_tokenizer(str(tok_root))
    adapter_root = model_root / "lora_adapter" if (model_root / "lora_adapter").exists() else model_root
    cfg_path = adapter_root / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_root}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        base_name = str(model_root / "base_model") if (model_root / "base_model").exists() else str(model_root)
    base_path = Path(str(base_name))
    if not base_path.is_dir():
        fallback = _ROOT / "models" / "AMPGuide"
        rel = _ROOT / str(base_name)
        if rel.is_dir():
            base_path = rel
        elif fallback.is_dir():
            base_path = fallback
    base = GPT2LMHeadModel.from_pretrained(str(base_path))
    base.config.eos_token_id = tok.eos_token_id
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_root))
    ds = str(device_str).strip() or "cuda"
    device = torch.device(ds)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA 但当前环境不可用 CUDA GPU")
    model.to(device).eval()
    return model, tok, device


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    x = logits.clone()
    if int(top_k) > 0:
        v, _ = torch.topk(x, k=min(int(top_k), x.size(-1)))
        x = torch.where(x < v[..., -1, None], torch.full_like(x, -float("inf")), x)
    if float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        csum = torch.cumsum(probs, dim=-1)
        mask = csum > float(top_p)
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, -float("inf")), sorted_logits)
        x = torch.full_like(x, -float("inf"))
        x.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return x


def _apply_no_repeat_ngram(logits: torch.Tensor, gen_ids: torch.Tensor, n: int) -> torch.Tensor:
    n = int(n)
    if n <= 0 or gen_ids.ndim != 2 or gen_ids.size(0) != 1:
        return logits
    ids = [int(x) for x in gen_ids[0].tolist()]
    if len(ids) < max(1, n - 1):
        return logits
    banned = set()
    if n == 1:
        banned.update(ids)
    else:
        prefix = tuple(ids[-(n - 1) :])
        for i in range(0, len(ids) - n + 1):
            if tuple(ids[i : i + n - 1]) == prefix:
                banned.add(int(ids[i + n - 1]))
    if banned:
        logits[:, list(banned)] = -float("inf")
    return logits


def _sample_next(logits: torch.Tensor, do_sample: bool = True) -> torch.Tensor:
    if not bool(do_sample):
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _apply_repetition_penalty(logits: torch.Tensor, gen_ids: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
    rp = float(repetition_penalty)
    if rp == 1.0 or gen_ids.numel() <= 0:
        return logits
    uniq = torch.unique(gen_ids)
    pos_mask = logits[:, uniq] > 0
    logits[:, uniq] = torch.where(pos_mask, logits[:, uniq] / rp, logits[:, uniq] * rp)
    return logits


def _combo_attempt_seed(seed: int, combo: str, attempt_idx: int) -> int:
    payload = f"{int(seed)}|{str(combo)}|{int(attempt_idx)}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def fused_combo_logits(
    model: Any,
    tok: GPT2TokenizerFast,
    device: torch.device,
    combo: str,
    generated_ids: torch.Tensor,
    attr_weights: Optional[Sequence[float]] = None,
    max_length: int = 512,
    default_length: Optional[int] = None,
    prompt_mode: str = "fused",
) -> torch.Tensor:
    mode = str(prompt_mode or "fused").strip().lower()
    if mode == "length_only":
        tasks = condition_to_length_task_names(combo, default_length=default_length)
    elif mode == "content_only":
        tasks = condition_to_content_task_names(combo, default_length=default_length)
    elif mode == "fused":
        tasks = condition_to_task_names(combo, default_length=default_length)
    else:
        raise ValueError(f"unknown prompt_mode: {prompt_mode}")
    gen_list = [int(x) for x in generated_ids[0].tolist()] if generated_ids.numel() > 0 else []
    weights = parse_attr_weights(",".join(str(x) for x in (attr_weights or [1.0, 1.0, 1.0, 1.0, 1.0])))
    acc = None
    total_w = 0.0
    for task in tasks:
        prompt_ids = tok(make_task_prompt(task), add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + gen_list
        if len(input_ids) > int(max_length):
            input_ids = input_ids[-int(max_length) :]
        tens = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=tens)
        logits = out.logits[:, -1, :]
        w = task_weight(task, weights)
        acc = logits * w if acc is None else (acc + logits * w)
        total_w += w
    if acc is None or total_w <= 0.0:
        raise RuntimeError("failed to produce fused logits")
    return acc / total_w


def generate_sequences_for_combos(
    model: Any,
    tok: GPT2TokenizerFast,
    device: torch.device,
    out_fasta: str,
    combos: Sequence[str],
    n_per_combo: int,
    min_seq_len: int,
    max_seq_len: int,
    min_new_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    restrict_aa_tokens: bool,
    seed: int,
    max_attempts_multiplier: int,
    attr_weights: Optional[Sequence[float]] = None,
    max_model_length: int = 512,
    do_sample: bool = True,
    shuffle_combos: bool = True,
    return_stats: bool = False,
    enforce_exact_length: bool = False,
    default_length: int = 24,
    prompt_mode: str = "fused",
) -> Any:
    if enforce_exact_length:
        raise ValueError("Forced length is no longer supported; use EOS stopping with enforce_exact_length=False")
    combos_in = [normalize_condition(str(c), default_length=default_length) for c in combos]
    combos_in = [c for c in combos_in if c.split(":", 1)[0] not in FORBIDDEN_ATTR_BITS]
    if not combos_in:
        raise ValueError("combos is empty after filtering")
    Path(out_fasta).parent.mkdir(parents=True, exist_ok=True)
    total_target = len(combos_in) * int(n_per_combo)
    rng = random.Random(int(seed))
    combo_order = list(combos_in)
    if bool(shuffle_combos):
        rng.shuffle(combo_order)

    eos_id = int(tok.eos_token_id)
    id_to_aa: Dict[int, str] = {}
    aa_allowed_ids = None
    if bool(restrict_aa_tokens):
        id_to_aa, aa_allowed_ids_cpu = _build_aa_char_maps(tok)
        aa_allowed_ids = aa_allowed_ids_cpu.to(device)

    written = 0
    reject_too_short = 0
    reject_invalid = 0
    reject_len_out_of_range = 0
    reject_len_mismatch = 0
    lengths: List[int] = []
    unique_sequences = set()
    per_combo_written: Dict[str, int] = {c: 0 for c in combo_order}
    per_combo_attempts: Dict[str, int] = {c: 0 for c in combo_order}
    per_combo_attempt_limit = max(1, int(max_attempts_multiplier) * int(n_per_combo))
    with open(out_fasta, "w", encoding="utf-8") as w:
        for combo in combo_order:
            bits, target_len = parse_condition(combo, default_length=default_length)
            gen_limit = int(max_new_tokens)
            while per_combo_written[combo] < int(n_per_combo) and per_combo_attempts[combo] < per_combo_attempt_limit:
                combo_attempt_idx = int(per_combo_attempts[combo])
                torch.manual_seed(_combo_attempt_seed(int(seed), combo, combo_attempt_idx))
                gen_ids = torch.empty((1, 0), dtype=torch.long, device=device)
                per_combo_attempts[combo] += 1
                stop = False
                for step_idx in range(gen_limit):
                    logits = fused_combo_logits(
                        model=model,
                        tok=tok,
                        device=device,
                        combo=combo,
                        generated_ids=gen_ids,
                        attr_weights=attr_weights,
                        max_length=int(max_model_length),
                        default_length=default_length,
                        prompt_mode=prompt_mode,
                    )
                    logits = logits / max(1e-6, float(temperature))
                    logits = _apply_repetition_penalty(logits, gen_ids, float(repetition_penalty))
                    logits = _apply_no_repeat_ngram(logits, gen_ids, int(no_repeat_ngram_size))
                    if aa_allowed_ids is not None:
                        masked = torch.full_like(logits, -float("inf"))
                        masked[:, aa_allowed_ids] = logits[:, aa_allowed_ids]
                        logits = masked
                    logits = _top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
                    nxt = _sample_next(logits, do_sample=bool(do_sample))
                    gen_ids = torch.cat([gen_ids, nxt], dim=1)
                    if int(nxt.item()) == eos_id:
                        stop = True
                        break
                if (not bool(enforce_exact_length)) and (not stop) and gen_ids.numel() < int(min_new_tokens):
                    reject_too_short += 1
                    continue
                ids_out = [int(x) for x in gen_ids[0].tolist() if int(x) != eos_id]
                if bool(restrict_aa_tokens):
                    seq = "".join(id_to_aa[i] for i in ids_out if i in id_to_aa)
                else:
                    seq = clean_seq(tok.decode(gen_ids[0], skip_special_tokens=True))
                if CANON_RX.fullmatch(seq or "") is None:
                    reject_invalid += 1
                    continue
                if len(seq) < int(min_seq_len) or len(seq) > int(max_seq_len):
                    reject_len_out_of_range += 1
                    continue
                gp, gn, fu, sf = condition_to_targets(combo)
                gp_txt = "X" if gp is None else str(gp)
                gn_txt = "X" if gn is None else str(gn)
                fu_txt = "X" if fu is None else str(fu)
                sf_txt = "X" if sf is None else str(sf)
                w.write(
                    f">seq_{written:07d}|combo={bits}:{target_len}|GP={gp_txt}|GN={gn_txt}|FU={fu_txt}|SF={sf_txt}|LEN={target_len}|scheme={GENERATION_SCHEME_HEADER}\n"
                )
                w.write(seq + "\n")
                written += 1
                per_combo_written[combo] += 1
                lengths.append(int(len(seq)))
                unique_sequences.add(seq)
                if written % 100 == 0:
                    print(f"[GEN] written={written}/{total_target}", flush=True)
    if written < total_target:
        detail = ", ".join(f"{c}:{per_combo_written[c]}/{int(n_per_combo)}(attempts={per_combo_attempts[c]})" for c in combo_order)
        raise RuntimeError(f"generate failed: written={written} target={total_target}; per_combo={detail}")
    if not bool(return_stats):
        return out_fasta
    total_attempts = int(sum(int(x) for x in per_combo_attempts.values()))
    mean_len = float(sum(lengths) / max(1, len(lengths))) if lengths else 0.0
    return {
        "out_fasta": str(out_fasta),
        "seed": int(seed),
        "do_sample": bool(do_sample),
        "shuffle_combos": bool(shuffle_combos),
        "combo_order": [str(x) for x in combo_order],
        "attempt_limit_per_combo": int(per_combo_attempt_limit),
        "written": int(written),
        "target": int(total_target),
        "attempts": total_attempts,
        "accept_rate": float(written) / float(max(1, total_attempts)),
        "reject_too_short": int(reject_too_short),
        "reject_invalid": int(reject_invalid),
        "reject_len_out_of_range": int(reject_len_out_of_range),
        "reject_len_mismatch": int(reject_len_mismatch),
        "written_per_combo": {str(k): int(v) for k, v in per_combo_written.items()},
        "attempts_per_combo": {str(k): int(v) for k, v in per_combo_attempts.items()},
        "unique_sequences": int(len(unique_sequences)),
        "unique_ratio": float(len(unique_sequences)) / float(max(1, written)),
        "length_min": int(min(lengths)) if lengths else 0,
        "length_max": int(max(lengths)) if lengths else 0,
        "length_mean": mean_len,
        "lengths": [int(x) for x in lengths],
    }


def unwrap_generation_model(model: Any) -> Any:
    """Return the underlying LM when training wraps PEFT with StopHead."""
    return getattr(model, "lm", model)


def _decode_sequence_once(
    model: Any,
    tok: GPT2TokenizerFast,
    device: torch.device,
    combo: str,
    prompt_mode: str,
    target_len: int,
    step_limit: int,
    enforce_exact_length: bool,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    restrict_aa_tokens: bool,
    attr_weights: Optional[Sequence[float]],
    default_length: int,
) -> Tuple[str, str]:
    if enforce_exact_length:
        raise ValueError("Forced length is no longer supported; use EOS stopping with enforce_exact_length=False")
    eos_id = int(tok.eos_token_id)
    id_to_aa: Dict[int, str] = {}
    aa_allowed_ids = None
    if bool(restrict_aa_tokens):
        id_to_aa, aa_allowed_ids_cpu = _build_aa_char_maps(tok)
        aa_allowed_ids = aa_allowed_ids_cpu.to(device)

    gen_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    stopped = False
    for step_idx in range(int(step_limit)):
        logits = fused_combo_logits(
            model=model,
            tok=tok,
            device=device,
            combo=combo,
            generated_ids=gen_ids,
            attr_weights=attr_weights,
            default_length=default_length,
            prompt_mode=prompt_mode,
        )
        logits = logits / max(1e-6, float(temperature))
        logits = _apply_repetition_penalty(logits, gen_ids, float(repetition_penalty))
        logits = _apply_no_repeat_ngram(logits, gen_ids, int(no_repeat_ngram_size))
        if aa_allowed_ids is not None:
            masked = torch.full_like(logits, -float("inf"))
            masked[:, aa_allowed_ids] = logits[:, aa_allowed_ids]
            logits = masked
        logits = _top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
        nxt = _sample_next(logits, do_sample=True)
        gen_ids = torch.cat([gen_ids, nxt], dim=1)
        if int(nxt.item()) == eos_id:
            stopped = True
            break

    ids_out = [int(x) for x in gen_ids[0].tolist() if int(x) != eos_id]
    if bool(restrict_aa_tokens):
        seq = "".join(id_to_aa[i] for i in ids_out if i in id_to_aa)
    else:
        seq = clean_seq(tok.decode(gen_ids[0], skip_special_tokens=True))
    return seq, "stop" if stopped else "max"


def generate_length_then_content_for_combos(
    model: Any,
    tok: GPT2TokenizerFast,
    device: torch.device,
    out_fasta: str,
    combos: Sequence[str],
    n_per_combo: int,
    min_seq_len: int,
    max_seq_len: int,
    min_new_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    restrict_aa_tokens: bool,
    seed: int,
    max_attempts_multiplier: int,
    attr_weights: Optional[Sequence[float]] = None,
    default_length: int = 24,
    require_phase1_exact: bool = False,
    return_stats: bool = False,
) -> Any:
    gen_model = unwrap_generation_model(model)
    combos_in = [normalize_condition(str(c), default_length=default_length) for c in combos]
    combos_in = [c for c in combos_in if c.split(":", 1)[0] not in FORBIDDEN_ATTR_BITS]
    if not combos_in:
        raise ValueError("combos is empty after filtering")

    Path(out_fasta).parent.mkdir(parents=True, exist_ok=True)
    total_target = len(combos_in) * int(n_per_combo)
    rng = random.Random(int(seed))
    combo_order = list(combos_in)
    rng.shuffle(combo_order)

    written = 0
    phase1_probes = 0
    phase1_exact = 0
    reject_phase1_len = 0
    reject_phase1_invalid = 0
    reject_phase2_invalid = 0
    reject_phase2_len = 0
    reject_len_out_of_range = 0
    lengths: List[int] = []
    unique_sequences = set()
    per_combo_written: Dict[str, int] = {c: 0 for c in combo_order}
    per_combo_attempts: Dict[str, int] = {c: 0 for c in combo_order}
    per_combo_attempt_limit = max(1, int(max_attempts_multiplier) * int(n_per_combo))

    with open(out_fasta, "w", encoding="utf-8") as w:
        for combo in combo_order:
            bits, target_len = parse_condition(combo, default_length=default_length)
            while per_combo_written[combo] < int(n_per_combo) and per_combo_attempts[combo] < per_combo_attempt_limit:
                attempt_idx = int(per_combo_attempts[combo])
                per_combo_attempts[combo] += 1
                torch.manual_seed(_combo_attempt_seed(int(seed), combo, attempt_idx))

                probe, _ = _decode_sequence_once(
                    gen_model,
                    tok,
                    device,
                    combo,
                    prompt_mode="length_only",
                    target_len=int(target_len),
                    step_limit=int(max_new_tokens),
                    enforce_exact_length=False,
                    min_new_tokens=int(min_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                    restrict_aa_tokens=bool(restrict_aa_tokens),
                    attr_weights=attr_weights,
                    default_length=int(default_length),
                )
                phase1_probes += 1
                if CANON_RX.fullmatch(probe or "") is None:
                    reject_phase1_invalid += 1
                if len(probe) == int(target_len):
                    phase1_exact += 1
                elif bool(require_phase1_exact):
                    reject_phase1_len += 1
                    continue

                seq, _ = _decode_sequence_once(
                    gen_model,
                    tok,
                    device,
                    combo,
                    prompt_mode="fused",
                    target_len=int(target_len),
                    step_limit=int(max_new_tokens),
                    enforce_exact_length=False,
                    min_new_tokens=int(min_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(no_repeat_ngram_size),
                    restrict_aa_tokens=bool(restrict_aa_tokens),
                    attr_weights=attr_weights,
                    default_length=int(default_length),
                )
                if CANON_RX.fullmatch(seq or "") is None:
                    reject_phase2_invalid += 1
                    continue
                if len(seq) < int(min_seq_len) or len(seq) > int(max_seq_len):
                    reject_len_out_of_range += 1
                    continue

                gp, gn, fu, sf = condition_to_targets(combo)
                gp_txt = "X" if gp is None else str(gp)
                gn_txt = "X" if gn is None else str(gn)
                fu_txt = "X" if fu is None else str(fu)
                sf_txt = "X" if sf is None else str(sf)
                w.write(
                    f">seq_{written:07d}|combo={bits}:{target_len}|GP={gp_txt}|GN={gn_txt}|FU={fu_txt}|SF={sf_txt}|LEN={target_len}|scheme=five_attr_lftg_length_then_content\n"
                )
                w.write(seq + "\n")
                written += 1
                per_combo_written[combo] += 1
                lengths.append(int(len(seq)))
                unique_sequences.add(seq)
                if written % 100 == 0:
                    print(f"[GEN-LFTG] written={written}/{total_target}", flush=True)

    if written < total_target:
        detail = ", ".join(f"{c}:{per_combo_written[c]}/{int(n_per_combo)}(attempts={per_combo_attempts[c]})" for c in combo_order)
        raise RuntimeError(f"LFTG generate failed: written={written} target={total_target}; per_combo={detail}")
    if not bool(return_stats):
        return out_fasta
    total_attempts = int(sum(int(x) for x in per_combo_attempts.values()))
    return {
        "out_fasta": str(out_fasta),
        "seed": int(seed),
        "scheme": "five_attr_lftg_length_then_content",
        "written": int(written),
        "target": int(total_target),
        "attempts": int(total_attempts),
        "accept_rate": float(written) / float(max(1, total_attempts)),
        "reject_phase1_len": int(reject_phase1_len),
        "reject_phase1_invalid": int(reject_phase1_invalid),
        "reject_phase2_invalid": int(reject_phase2_invalid),
        "reject_phase2_len": int(reject_phase2_len),
        "reject_len_out_of_range": int(reject_len_out_of_range),
        "phase1_probes": int(phase1_probes),
        "phase1_exact": int(phase1_exact),
        "phase1_exact_rate": float(phase1_exact) / float(max(1, phase1_probes)),
        "written_per_combo": {str(k): int(v) for k, v in per_combo_written.items()},
        "attempts_per_combo": {str(k): int(v) for k, v in per_combo_attempts.items()},
        "unique_sequences": int(len(unique_sequences)),
        "unique_ratio": float(len(unique_sequences)) / float(max(1, written)),
        "length_min": int(min(lengths)) if lengths else 0,
        "length_max": int(max(lengths)) if lengths else 0,
        "length_mean": float(sum(lengths) / max(1, len(lengths))) if lengths else 0.0,
        "lengths": [int(x) for x in lengths],
    }


def build_length_diagnostic_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Collect length-oriented slices already computed by the eval protocol."""
    lftg = summary.get("lftg") or {}
    lenonly = summary.get("lenonly_free") or {}
    report: Dict[str, Any] = {}
    if lftg:
        report["lftg_by_target"] = lftg.get("length_control_by_target", {})
        report["lftg_sensitivity_hint"] = lftg.get("length_control_by_target", {})
    if lenonly:
        report["lenonly_by_target"] = lenonly.get("length_control_by_target", {})
    return report
