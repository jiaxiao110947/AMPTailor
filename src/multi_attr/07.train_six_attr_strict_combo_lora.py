#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
import inspect
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from transformers import (
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    _PEFT_AVAILABLE = False

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_LIB = _load_module_from_path("five_attr_classification_lora_lib_07", _SCRIPTS_DIR / "07.five_attr_classification_lora_lib.py")
ATTRS = _LIB.ATTRS
COMBOS_14 = _LIB.COMBOS_14
REGULAR_CONDITIONS_6 = _LIB.REGULAR_CONDITIONS_6
DEFAULT_EVAL_LENGTHS = _LIB.DEFAULT_EVAL_LENGTHS
LENGTH_TIERS = _LIB.LENGTH_TIERS
EOS_TEXT = _LIB.EOS_TEXT
BASE_TASKS = _LIB.BASE_TASKS
expand_conditions_with_lengths = _LIB.expand_conditions_with_lengths
expand_lenonly_targets = _LIB.expand_lenonly_targets
make_length_task = _LIB.make_length_task
combos_to_jsonable = _LIB.combos_to_jsonable
generate_sequences_for_combos = _LIB.generate_sequences_for_combos
generate_length_then_content_for_combos = _LIB.generate_length_then_content_for_combos
build_length_diagnostic_report = _LIB.build_length_diagnostic_report
compute_length_stop_loss = _LIB.compute_length_stop_loss
StopHead = _LIB.StopHead
unwrap_generation_model = _LIB.unwrap_generation_model
is_length_task = _LIB.is_length_task
make_task_prompt = _LIB.make_task_prompt
parse_attr_weights = _LIB.parse_attr_weights
parse_combo_text = _LIB.parse_combo_text
parse_length_list = _LIB.parse_length_list
task_weight = _LIB.task_weight

try:
    _CB = _load_module_from_path("unified_single_attr_balanced_eval_callback_06", Path(__file__).resolve().parent / "06.unified_single_attr_balanced_eval_callback.py")
    UnifiedFourAttrSingleLoRAEvalCallback = _CB.UnifiedFourAttrSingleLoRAEvalCallback
except Exception:
    UnifiedFourAttrSingleLoRAEvalCallback = None

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

UNIPROT_SCRIPTS = (_ROOT / "src" / "pretrain")
if UNIPROT_SCRIPTS.is_dir() and str(UNIPROT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UNIPROT_SCRIPTS))
try:
    from eval_throttle import should_run_heavy_eval  # noqa: E402
    from length_eval_callback import LengthEvalCallback  # noqa: E402
except Exception:
    LengthEvalCallback = None  # type: ignore

    def should_run_heavy_eval(step: int, eval_steps: int) -> bool:  # type: ignore
        return True

PROJ_ROOT = _ROOT
SCRIPTS_ROOT = PROJ_ROOT / "src" / "multi_attr"
DEFAULT_CSV_PATH = str(_ROOT / "data" / "corpus" / "sequence_species_mic_binned_for_six_attr.csv")
DEFAULT_BASE_MODEL_DIR = str(_ROOT / "models/AMPGuide")
DEFAULT_TOKENIZER_DIR = str(_ROOT / "models/AMPGuide")
DEFAULT_COMBO_EVAL_SCRIPT = str(Path(__file__).resolve().parent / "07.eval_checkpoint_protocol_five_attr_single_lora.py")
DEFAULT_AMPSORTER_SCRIPT = ""
DEFAULT_AMPSORTER_BACKBONE = ""
DEFAULT_AMPSORTER_TOKENIZER = DEFAULT_TOKENIZER_DIR
DEFAULT_AMPSORTER_CLASSIFIER_PT = ""
DEFAULT_TOX_SCRIPT = ""
DEFAULT_TOX_BACKBONE = ""
DEFAULT_TOX_TOKENIZER = DEFAULT_TOKENIZER_DIR
DEFAULT_TOX_CLASSIFIER_PT = ""
DEFAULT_IAMPCN_REPO = ""
DEFAULT_IAMPCN_PY = ""
DEFAULT_PROTEOGPT_PY = "python"

CANON_RX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
ATTR_ALIASES = {
    "gram_pos": ["gram_pos", "Gram+", "Gram_pos", "anti_gram_pos_final", "革兰氏阳性菌"],
    "gram_neg": ["gram_neg", "Gram-", "Gram_neg", "anti_gram_neg_final", "革兰氏阴性菌"],
    "fungus": ["fungus", "Fungus", "anti_fungi_final", "真菌"],
    "safety": ["safety", "safety_label", "safe", "toxic", "安全性"],
    "length": ["length", "seq_len", "长度"],
    "mic_bin": ["mic_bin", "MIC_bin"],
}
MIC_TASKS = ["MIC0", "MIC1", "MIC2", "MIC3", "MIC4"]
_MIC_LIB_PATH = Path(__file__).resolve().parent / "07.multi_attr_species_mic_lib.py"
_mic_spec = importlib.util.spec_from_file_location("multi_attr_species_mic_lib_07_train", str(_MIC_LIB_PATH))
if _mic_spec is None or _mic_spec.loader is None:
    raise ImportError(f"cannot load {_MIC_LIB_PATH}")
_MIC_LIB = importlib.util.module_from_spec(_mic_spec)
_mic_spec.loader.exec_module(_MIC_LIB)
_COMBO_LIB = _load_module_from_path(
    "six_attr_strict_combo_lib_07_train",
    Path(__file__).resolve().parent / "07.six_attr_strict_combo_lib.py",
)


def _hash_to_unit_interval(s: str, seed: str) -> float:
    import hashlib
    h = hashlib.md5((str(seed) + "|" + str(s)).encode("utf-8")).hexdigest()
    return float(int(h[:12], 16)) / float(1 << 48)


def assign_split(seed: str, seq: str, val_frac: float, test_frac: float) -> str:
    u = _hash_to_unit_interval(seq, seed)
    train_cut = 1.0 - float(val_frac) - float(test_frac)
    val_cut = 1.0 - float(test_frac)
    if u < train_cut:
        return "train"
    if u < val_cut:
        return "val"
    return "test"


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _mic_um_to_bin(mic_um: float) -> float:
    if pd.isna(mic_um):
        return float("nan")
    v = float(mic_um)
    if v <= 1.0:
        return 0.0
    if v <= 4.0:
        return 1.0
    if v <= 16.0:
        return 2.0
    if v <= 64.0:
        return 3.0
    return 4.0


def load_clean_csv_multi(
    csv_path: str,
    split_seed: str,
    val_frac: float,
    test_frac: float,
    min_seq_len: int = 1,
) -> pd.DataFrame:
    probe = pd.read_csv(str(csv_path), nrows=1)
    rowlevel = (
        _resolve_column(probe, _MIC_LIB.COLUMN_ALIASES["species"]) is not None
        or "species_id" in probe.columns
        or "species" in probe.columns
    )
    if rowlevel:
        return _load_clean_csv_rowlevel(csv_path, split_seed, val_frac, test_frac, min_seq_len)

    df = pd.read_csv(str(csv_path))
    seq_col = _resolve_column(df, ["sequence", "sequence_clean", "序列"])
    if seq_col is None:
        raise RuntimeError("CSV must contain column: sequence or sequence_clean")
    df["sequence"] = df[seq_col].astype(str).str.strip().str.upper()
    optional_attrs = {"mic_bin"}
    for attr in ATTRS:
        if attr == "length":
            continue
        col = _resolve_column(df, ATTR_ALIASES.get(attr, [attr]))
        if col is None:
            if attr in optional_attrs:
                continue
            raise RuntimeError(f"CSV must contain column for {attr}")
        if attr == "safety":
            raw = df[col].astype(str).str.strip()
            missing = df[col].isna() | raw.eq("") | raw.str.upper().isin(["NA", "NAN", "NONE", "NULL", "N/A"])
            df[attr] = pd.to_numeric(df[col], errors="coerce")
            df[attr] = df[attr].where(~missing, other=float("nan"))
            df[attr] = df[attr].where(df[attr].isin([0.0, 1.0]), other=float("nan"))
        else:
            df[attr] = pd.to_numeric(df[col], errors="coerce")
    mic_um_col = _resolve_column(df, ["MIC_uM", "mic_um", "mic_uM"])
    if "mic_bin" not in df.columns and mic_um_col is not None:
        df["mic_bin"] = pd.to_numeric(df[mic_um_col], errors="coerce").apply(_mic_um_to_bin)
    length_col = _resolve_column(df, ATTR_ALIASES["length"])
    if length_col is not None:
        df["length"] = pd.to_numeric(df[length_col], errors="coerce")
    df = df[df["sequence"].str.len() > 0]
    df = df[df["sequence"].apply(lambda s: CANON_RX.fullmatch(s) is not None)]
    if int(min_seq_len) > 1:
        df = df[df["sequence"].str.len() >= int(min_seq_len)]
    # 不因 gram 全 0 丢弃行
    if "length" not in df.columns or df["length"].isna().any():
        df["length"] = df["sequence"].str.len()
    group_cols = [a for a in ["gram_pos", "gram_neg", "fungus", "length", "safety", "mic_bin"] if a in df.columns]
    agg_map = {c: "max" for c in group_cols if c not in {"safety", "mic_bin"}}
    if "safety" in group_cols:
        def _agg_safety(series: pd.Series) -> float:
            valid = series.dropna()
            return float(valid.max()) if len(valid) else float("nan")
        agg_map["safety"] = _agg_safety
    if "mic_bin" in group_cols:
        def _agg_mic_bin(series: pd.Series) -> float:
            valid = series.dropna()
            return float(valid.min()) if len(valid) else float("nan")
        agg_map["mic_bin"] = _agg_mic_bin
    df = df.groupby("sequence", as_index=False).agg(agg_map)
    df["length"] = df["length"].astype(int)
    df["split"] = df["sequence"].apply(lambda s: assign_split(split_seed, s, float(val_frac), float(test_frac)))
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        seq = str(r["sequence"])
        split = str(r["split"])
        seq_len = int(r["length"])
        task = _COMBO_LIB.row_to_combo_task(r)
        if task:
            rows.append({"sequence": seq, "task": task, "split": split})
    if not rows:
        raise RuntimeError("No flattened samples after cleaning")
    return pd.DataFrame(rows)


def _load_clean_csv_rowlevel(
    csv_path: str,
    split_seed: str,
    val_frac: float,
    test_frac: float,
    min_seq_len: int = 1,
) -> pd.DataFrame:
    row_df, _species_mapping = _MIC_LIB.load_rowlevel_table(str(csv_path), int(min_seq_len))
    row_df["split"] = row_df["sequence"].apply(
        lambda s: assign_split(split_seed, s, float(val_frac), float(test_frac))
    )
    flat_df = _COMBO_LIB.rows_to_strict_combo_tasks(row_df)
    stats = flat_df.attrs.get("strict_combo_stats") or {}
    print(f"[DATA] strict_combo {stats}", flush=True)
    return flat_df[["sequence", "task", "split"]]


def compute_multi_attr_weights(train_flat: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    stats_per_task: Dict[str, Dict[str, Any]] = {}
    weight_map: Dict[str, float] = {}
    total = int(len(train_flat))
    tasks_s = train_flat["task"].astype(str)
    n_combo = int(tasks_s.str.startswith("COMBO:").sum())
    bit_counts = (
        tasks_s.str.extract(r"^COMBO:([01]{4}):", expand=False).value_counts().to_dict()
        if n_combo
        else {}
    )
    weight_map["COMBO"] = 1.0
    stats_per_task["COMBO"] = {
        "count": n_combo,
        "weight": 1.0,
        "unique_combo_tasks": int(tasks_s.nunique()),
        "combo_bit_counts": {str(k): int(v) for k, v in bit_counts.items()},
        "n_rows": total,
    }
    return weight_map, stats_per_task


def _parse_length_balance_bins(text: str) -> List[Tuple[int, int]]:
    bins: List[Tuple[int, int]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(item)
        if lo > hi:
            lo, hi = hi, lo
        bins.append((int(lo), int(hi)))
    return bins or [(6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40)]


def _length_bucket_id(seq_len: int, bins: Sequence[Tuple[int, int]]) -> str:
    n = int(seq_len)
    for lo, hi in bins:
        if int(lo) <= n <= int(hi):
            return f"{int(lo)}-{int(hi)}"
    if bins and n < int(bins[0][0]):
        return f"<{int(bins[0][0])}"
    if bins:
        return f">{int(bins[-1][1])}"
    return "all"


class MultiAttrTrainIter(IterableDataset):
    def __init__(
        self,
        flat_df: pd.DataFrame,
        length_balance_enable: bool = False,
        length_balance_bins: str = "6-10,11-15,16-20,21-25,26-30,31-35,36-40",
    ):
        super().__init__()
        self.items: List[Tuple[str, str]] = []
        self.length_balance_enable = bool(length_balance_enable)
        self.length_bins = _parse_length_balance_bins(length_balance_bins)
        self.bucket_to_indices: Dict[str, List[int]] = {}
        self.epoch = 0
        for _, r in flat_df.iterrows():
            task = str(r["task"]).upper()
            self.items.append((str(r["sequence"]), task))
            bucket = _length_bucket_id(len(str(r["sequence"])), self.length_bins)
            self.bucket_to_indices.setdefault(bucket, []).append(len(self.items) - 1)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def info(self) -> str:
        if not self.length_balance_enable:
            return f"[MultiTaskTrainIter] N={len(self.items)} base_tasks={BASE_TASKS}+LEN{{n}}"
        bucket_stats = ",".join(f"{k}:{len(v)}" for k, v in sorted(self.bucket_to_indices.items()))
        return f"[MultiTaskTrainIter] N={len(self.items)} length_balance=1 buckets={bucket_stats}"

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.items:
            return iter(())
        rng = np.random.default_rng(42 + int(self.epoch))
        if self.length_balance_enable:
            bucket_names = [k for k, v in sorted(self.bucket_to_indices.items()) if v]
            bucket_lists = {k: list(v) for k, v in self.bucket_to_indices.items() if v}
            for v in bucket_lists.values():
                rng.shuffle(v)
            idxs_list: List[int] = []
            while any(bucket_lists.get(k) for k in bucket_names):
                order = list(bucket_names)
                rng.shuffle(order)
                for k in order:
                    if bucket_lists.get(k):
                        idxs_list.append(int(bucket_lists[k].pop()))
            idxs = np.asarray(idxs_list, dtype=np.int64)
        else:
            idxs = np.arange(len(self.items), dtype=np.int64)
            rng.shuffle(idxs)
        wi = get_worker_info()
        if wi is not None and int(wi.num_workers) > 1:
            if self.length_balance_enable:
                idxs = idxs[int(wi.id) :: int(wi.num_workers)]
            else:
                per_worker = len(idxs) // int(wi.num_workers)
                start = int(wi.id) * per_worker
                end = start + per_worker if int(wi.id) < int(wi.num_workers) - 1 else len(idxs)
                idxs = idxs[start:end]
        for i in idxs:
            seq, task = self.items[int(i)]
            yield {"prompt": _COMBO_LIB.make_strict_combo_prompt(task), "sequence": seq, "task": task}


class MultiAttrEvalSet(Dataset):
    def __init__(self, flat_df: pd.DataFrame, max_samples: int = 0, seed: int = 42):
        if int(max_samples) > 0 and len(flat_df) > int(max_samples):
            flat_df = flat_df.sample(n=int(max_samples), random_state=int(seed)).copy()
        self.items: List[Dict[str, Any]] = []
        for _, r in flat_df.iterrows():
            task = str(r["task"]).upper()
            self.items.append(
                {"prompt": _COMBO_LIB.make_strict_combo_prompt(task), "sequence": str(r["sequence"]), "task": task}
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.items[int(i)].copy()


class CollatorPromptMaskWithPos:
    def __init__(
        self,
        tok: GPT2TokenizerFast,
        eos_text: str,
        pad_to_multiple_of: int = 8,
        max_length: int = 512,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        self.tok = tok
        self.eos_text = str(eos_text)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.max_length = int(max_length)
        self.task_weights = dict(task_weights or {})
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        seq_token_lists: List[List[int]] = []
        prompt_lens: List[int] = []
        for item in batch:
            p_ids = self.tok(str(item["prompt"]), add_special_tokens=False)["input_ids"]
            s_ids = self.tok(str(item["sequence"]) + self.eos_text, add_special_tokens=False)["input_ids"]
            ids = (list(p_ids) + list(s_ids))[: self.max_length]
            seq_token_lists.append(ids)
            prompt_lens.append(min(len(p_ids), len(ids)))
        max_len = max(len(x) for x in seq_token_lists)
        if self.pad_to_multiple_of > 1:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        pad_id = int(self.tok.pad_token_id)
        bs = len(batch)
        input_ids = torch.full((bs, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), fill_value=-100, dtype=torch.long)
        for i, ids in enumerate(seq_token_lists):
            real_len = len(ids)
            pad_len = max_len - real_len
            input_ids[i, pad_len:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, pad_len:] = 1
            labels[i, pad_len:] = torch.tensor(ids, dtype=torch.long)
            end = min(max_len, pad_len + int(prompt_lens[i]))
            if end > pad_len:
                labels[i, pad_len:end] = -100
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids[attention_mask == 0] = 0
        tasks: List[str] = []
        target_seq_lens: List[int] = []
        sample_weights: List[float] = []
        for item in batch:
            task = str(item.get("task", "")).strip().upper()
            tasks.append(task)
            if self.task_weights:
                sample_weights.append(float(task_weight(task, self.task_weights)))
            else:
                sample_weights.append(1.0)
            target_len = int(_COMBO_LIB.combo_target_length(task))
            if target_len > 0:
                target_seq_lens.append(target_len)
            elif is_length_task(task):
                try:
                    target_seq_lens.append(int(_LIB.length_from_task(task)))
                except Exception:
                    target_seq_lens.append(int(len(str(item.get("sequence", "")))))
            else:
                target_seq_lens.append(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "target_seq_lens": torch.tensor(target_seq_lens, dtype=torch.long),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float),
        }


def _strict_check_eos(tok: GPT2TokenizerFast, eos_text: str) -> None:
    eos_ids = tok.encode(eos_text, add_special_tokens=False)
    if len(eos_ids) != 1:
        raise RuntimeError(f"eos_text is not a single token: {eos_text} -> {eos_ids}")


def _safe_exp(x: float) -> float:
    try:
        return float(math.exp(min(float(x), 50.0)))
    except Exception:
        return float("nan")


def _trainer_tokenizer_kwargs(tok: GPT2TokenizerFast) -> Dict[str, Any]:
    sig = inspect.signature(Trainer.__init__)
    return {"processing_class": tok} if "processing_class" in sig.parameters else {"tokenizer": tok}


class FiveAttrModelWithStopHead(torch.nn.Module):
    """Peft 包装 + 可训练 StopHead（仅 LEN 样本参与 stop loss）。"""
    can_return_loss = True

    def __init__(self, lm_model: Any, hidden_size: int, stop_dropout: float = 0.05):
        super().__init__()
        self.lm = lm_model
        self.stop_head = StopHead(int(hidden_size), dropout=float(stop_dropout))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.lm, name)

    def forward(self, *args, **kwargs):
        return self.lm(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        if hasattr(self.lm, "save_pretrained"):
            kwargs.setdefault("safe_serialization", False)
            self.lm.save_pretrained(save_directory, **kwargs)


class TrainerWithMultiAttrAuc(Trainer):
    def __init__(
        self,
        eos_text: str,
        auc_max_samples: int = 1500,
        stop_loss_weight: float = 1.0,
        stop_head_enable: bool = True,
        **kwargs,
    ):
        self._eos_text = str(eos_text)
        self._auc_max_samples = int(auc_max_samples)
        self.stop_loss_weight = float(stop_loss_weight)
        self.stop_head_enable = bool(stop_head_enable)
        super().__init__(**kwargs)
        # FiveAttrModelWithStopHead.forward 为 *args/**kwargs，HF 4.57 无法推断 labels，
        # 导致 evaluate 不计算 loss、save 时 metric_for_best_model=eval_loss 崩溃。
        if isinstance(self.model, FiveAttrModelWithStopHead):
            self.label_names = list(self.args.label_names or ["labels"])
            self.can_return_loss = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        target_seq_lens = inputs.pop("target_seq_lens", None)
        sample_weights = inputs.pop("sample_weights", None)
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if sample_weights is not None:
            flat_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            token_mask = shift_labels.ne(-100).float()
            per_sample = (flat_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
            weights = sample_weights.to(per_sample.device).float()
            valid = token_mask.sum(dim=1).gt(0).float()
            loss_ce = (per_sample * weights * valid).sum() / (weights * valid).sum().clamp(min=1.0)
        else:
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        loss = loss_ce
        if (
            self.stop_head_enable
            and target_seq_lens is not None
            and hasattr(model, "stop_head")
            and outputs.hidden_states is not None
        ):
            stop_loss, stop_stats = compute_length_stop_loss(
                outputs.hidden_states[-1],
                labels,
                target_seq_lens,
                model.stop_head,
            )
            if float(stop_stats.get("stop_loss_n_terms", 0.0)) > 0.0:
                loss = loss + self.stop_loss_weight * stop_loss
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Peft+StopHead 包装时只保存内层 LoRA，避免 tie_word_embeddings 触发 safetensors 共享权重报错。"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        out = Path(str(output_dir))
        out.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, FiveAttrModelWithStopHead):
            self.model.lm.save_pretrained(str(out), safe_serialization=True)
            torch.save(self.model.stop_head.state_dict(), str(out / "stop_head.pt"))
            return
        super().save_model(output_dir, _internal_call=_internal_call)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """FiveAttrModelWithStopHead 需走 PEFT load_adapter，否则 Trainer 会误用 sharded 全量加载。"""
        if model is None:
            model = self.model
        if isinstance(model, FiveAttrModelWithStopHead):
            ckpt = str(resume_from_checkpoint)
            peft_model = model.lm
            if hasattr(peft_model, "load_adapter"):
                if hasattr(peft_model, "active_adapters"):
                    active = peft_model.active_adapters[0]
                else:
                    active = peft_model.active_adapter
                peft_model.load_adapter(ckpt, active, is_trainable=True)
                print(f"[Resume] PEFT adapter loaded from {ckpt}", flush=True)
            sh = Path(ckpt) / "stop_head.pt"
            if sh.exists():
                model.stop_head.load_state_dict(torch.load(str(sh), map_location="cpu", weights_only=True))
                print(f"[Resume] StopHead loaded from {sh}", flush=True)
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """绕过 transformers 对 torch 元数据版本的误报（runtime 2.9.x / metadata 2.5.x）。"""
        if checkpoint is None or not isinstance(self.model, FiveAttrModelWithStopHead):
            return super()._load_optimizer_and_scheduler(checkpoint)
        ckpt = Path(str(checkpoint))
        opt_file = ckpt / "optimizer.pt"
        sched_file = ckpt / "scheduler.pt"
        if self.optimizer is not None and opt_file.is_file():
            self.optimizer.load_state_dict(torch.load(str(opt_file), map_location="cpu", weights_only=False))
            print(f"[Resume] optimizer loaded from {opt_file}", flush=True)
        if self.lr_scheduler is not None and sched_file.is_file():
            self.lr_scheduler.load_state_dict(torch.load(str(sched_file), weights_only=False))
            print(f"[Resume] scheduler loaded from {sched_file}", flush=True)


class SetEpochCB(TrainerCallback):
    def __init__(self, ds: Any):
        self.ds = ds

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(0 if state.epoch is None else int(state.epoch))
        return control


class UnifiedWandbCallback(TrainerCallback):
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._wandb = None
        self._last_lr: Optional[float] = None
        self._metrics_defined = False
        if self.enabled:
            import wandb
            self._wandb = wandb

    def _safe_step(self, step: int) -> int:
        return max(0, int(step))

    @staticmethod
    def _wandb_key(text: str) -> str:
        return str(text).replace(":", "_").replace("/", "_")

    def _setup_wandb_metrics(self) -> None:
        if not self.enabled or self._wandb is None or self._metrics_defined:
            return
        if getattr(self._wandb, "run", None) is None:
            return
        self._wandb.define_metric("trainer/global_step")
        self._wandb.define_metric("train/*", step_metric="trainer/global_step")
        self._wandb.define_metric("eval/*", step_metric="trainer/global_step")
        self._wandb.define_metric("combo/*", step_metric="trainer/global_step")
        self._wandb.define_metric("quality/*", step_metric="trainer/global_step")
        self._metrics_defined = True

    def _wandb_log(self, payload: Dict[str, float]) -> None:
        if not payload or self._wandb is None or getattr(self._wandb, "run", None) is None:
            return
        self._setup_wandb_metrics()
        # 用 trainer/global_step 作为 W&B step，train/eval/length 同一步合并到一行，默认图表可见。
        step = int(payload.get("trainer/global_step", 0))
        self._wandb.log(payload, step=step, commit=True)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.enabled and self._wandb is not None and getattr(self._wandb, "run", None) is None:
            self._wandb.init(project=os.environ.get("WANDB_PROJECT"), group=os.environ.get("WANDB_GROUP"), name=os.environ.get("WANDB_RUN_NAME"))
        if self.enabled and self._wandb is not None and getattr(self._wandb, "run", None) is not None:
            self._setup_wandb_metrics()
            try:
                run = self._wandb.run
                if getattr(run, "url", None):
                    print(f"[W&B] run_url={run.url}", flush=True)
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or self._wandb is None or getattr(self._wandb, "run", None) is None or not logs:
            return
        # eval_loss 由 on_evaluate 与 combo 指标一起写入，避免同一步重复/乱序。
        if logs.get("eval_loss") is not None and logs.get("loss") is None:
            return
        step = self._safe_step(int(getattr(state, "global_step", 0)))
        payload: Dict[str, float] = {"trainer/global_step": float(step)}
        if logs.get("loss") is not None:
            payload["train/loss"] = float(logs["loss"])
            payload["train/ppl"] = _safe_exp(float(logs["loss"]))
        if logs.get("learning_rate") is not None:
            self._last_lr = float(logs["learning_rate"])
            payload["train/lr"] = self._last_lr
        self._wandb_log(payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.enabled or self._wandb is None or getattr(self._wandb, "run", None) is None or not metrics:
            return
        step = self._safe_step(int(getattr(state, "global_step", 0)))
        payload: Dict[str, float] = {"trainer/global_step": float(step)}
        if metrics.get("eval_loss") is not None:
            payload["eval/loss"] = float(metrics["eval_loss"])
            payload["eval/ppl"] = _safe_exp(float(metrics["eval_loss"]))
        if metrics.get("eval_runtime") is not None:
            payload["eval/runtime_s"] = float(metrics["eval_runtime"])
        if metrics.get("eval_samples_per_second") is not None:
            payload["eval/samples_per_s"] = float(metrics["eval_samples_per_second"])
        if metrics.get("eval_macro_auc") is not None:
            payload["eval/macro_auc"] = float(metrics["eval_macro_auc"])
        if metrics.get("eval_combo_adherence_mean") is not None:
            payload["eval/combo_adherence_mean"] = float(metrics["eval_combo_adherence_mean"])
        if metrics.get("eval_combo_adherence_min") is not None:
            payload["eval/combo_adherence_min"] = float(metrics["eval_combo_adherence_min"])
        if metrics.get("eval_length_control_variance") is not None:
            payload["eval/length_control_variance"] = float(metrics["eval_length_control_variance"])
        if metrics.get("eval_regular6_length_control_variance") is not None:
            payload["eval/regular6/length_control_variance"] = float(metrics["eval_regular6_length_control_variance"])
        if metrics.get("eval_strict14_length_control_variance") is not None:
            payload["eval/strict14/length_control_variance"] = float(metrics["eval_strict14_length_control_variance"])
        for attr in ATTRS:
            k = f"eval_auc_{attr}"
            if metrics.get(k) is not None:
                payload[f"eval/auc_{attr}"] = float(metrics[k])
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if k.startswith("combo/") or k.startswith("quality/"):
                payload[k] = float(v)
            elif k.startswith("eval_combo_") and k not in {"eval_combo_adherence_mean", "eval_combo_adherence_min"}:
                payload[f"eval/combo/{self._wandb_key(k[len('eval_combo_'):])}"] = float(v)
            elif k.startswith("eval_len_regular6_"):
                suffix = self._wandb_key(k[len("eval_len_regular6_") :])
                payload[f"eval/len_regular6/{suffix}"] = float(v)
            elif k.startswith("eval_regular6_"):
                suffix = self._wandb_key(k[len("eval_regular6_") :])
                payload[f"eval/regular6/{suffix}"] = float(v)
            elif k.startswith("eval_fused_free_"):
                suffix = self._wandb_key(k[len("eval_fused_free_") :])
                payload[f"eval/fused_free/{suffix}"] = float(v)
            elif k.startswith("eval_lenonly_free_"):
                suffix = self._wandb_key(k[len("eval_lenonly_free_") :])
                payload[f"eval/lenonly_free/{suffix}"] = float(v)
            elif k.startswith("eval_lftg_"):
                suffix = self._wandb_key(k[len("eval_lftg_") :])
                payload[f"eval/lftg/{suffix}"] = float(v)
            elif k.startswith("eval_fused_enforce1_"):
                suffix = self._wandb_key(k[len("eval_fused_enforce1_") :])
                payload[f"eval/fused_enforce1/{suffix}"] = float(v)
            elif k.startswith("eval_strict14_"):
                suffix = self._wandb_key(k[len("eval_strict14_") :])
                payload[f"eval/strict14/{suffix}"] = float(v)
        self._wandb_log(payload)
        n_eval = sum(1 for key in payload if key.startswith("eval/"))
        loss_s = f" eval/loss={payload['eval/loss']:.4f}" if "eval/loss" in payload else ""
        print(f"[W&B] logged {n_eval} eval metrics @ step={step}{loss_s}", flush=True)


class StagedEvalScheduleCallback(TrainerCallback):
    """前 eval_steps_early_until 步每 eval_steps_early 评估/保存，之后每 eval_steps_late 步。"""

    def __init__(self, early_step: int = 10, late_step: int = 100, until_step: int = 100):
        self.early_step = max(1, int(early_step))
        self.late_step = max(1, int(late_step))
        self.until_step = max(0, int(until_step))

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0))
        if step <= 0:
            return
        if step <= self.until_step:
            if step % self.early_step == 0:
                control.should_evaluate = True
                control.should_save = True
        elif step % self.late_step == 0:
            control.should_evaluate = True
            control.should_save = True


class MetricEarlyStoppingCallback(TrainerCallback):
    """Early stop on a metric already present in evaluate() metrics.

    This avoids relying on load_best_model_at_end, which is intentionally disabled in
    these runs because older torch/transformers checkpoints can trip loader guards.
    """

    def __init__(self, metric_name: str, greater_is_better: bool = True, patience: int = 0, min_delta: float = 0.0):
        self.metric_name = str(metric_name)
        self.greater_is_better = bool(greater_is_better)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.bad_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.patience <= 0 or not metrics or self.metric_name not in metrics:
            return control
        cur = float(metrics[self.metric_name])
        if self.best is None:
            self.best = cur
            self.bad_count = 0
            print(f"[EarlyStop] metric={self.metric_name} init_best={cur:.6f}", flush=True)
            return control
        improved = (cur > self.best + self.min_delta) if self.greater_is_better else (cur < self.best - self.min_delta)
        if improved:
            print(f"[EarlyStop] metric={self.metric_name} improved {self.best:.6f} -> {cur:.6f}", flush=True)
            self.best = cur
            self.bad_count = 0
        else:
            self.bad_count += 1
            print(
                f"[EarlyStop] metric={self.metric_name} no_improve={self.bad_count}/{self.patience} "
                f"cur={cur:.6f} best={self.best:.6f}",
                flush=True,
            )
            if self.bad_count >= self.patience:
                control.should_training_stop = True
        return control


def _payload_summary_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    pas = payload.get("per_attr_satisfaction") or {}
    return {
        "combo_adherence_mean": float(payload.get("combo_adherence_mean", payload.get("strict_adherence_mean", 0.0))),
        "combo_adherence_min": float(payload.get("combo_adherence_min", 0.0)),
        "attr_mean_satisfaction": float(payload.get("attr_mean_satisfaction", 0.0)),
        "per_attr_satisfaction": {a: float(r.get("satisfaction", 0.0)) for a, r in pas.items()},
        "per_combo_adherence": {
            c: float(r.get("adherence", 0.0)) for c, r in (payload.get("per_combo") or {}).items()
        },
        "length_control_exact_rate": float(payload.get("length_control_exact_rate", 0.0)),
        "length_control_mae": float(payload.get("length_control_mae", 0.0)),
        "n_generated": float(payload.get("clean_kept", 0.0)),
    }


class ValidationMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or metrics.get("eval_loss") is None:
            return
        metrics["eval_ppl"] = _safe_exp(float(metrics["eval_loss"]))


def _merge_combo_payload_metrics(metrics: Dict[str, Any], payload: Dict[str, Any], prefix: str) -> None:
    metrics[f"{prefix}_combo_adherence_mean"] = float(
        payload.get("combo_adherence_mean", payload.get("strict_adherence_mean", 0.0))
    )
    metrics[f"{prefix}_combo_adherence_min"] = float(payload.get("combo_adherence_min", 0.0))
    metrics[f"{prefix}_attr_mean_satisfaction"] = float(payload.get("attr_mean_satisfaction", 0.0))
    metrics[f"{prefix}_length_control_variance"] = float(payload.get("length_control_variance", 0.0))
    metrics[f"{prefix}_length_control_std"] = float(payload.get("length_control_std", 0.0))
    metrics[f"{prefix}_length_control_mae"] = float(payload.get("length_control_mae", 0.0))
    metrics[f"{prefix}_length_control_rmse"] = float(payload.get("length_control_rmse", 0.0))
    metrics[f"{prefix}_length_control_mean_dev"] = float(payload.get("length_control_mean_dev", 0.0))
    metrics[f"{prefix}_length_control_exact_rate"] = float(payload.get("length_control_exact_rate", 0.0))
    pas = payload.get("per_attr_satisfaction", {}) or {}
    for attr, rec in pas.items():
        metrics[f"{prefix}_attr_{attr}"] = float(rec.get("satisfaction", 0.0))
    for tier_id, _, _, _ in LENGTH_TIERS:
        tier = (payload.get("length_control_by_tier") or {}).get(tier_id, {})
        if float(tier.get("n", 0.0)) > 0:
            metrics[f"{prefix}_len_tier_{tier_id}_mae"] = float(tier.get("length_control_mae", 0.0))
            metrics[f"{prefix}_len_tier_{tier_id}_exact_rate"] = float(tier.get("length_control_exact_rate", 0.0))
    for target, rec in (payload.get("length_control_by_target") or {}).items():
        metrics[f"{prefix}_len_{target}_mae"] = float(rec.get("length_control_mae", 0.0))
        metrics[f"{prefix}_len_{target}_exact_rate"] = float(rec.get("length_control_exact_rate", 0.0))


class ComboEvalCallback(TrainerCallback):
    """训练中每 eval_steps 执行：六组合(regular6)与14 combo(strict14)分别生成 FASTA、独立判别打分、计算满足率。"""

    def __init__(
        self,
        enabled: bool,
        tokenizer: GPT2TokenizerFast,
        output_dir: Path,
        eval_script: str,
        eval_cmd_args: Sequence[str],
        regular_combos: Sequence[str],
        strict_combos: Sequence[str],
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
        max_attempts_multiplier: int,
        attr_weights: Sequence[float],
        timeout: int,
        default_length: int,
        enforce_exact_length: bool = True,
        regular6_only: bool = False,
        run_enforce1: bool = False,
        lenonly_enable: bool = True,
        lenonly_targets: Optional[Sequence[str]] = None,
        lenonly_n_per: int = 0,
        prompt_mode: str = "fused",
        use_lftg: bool = False,
        len_regular6_enable: bool = True,
        len_regular6_combos: Optional[Sequence[str]] = None,
    ):
        self.enabled = bool(enabled)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.eval_script = str(eval_script)
        self.eval_cmd_args = list(eval_cmd_args)
        self.regular_combos = list(regular_combos)
        self.strict_combos = list(strict_combos)
        self.regular6_only = bool(regular6_only)
        self.n_per_combo = int(n_per_combo)
        self.min_seq_len = int(min_seq_len)
        self.max_seq_len = int(max_seq_len)
        self.min_new_tokens = int(min_new_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.restrict_aa_tokens = bool(restrict_aa_tokens)
        self.max_attempts_multiplier = int(max_attempts_multiplier)
        self.attr_weights = [float(x) for x in attr_weights]
        self.timeout = int(timeout)
        self.default_length = int(default_length)
        self.enforce_exact_length = bool(enforce_exact_length)
        self.run_enforce1 = bool(run_enforce1)
        self.lenonly_enable = bool(lenonly_enable)
        self.lenonly_targets = list(lenonly_targets or [])
        self.lenonly_n_per = int(lenonly_n_per) if int(lenonly_n_per) > 0 else int(n_per_combo)
        self.prompt_mode = str(prompt_mode)
        self.use_lftg = bool(use_lftg)
        self.len_regular6_enable = bool(len_regular6_enable)
        self.len_regular6_combos = list(len_regular6_combos or [])
        self._last_step = -1
        self._cached_step = -1
        self._cached_metrics: Dict[str, Any] = {}

    def _apply_combo_summary_to_metrics(self, metrics: Dict[str, Any], summary: Dict[str, Any]) -> None:
        payload_lftg = summary.get("lftg") or {}
        payload_lenonly_free = summary.get("lenonly_free") or {}
        payload_fused_free = summary.get("fused_free") or {}
        payload_fused_enforce1 = summary.get("fused_enforce1") or {}
        if payload_lftg:
            _merge_combo_payload_metrics(metrics, payload_lftg, "eval_lftg")
            metrics["eval_lftg_phase1_exact_rate"] = float(
                (payload_lftg.get("lftg_generation") or {}).get("phase1_exact_rate", 0.0)
            )
        if payload_fused_free:
            _merge_combo_payload_metrics(metrics, payload_fused_free, "eval_fused_free")
        if payload_lenonly_free:
            _merge_combo_payload_metrics(metrics, payload_lenonly_free, "eval_lenonly_free")
        if payload_fused_enforce1:
            _merge_combo_payload_metrics(metrics, payload_fused_enforce1, "eval_fused_enforce1")
        if payload_lenonly_free and payload_lftg:
            len_mae = float(payload_lenonly_free.get("length_control_mae", 0.0))
            combo = float(payload_lftg.get("combo_adherence_mean", payload_lftg.get("strict_adherence_mean", 0.0)))
            attr = float(payload_lftg.get("attr_mean_satisfaction", 0.0))
            metrics["eval_aplus_best_score"] = float(-len_mae + 0.5 * combo + 0.2 * attr)
            safety = float(((payload_lftg.get("per_attr_satisfaction") or {}).get("safety") or {}).get("satisfaction", 0.0))
            metrics["eval_bpp_best_score"] = float(2.0 * combo + 0.5 * attr - 0.05 * len_mae + 0.5 * safety)
        primary = payload_lftg or payload_lenonly_free or payload_fused_free
        metrics["eval_regular6_adherence_mean"] = float(
            (payload_lftg or payload_fused_free).get("combo_adherence_mean", 0.0)
        )
        metrics["eval_regular6_adherence_min"] = float(
            (payload_lftg or payload_fused_free).get("combo_adherence_min", 0.0)
        )
        metrics["eval_regular6_attr_mean_satisfaction"] = float(
            (payload_lftg or payload_fused_free).get("attr_mean_satisfaction", 0.0)
        )
        metrics["eval_regular6_strict_adherence_mean"] = metrics["eval_regular6_adherence_mean"]
        metrics["eval_combo_adherence_mean"] = metrics["eval_regular6_adherence_mean"]
        metrics["eval_combo_adherence_min"] = metrics["eval_regular6_adherence_min"]
        primary_len = payload_lftg or payload_fused_free or {}
        metrics["eval_regular6_length_control_variance"] = float(primary_len.get("length_control_variance", 0.0))
        metrics["eval_regular6_length_control_std"] = float(primary_len.get("length_control_std", 0.0))
        metrics["eval_regular6_length_control_mae"] = float(primary_len.get("length_control_mae", 0.0))
        metrics["eval_regular6_length_control_exact_rate"] = float(primary_len.get("length_control_exact_rate", 0.0))
        metrics["eval_length_control_variance"] = metrics["eval_regular6_length_control_variance"]

    def _restore_combo_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._cached_step == step and self._cached_metrics:
            metrics.update(self._cached_metrics)
            return
        eval_root = self.output_dir / "combo_eval_callback" / f"step_{step:07d}"
        summary_path = eval_root / "eval_summary.json"
        if not summary_path.exists():
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._apply_combo_summary_to_metrics(metrics, summary)

    def _cache_combo_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        self._cached_step = int(step)
        self._cached_metrics = {
            k: v for k, v in metrics.items() if str(k).startswith("eval_") and isinstance(v, (int, float))
        }

    def _run_lftg(
        self,
        model,
        gen_fa: Path,
        out_dir: Path,
        combos: Sequence[str],
        step: int,
        args,
        seed_offset: int,
        n_per_combo: int,
    ) -> Dict[str, Any]:
        gen_model = unwrap_generation_model(model)
        device = next(gen_model.parameters()).device
        gen_stats = generate_length_then_content_for_combos(
            model=model,
            tok=self.tokenizer,
            device=device,
            out_fasta=str(gen_fa),
            combos=combos,
            n_per_combo=int(n_per_combo),
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            restrict_aa_tokens=self.restrict_aa_tokens,
            seed=int(getattr(args, "seed", 42)) + step + seed_offset,
            max_attempts_multiplier=self.max_attempts_multiplier,
            attr_weights=self.attr_weights,
            default_length=self.default_length,
            require_phase1_exact=False,
            return_stats=True,
        )
        n_lines = sum(1 for _ in open(gen_fa, "r", encoding="utf-8"))
        p1_rate = float(gen_stats.get("phase1_exact_rate", 0.0))
        print(
            f"[ComboEval] step={step} {gen_fa.parent.name}/{gen_fa.name} scheme=LFTG "
            f"records~{n_lines // 2} phase1_exact_rate={p1_rate:.4f}",
            flush=True,
        )
        cmd = [sys.executable, self.eval_script, "--in_fasta", str(gen_fa), "--out_dir", str(out_dir)] + self.eval_cmd_args
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:] or "combo eval failed")
        payload = json.loads((out_dir / "eval_payload.json").read_text(encoding="utf-8"))
        payload["lftg_generation"] = gen_stats
        return payload

    def _run_one(
        self,
        model,
        gen_fa: Path,
        out_dir: Path,
        combos: Sequence[str],
        step: int,
        args,
        seed_offset: int,
        *,
        enforce_exact_length: bool,
        n_per_combo: int,
        prompt_mode: str,
    ) -> Dict[str, Any]:
        """生成一个 FASTA、跑判别、返回 payload。"""
        generate_sequences_for_combos(
            model=model,
            tok=self.tokenizer,
            device=next(model.parameters()).device,
            out_fasta=str(gen_fa),
            combos=combos,
            n_per_combo=int(n_per_combo),
            min_seq_len=self.min_seq_len,
            max_seq_len=self.max_seq_len,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            restrict_aa_tokens=self.restrict_aa_tokens,
            seed=int(getattr(args, "seed", 42)) + step + seed_offset,
            max_attempts_multiplier=self.max_attempts_multiplier,
            attr_weights=self.attr_weights,
            enforce_exact_length=bool(enforce_exact_length),
            default_length=self.default_length,
            prompt_mode=str(prompt_mode),
        )
        n_lines = sum(1 for _ in open(gen_fa, "r", encoding="utf-8"))
        print(
            f"[ComboEval] step={step} {gen_fa.parent.name}/{gen_fa.name} "
            f"enforce={int(enforce_exact_length)} mode={prompt_mode} records~{n_lines // 2}",
            flush=True,
        )
        cmd = [sys.executable, self.eval_script, "--in_fasta", str(gen_fa), "--out_dir", str(out_dir)] + self.eval_cmd_args
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:] or "combo eval failed")
        return json.loads((out_dir / "eval_payload.json").read_text(encoding="utf-8"))

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not self.enabled or metrics is None:
            return
        step = int(getattr(state, "global_step", 0))
        if step <= self._last_step:
            self._restore_combo_metrics(metrics, step)
            return
        eval_steps = int(getattr(args, "eval_steps", 10) or 10)
        if not should_run_heavy_eval(step, eval_steps):
            self._last_step = step
            self._restore_combo_metrics(metrics, step)
            print(f"[ComboEval] step={step} skip heavy eval (throttle/alternate)", flush=True)
            return
        if model is None:
            return
        self._last_step = step
        eval_root = self.output_dir / "combo_eval_callback" / f"step_{step:07d}"
        eval_root.mkdir(parents=True, exist_ok=True)
        was_training = bool(model.training)
        try:
            model.eval()
            payload_regular: Dict[str, Any] = {}
            payload_len_regular6: Dict[str, Any] = {}

            regular_dir = eval_root / "regular6"
            regular_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[ComboEval] step={step} regular6 attr-only "
                f"({len(self.regular_combos)} conditions, {self.n_per_combo} each)",
                flush=True,
            )
            payload_regular = self._run_one(
                model,
                regular_dir / "generated.fa",
                regular_dir,
                self.regular_combos,
                step,
                args,
                seed_offset=0,
                enforce_exact_length=False,
                n_per_combo=self.n_per_combo,
                prompt_mode=self.prompt_mode,
            )

            if self.len_regular6_enable and self.len_regular6_combos:
                len_dir = eval_root / "len_regular6"
                len_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[ComboEval] step={step} len_regular6 "
                    f"({len(self.len_regular6_combos)} conditions, {self.n_per_combo} each)",
                    flush=True,
                )
                payload_len_regular6 = self._run_one(
                    model,
                    len_dir / "generated.fa",
                    len_dir,
                    self.len_regular6_combos,
                    step,
                    args,
                    seed_offset=5000,
                    enforce_exact_length=True,
                    n_per_combo=self.n_per_combo,
                    prompt_mode=self.prompt_mode,
                )
                metrics["eval_len_regular6_adherence_mean"] = float(
                    payload_len_regular6.get("combo_adherence_mean", payload_len_regular6.get("strict_adherence_mean", 0.0))
                )
                metrics["eval_len_regular6_attr_mean_satisfaction"] = float(
                    payload_len_regular6.get("attr_mean_satisfaction", 0.0)
                )
                metrics["eval_len_regular6_length_control_exact_rate"] = float(
                    payload_len_regular6.get("length_control_exact_rate", 0.0)
                )

            m6 = float(payload_regular.get("combo_adherence_mean", payload_regular.get("strict_adherence_mean", 0.0)))
            pas6 = payload_regular.get("per_attr_satisfaction", {})
            attr_mean6 = float(
                payload_regular.get("attr_mean_satisfaction")
                or (sum(float(r.get("satisfaction", 0)) for r in pas6.values()) / max(1, len(pas6)) if pas6 else 0.0)
            )
            metrics["eval_regular6_adherence_mean"] = m6
            metrics["eval_regular6_adherence_min"] = float(payload_regular.get("combo_adherence_min", 0.0))
            metrics["eval_regular6_attr_mean_satisfaction"] = attr_mean6
            metrics["eval_regular6_strict_adherence_mean"] = m6
            metrics["eval_combo_adherence_mean"] = m6
            metrics["eval_combo_adherence_min"] = float(payload_regular.get("combo_adherence_min", 0.0))
            metrics["eval_strict14_adherence_mean"] = 0.0
            metrics["eval_strict14_adherence_min"] = 0.0
            metrics["eval_strict14_attr_mean_satisfaction"] = 0.0
            metrics["eval_strict14_strict_adherence_mean"] = 0.0

            summary: Dict[str, Any] = {
                "step": step,
                "regular6_only": bool(self.regular6_only),
                "eval_categories": {
                    "validation_ce": "trainer metrics: eval_loss, eval_ppl, eval_runtime",
                    "length_eval": "eval_stage/step_*/05_length_eval/length_eval.summary.json",
                    "combo_attr_only": "regular6 — 5-attr fused prompt, no fixed length",
                    "combo_len_attr": "len_regular6 — LEN+4-attr combo prompt",
                },
                "regular6": _payload_summary_block(payload_regular),
                "len_regular6": _payload_summary_block(payload_len_regular6) if payload_len_regular6 else {},
            }
            (eval_root / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[ComboEval] step={step} regular6 adh={m6:.4f} attr_mean={attr_mean6:.4f}",
                flush=True,
            )
            if payload_len_regular6:
                print(
                    f"[ComboEval] step={step} len_regular6 adh="
                    f"{float(payload_len_regular6.get('combo_adherence_mean', 0.0)):.4f} "
                    f"len_exact={float(payload_len_regular6.get('length_control_exact_rate', 0.0)):.4f}",
                    flush=True,
                )
            self._cache_combo_metrics(metrics, step)
            if was_training:
                model.train()
        except Exception as e:
            print(f"[WARN] ComboEvalCallback failed at step={step}: {e}", flush=True)
            metrics["eval_regular6_adherence_mean"] = 0.0
            metrics["eval_regular6_adherence_min"] = 0.0
            metrics["eval_strict14_adherence_mean"] = 0.0
            metrics["eval_strict14_adherence_min"] = 0.0
            metrics["eval_combo_adherence_mean"] = 0.0
            metrics["eval_combo_adherence_min"] = 0.0
            metrics["eval_len_regular6_adherence_mean"] = 0.0
            try:
                if was_training:
                    model.train()
            except Exception:
                pass


UNLIMITED_MAX_STEPS = 999_999_999


def resolve_max_steps(max_steps: int) -> int:
    ms = int(max_steps)
    return int(UNLIMITED_MAX_STEPS) if ms <= 0 else ms


def build_training_args(
    a: argparse.Namespace,
    outdir: Path,
    use_auc_for_best: bool,
    *,
    do_eval: bool = True,
) -> TrainingArguments:
    resolved_max_steps = resolve_max_steps(int(a.max_steps))
    staged = int(getattr(a, "use_staged_eval_schedule", 0)) == 1 and do_eval
    eval_steps = resolved_max_steps + 1 if staged else int(a.eval_steps)
    save_steps = resolved_max_steps + 1 if staged else int(a.save_steps)
    kwargs: Dict[str, Any] = dict(
        output_dir=str(outdir),
        max_steps=resolved_max_steps,
        per_device_train_batch_size=int(a.train_bs),
        per_device_eval_batch_size=int(a.eval_bs),
        gradient_accumulation_steps=int(a.grad_accum),
        learning_rate=float(a.lr),
        weight_decay=float(a.weight_decay),
        warmup_ratio=float(a.warmup_ratio),
        lr_scheduler_type=str(a.lr_scheduler),
        logging_steps=int(a.logging_steps),
        eval_strategy="steps" if do_eval else "no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(a.save_total_limit),
        load_best_model_at_end=(bool(a.load_best_model_at_end) and do_eval),
        seed=int(a.seed),
        dataloader_num_workers=int(a.num_workers),
        max_grad_norm=float(a.max_grad_norm),
        optim=str(a.optim),
        label_smoothing_factor=float(a.label_smoothing),
        # W&B 仅走 UnifiedWandbCallback（显式 step），避免与 HF 内置 report_to=wandb 双写导致重复曲线与 step 乱序。
        report_to="none",
        remove_unused_columns=False,
    )
    if do_eval:
        kwargs["eval_steps"] = eval_steps
    best_metric = str(getattr(a, "best_metric", "eval_loss")).strip()
    if bool(a.load_best_model_at_end) and do_eval:
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    elif best_metric:
        kwargs["metric_for_best_model"] = best_metric
        kwargs["greater_is_better"] = bool(int(getattr(a, "best_metric_greater_is_better", 1)) == 1)
    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    if "safe_serialization" in supported:
        kwargs["safe_serialization"] = False
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return TrainingArguments(**filtered)


def _collect_combo_eval_args(a: argparse.Namespace) -> List[str]:
    pairs = [
        ("--ampsorter_script", a.combo_eval_ampsorter_script),
        ("--ampsorter_py", a.combo_eval_ampsorter_py),
        ("--ampsorter_backbone", a.combo_eval_ampsorter_backbone),
        ("--ampsorter_tokenizer", a.combo_eval_ampsorter_tokenizer),
        ("--ampsorter_classifier_pt", a.combo_eval_ampsorter_classifier_pt),
        ("--tox_script", a.combo_eval_tox_script),
        ("--tox_py", a.combo_eval_tox_py),
        ("--tox_backbone", a.combo_eval_tox_backbone),
        ("--tox_tokenizer", a.combo_eval_tox_tokenizer),
        ("--tox_classifier_pt", a.combo_eval_tox_classifier_pt),
        ("--iampcn_repo", a.combo_eval_iampcn_repo),
        ("--iampcn_py", a.combo_eval_iampcn_py),
        ("--iampcn_device", a.combo_eval_iampcn_device),
        ("--tox_thresh", a.combo_eval_tox_thresh),
        ("--iampcn_thresh", a.combo_eval_iampcn_thresh),
        ("--min_len", a.combo_eval_min_len),
        ("--max_len", a.combo_eval_max_len),
        ("--timeout", a.combo_eval_timeout),
    ]
    required = {
        "--ampsorter_script",
        "--ampsorter_py",
        "--ampsorter_backbone",
        "--ampsorter_tokenizer",
        "--ampsorter_classifier_pt",
        "--tox_script",
        "--tox_py",
        "--tox_backbone",
        "--tox_tokenizer",
        "--tox_classifier_pt",
        "--iampcn_repo",
        "--iampcn_py",
    }
    missing = [k for k, v in pairs if k in required and not str(v).strip()]
    if missing:
        return []
    out: List[str] = []
    for k, v in pairs:
        out.extend([k, str(v)])
    return out


def main() -> None:
    default_csv = DEFAULT_CSV_PATH
    default_eval_script = DEFAULT_COMBO_EVAL_SCRIPT
    p = argparse.ArgumentParser("07.train_six_attr_strict_combo_lora.py")
    p.add_argument("--csv_path", default=default_csv)
    p.add_argument("--init_model_dir", default=DEFAULT_BASE_MODEL_DIR)
    p.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_seed", default="split_v1")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument(
        "--train_only",
        action="store_true",
        help="全部数据用于训练，关闭 val/test 划分",
    )
    p.add_argument(
        "--full_train_heavy_eval",
        type=int,
        default=0,
        help="1=train_only 但仍每 eval_steps 跑 combo/length 重评估",
    )
    p.add_argument("--min_seq_len", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--pad_to_multiple_of", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--use_staged_eval_schedule", type=int, default=1, help="1=前100步每10步eval，之后每100步")
    p.add_argument("--eval_steps_early", type=int, default=10)
    p.add_argument("--eval_steps_late", type=int, default=100)
    p.add_argument("--eval_steps_early_until", type=int, default=100)
    p.add_argument("--best_metric", default="eval_loss")
    p.add_argument("--best_metric_greater_is_better", type=int, default=0)
    p.add_argument("--stop_head_enable", type=int, default=1)
    p.add_argument("--stop_head_dropout", type=float, default=0.05)
    p.add_argument("--stop_loss_weight", type=float, default=1.0)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--load_best_model_at_end", action="store_true")
    p.add_argument("--overwrite_output_dir", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--label_smoothing", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=4)
    p.add_argument("--aplus_metric_early_stop", type=int, default=0, help="1=use metric-only early stopping without loading best model")
    p.add_argument("--aplus_early_stop_metric", default="eval_aplus_best_score")
    p.add_argument("--aplus_early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--eval_max_samples", type=int, default=5000)
    p.add_argument("--auc_max_samples", type=int, default=1500)
    p.add_argument("--use_auc_early_stop", action="store_true")
    p.add_argument("--wandb_enable", action="store_true")
    p.add_argument("--wandb_project", default="AMP_FiveAttrSingleLoRA")
    p.add_argument("--wandb_group", default="five_attr_single_lora")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.08)
    p.add_argument("--lora_target_modules", type=str, default="c_attn,c_proj,c_fc")
    p.add_argument("--lora_modules_to_save", type=str, default="lm_head")
    p.add_argument("--lora_no_lm_head", action="store_true")
    p.add_argument("--lora_use_dora", action="store_true")
    p.add_argument("--lora_no_rslora", action="store_true")
    p.add_argument("--train_attr_weights", type=str, default="1,1,1,1,1", help="gp,gn,fu,safety,length CE weights")
    p.add_argument("--length_balance_enable", type=int, default=0, help="1=round-robin train samples by sequence length buckets")
    p.add_argument("--length_balance_bins", default="6-10,11-15,16-20,21-25,26-30,31-35,36-40")
    p.add_argument("--combo_eval_enable", action="store_true")
    p.add_argument("--use_external_unified_eval_callback", action="store_true")
    p.add_argument("--combo_eval_script", default=default_eval_script)
    p.add_argument("--combo_eval_combos", type=str, default=",".join(COMBOS_14))
    p.add_argument("--combo_eval_n_per_combo", type=int, default=100)
    p.add_argument("--combo_eval_gen_min_seq_len", type=int, default=1)
    p.add_argument("--combo_eval_gen_max_seq_len", type=int, default=128)
    p.add_argument("--combo_eval_gen_min_new_tokens", type=int, default=1)
    p.add_argument("--combo_eval_gen_max_new_tokens", type=int, default=128)
    p.add_argument("--combo_eval_gen_temperature", type=float, default=1.0)
    p.add_argument("--combo_eval_gen_top_p", type=float, default=0.95)
    p.add_argument("--combo_eval_gen_top_k", type=int, default=0)
    p.add_argument("--combo_eval_gen_repetition_penalty", type=float, default=1.0)
    p.add_argument("--combo_eval_gen_no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--combo_eval_gen_restrict_aa_tokens", type=int, default=0)
    p.add_argument("--combo_eval_gen_max_attempts_multiplier", type=int, default=20)
    p.add_argument("--combo_eval_attr_weights", type=str, default="1,1,1,1,5")
    p.add_argument("--combo_eval_default_length", type=int, default=24)
    p.add_argument(
        "--combo_eval_lengths",
        type=str,
        default="",
        help="逗号分隔训练期 combo eval 目标长度；空则使用 DEFAULT_EVAL_LENGTHS 多档",
    )
    p.add_argument("--combo_eval_regular6_only", action="store_true", help="训练中 combo eval 仅跑 regular6，跳过 strict14")
    p.add_argument("--combo_eval_enforce_exact_length", type=int, default=0, help="fused 生成是否硬截断长度；0=自由停步")
    p.add_argument("--combo_eval_run_enforce1", type=int, default=0, help="1=额外跑 fused enforce=1 对照")
    p.add_argument("--combo_eval_lenonly_enable", type=int, default=0, help="0=关闭 lenonly ablation（与 length_eval 冗余）")
    p.add_argument("--combo_eval_lenonly_n_per", type=int, default=0, help="0=与 n_per_combo 相同")
    p.add_argument("--combo_eval_lenonly_attr_bits", default="XX10")
    p.add_argument("--combo_eval_use_lftg", type=int, default=0, help="0=关闭 LFTG 多长度 combo（与 len_regular6 冗余）")
    p.add_argument("--combo_eval_len_regular6_enable", action="store_true", help="额外跑 len_regular6（固定长度+6组合）")
    p.add_argument(
        "--combo_eval_len_regular6_lengths",
        type=str,
        default="6,8,10,12,16,20,24,28,32,40",
        help="len_regular6 目标长度，逗号分隔；默认与 length_eval 十档一致",
    )
    p.add_argument("--combo_eval_ampsorter_script", default=DEFAULT_AMPSORTER_SCRIPT)
    p.add_argument("--combo_eval_ampsorter_py", default=DEFAULT_PROTEOGPT_PY)
    p.add_argument("--combo_eval_ampsorter_backbone", default=DEFAULT_AMPSORTER_BACKBONE)
    p.add_argument("--combo_eval_ampsorter_tokenizer", default=DEFAULT_AMPSORTER_TOKENIZER)
    p.add_argument("--combo_eval_ampsorter_classifier_pt", default=DEFAULT_AMPSORTER_CLASSIFIER_PT)
    p.add_argument("--combo_eval_tox_script", default=DEFAULT_TOX_SCRIPT)
    p.add_argument("--combo_eval_tox_py", default=DEFAULT_PROTEOGPT_PY)
    p.add_argument("--combo_eval_tox_backbone", default=DEFAULT_TOX_BACKBONE)
    p.add_argument("--combo_eval_tox_tokenizer", default=DEFAULT_TOX_TOKENIZER)
    p.add_argument("--combo_eval_tox_classifier_pt", default=DEFAULT_TOX_CLASSIFIER_PT)
    p.add_argument("--combo_eval_iampcn_repo", default=DEFAULT_IAMPCN_REPO)
    p.add_argument("--combo_eval_iampcn_py", default=DEFAULT_IAMPCN_PY)
    p.add_argument("--combo_eval_iampcn_device", default="auto")
    p.add_argument("--combo_eval_tox_thresh", type=float, default=0.5)
    p.add_argument("--combo_eval_iampcn_thresh", type=float, default=0.5)
    p.add_argument("--combo_eval_min_len", type=int, default=1)
    p.add_argument("--combo_eval_max_len", type=int, default=128)
    p.add_argument("--combo_eval_timeout", type=int, default=3600)
    p.add_argument(
        "--length_eval_enable",
        action="store_true",
        help="训练中额外跑 LEN{n}: 条件生成长度评估（与 uniprot length 三模型一致）",
    )
    a = p.parse_args()

    if not _PEFT_AVAILABLE:
        raise RuntimeError("peft is not installed")

    set_seed(int(a.seed))
    outdir = Path(str(a.output_dir))
    outdir.mkdir(parents=True, exist_ok=True)

    train_only = bool(getattr(a, "train_only", False))
    full_train_heavy_eval = int(getattr(a, "full_train_heavy_eval", 0)) == 1
    heavy_eval_enabled = (not train_only) or full_train_heavy_eval
    val_frac = 0.0 if train_only else float(a.val_frac)
    test_frac = 0.0 if train_only else float(a.test_frac)

    df_flat = load_clean_csv_multi(
        str(a.csv_path),
        str(a.split_seed),
        val_frac,
        test_frac,
        int(a.min_seq_len),
    )
    train_flat = df_flat[df_flat["split"] == "train"].copy()
    val_flat = df_flat[df_flat["split"] == "val"].copy()
    test_flat = df_flat[df_flat["split"] == "test"].copy()
    if train_only:
        print(f"[DATA] train_only=1 all_rows={len(train_flat)} (no val/test split)", flush=True)
    else:
        print(f"[DATA] train={len(train_flat)} val={len(val_flat)} test={len(test_flat)}", flush=True)
    if full_train_heavy_eval:
        print(
            f"[DATA] full_train_heavy_eval=1 -> combo/length eval every {int(a.eval_steps)} steps "
            f"(eval trigger subset max={int(a.eval_max_samples)})",
            flush=True,
        )

    weight_map, stats_per_task = compute_multi_attr_weights(train_flat)
    for task in list(stats_per_task.keys()):
        st = stats_per_task[task]
        extra = ""
        if "combo_bit_counts" in st:
            extra = f" bits={st['combo_bit_counts']}"
        print(f"[STAT] {task}: count={st['count']} weight={st['weight']:.3f}{extra}", flush=True)
    use_auc_for_best = False

    train_weight_map = parse_attr_weights(str(getattr(a, "train_attr_weights", "1,1,1,1,1")))
    print(
        "[TRAIN_WEIGHTS] "
        + ",".join(
            [
                f"GP={train_weight_map['GP1']}",
                f"GN={train_weight_map['GN1']}",
                f"FU={train_weight_map['FU1']}",
                f"SF={train_weight_map['SF0']}",
                f"LEN={train_weight_map['LENGTH']}",
            ]
        ),
        flush=True,
    )
    train_ds = MultiAttrTrainIter(
        train_flat,
        length_balance_enable=(int(getattr(a, "length_balance_enable", 0)) == 1),
        length_balance_bins=str(getattr(a, "length_balance_bins", "6-10,11-15,16-20,21-25,26-30,31-35,36-40")),
    )
    val_ds = None
    test_ds = None
    do_eval = False
    if not train_only:
        val_ds = MultiAttrEvalSet(val_flat, max_samples=int(a.eval_max_samples), seed=int(a.seed))
        test_ds = MultiAttrEvalSet(test_flat, max_samples=int(a.eval_max_samples), seed=int(a.seed))
        do_eval = True
    elif full_train_heavy_eval:
        val_ds = MultiAttrEvalSet(train_flat, max_samples=int(a.eval_max_samples), seed=int(a.seed))
        do_eval = True
    print(train_ds.info(), flush=True)

    if bool(a.tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if bool(a.wandb_enable):
        os.environ["WANDB_PROJECT"] = str(a.wandb_project)
        os.environ["WANDB_GROUP"] = str(a.wandb_group)
        os.environ["WANDB_RUN_NAME"] = str(a.wandb_run_name) or outdir.name

    tok = GPT2TokenizerFast.from_pretrained(str(a.tokenizer_dir))
    if tok.eos_token is None:
        tok.eos_token = EOS_TEXT
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    _strict_check_eos(tok, str(tok.eos_token))

    model = GPT2LMHeadModel.from_pretrained(str(a.init_model_dir))
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False
    for k in ("resid_pdrop", "attn_pdrop", "embd_pdrop"):
        if hasattr(model.config, k):
            setattr(model.config, k, float(a.dropout))

    target_mods = [x.strip() for x in str(a.lora_target_modules).split(",") if x.strip()]
    modules_to_save = None if bool(a.lora_no_lm_head) else [x.strip() for x in str(a.lora_modules_to_save).split(",") if x.strip()]
    lora_kw: Dict[str, Any] = dict(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(a.lora_r),
        lora_alpha=int(a.lora_alpha),
        lora_dropout=float(a.lora_dropout),
        target_modules=target_mods,
        modules_to_save=modules_to_save,
        bias="none",
        fan_in_fan_out=True,
        use_rslora=not bool(a.lora_no_rslora),
        init_lora_weights=True,
    )
    if bool(a.lora_use_dora):
        lora_kw["use_dora"] = True
    try:
        model = get_peft_model(model, LoraConfig(**lora_kw))
    except Exception as e:
        if bool(a.lora_use_dora) and "Conv1D" in str(e):
            lora_kw.pop("use_dora", None)
            model = get_peft_model(model, LoraConfig(**lora_kw))
        else:
            raise
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] single adapter trainable={trainable:,}/{total:,} ({100*trainable/max(1,total):.2f}%)", flush=True)

    stop_head_enable = int(getattr(a, "stop_head_enable", 1)) == 1
    if stop_head_enable:
        model = FiveAttrModelWithStopHead(
            model,
            hidden_size=int(model.config.n_embd),
            stop_dropout=float(a.stop_head_dropout),
        )
        sh_trainable = sum(p.numel() for p in model.stop_head.parameters() if p.requires_grad)
        print(f"[StopHead] enabled trainable={sh_trainable:,} stop_loss_weight={float(a.stop_loss_weight)}", flush=True)

    callbacks: List[TrainerCallback] = [SetEpochCB(train_ds)]
    if train_only and not full_train_heavy_eval:
        print("[TRAIN] train_only=1 -> skip validation/combo/length eval callbacks", flush=True)
    if heavy_eval_enabled and int(getattr(a, "use_staged_eval_schedule", 0)) == 1:
        callbacks.append(
            StagedEvalScheduleCallback(
                early_step=int(a.eval_steps_early),
                late_step=int(a.eval_steps_late),
                until_step=int(a.eval_steps_early_until),
            )
        )
        print(
            f"[SCHED] staged eval/save: every {int(a.eval_steps_early)} until step {int(a.eval_steps_early_until)}, "
            f"then every {int(a.eval_steps_late)}",
            flush=True,
        )
    combo_eval_args = _collect_combo_eval_args(a)
    combo_eval_default_length = int(a.combo_eval_default_length)
    combo_eval_lengths = parse_length_list(str(a.combo_eval_lengths), single_fallback=combo_eval_default_length)
    combo_weight_map = parse_attr_weights(str(a.combo_eval_attr_weights))
    combo_weight_list = [
        combo_weight_map["GP1"],
        combo_weight_map["GN1"],
        combo_weight_map["FU1"],
        combo_weight_map["SF0"],
        combo_weight_map["LENGTH"],
    ]
    if bool(a.use_external_unified_eval_callback) and heavy_eval_enabled:
        if UnifiedFourAttrSingleLoRAEvalCallback is None:
            print("[WARN] external unified eval callback import failed; fallback to internal callback", flush=True)
        else:
            callbacks.append(
                UnifiedFourAttrSingleLoRAEvalCallback(
                    tokenizer=tok,
                    enabled=True,
                    combos=parse_combo_text(str(a.combo_eval_combos), default_length=combo_eval_default_length),
                )
            )
    elif bool(a.combo_eval_enable) and heavy_eval_enabled:
        if not combo_eval_args:
            print("[WARN] combo_eval_enable=1 but evaluator paths are incomplete; callback skipped", flush=True)
        else:
            len_regular6_lengths = parse_length_list(
                str(getattr(a, "combo_eval_len_regular6_lengths", "6,8,10,12,16,20,24,28,32,40")),
                single_fallback=combo_eval_default_length,
            )
            len_regular6_combos = (
                expand_conditions_with_lengths(REGULAR_CONDITIONS_6, len_regular6_lengths)
                if bool(getattr(a, "combo_eval_len_regular6_enable", False))
                else []
            )
            callbacks.append(
                ComboEvalCallback(
                    enabled=True,
                    tokenizer=tok,
                    output_dir=outdir,
                    eval_script=str(a.combo_eval_script),
                    eval_cmd_args=combo_eval_args,
                    regular_combos=list(REGULAR_CONDITIONS_6),
                    strict_combos=[] if bool(a.combo_eval_regular6_only) else list(COMBOS_14),
                    n_per_combo=int(a.combo_eval_n_per_combo),
                    min_seq_len=int(a.combo_eval_gen_min_seq_len),
                    max_seq_len=int(a.combo_eval_gen_max_seq_len),
                    min_new_tokens=int(a.combo_eval_gen_min_new_tokens),
                    max_new_tokens=int(a.combo_eval_gen_max_new_tokens),
                    temperature=float(a.combo_eval_gen_temperature),
                    top_p=float(a.combo_eval_gen_top_p),
                    top_k=int(a.combo_eval_gen_top_k),
                    repetition_penalty=float(a.combo_eval_gen_repetition_penalty),
                    no_repeat_ngram_size=int(a.combo_eval_gen_no_repeat_ngram_size),
                    restrict_aa_tokens=(int(a.combo_eval_gen_restrict_aa_tokens) == 1),
                    max_attempts_multiplier=int(a.combo_eval_gen_max_attempts_multiplier),
                    attr_weights=combo_weight_list,
                    timeout=int(a.combo_eval_timeout),
                    default_length=combo_eval_default_length,
                    enforce_exact_length=False,
                    regular6_only=bool(a.combo_eval_regular6_only),
                    run_enforce1=False,
                    lenonly_enable=False,
                    lenonly_targets=[],
                    lenonly_n_per=0,
                    prompt_mode="fused",
                    use_lftg=False,
                    len_regular6_enable=bool(getattr(a, "combo_eval_len_regular6_enable", False)),
                    len_regular6_combos=len_regular6_combos,
                )
            )
            print(
                f"[ComboEval] regular6={len(REGULAR_CONDITIONS_6)}×{int(a.combo_eval_n_per_combo)}; "
                f"len_regular6={len(len_regular6_combos)}×{int(a.combo_eval_n_per_combo)} "
                f"(lengths={len_regular6_lengths}); early_stop=eval_loss",
                flush=True,
            )
    if heavy_eval_enabled:
        callbacks.append(ValidationMetricsCallback())
    if heavy_eval_enabled and int(getattr(a, "aplus_metric_early_stop", 0)) == 1 and int(a.early_stop_patience) > 0:
        callbacks.append(
            MetricEarlyStoppingCallback(
                metric_name=str(getattr(a, "aplus_early_stop_metric", "eval_aplus_best_score")),
                greater_is_better=(int(getattr(a, "best_metric_greater_is_better", 1)) == 1),
                patience=int(a.early_stop_patience),
                min_delta=float(getattr(a, "aplus_early_stop_min_delta", 0.0)),
            )
        )
    elif heavy_eval_enabled and int(a.early_stop_patience) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(a.early_stop_patience)))
    if heavy_eval_enabled and bool(getattr(a, "length_eval_enable", False)):
        if LengthEvalCallback is None:
            print("[WARN] length_eval_enable=1 but LengthEvalCallback import failed; skipped", flush=True)
        else:
            callbacks.append(LengthEvalCallback(tokenizer=tok, enabled=True))
            print(
                "[LengthEval] enabled "
                f"(LENGTH_EVAL_LENGTHS={os.environ.get('LENGTH_EVAL_LENGTHS', 'default')} "
                f"n_per={os.environ.get('LENGTH_EVAL_N_PER_LEN', '10')})",
                flush=True,
            )
    callbacks.append(UnifiedWandbCallback(bool(a.wandb_enable)))

    train_collator = CollatorPromptMaskWithPos(
        tok=tok,
        eos_text=str(tok.eos_token),
        pad_to_multiple_of=int(a.pad_to_multiple_of),
        max_length=int(a.max_length),
        task_weights=train_weight_map,
    )
    trainer_kw: Dict[str, Any] = dict(
        model=model,
        args=build_training_args(a, outdir, use_auc_for_best=use_auc_for_best, do_eval=do_eval),
        train_dataset=train_ds,
        data_collator=train_collator,
        callbacks=callbacks,
        eos_text=str(tok.eos_token),
        auc_max_samples=int(a.auc_max_samples),
        stop_loss_weight=float(a.stop_loss_weight),
        stop_head_enable=stop_head_enable,
        **_trainer_tokenizer_kwargs(tok),
    )
    if do_eval and val_ds is not None:
        eval_collator = CollatorPromptMaskWithPos(
            tok=tok,
            eos_text=str(tok.eos_token),
            pad_to_multiple_of=int(a.pad_to_multiple_of),
            max_length=int(a.max_length),
            task_weights=None,
        )
        trainer_kw["eval_dataset"] = val_ds
        if "eval_data_collator" in inspect.signature(Trainer.__init__).parameters:
            trainer_kw["eval_data_collator"] = eval_collator
    trainer = TrainerWithMultiAttrAuc(**trainer_kw)

    print("[TRAIN] start", flush=True)
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(str(outdir))
    if last_ckpt:
        print(f"[TRAIN] resume from {last_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=last_ckpt)

    if do_eval and val_ds is not None and not train_only:
        print("VAL :", trainer.evaluate(eval_dataset=val_ds), flush=True)
    if not train_only and test_ds is not None:
        print("TEST:", trainer.evaluate(eval_dataset=test_ds), flush=True)

    trainer.save_model(str(outdir))
    tok.save_pretrained(str(outdir))
    if isinstance(trainer.model, FiveAttrModelWithStopHead):
        torch.save(trainer.model.stop_head.state_dict(), str(outdir / "stop_head.pt"))
        print(f"[StopHead] saved: {outdir / 'stop_head.pt'}", flush=True)

    if isinstance(trainer.model, FiveAttrModelWithStopHead):
        peft_model = trainer.model.lm
    elif isinstance(trainer.model, PeftModel):
        peft_model = trainer.model
    else:
        peft_model = None

    if peft_model is not None and isinstance(peft_model, PeftModel):
        base_model = getattr(peft_model, "base_model", None)
        if base_model is not None and hasattr(base_model, "save_pretrained"):
            (outdir / "base_model").mkdir(parents=True, exist_ok=True)
            try:
                base_model.save_pretrained(str(outdir / "base_model"), safe_serialization=True)
            except Exception as e:
                try:
                    if hasattr(base_model, "config") and hasattr(base_model.config, "tie_word_embeddings"):
                        base_model.config.tie_word_embeddings = False
                    base_model.save_pretrained(str(outdir / "base_model"), safe_serialization=True)
                except Exception:
                    print(f"[WARN] base_model save_pretrained failed; skipped. err={e}", flush=True)
        (outdir / "lora_adapter").mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(outdir / "lora_adapter"), safe_serialization=False)
        print(f"[LoRA] saved single adapter: {outdir / 'lora_adapter'}", flush=True)

    cfg = vars(a).copy()
    cfg["stats_per_task"] = stats_per_task
    cfg["use_auc_for_best"] = use_auc_for_best
    cfg["combo_eval_combos_resolved"] = combos_to_jsonable(
        expand_conditions_with_lengths(COMBOS_14, parse_length_list(str(a.combo_eval_lengths), single_fallback=int(a.combo_eval_default_length)))
    )
    cfg["combo_eval_lengths_resolved"] = parse_length_list(str(a.combo_eval_lengths), single_fallback=int(a.combo_eval_default_length))
    cfg["train_only"] = train_only
    cfg["full_train_heavy_eval"] = full_train_heavy_eval
    cfg["task_mode"] = "six_attr_strict_combo_joint_prompt"
    cfg["length_prompt_format"] = "LEN{n}:"
    cfg["combo_eval_lenonly_targets"] = expand_lenonly_targets(
        parse_length_list(str(a.combo_eval_lengths), single_fallback=int(a.combo_eval_default_length)),
        attr_bits=str(getattr(a, "combo_eval_lenonly_attr_bits", "XX10")),
    )
    (outdir / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {outdir}", flush=True)


if __name__ == "__main__":
    main()
