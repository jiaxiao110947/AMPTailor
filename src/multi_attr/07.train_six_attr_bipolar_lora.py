#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train six-attr LoRA with bipolar GP/GN/FU prefixes. Original trainer is unchanged."""
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

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent
_bip = importlib.util.spec_from_file_location("bipolar_lib_train", str(_SCRIPTS / "07.six_attr_bipolar_prefix_lib.py"))
B = importlib.util.module_from_spec(_bip)
_bip.loader.exec_module(B)

_tr = importlib.util.spec_from_file_location("train5", str(_SCRIPTS / "07.train_five_attr_single_lora.py"))
T = importlib.util.module_from_spec(_tr)
_tr.loader.exec_module(T)

T.BASE_TASKS = list(B.BASE_TASKS)
T.make_task_prompt = B.make_task_prompt
T._LIB.make_task_prompt = B.make_task_prompt
T._LIB.FIXED_TASK_RX = B.FIXED_TASK_RX
T._LIB.BASE_TASKS = list(B.BASE_TASKS)
T._LIB.condition_to_task_names = B.condition_to_task_names
T.parse_attr_weights = B.parse_attr_weights
T.task_weight = B.task_weight
T._MIC_LIB.rows_to_flat_tasks = B.rows_to_flat_tasks

_INV_FREQ_CACHE: Dict[str, float] = {}
_orig_compute_weights = T.compute_multi_attr_weights
_orig_parse_attr_weights = T.parse_attr_weights


def compute_multi_attr_weights(train_flat):
    weight_map, stats_per_task = _orig_compute_weights(train_flat)
    _INV_FREQ_CACHE.clear()
    _INV_FREQ_CACHE.update(weight_map)
    if "LEN" in _INV_FREQ_CACHE and "LENGTH" not in _INV_FREQ_CACHE:
        _INV_FREQ_CACHE["LENGTH"] = float(_INV_FREQ_CACHE["LEN"])
    return weight_map, stats_per_task


def parse_attr_weights(text: str):
    raw = str(text).strip()
    key = raw.lower()
    if key in ("invfreq", "inv"):
        if not _INV_FREQ_CACHE:
            raise RuntimeError("invfreq requested before task stats were computed")
        return dict(_INV_FREQ_CACHE)
    if key in ("invfreq_clip", "invclip"):
        if not _INV_FREQ_CACHE:
            raise RuntimeError("invfreq_clip requested before task stats were computed")
        return {k: float(min(4.0, max(0.5, float(v)))) for k, v in _INV_FREQ_CACHE.items()}
    if key.startswith("negx"):
        mul = float(key[4:] or "3")
        out = {
            "GP0": mul,
            "GP1": 1.0,
            "GN0": mul,
            "GN1": 1.0,
            "FU0": mul,
            "FU1": 1.0,
            "SF0": 1.0,
            "SF1": 1.0,
            "LENGTH": 1.0,
            "LEN": 1.0,
        }
        return out
    return _orig_parse_attr_weights(raw)


T.compute_multi_attr_weights = compute_multi_attr_weights
T.parse_attr_weights = parse_attr_weights

_orig_seq_flatten = T.load_clean_csv_multi
_SCORE_PATH = (_ROOT / "src" / "generation" / "score_expert_discriminators.py")
_spec_score = importlib.util.spec_from_file_location("score_en_train", str(_SCORE_PATH))
SCORE = importlib.util.module_from_spec(_spec_score)
_spec_score.loader.exec_module(SCORE)

PACK = (_ROOT / "models" / "discriminators")
HEADS = {
    "gp": PACK / "gram_positive",
    "gn": PACK / "gram_negative",
    "fu": PACK / "antifungal",
    "sf": PACK / "safety",
}


def load_clean_csv_multi(csv_path, split_seed, val_frac, test_frac, min_seq_len=1):
    df_flat = _orig_seq_flatten(csv_path, split_seed, val_frac, test_frac, min_seq_len)
    tasks = set(df_flat["task"].astype(str))
    if not any(t in tasks for t in ("GP0", "GN0", "FU0")):
        raise RuntimeError(f"bipolar flatten failed, tasks={sorted(tasks)[:20]}")
    return df_flat


T.load_clean_csv_multi = load_clean_csv_multi


def _length_stats(pairs):
    if not pairs:
        return {
            "length_control_variance": 0.0,
            "length_control_std": 0.0,
            "length_control_mae": 0.0,
            "length_control_rmse": 0.0,
            "length_control_mean_dev": 0.0,
            "length_control_exact_rate": 0.0,
        }
    diffs = np.array([float(a - b) for a, b in pairs], dtype=np.float64)
    absd = np.abs(diffs)
    return {
        "length_control_variance": float(np.var(diffs)),
        "length_control_std": float(np.std(diffs)),
        "length_control_mae": float(np.mean(absd)),
        "length_control_rmse": float(np.sqrt(np.mean(diffs * diffs))),
        "length_control_mean_dev": float(np.mean(diffs)),
        "length_control_exact_rate": float(np.mean([1.0 if a == b else 0.0 for a, b in pairs])),
    }


def payload_from_english_fasta(fasta, out_dir, models, toks, device, thr: float = 0.5) -> Dict[str, Any]:
    items = SCORE.read_fasta(str(fasta))
    seqs = [s for _, s in items]
    probs: Dict[str, np.ndarray] = {}
    for name in HEADS:
        tok = toks[name]
        loader = SCORE.make_loader(seqs, tok, int(tok.pad_token_id), 80, 256, 0)
        probs[name] = SCORE.predict_p1(models[name], loader, device)
    rows = []
    length_pairs = []
    for i, (pid, seq) in enumerate(items):
        bits, tlen = SCORE.parse_combo(pid)
        rows.append(
            {
                "id": pid.split("|", 1)[0],
                "header": pid,
                "sequence": seq,
                "length": len(seq),
                "bits": bits,
                "target_len": tlen,
                "p_gp": float(probs["gp"][i]),
                "p_gn": float(probs["gn"][i]),
                "p_fu": float(probs["fu"][i]),
                "p_sf": float(probs["sf"][i]),
                "p_amp": float(probs["amp"][i]) if "amp" in probs else 0.0,
            }
        )
        if tlen is not None:
            length_pairs.append((int(len(seq)), int(tlen)))
    stats = SCORE.adherence_from_rows(rows, thr)
    attr_map = {
        "gp": "gram_pos",
        "gn": "gram_neg",
        "fu": "fungus",
        "sf": "safety",
        "length": "length",
    }
    per_attr_satisfaction = {}
    for src, dst in attr_map.items():
        per_attr = stats.get("per_attr") or {}
        count = int((stats.get("per_attr_n") or {}).get(src, 0))
        if src not in per_attr or count <= 0:
            continue
        per_attr_satisfaction[dst] = {"satisfaction": float(per_attr[src]), "n_common": float(count)}
    lc = _length_stats(length_pairs)
    payload = {
        "scorer": "english_classifiers",
        "threshold": float(thr),
        "n": float(stats.get("n", 0)),
        "combo_adherence_mean": float(stats.get("combo_adherence_mean", 0.0)),
        "combo_adherence_min": float(stats.get("combo_adherence_min", 0.0)),
        "strict_adherence_mean": float(stats.get("combo_adherence_mean", 0.0)),
        "attr_mean_satisfaction": float(stats.get("attr_mean_satisfaction", 0.0)),
        "amp_pos_rate": float(stats.get("amp_pos_rate", 0.0)),
        "precise_amp_attr_rate": float(stats.get("precise_amp_attr_rate", 0.0)),
        "per_attr_satisfaction": per_attr_satisfaction,
        "per_combo": stats.get("per_combo") or {},
        **lc,
        "length_control": lc,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


class ComboEvalEnglishDisc(T.ComboEvalCallback):
    """Same generation as ComboEvalCallback, scored by english_classifiers instead of AmpSorter/iAMPCN/Tox."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disc_models = None
        self._disc_toks = None

    def _ensure_disc(self, device):
        if self._disc_models is not None:
            return
        print("[ComboEval] loading expert-backbone discriminators GP/GN/FU/SF", flush=True)
        models = {}
        toks = {}
        for name, d in HEADS.items():
            tok = SCORE.load_eos_tokenizer(str(d / "tokenizer"), local_files_only=True)
            tok.padding_side = "left"
            tok.pad_token = tok.eos_token
            toks[name] = tok
            models[name] = SCORE.load_binary(str(d / "backbone"), str(d / "weights/best.pt"), device)
            print(f"[ComboEval] loaded disc {name}", flush=True)
        self._disc_models = models
        self._disc_toks = toks

    def _run_one(
        self,
        model,
        gen_fa,
        out_dir,
        combos,
        step,
        args,
        seed_offset,
        *,
        enforce_exact_length,
        n_per_combo,
        prompt_mode,
    ):
        gen_model = T.unwrap_generation_model(model)
        device = next(gen_model.parameters()).device
        T.generate_sequences_for_combos(
            model=gen_model,
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
            enforce_exact_length=bool(enforce_exact_length),
            default_length=self.default_length,
            prompt_mode=str(prompt_mode),
        )
        n_lines = sum(1 for _ in open(gen_fa, "r", encoding="utf-8"))
        print(
            f"[ComboEval] step={step} {gen_fa.parent.name}/{gen_fa.name} "
            f"enforce={int(enforce_exact_length)} mode={prompt_mode} records~{n_lines // 2} "
            f"scorer=english_classifiers",
            flush=True,
        )
        self._ensure_disc(device)
        return payload_from_english_fasta(
            gen_fa, out_dir, self._disc_models, self._disc_toks, device, thr=0.5
        )

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, model=model, **kwargs)
        if metrics is None:
            return
        step = int(getattr(state, "global_step", 0))
        eval_root = self.output_dir / "combo_eval_callback" / f"step_{step:07d}"
        for name in ("regular6", "len_regular6"):
            path = eval_root / name / "eval_payload.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            prefix = f"eval_{name}"
            metrics[f"{prefix}_amp_pos_rate"] = float(payload.get("amp_pos_rate", 0.0))
            metrics[f"{prefix}_precise_rate"] = float(payload.get("precise_amp_attr_rate", 0.0))
            pas = payload.get("per_attr_satisfaction") or {}
            for attr, rec in pas.items():
                if float(rec.get("n_common", 0)) > 0:
                    metrics[f"{prefix}_attr_{attr}"] = float(rec["satisfaction"])
        self._cache_combo_metrics(metrics, step)
        try:
            import wandb

            if getattr(wandb, "run", None) is not None:
                wb = {
                    "trainer/global_step": float(step),
                    "eval/combo_adherence_mean": float(metrics.get("eval_combo_adherence_mean", 0.0)),
                    "eval/regular6/adherence_mean": float(metrics.get("eval_regular6_adherence_mean", 0.0)),
                    "eval/regular6/attr_mean_satisfaction": float(metrics.get("eval_regular6_attr_mean_satisfaction", 0.0)),
                    "eval/regular6/amp_pos_rate": float(metrics.get("eval_regular6_amp_pos_rate", 0.0)),
                    "eval/len_regular6/adherence_mean": float(metrics.get("eval_len_regular6_adherence_mean", 0.0)),
                    "eval/len_regular6/attr_mean_satisfaction": float(metrics.get("eval_len_regular6_attr_mean_satisfaction", 0.0)),
                    "eval/len_regular6/length_control_exact_rate": float(metrics.get("eval_len_regular6_length_control_exact_rate", 0.0)),
                }
                if metrics.get("eval_loss") is not None:
                    wb["eval/loss"] = float(metrics["eval_loss"])
                    wb["eval/ppl"] = float(T._safe_exp(float(metrics["eval_loss"])))
                wandb.log(wb, commit=True)
                print(f"[W&B] combo metrics flushed @ step={step} (no wandb step=)", flush=True)
        except Exception as e:
            print(f"[W&B] combo flush failed: {e}", flush=True)


T.ComboEvalCallback = ComboEvalEnglishDisc

if __name__ == "__main__":
    T.main()
