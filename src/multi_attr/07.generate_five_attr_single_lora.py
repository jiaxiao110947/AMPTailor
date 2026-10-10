#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LIB = _load_module_from_path("five_attr_classification_lora_lib_07", Path(__file__).resolve().parent / "07.five_attr_classification_lora_lib.py")
COMBOS_14 = _LIB.COMBOS_14
REGULAR_CONDITIONS_6 = _LIB.REGULAR_CONDITIONS_6
GENERATION_SCHEME_HEADER = _LIB.GENERATION_SCHEME_HEADER
combos_to_jsonable = _LIB.combos_to_jsonable
expand_conditions_with_lengths = _LIB.expand_conditions_with_lengths
generate_sequences_for_combos = _LIB.generate_sequences_for_combos
_FT = _load_module_from_path(
    "five_attr_finetune_lib_generate",
    Path(__file__).resolve().parent / "07.five_attr_finetune_lib.py",
)


def load_single_lora_model(model_dir: str, tokenizer_dir: str = "", device_str: str = "cuda"):
    """LoRA adapter 或 full / freeze_top / bitfit 全参 checkpoint 都能加载。"""
    tok_dir = str(tokenizer_dir).strip()
    if not tok_dir:
        tok_dir = str(Path(__file__).resolve().parents[2] / "models" / "AMPGuide")
    return _FT.load_five_attr_model(model_dir, tokenizer_dir=tok_dir, device_str=device_str)


parse_attr_weights = _LIB.parse_attr_weights
parse_combo_text = _LIB.parse_combo_text
select_combos = _LIB.select_combos


def _parse_lengths(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("lengths must not be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("all lengths must be positive")
    return vals


def _gen_kw(a: argparse.Namespace) -> Dict[str, Any]:
    weight_map = parse_attr_weights(str(a.attr_weights))
    return dict(
        n_per_combo=int(a.n_per_combo),
        min_seq_len=int(a.min_seq_len),
        max_seq_len=int(a.max_seq_len),
        min_new_tokens=int(a.min_new_tokens),
        max_new_tokens=int(a.max_new_tokens),
        temperature=float(a.temperature),
        top_p=float(a.top_p),
        top_k=int(a.top_k),
        repetition_penalty=float(a.repetition_penalty),
        no_repeat_ngram_size=int(a.no_repeat_ngram_size),
        restrict_aa_tokens=(int(a.restrict_aa_tokens) == 1),
        do_sample=(int(a.do_sample) == 1),
        shuffle_combos=(int(a.shuffle_combos) == 1),
        seed=int(a.seed),
        max_attempts_multiplier=int(a.max_attempts_multiplier),
        attr_weights=[
            weight_map["GP1"],
            weight_map["GN1"],
            weight_map["FU1"],
            weight_map["SF0"],
            weight_map["LENGTH"],
        ],
        return_stats=True,
        enforce_exact_length=(int(a.enforce_exact_length) == 1),
        default_length=int(a.length),
    )


def _resolve_run_list(a: argparse.Namespace) -> List[tuple[str, List[str], Path]]:
    lengths = _parse_lengths(a.lengths) if str(a.lengths).strip() else [int(a.length)]
    base = Path(str(a.out_fasta))

    if str(a.combos).strip():
        combos = parse_combo_text(str(a.combos), default_length=lengths[0])
        return [("custom", combos, base)]

    if bool(a.regular6_and_strict14) or (bool(a.use_regular6) and bool(a.use_all_14)):
        return [
            ("regular6", expand_conditions_with_lengths(REGULAR_CONDITIONS_6, lengths), base.with_name(f"{base.stem}.regular6{base.suffix}")),
            ("strict14", expand_conditions_with_lengths(COMBOS_14, lengths), base.with_name(f"{base.stem}.strict14{base.suffix}")),
        ]
    if bool(a.use_regular6):
        return [("regular6", expand_conditions_with_lengths(REGULAR_CONDITIONS_6, lengths), base)]
    if bool(a.use_all_14):
        return [("strict14", expand_conditions_with_lengths(COMBOS_14, lengths), base)]

    combos = select_combos(
        gram_pos=str(a.gram_pos),
        gram_neg=str(a.gram_neg),
        fungus=str(a.fungus),
        safety=str(a.safety),
        toxic=str(a.toxic),
        target_lengths=lengths,
    )
    return [("grid", combos, base)]


def main() -> None:
    p = argparse.ArgumentParser("07.generate_five_attr_single_lora.py")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tokenizer_dir", default="")
    p.add_argument("--out_fasta", required=True)
    p.add_argument("--combos", default="", help="条件列表，如 1010:24,1110:32（4-bit属性:目标长度）")
    p.add_argument("--gram_pos", choices=["0", "1", "all"], default="all")
    p.add_argument("--gram_neg", choices=["0", "1", "all"], default="all")
    p.add_argument("--fungus", choices=["0", "1", "all"], default="all")
    p.add_argument("--safety", choices=["0", "1", "all"], default="all", help="安全性标签 0=safe 1=unsafe")
    p.add_argument("--toxic", choices=["0", "1", "all"], default="all", help="(deprecated) 同 --safety")
    p.add_argument("--length", type=int, default=24, help="单目标长度（与 --lengths 二选一）")
    p.add_argument("--lengths", default="", help="多个目标长度，逗号分隔，如 16,24,32")
    p.add_argument("--use_all_14", action="store_true", help="14 种严格 4-bit 属性组合 × 指定长度")
    p.add_argument("--use_regular6", action="store_true", help="6 种 regular 条件 × 指定长度")
    p.add_argument("--regular6_and_strict14", action="store_true")
    p.add_argument("--n_per_combo", type=int, default=100)
    p.add_argument("--min_seq_len", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--min_new_tokens", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--enforce_exact_length", type=int, default=1, help="1=严格生成指定长度")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--restrict_aa_tokens", type=int, default=1)
    p.add_argument("--do_sample", type=int, default=1)
    p.add_argument("--shuffle_combos", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_attempts_multiplier", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attr_weights", type=str, default="1,1,1,1,1", help="gp,gn,fu,safety,length")
    a = p.parse_args()

    run_list = _resolve_run_list(a)
    model, tok, device = load_single_lora_model(str(a.model_dir), str(a.tokenizer_dir), device_str=str(a.device))
    kw = _gen_kw(a)
    all_meta: Dict[str, Any] = {"model_dir": str(a.model_dir), "runs": []}

    for tag, combos, out_fa in run_list:
        out_fa.parent.mkdir(parents=True, exist_ok=True)
        stats = generate_sequences_for_combos(
            model=model,
            tok=tok,
            device=device,
            out_fasta=str(out_fa),
            combos=combos,
            **kw,
        )
        meta = {
            "tag": tag,
            "model_dir": str(a.model_dir),
            "tokenizer_dir": str(a.tokenizer_dir or a.model_dir),
            "out_fasta": str(out_fa),
            "combos": combos_to_jsonable(combos),
            "n_per_combo": int(a.n_per_combo),
            "scheme": GENERATION_SCHEME_HEADER,
            "attr_weights": parse_attr_weights(str(a.attr_weights)),
            "enforce_exact_length": bool(int(a.enforce_exact_length) == 1),
            "default_length": int(a.length),
            "lengths": _parse_lengths(a.lengths) if str(a.lengths).strip() else [int(a.length)],
            "do_sample": bool(int(a.do_sample) == 1),
            "shuffle_combos": bool(int(a.shuffle_combos) == 1),
            "stats": stats,
        }
        (out_fa.parent / f"{out_fa.stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        all_meta["runs"].append(meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

    if len(run_list) > 1:
        base = Path(str(a.out_fasta))
        summary_path = base.parent / f"{base.stem}.both.meta.json"
        summary_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(all_meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
