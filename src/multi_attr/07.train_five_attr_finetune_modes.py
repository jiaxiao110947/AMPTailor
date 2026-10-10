#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六属性微调模式训练入口（full / freeze_top / bitfit / lora）。

不修改 07.train_five_attr_single_lora.py：
- lora 模式：子进程调用原脚本
- 其他模式：monkey-patch 原脚本后调用其 main()
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ORIG_PY = SCRIPTS / "07.train_five_attr_single_lora.py"
LIB_PY = SCRIPTS / "07.five_attr_finetune_lib.py"

_FINETUNE_FLAGS = frozenset(
    {
        "--finetune_mode",
        "--freeze_bottom_layers",
        "--gradient_checkpointing",
        "--bf16",
    }
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip_finetune_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in _FINETUNE_FLAGS:
            skip = True
            continue
        if any(arg.startswith(f"{f}=") for f in _FINETUNE_FLAGS):
            continue
        out.append(arg)
    return out


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--finetune_mode", choices=["lora", "full", "freeze_top", "bitfit"], default="full")
    pre.add_argument("--freeze_bottom_layers", type=int, default=10)
    pre.add_argument("--gradient_checkpointing", type=int, default=1)
    pre.add_argument("--bf16", type=int, default=1)
    pre_args, remaining = pre.parse_known_args()

    mode = str(pre_args.finetune_mode).strip().lower()
    if mode == "lora":
        cmd = [sys.executable, str(ORIG_PY)] + _strip_finetune_argv(remaining)
        print(f"[FinetuneEntry] delegate lora -> {' '.join(cmd[:6])} ...", flush=True)
        subprocess.run(cmd, check=True)
        return

    ft = _load_module(LIB_PY, "five_attr_finetune_lib")
    orig = _load_module(ORIG_PY, "five_attr_train_orig")

    orig.build_training_args = ft.wrap_build_training_args(orig.build_training_args, pre_args)
    orig.get_peft_model = ft.make_get_peft_replacement(pre_args)
    orig.TrainerWithMultiAttrAuc = ft.make_trainer_subclass(
        orig.TrainerWithMultiAttrAuc,
        orig.FiveAttrModelWithStopHead,
    )
    orig._PEFT_AVAILABLE = True

    sys.argv = [str(ORIG_PY)] + _strip_finetune_argv(remaining)
    print(f"[FinetuneEntry] mode={mode} patched main argv_len={len(sys.argv)}", flush=True)
    orig.main()


if __name__ == "__main__":
    main()
