#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Throttle expensive generation-based eval callbacks."""

import os


def heavy_eval_every_n() -> int:
    try:
        return max(1, int(os.environ.get("HEAVY_EVAL_EVERY_N", "1")))
    except Exception:
        return 1


def _heavy_eval_alternate() -> bool:
    return os.environ.get("HEAVY_EVAL_ALTERNATE", "").strip().lower() in ("1", "true", "yes")


def _heavy_eval_phase_offset() -> int:
    try:
        return int(os.environ.get("HEAVY_EVAL_PHASE_OFFSET", "0"))
    except Exception:
        return 0


def should_run_heavy_eval(step: int, eval_steps: int) -> bool:
    """Return True if generation-based eval (combo/length) should run this round."""
    if os.environ.get("HEAVY_EVAL_FORCE", "").strip() in ("1", "true", "yes"):
        return True
    es = int(eval_steps)
    if step <= 0 or es <= 0:
        return True
    eval_round = int(step) // es
    if _heavy_eval_alternate():
        return (eval_round % 2) == (_heavy_eval_phase_offset() % 2)
    every = heavy_eval_every_n()
    if every <= 1:
        return True
    return (eval_round % every) == 0
