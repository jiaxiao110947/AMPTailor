#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare six-attr data with GP0/GN0/FU0 (same 0/1 scheme as SF / discriminators)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_bip = importlib.util.spec_from_file_location("bipolar_lib_prep", str(_SCRIPTS / "07.six_attr_bipolar_prefix_lib.py"))
B = importlib.util.module_from_spec(_bip)
_bip.loader.exec_module(B)

_prep = importlib.util.spec_from_file_location("prep5", str(_SCRIPTS / "07.prepare_five_attr_single_lora_data.py"))
P = importlib.util.module_from_spec(_prep)
_prep.loader.exec_module(P)

P._MIC_LIB.rows_to_flat_tasks = B.rows_to_flat_tasks
P.build_flat_five_attr = lambda df: B.rows_to_flat_tasks(df)

if __name__ == "__main__":
    P.main()
