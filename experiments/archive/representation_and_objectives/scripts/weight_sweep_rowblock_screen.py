#!/usr/bin/env python3
"""research: row-block-including fixed screens for the FW same-initialization sweep.

This wraps `weight_sweep_first_screen.py` but writes to a separate output
root so the compact->interleaved first-screen evidence is preserved.  It evaluates
only the two predeclared row-block-containing mixtures after row-block's endpoint
showed hard GlobalPIQA_parallel and EWoK four-cell movement.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
BASE = WS / "scripts/weight_sweep_first_screen.py"

spec = importlib.util.spec_from_file_location("weight_sweep_first_screen_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)  # type: ignore[union-attr]

mod.OUT_ROOT = WS / "data/weight_sweep_rowblock_screen"
mod.ZERO_OUT = mod.OUT_ROOT / "zero_surface"
mod.EWOK_OUT = mod.OUT_ROOT / "official_ewok"
mod.MARGIN_OUT = mod.OUT_ROOT / "globalpiqa_margin"
mod.LOG_ROOT = mod.OUT_ROOT / "logs"
mod.NOTE = (ROOT / 'research/notes/representation_and_objectives/weight_sweep_rowblock_screen.md')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--targets", "cr_a0p25", "cir_i0p25_r0p25"])
    mod.main()
