#!/usr/bin/env python3
"""research wrapper: fast evaluation for the research-regime b256/fixed-seq pair.

This imports the known-good research official-compatible fast evaluator and only
changes the run map/output paths.  The scientific purpose is a minimal paired
comparison of fixed WWM vs unconditional WWM->token under the inherited strong
research-like regime (fixed seq256, batch256), because the earlier seq-scheduled
100M screen may not be directly aligned to the INITIAL_MODEL_STUDIES 40.7028 coordinate.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path("experiments/archive/compact_experience")
BASE = ROOT / "training/runs"
EVAL_SRC = ROOT / "scripts/eval_curriculum_100M_fast.py"

spec = importlib.util.spec_from_file_location("eval_curriculum_100M_fast", EVAL_SRC)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {EVAL_SRC}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Replace only the scientific objects and evidence paths; preserve all evaluator
# invocation details that have already been validated in Steps 8, 10, and 11.
mod.RUNS = {
    "wwm_fixed": BASE / "wwm_fixed_100M_b256_seq256_seed43",
    "wwm_to_token": BASE / "wwm_to_token_100M_b256_seq256_seed43",
}
mod.DEFAULT_FINAL_TARGETS = ["wwm_fixed@chck_100M", "wwm_to_token@chck_100M"]
mod.DEFAULT_TRAJ_TARGETS = [
    f"{arm}@{ck}"
    for arm in ["wwm_fixed", "wwm_to_token"]
    for ck in ["chck_70M", "chck_80M", "chck_100M"]
]
mod.OUT_ROOT = ROOT / "data/b256_pair_eval"
mod.NOTE_PATH = (ROOT.parents[2] / 'research/notes/compact_experience/b256_fixedseq_pair_eval_raw.md')
mod.FINAL_JSON_PATH = mod.OUT_ROOT / "b256_pair_100M_eval_summary.json"
mod.TRAJ_JSON_PATH = mod.OUT_ROOT / "b256_pair_trajectory_eval_summary.json"

if __name__ == "__main__":
    mod.main()
