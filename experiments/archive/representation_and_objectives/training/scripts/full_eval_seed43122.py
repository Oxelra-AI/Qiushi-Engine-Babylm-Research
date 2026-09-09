#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for compact_view_reinvest seed43122.

Reuses the robust COMPACT_EXPERIENCE full evaluator, pointing it at the trained seed43122
run directory, writing all evaluation artifacts into a dedicated local output dir.
The seed43122 trained run stays read-only for the model weights.
"""
from __future__ import annotations

import pathlib
import sys

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

A01_WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A01_RUN_BASE = A01_WORKSPACE / "training" / "runs"

base.OUT_ROOT = A01_WORKSPACE / "data" / "seed43122_full_eval"
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "compact_view_reinvest_seed43122": {
        "run_dir": A01_RUN_BASE / "repl_compact_view_reinvest_seed43122",
        "endpoint": "chck_100M",
        "description": (
            "A01 independent-seed replication of compact_view_reinvest; identical "
            "corpus and recipe as seed43022 but different initialization seed "
            "(seed43122). Full official-compatible evaluation to measure robustness "
            "of the 42.0331 official-coordinate endpoint across initialization."
        ),
        "family": "representation_and_objectives_density_reinvest_seed43122",
    }
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
