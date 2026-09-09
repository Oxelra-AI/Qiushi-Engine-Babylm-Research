#!/usr/bin/env python3
"""research full official-style evaluation wrapper for the AoA-safe developmental
first-pass clean-Qwen arms (dual seed).

Reuses the corrected research runner (which imports babylm_official_scoring.py, so
AoA is scaled to leaderboard units).  Baselines for comparison are the research/research
clean-Qwen arms already evaluated under corrected scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/devcurr_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "qwen_devcurr_firstpass_seed43022": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research AoA-safe developmental first-pass order over the exact research clean-Qwen 10M multiset; DeBERTa-v2 8x480 baseline16k WWM extra_init_seed=43022 train_rng_seed=43023.",
        "family": "debertav2_8x480_16k_step034_devcurr",
    },
    "qwen_devcurr_firstpass_seed43122": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43122'),
        "endpoint": "chck_100M",
        "description": "research second-seed AoA-safe developmental first-pass order; extra_init_seed=43122 train_rng_seed=43123.",
        "family": "debertav2_8x480_16k_step034_devcurr",
    },
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
