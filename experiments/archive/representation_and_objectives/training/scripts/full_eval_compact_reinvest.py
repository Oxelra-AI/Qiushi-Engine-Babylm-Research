#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for compact reinvest.

The seed43022 compact_view_reinvest checkpoint is already trained and has the
strongest measured fast no-AoA surface. This wrapper reuses the
robust COMPACT_EXPERIENCE full evaluator but writes all evaluation artifacts into a separate
local output directory, leaving the trained run read-only.
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
A02_RUN_BASE = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"

base.OUT_ROOT = A01_WORKSPACE / "data" / "compact_reinvest_full_eval"
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "compact_view_reinvest": {
        "run_dir": A02_RUN_BASE / "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
        "endpoint": "chck_100M",
        "description": (
            "frontier_consolidation compact generated same-source views with saved word budget "
            "reinvested into additional compact FineWeb source-view packets; "
            "DeBERTa-v2 8x480, baseline16k, fixed seq256, fixed whole-word masking, "
            "AdamW, seed43/extra_init_seed43022/train_rng_seed43023."
        ),
        "family": "frontier_consolidation_density_medium_riskhard_cleanqwen_overlay_reinvest",
    }
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
