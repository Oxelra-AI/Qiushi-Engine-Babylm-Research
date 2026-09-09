#!/usr/bin/env python3
"""research full official-compatible evaluation wrapper for density compact candidates.

This wrapper reuses the robust COMPACT_EXPERIENCE full evaluator but points it at frontier_consolidation
risk-hard medium density runs.  The immediate protected target is
`compact_view_core`: the shared-core compact semantic-view model that produced the
strong research no-AoA task-family gain over matched repetition.  Reinvestment is
kept as a separate extension target and should not be interpreted as deciding the
compact-core mechanism.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

THIS_SCRIPT_DIR = _public_path('experiments/archive/frontier_consolidation/scripts')
THIS_WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
USER_ROOT = _public_path('.')
COMPACT_EXPERIENCE_SCRIPTS = _public_path('experiments/archive/compact_experience/scripts')
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

RUN_BASE = _public_path('experiments/archive/frontier_consolidation/training/runs')
base.OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/density_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "compact_view_core": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": (
            "frontier_consolidation medium risk-hard clean-Qwen row-holdout overlay: compact "
            "Qwen3.5 FineWeb semantic views on the shared source-aligned core, "
            "neutral clean-Qwen top-up, DeBERTa-v2 8x480, baseline16k, fixed "
            "seq256, WWM, seed43/extra_init_seed43022/train_rng_seed43023."
        ),
        "family": "frontier_consolidation_density_medium_riskhard_cleanqwen_overlay_core",
    },
    "compact_repeat_core": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": (
            "Matched source-repetition control for compact_view_core on the same "
            "shared FineWeb core and neutral top-up."
        ),
        "family": "frontier_consolidation_density_medium_riskhard_cleanqwen_overlay_core",
    },
    "compact_view_reinvest": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": (
            "Compact semantic views with saved word budget reinvested into "
            "additional compact source-view packets; extension target, not the "
            "verdict on compact-core view utility."
        ),
        "family": "frontier_consolidation_density_medium_riskhard_cleanqwen_overlay_reinvest",
    },
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
