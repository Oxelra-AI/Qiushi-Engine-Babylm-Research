#!/usr/bin/env python3
"""Corrected full nine-column evaluator for research official-geometry candidates.

Targets must be frozen by a no-AoA trajectory ranking before use. This delegates to the
corrected research full evaluator, adding SuperGLUE and AoA only as final measurement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
import sys
from typing import Any, Dict

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/official_geometry_full_eval_candidates')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"

CKPTS = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]
RUNS = {
    "official160_b256": "official160_b256_16k_seed43022",
    "official_cap120geom_b256": "official_cap120geom_b256_16k_seed43022",
    "official_cap120geom_b286_lr2515": "official_cap120geom_b286_lr2515_16k_seed43022",
    "official_cap120geom_b288_lr2515": "official_cap120geom_b288_lr2515_16k_seed43022",
    "official_cap120geom_b295_lr2442": "official_cap120geom_b295_lr2442_16k_seed43022",
    "official160_b223_lr2809": "official160_b223_lr2809_16k_seed43022",
}
CANDIDATES: Dict[str, Dict[str, Any]] = {}
for family, run in RUNS.items():
    for m in CKPTS:
        CANDIDATES[f"{family}_{m}M"] = {
            "run_dir": RUN_BASE / run,
            "endpoint": f"chck_{m}M",
            "description": (
                f"research official-only geometry candidate {family} chck_{m}M. "
                "Selected only by no-AoA trajectory ranking; full eval adds SuperGLUE/AoA as final measurement."
            ),
            "family": family,
        }

base.TARGETS = CANDIDATES
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    if "--list-candidates" in sys.argv:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(CANDIDATES.items())}, indent=2, ensure_ascii=False))
        return
    base.main()

if __name__ == "__main__":
    main()
