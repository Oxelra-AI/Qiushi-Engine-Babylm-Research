#!/usr/bin/env python3
"""Corrected full nine-column evaluator for selected trajectory-screen checkpoints.

This wrapper dynamically registers candidate checkpoint endpoints and then delegates all work to
full_overall_eval_runner.py, which uses babylm_official_scoring.py for AoA leaderboard-unit
scoring. Candidate selection should come from the no-AoA trajectory ranker; AoA is measured here only
as a final official-style column, never as a training or schedule-design signal.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/full_eval_candidates')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"

CANDIDATES: Dict[str, Dict[str, Any]] = {
    # Existing clean-Qwen family. Endpoint can be any saved chck_*M; model_root contains full ladder for AoA.
    "clean_qwen_seed43022_100M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research clean-Qwen aligned seed43022 checkpoint selected from no-AoA trajectory screen.",
        "family": "clean_qwen_aligned",
    },
    "clean_qwen_seed43122_100M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122'),
        "endpoint": "chck_100M",
        "description": "research clean-Qwen aligned second seed checkpoint selected from no-AoA trajectory screen.",
        "family": "clean_qwen_aligned",
    },
    # research developmental first-pass family. Endpoint can be overridden through --endpoint-map.
    "devcurr_seed43022_100M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research developmental first-pass clean-Qwen seed43022 checkpoint selected from no-AoA trajectory screen.",
        "family": "qwen_devcurr_firstpass",
    },
    "devcurr_seed43122_100M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_devcurr_firstpass_16k_seed43122'),
        "endpoint": "chck_100M",
        "description": "research developmental first-pass clean-Qwen seed43122 checkpoint selected from no-AoA trajectory screen.",
        "family": "qwen_devcurr_firstpass",
    },
}

# Programmatic aliases for frequent candidate endpoints. These keep target names unique because
# research per-target payloads are keyed by target name.
for seed, run in [
    ("43022", "qwen_clean_aligned_16k_seed43022"),
    ("43122", "qwen_clean_aligned_16k_seed43122"),
]:
    for m in [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]:
        CANDIDATES[f"clean_qwen_seed{seed}_{m}M"] = {
            "run_dir": RUN_BASE / run,
            "endpoint": f"chck_{m}M",
            "description": f"Clean-Qwen aligned seed{seed} chck_{m}M selected/available for corrected full eval.",
            "family": "clean_qwen_aligned",
        }
for seed, run in [
    ("43022", "qwen_devcurr_firstpass_16k_seed43022"),
    ("43122", "qwen_devcurr_firstpass_16k_seed43122"),
]:
    for m in [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]:
        CANDIDATES[f"devcurr_seed{seed}_{m}M"] = {
            "run_dir": RUN_BASE / run,
            "endpoint": f"chck_{m}M",
            "description": f"research devcurr first-pass seed{seed} chck_{m}M selected/available for corrected full eval.",
            "family": "qwen_devcurr_firstpass",
        }

base.TARGETS = CANDIDATES
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    # Preserve research CLI, but print a compact candidate map on --list-candidates.
    if "--list-candidates" in sys.argv:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(CANDIDATES.items())}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
