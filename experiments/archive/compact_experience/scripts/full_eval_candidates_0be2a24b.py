#!/usr/bin/env python3
"""research corrected full nine-column evaluator for bidirectional and contextual candidates.

Targets must be frozen by no-AoA trajectory screens before this wrapper is used.  This
script delegates to full_overall_eval_runner.py, which uses the corrected
BabyLM official-style scoring helper where AoA enters in leaderboard units.  AoA is
measured only inside this full evaluation, not as a training, checkpoint, or schedule
selection signal.
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
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/full_eval_candidates')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"

CKPTS = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]

CANDIDATES: Dict[str, Dict[str, Any]] = {}

for seed, run in [
    ("43022", "qwen_bidirectional_pair_order_16k_seed43022"),
    ("43122", "qwen_bidirectional_pair_order_16k_seed43122"),
]:
    for m in CKPTS:
        CANDIDATES[f"bidir_seed{seed}_{m}M"] = {
            "run_dir": RUN_BASE / run,
            "endpoint": f"chck_{m}M",
            "description": (
                f"research bidirectional same-window pair-order seed{seed} chck_{m}M. "
                "Candidate should be selected from research no-AoA trajectory ranking; "
                "full eval here adds SuperGLUE and AoA as final measurement."
            ),
            "family": "qwen_bidirectional_pair_order",
        }

for m in CKPTS:
    CANDIDATES[f"context_cap120_treat_seed43022_{m}M"] = {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_context_onepair_cap120_16k_seed43022'),
        "endpoint": f"chck_{m}M",
        "description": (
            f"research contextual one-pair cap-120 treatment seed43022 chck_{m}M. "
            "Candidate should be selected from no-AoA treatment/control trajectory comparison."
        ),
        "family": "qwen_contextual_onepair_cap120",
    }
    CANDIDATES[f"context_cap120_control_seed43022_{m}M"] = {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_context_lengthmatched_cap120_16k_seed43022'),
        "endpoint": f"chck_{m}M",
        "description": (
            f"research length-sequence-matched official contextual cap-120 control seed43022 chck_{m}M. "
            "Evaluated with the treatment checkpoint chosen from no-AoA trajectory comparison."
        ),
        "family": "official_context_lengthmatched_cap120",
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
