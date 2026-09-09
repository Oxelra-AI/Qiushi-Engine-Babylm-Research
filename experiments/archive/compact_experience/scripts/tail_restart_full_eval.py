#!/usr/bin/env python3
"""Full official-style evaluation wrapper for research clean-tail restart ladder."""
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
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_restart_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"

TARGETS: Dict[str, Dict[str, Any]] = {
    "clean_tail_restart_ladder_100M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/clean_tail_restart_ladder_seed43044'),
        "endpoint": "chck_100M",
        "description": "Clean-Qwen seed43022 true-parent 80M -> 100M untouched optimizer/LR restart with full AoA ladder symlinked from parent.",
        "family": "clean_parent_tail_restart",
    },
}
base.TARGETS = TARGETS
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    if "--list-candidates" in sys.argv:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(TARGETS.items())}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
