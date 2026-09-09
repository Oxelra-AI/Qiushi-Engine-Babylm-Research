#!/usr/bin/env python3
"""Full official-style evaluation wrapper for one research masking arm."""
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
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/mask_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
VALID = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]


def make_targets(arm: str) -> Dict[str, Dict[str, Any]]:
    return {
        f"mask_{arm}_100M": {
            "run_dir": RUN_BASE / f"mask_{arm}",
            "endpoint": "chck_100M",
            "description": f"research {arm} masking continuation from clean-Qwen chck_80M with parent AoA ladder attached.",
            "family": "masking_continuation_with_parent_ladder",
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", choices=VALID, default="uniform_control")
    known, rest = ap.parse_known_args()
    base.TARGETS = make_targets(known.arm)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    sys.argv = [sys.argv[0]] + rest
    if "--list-candidates" in rest:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(base.TARGETS.items())}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
