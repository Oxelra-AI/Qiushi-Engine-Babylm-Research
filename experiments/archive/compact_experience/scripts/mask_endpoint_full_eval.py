#!/usr/bin/env python3
"""Endpoint-aware full official-style evaluation for a research masking arm."""
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
EVALROOT_BASE = _public_path('experiments/archive/compact_experience/data/mask_endpoint_ladders')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
VALID_ENDPOINTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]
VALID_ARMS = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]


def endpoint_to_m(endpoint: str) -> int:
    if not (endpoint.startswith("chck_") and endpoint.endswith("M")):
        raise ValueError(endpoint)
    return int(endpoint[len("chck_"):-1])


def target_name(arm: str, endpoint: str) -> str:
    return f"mask_{arm}_{endpoint_to_m(endpoint)}M"


def eval_root_for(arm: str, endpoint: str) -> pathlib.Path:
    return EVALROOT_BASE / f"mask_{arm}_endpoint_{endpoint_to_m(endpoint)}M"


def make_targets(arm: str, endpoint: str) -> Dict[str, Dict[str, Any]]:
    return {
        target_name(arm, endpoint): {
            "run_dir": eval_root_for(arm, endpoint),
            "endpoint": endpoint,
            "description": f"Endpoint-consistent full evaluation root for research {arm} masking continuation frozen at {endpoint}; parent ladder is attached and later AoA-required names are plateaued at the endpoint.",
            "family": "endpoint_consistent_masking_continuation",
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm", choices=VALID_ARMS, required=True)
    ap.add_argument("--endpoint", choices=VALID_ENDPOINTS, required=True)
    known, rest = ap.parse_known_args()
    base.TARGETS = make_targets(known.arm, known.endpoint)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    sys.argv = [sys.argv[0]] + rest
    if "--list-candidates" in rest:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(base.TARGETS.items())}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
