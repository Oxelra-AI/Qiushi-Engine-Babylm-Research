#!/usr/bin/env python3
"""Endpoint-aware full official-style evaluation for research/046 tail restart.

Targets are not hard-coded to chck_100M.  Each target points at an endpoint-
consistent evaluation root made by make_endpoint_consistent_ladder.py.
For chck_85M/chck_95M, SuperGLUE evaluates the explicit nonstandard endpoint
child.  For AoA, the root provides the required strict-small names, with later
standard names plateaued at the endpoint to avoid using future-trained weights.
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
EVALROOT_BASE = _public_path('experiments/archive/compact_experience/data/tail_endpoint_ladders')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_endpoint_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
VALID_ENDPOINTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]


def endpoint_to_m(endpoint: str) -> int:
    if not (endpoint.startswith("chck_") and endpoint.endswith("M")):
        raise ValueError(endpoint)
    return int(endpoint[len("chck_"):-1])


def eval_root_for(endpoint: str) -> pathlib.Path:
    return EVALROOT_BASE / f"tail_restart_seed43044_endpoint_{endpoint_to_m(endpoint)}M"


def target_name(endpoint: str) -> str:
    return f"tail_restart_seed43044_{endpoint_to_m(endpoint)}M"


def make_targets(endpoints: list[str]) -> Dict[str, Dict[str, Any]]:
    targets: Dict[str, Dict[str, Any]] = {}
    for endpoint in endpoints:
        run_dir = eval_root_for(endpoint)
        targets[target_name(endpoint)] = {
            "run_dir": run_dir,
            "endpoint": endpoint,
            "description": f"Endpoint-consistent full evaluation root for clean-Qwen seed43022 80M tail restart frozen at {endpoint}; later AoA-required standard names are plateaued at the endpoint rather than using future exposure.",
            "family": "endpoint_consistent_tail_restart",
        }
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--endpoint", choices=VALID_ENDPOINTS, default="chck_90M")
    ap.add_argument("--endpoints", nargs="*", choices=VALID_ENDPOINTS, default=None)
    known, rest = ap.parse_known_args()
    endpoints = known.endpoints or [known.endpoint]
    base.TARGETS = make_targets(endpoints)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    sys.argv = [sys.argv[0]] + rest
    if "--list-candidates" in rest:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(base.TARGETS.items())}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
