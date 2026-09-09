#!/usr/bin/env python3
"""Generic endpoint-aware full official-style evaluation wrapper.

This wrapper lets later steps full-evaluate a dynamically materialized endpoint
ladder (for example the seed43122 inverse/uniform replication) without editing the
large research evaluator's static TARGETS dictionary. It expects the run_dir to
contain hf_model/<endpoint> plus any AoA-required checkpoint names.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import pathlib
import sys
from typing import Any, Dict

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval')


def make_target(eval_target: str, run_dir: pathlib.Path, endpoint: str, family: str, description: str) -> Dict[str, Dict[str, Any]]:
    return {
        eval_target: {
            "run_dir": run_dir,
            "endpoint": endpoint,
            "description": description,
            "family": family,
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--eval_target", required=True, help="Dynamic target name to store in the per-target JSON.")
    ap.add_argument("--run_dir", required=True, help="Evaluation root containing hf_model/<endpoint>.")
    ap.add_argument("--endpoint", required=True, help="Checkpoint name to evaluate, e.g. chck_100M or chck_95M.")
    ap.add_argument("--family", default="custom_endpoint")
    ap.add_argument("--description", default="Dynamic endpoint-consistent BabyLM strict-small evaluation target.")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    known, rest = ap.parse_known_args()

    run_dir = pathlib.Path(known.run_dir)
    model_path = run_dir / "hf_model" / known.endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    base.OUT_ROOT = pathlib.Path(known.out_root)
    base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
    base.TARGETS = make_target(known.eval_target, run_dir, known.endpoint, known.family, known.description)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

    # The underlying evaluator requires --target and handles --gpu/--columns/--force.
    sys.argv = [sys.argv[0], "--target", known.eval_target] + rest
    if "--list-candidates" in rest:
        import json
        print(json.dumps({known.eval_target: {"run_dir": str(run_dir), "endpoint": known.endpoint, "model_path": str(model_path), "family": known.family}}, indent=2))
        return
    base.main()


if __name__ == "__main__":
    main()
