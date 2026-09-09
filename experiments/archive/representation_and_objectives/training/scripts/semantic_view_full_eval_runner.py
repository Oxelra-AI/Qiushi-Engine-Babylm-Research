#!/usr/bin/env python3
"""Full official-style evaluator for selected REPRESENTATION_FRONTIER_STUDIES semantic-view endpoints.

This wrapper reuses COMPACT_EXPERIENCE's robust research evaluator but defines dynamic targets
inside representation_and_objectives.  It can run all columns, or only SuperGLUE/AoA after the
zero-shot/Reading columns have been prefilled from the matched no-AoA trajectory.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

COMPACT_EXPERIENCE_SCRIPTS = pathlib.Path("experiments/archive/compact_experience/scripts")
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS.resolve()))
import full_overall_eval_runner as base  # noqa: E402

OUT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/semantic_view_full_eval")
RUNS = pathlib.Path("experiments/archive/representation_and_objectives/training/runs")
RUN_DIRS = {
    "semantic_view_treatment": RUNS / "semantic_view_treatment_8x480_16k_wwm_seed43022",
    "original_packet_local": RUNS / "original_packet_local_8x480_16k_wwm_seed43022",
}


def configure(target: str, endpoint: str, eval_target: str, description: str) -> None:
    run_dir = RUN_DIRS[target]
    model_root = run_dir / "hf_model"
    model_path = model_root / endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    base.OUT_ROOT = OUT_ROOT
    base.PER_TARGET_DIR = OUT_ROOT / "per_target"
    base.TARGETS = {
        eval_target: {
            "run_dir": run_dir,
            "endpoint": endpoint,
            "description": description,
            "family": "REPRESENTATION_FRONTIER_STUDIES_semantic_view_packet_local_contrast",
        }
    }
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--eval-target", default="", help="target name in per_target JSON; default target__endpoint")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--columns", nargs="*", default=None, help="Subset among zero-shot columns, Reading, SuperGLUE, AoA. Default: all official columns.")
    ap.add_argument("--description", default="Semantic-view contrast selected endpoint full official-style evaluation.")
    args = ap.parse_args()
    eval_target = args.eval_target or f"{args.target}__{args.endpoint}"
    configure(args.target, args.endpoint, eval_target, args.description)
    argv = [sys.argv[0], "--target", eval_target, "--gpu", str(args.gpu)]
    if args.force:
        argv.append("--force")
    if args.columns:
        argv.append("--columns")
        argv.extend(args.columns)
    old_argv = sys.argv
    try:
        sys.argv = argv
        base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
