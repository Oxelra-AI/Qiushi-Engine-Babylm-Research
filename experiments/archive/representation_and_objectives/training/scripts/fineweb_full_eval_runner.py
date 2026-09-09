#!/usr/bin/env python3
"""Full official-style evaluator wrapper for selected repaired FineWeb endpoints.

It reuses the robust COMPACT_EXPERIENCE research evaluator, but injects target definitions for
representation_and_objectives's repaired FineWeb seqsafe96 treatment/control runs and writes into a
research output root.  Intended use is after no-AoA endpoint selection and prefill,
running only SuperGLUE and AoA unless a full recomputation is explicitly requested.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

COMPACT_EXPERIENCE_SCRIPTS = pathlib.Path("experiments/archive/compact_experience/scripts")
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS.resolve()))
import full_overall_eval_runner as base  # noqa: E402

OUT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval")
RUNS = pathlib.Path("experiments/archive/representation_and_objectives/training/runs")
RUN_DIRS = {
    "fineweb_seqsafe96_treatment_repairseq": RUNS / "cleanqwen_fineweb_seqsafe96_repairseq_8x480_16k_wwm_seed43022",
    "fineweb_seqsafe96_control_repairseq": RUNS / "cleanqwen_official_seqsafe96_control_repairseq_8x480_16k_wwm_seed43022",
}


def configure(source_target: str, endpoint: str, eval_target: str, description: str) -> None:
    run_dir = RUN_DIRS[source_target]
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
            "family": "REPRESENTATION_FRONTIER_STUDIES_fineweb_seqsafe96_source_breadth_repairseq",
            "source_target": source_target,
        }
    }
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-target", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--eval-target", default="", help="target name in per_target JSON; default source-target__endpoint")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--columns", nargs="*", default=None, help="Subset among zero-shot columns, Reading, SuperGLUE, AoA. Default: all official columns.")
    ap.add_argument("--description", default="FineWeb seqsafe96 repaired source-breadth selected endpoint full official-style evaluation.")
    args = ap.parse_args()
    eval_target = args.eval_target or f"{args.source_target}__{args.endpoint}"
    configure(args.source_target, args.endpoint, eval_target, args.description)
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
