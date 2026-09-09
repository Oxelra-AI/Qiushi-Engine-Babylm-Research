#!/usr/bin/env python3
"""research FW shared-anchor official-compatible evaluator wrapper.

This wrapper defines the three relevant trained endpoints for the compact-vs-breadth
mechanism-scale comparison and reuses the robust COMPACT_EXPERIENCE full evaluator. It is used
only after a model checkpoint exists. Current-pristine EWoK and min-context-zero
AoA are intentionally handled by the research posttrain controller, not by asking
this wrapper to run the stale/default EWoK or embedded AoA path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

A01_WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WORKSPACE = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
PRISTINE_STRICT = A01_WORKSPACE / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
GLOBALPIQA_GENERATED = A01_WORKSPACE / "data" / "globalpiqa_official_lineage" / "official_dl_scratch" / "generated_by_current_official_dl" / "evaluation_data" / "full_eval"
OUT_ROOT = A01_WORKSPACE / "data" / "fw_shared_anchor_full_eval"

TARGET_REGISTRY = {
    "fw_compact_fullbatch_seed43022": {
        "run_dir": A02_WORKSPACE / "training" / "runs" / "fw_compact_view_shared16k_seed43022",
        "endpoint": "chck_100M",
        "description": (
            "A02 full-batch compact-view anchor: A01 research-104 compact 100M stream "
            "(c8d7f24b...), shared legal 16k tokenizer (e70d167f...), DeBERTa-v2 8x480, "
            "fixed WWM 0.15, AdamW, seeds 43/43022/43023. Used by A01 research as the "
            "common compact anchor to avoid a duplicate compact training run."
        ),
        "family": "fw_shared16k_fullbatch_compact_anchor",
    },
    "fw_breadth_rowblock_fullbatch_seed43022": {
        "run_dir": A02_WORKSPACE / "training" / "runs" / "fw_source_breadth_shared16k_seed43022",
        "endpoint": "chck_100M",
        "description": (
            "A02 full-batch row-block whole-sentence source-breadth arm: same shared filler, "
            "common FineWeb source words, tokenizer, recipe, and seeds as the compact anchor, "
            "with independent FineWeb companion sentences in the research row-block layout."
        ),
        "family": "fw_shared16k_fullbatch_rowblock_breadth",
    },
    "fw_breadth_interleaved_fullbatch_seed43022": {
        "run_dir": A01_WORKSPACE / "training" / "runs" / "fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022",
        "endpoint": "chck_100M",
        "description": (
            "A01 full-batch interleaved whole-sentence source-breadth arm: same independent "
            "FineWeb companion sentences as the row-block breadth arm but placed between the "
            "same common source segments to better match compact source/rewrite alternation. "
            "Trained on the A02 full-batch coordinate so the A02 compact model is a valid anchor."
        ),
        "family": "fw_shared16k_fullbatch_interleaved_breadth",
    },
}


def configure(target: str, endpoint: str | None = None) -> None:
    spec = dict(TARGET_REGISTRY[target])
    if endpoint:
        spec["endpoint"] = endpoint
    model_path = spec["run_dir"] / "hf_model" / spec["endpoint"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    base.STRICT = PRISTINE_STRICT
    for task in base.ZERO_SHOT_TASKS:
        if task["column"] == "GlobalPIQA_parallel":
            task["data_path"] = str((GLOBALPIQA_GENERATED / "global_piqa_parallel").resolve())
        elif task["column"] == "GlobalPIQA_nonparallel":
            task["data_path"] = str((GLOBALPIQA_GENERATED / "global_piqa_nonparallel").resolve())

    base.OUT_ROOT = OUT_ROOT
    base.PER_TARGET_DIR = OUT_ROOT / "per_target"
    base.TARGETS = {target: spec}
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGET_REGISTRY), required=True)
    ap.add_argument("--endpoint", default="chck_100M")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--columns", nargs="*", default=None,
        help="Subset among BLiMP, Supplement, Entity, COMPS, GlobalPIQA_parallel, GlobalPIQA_nonparallel, Reading, SuperGLUE. Do not use this wrapper for EWoK/AoA in the final pristine coordinate.",
    )
    args = ap.parse_args()

    if args.columns and any(c in {"EWoK", "AoA"} for c in args.columns):
        raise SystemExit("Use the research posttrain controller for pristine EWoK and min-context-zero AoA; this wrapper should not run those columns directly.")
    configure(args.target, args.endpoint)

    argv = [sys.argv[0], "--target", args.target, "--gpu", str(args.gpu)]
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
