#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for the
legal40k compact_view_reinvest true 12x384 depth seed43022 run.

This wrapper reuses the same robust COMPACT_EXPERIENCE full evaluator and pristine official
coordinate used for research legal40k evaluation.  EWoK and AoA are intentionally
computed by the research post-training controller on the current 7,618-row EWoK
and min_context=0 AoA coordinate, then staged by the research-hardened collator.

Scientific interpretation: relative to the completed research legal40k 8x480
fixed-WWM endpoint, this route changes the DeBERTa-v2 geometry only
(8x480/FFN1920 -> 12x384/FFN1280, expected 38,421,952 params), preserving the
legal40k tokenizer, compact_view_reinvest stream, AdamW/cosine schedule,
fixed WWM 0.15 objective, sequence length 256, effective batch 256, data order,
and seed identity.
"""
from __future__ import annotations

import pathlib
import sys

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

A01_WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A01_RUN_BASE = A01_WORKSPACE / "training" / "runs"
PRISTINE_STRICT = A01_WORKSPACE / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
GLOBALPIQA_GENERATED = A01_WORKSPACE / "data" / "globalpiqa_official_lineage" / "official_dl_scratch" / "generated_by_current_official_dl" / "evaluation_data" / "full_eval"

base.STRICT = PRISTINE_STRICT
for _task in base.ZERO_SHOT_TASKS:
    if _task["column"] == "GlobalPIQA_parallel":
        _task["data_path"] = str((GLOBALPIQA_GENERATED / "global_piqa_parallel").resolve())
    elif _task["column"] == "GlobalPIQA_nonparallel":
        _task["data_path"] = str((GLOBALPIQA_GENERATED / "global_piqa_nonparallel").resolve())

base.OUT_ROOT = A01_WORKSPACE / "data" / "legal40k_12x384_depth_seed43022_full_eval"
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "legal40k_12x384_depth_seed43022": {
        "run_dir": A01_RUN_BASE / "legal40k_12x384_depth_compact_view_reinvest_seed43022",
        "endpoint": "chck_100M",
        "description": (
            "Compact_view_reinvest seed43022 retrained with the same legal 40k "
            "byte-BPE tokenizer and 100M stream as research, but using true "
            "leader-style 12-layer hidden-384 DeBERTa-v2 geometry with "
            "intermediate_size=1280.  This isolates depth-over-width relative to "
            "the completed legal40k 8x480 fixed-WWM run while preserving data, "
            "tokenizer, AdamW schedule, fixed WWM, sequence length, effective "
            "batch, and seed identity."
        ),
        "family": "representation_and_objectives_legal40k_12x384_depth_seed43022",
    }
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
