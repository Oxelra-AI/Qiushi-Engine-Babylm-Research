#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for the
legal-40k compact_view_reinvest seed43122 run.

Same as the seed43022 wrapper, pointed at the seed43122 legal 40k run. EWoK and
AoA are recomputed separately by the research controller on the current official
coordinate before hardened pristine collation.
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

base.OUT_ROOT = A01_WORKSPACE / "data" / "legal40k_accum_seed43122_full_eval"
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "legal40k_reinvest_seed43122": {
        "run_dir": A01_RUN_BASE / "legal40k_accum_compact_view_reinvest_seed43122",
        "endpoint": "chck_100M",
        "description": (
            "Compact_view_reinvest seed43122 retrained with a legal 40k byte-BPE "
            "tokenizer learned only from the same compliant 10M pool. The initial "
            "batch-256 launch OOMed, so this active run uses microbatch-64 gradient "
            "accumulation to retain effective batch 256 and the same LR/update schedule. "
            "It tests the same legal representation package as seed43022 with RNG seeds "
            "43122/43123; all data, architecture, optimizer, and schedule factors held fixed."
        ),
        "family": "representation_and_objectives_legal40k_reinvest_seed43122",
    }
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
