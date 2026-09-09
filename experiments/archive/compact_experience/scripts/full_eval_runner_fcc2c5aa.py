#!/usr/bin/env python3
"""research official-style full evaluation wrapper for new clean-Qwen mechanism controls.

Targets:
  * selected_original_dup_all: all research selected official originals duplicated once,
    no Qwen words, source totals matched to the clean-Qwen effective source mix.
  * qwen_separated_pair: same selected originals and Qwen rewrite multiset as the
    aligned treatment, but originals and rewrites are in separate rows/windows.

The evaluator reuses the research official-style runner and writes outputs under
data/mechanism_eval/.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/mechanism_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "selected_original_dup_all": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/selected_original_dup_all_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research mechanism control: all research selected official originals duplicated once; no Qwen words; source totals matched to the clean-Qwen effective source mixture; DeBERTa-v2 8x480 baseline16k WWM seed43022.",
        "family": "debertav2_8x480_16k_step032_mechanism",
    },
    "qwen_separated_pair": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_separated_pair_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research mechanism control: same selected originals and Qwen rewrites as clean aligned treatment, but original/rewrite sides are placed in separate rows/windows; source totals matched; DeBERTa-v2 8x480 baseline16k WWM seed43022.",
        "family": "debertav2_8x480_16k_step032_mechanism",
    },
}
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
