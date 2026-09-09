#!/usr/bin/env python3
"""research full official-style evaluation wrapper.

Reuses the robust research evaluator but points it at the clean research matched
arms:
  * official_lengthmatched: official-only control with the same row-length
    sequence as the treatment corpus
  * qwen_clean_aligned: complete-pair Qwen-aligned treatment corpus
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
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = {
    "official_lengthmatched": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research official-only control chunked to the same row-length sequence as the clean Qwen treatment; DeBERTa-v2 8x480, baseline16k, WWM, seed43022.",
        "family": "debertav2_8x480_16k_step028_clean_matched",
    },
    "qwen_clean_aligned": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research clean local-Qwen-aligned corpus with complete original--rewrite pair boundaries plus official filler; compliance remains to be verified against final 2026 rules before any submission; DeBERTa-v2 8x480, baseline16k, WWM, seed43022.",
        "family": "debertav2_8x480_16k_step028_clean_matched",
    },
    "qwen_shuffled_control": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_shuffled_control_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research follow-up control: same selected originals and Qwen rewrite multiset as the clean treatment, but rewrites are shuffled within source/length bins to break pair correspondence.",
        "family": "debertav2_8x480_16k_step028_clean_matched",
    },
    "official_sourcematched": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_sourcematched_16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "research follow-up control: official-only text matching the clean-Qwen treatment row-length sequence and effective source word counts; separates source/domain reweighting from generated-pair effects.",
        "family": "debertav2_8x480_16k_step028_clean_matched",
    },
      "official_original_dup": {
          "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_original_dup_16k_seed43022'),
          "endpoint": "chck_100M",
          "description": "research follow-up control: official-only original+original rows based on selected research originals, testing within-row redundancy against original+Qwen-rewrite rows.",
          "family": "debertav2_8x480_16k_step028_clean_matched",
      },
      "official_lengthmatched_seed43122": {
          "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43122'),
          "endpoint": "chck_100M",
          "description": "research second-seed official-only row-length-matched control; DeBERTa-v2 8x480, baseline16k, WWM, extra_init_seed=43122, train_rng_seed=43123.",
          "family": "debertav2_8x480_16k_step029_second_seed",
      },
      "qwen_clean_aligned_seed43122": {
          "run_dir": _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122'),
          "endpoint": "chck_100M",
          "description": "research second-seed clean local-Qwen-aligned treatment using the same research clean corpus; DeBERTa-v2 8x480, baseline16k, WWM, extra_init_seed=43122, train_rng_seed=43123.",
          "family": "debertav2_8x480_16k_step029_second_seed",
      },
  }
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}

if __name__ == "__main__":
    base.main()
