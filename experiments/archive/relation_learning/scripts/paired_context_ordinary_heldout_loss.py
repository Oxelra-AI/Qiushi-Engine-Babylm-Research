#!/usr/bin/env python3
"""research: ordinary-heldout MLM loss for COMPACT_EXPERIENCE qwen relation design arms.

This wrapper reuses research's deterministic-mask heldout scorer but patches the
arm map to the five COMPACT_EXPERIENCE seed43022 relation-design coordinates.  Its purpose is
to separate relation-specific T/U/N effects from broad held-out language fit and
to bound the row-count mismatch of the separated arm.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/paired_context_ordinary_heldout_loss.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
sys.path.insert(0, str(WS / "scripts"))
import ordinary_heldout_price_probe as base  # noqa: E402

base.OUT = WS / "data/paired_context_ordinary_heldout_loss"
base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/paired_context_ordinary_heldout_loss.md')
base.ARM_CONFIGS = {
    "OFF": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43022",
    "DUP": COMPACT_EXPERIENCE / "training/runs/selected_original_dup_all_16k_seed43022",
    "SHUF": COMPACT_EXPERIENCE / "training/runs/qwen_shuffled_control_16k_seed43022",
    "SEP": COMPACT_EXPERIENCE / "training/runs/qwen_separated_pair_16k_seed43022",
    "ALN": COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022",
}
base.ROLE = {k: k for k in base.ARM_CONFIGS}
base.CONTRASTS = [
    ("ALN", "OFF"), ("ALN", "SEP"), ("ALN", "SHUF"),
    ("SHUF", "OFF"), ("SEP", "OFF"), ("DUP", "OFF"),
    ("ALN", "DUP"), ("DUP", "ALN"), ("SHUF", "SEP"), ("SEP", "SHUF"),
]

if __name__ == "__main__":
    base.main()
