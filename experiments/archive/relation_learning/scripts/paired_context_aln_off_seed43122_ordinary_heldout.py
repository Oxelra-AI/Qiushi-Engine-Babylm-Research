#!/usr/bin/env python3
"""research: ordinary-heldout MLM loss for COMPACT_EXPERIENCE ALN vs OFF at seed43122.

This is the broad-fit companion to the seed43122 forward-pass replication of the
ALN-vs-OFF reach crossing.  It reuses the research deterministic-mask heldout
scorer and patches the arm map to the second-seed COMPACT_EXPERIENCE official-only and
aligned-Qwen models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/paired_context_aln_off_seed43122_ordinary_heldout.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
sys.path.insert(0, str(WS / "scripts"))
import ordinary_heldout_price_probe as base  # noqa: E402

base.OUT = WS / "data/paired_context_aln_off_seed43122_ordinary_heldout"
base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/paired_context_aln_off_seed43122_ordinary_heldout.md')
base.ARM_CONFIGS = {
    "OFF": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122",
    "ALN": COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43122",
}
base.ROLE = {k: k for k in base.ARM_CONFIGS}
base.CONTRASTS = [("ALN", "OFF")]

if __name__ == "__main__":
    base.main()
