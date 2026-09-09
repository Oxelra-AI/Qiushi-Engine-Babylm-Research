#!/usr/bin/env python3
"""research: replicate the COMPACT_EXPERIENCE ALN-vs-OFF crossing at seed43122.

This wrapper reuses the validated research scorer but patches the arm map to the
second-seed COMPACT_EXPERIENCE aligned-Qwen and official-only DeBERTa coordinates.  It scores
only ALN and OFF because SHUF/DUP/SEP are not available at seed43122.  The
purpose is to protect the newly important crossing:
  - local aligned Qwen restatement strongly affects the near-register
    WikiLarge/SimpleWiki source-recurring readout;
  - the same first-tranche Qwen block is near-null on the later FineWeb-compact
    nonoverlap readout.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/replicate_paired_context_aln_off_seed43122.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
sys.path.insert(0, str(WS / "scripts"))
import score_paired_context_relation_design as base  # noqa: E402

base.DEFAULT_OUT = WS / "data/paired_context_aln_off_seed43122_probe"
base.DEFAULT_NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/paired_context_aln_off_seed43122_probe.md')
base.ARMS = {
    "OFF": {
        "role": "OFF",
        "description": "seed43122 official_lengthmatched no qwen-pair practice",
        "run": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43122",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/official_lengthmatched_seed43122.json",
    },
    "ALN": {
        "role": "ALN",
        "description": "seed43122 qwen_clean_aligned local original+own Qwen rewrite",
        "run": COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43122",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/qwen_clean_aligned_seed43122.json",
    },
}
base.CONTRASTS = [("ALNminusOFF", "ALN", "OFF")]

if __name__ == "__main__":
    base.main()
