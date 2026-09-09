#!/usr/bin/env python3
"""research: official Entity evaluation for seed43122 split DeBERTa arms.

This is the second-seed behavioral-locality readout requested after the compact
T/U split result replicated.  It reuses the official BabyLM Entity evaluator
wrapper from research, but points to the seed43122 REPEAT_SPLIT and VIEW_SPLIT
checkpoints.  It performs no upload and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/eval_split_entity_seed43122.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import eval_split_entity_official as base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
RUNS = WS / "training/runs"

base.OUT = WS / "data/split_entity_official_seed43122"
base.ARM_CONFIGS = {
    "D_RS_43122": {
        "run_dir": RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122",
        "role": "RS",
        "seed": 43122,
        "description": "REPEAT_SPLIT seed43122",
    },
    "D_VS_43122": {
        "run_dir": RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122",
        "role": "VS",
        "seed": 43122,
        "description": "VIEW_SPLIT seed43122",
    },
}

if __name__ == "__main__":
    base.main()
