#!/usr/bin/env python3
"""research: score seed43222 parallel CLEAN on the research held-out probes.

Imports the dynamic research probe scorer, redirects D_C_43222 to the completed
research parallel CLEAN directory, and writes a separate research output directory.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_parallel_clean_probes.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

WS = _public_path('experiments/archive/relation_learning')
base.ARM_CONFIGS['D_C_43222'] = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel')
base.OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_probes')

if __name__ == '__main__':
    base.main()
