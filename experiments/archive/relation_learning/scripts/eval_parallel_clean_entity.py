#!/usr/bin/env python3
"""research: evaluate seed43222 CLEAN using the completed parallel CLEAN run.

This imports the research Entity evaluator but redirects D_C_43222 to the
research completed parallel CLEAN directory. It writes to a separate research
output directory so the baseline provenance is explicit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/eval_parallel_clean_entity.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
import eval_seed43222_entity as base  # noqa: E402

WS = _public_path('experiments/archive/relation_learning')
RUNS = _public_path('experiments/archive/relation_learning/training/runs')
base.OUT = _public_path('experiments/archive/relation_learning/data/seed43222_parallel_clean_entity_eval')
base.ARM_CONFIGS['D_C_43222']['run_dir'] = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel')
base.ARM_CONFIGS['D_C_43222']['description'] = 'DeBERTa MAX CLEAN seed43222 completed in research parallel duplicate run, same recipe as research CLEAN'

if __name__ == '__main__':
    # Reuse the imported CLI; user should pass --arms D_C_43222.
    base.main()
