#!/usr/bin/env python3
"""research: score the 1.82x DeBERTa VIEW/REPEAT pair with the held-out relation probes.

This is a lightweight wrapper around the research held-out copy/rewrite/Entity-cue
probe scorer. It adds the existing frontier_consolidation intermediate-dose DeBERTa arms and
leaves the probe construction unchanged, so the late V-R direction can be compared
with the 2.64x MAX relation signature while the split-control trainings run.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_dose1p82_deberta_probes.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

RUNS = _public_path('experiments/archive/frontier_consolidation/training/runs')
base.ARM_CONFIGS.clear()
base.ARM_CONFIGS.update({
    "D182_V_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose1p82x_matched_rowholdout_deberta100M_seed43022'),
    "D182_R_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose1p82x_matched_rowholdout_deberta100M_seed43022'),
})
base.OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/dose1p82_deberta_probes')

if __name__ == "__main__":
    base.main()
