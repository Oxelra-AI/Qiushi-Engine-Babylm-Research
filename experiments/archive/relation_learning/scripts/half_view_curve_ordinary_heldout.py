#!/usr/bin/env python3
"""research: ordinary-heldout MLM loss for the C/R/V/HM/HV curve family.

This broad-fit readout is the companion to `score_half_view_curve.py`.
It helps distinguish changes in source-specific A_T=N-T from broad movement of
T/U/N target fit under the zero-dose, half-dose, full-dose, hash-mixed, and full
exact-recurrence arms.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/half_view_curve_ordinary_heldout.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive/frontier_consolidation"
sys.path.insert(0, str(WS / "scripts"))
import ordinary_heldout_price_probe as base  # noqa: E402

base.OUT = WS / "data/half_view_curve_ordinary_heldout"
base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/half_view_curve_ordinary_heldout.md')
base.ARM_CONFIGS = {
    "C": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "R": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "V": REPRESENTATION_FRONTIER_STUDIES / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "HM": WS / "training/runs/full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "HV": WS / "training/runs/full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022",
}
base.ROLE = {k: k for k in base.ARM_CONFIGS}
base.CONTRASTS = [
    ("R", "C"), ("V", "C"), ("HV", "C"), ("HM", "C"),
    ("HV", "V"), ("V", "HV"), ("HM", "HV"), ("HV", "HM"),
    ("HM", "V"), ("HM", "R"),
]

if __name__ == "__main__":
    base.main()
