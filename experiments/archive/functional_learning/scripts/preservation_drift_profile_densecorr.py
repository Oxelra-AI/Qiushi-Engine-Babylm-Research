#!/usr/bin/env python3
"""research wrapper: ordinary-WWM parent-function drift including dense-corruption preservation.

Scores the delivered dense-corruption preservation endpoint on the same ordinary
full-row WWM rendering used to interpret clean preservation. This asks whether
dense-corruption KL preserved ordinary inherited function, suppressed acquisition,
or changed the retention/acquisition balance relative to exact (M,S) and clean.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import pathlib, sys
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import preservation_drift_profile as base  # noqa: E402
base.MODEL_SPECS.update({
    "ordinary_inherited_wwm_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080'),
    "clean_pres_lambda1_eval_seed62064": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080'),
    "clean_pres_lambda1_eval_seed62065": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_seed62065_full80/checkpoints/update_0080'),
    "densecorr_pres_lambda1_seed62064": _public_path('experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80/checkpoints/update_0080'),
})
if __name__ == "__main__":
    base.main()
