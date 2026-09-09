#!/usr/bin/env python3
"""research wrapper: full-row ordinary-WWM drift profile including ordinary control.

Adds the verified research ordinary `inherited_wwm` endpoint to the research
parent-function drift profile on ordinary 15% WWM Qwen-pair full-row renderings.
This tests whether ordinary continuation drifts from coherent86 on the same
rendering used by the clean preservation KL, and compares it with exact (M,S)
and clean endpoints.
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
})
if __name__ == "__main__":
    base.main()
