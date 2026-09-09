#!/usr/bin/env python3
"""research wrapper: full-row preservation drift profile including clean endpoints."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preservation_drift_profile as base  # noqa: E402

base.MODEL_SPECS.update({
    "clean_pres_lambda1_eval_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080'),
    "clean_pres_lambda1_train_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_train_full80/checkpoints/update_0080'),
})

if __name__ == "__main__":
    base.main()
