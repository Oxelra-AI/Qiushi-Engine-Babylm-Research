#!/usr/bin/env python3
"""research wrapper: CDI endpoint temperature/rank profile including clean preservation endpoints.

Extends the research CDI endpoint diagnostic with research clean preservation
checkpoints and uses research fitted temperatures when available. This is endpoint
CDI masked-token NLL/rank profiling, not official AoA scoring.
"""
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

import cdi_endpoint_temperature_rank_profile_five_models as base  # noqa: E402

base.DEFAULT_TEMP = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_clean/temperature_source_readout.json')
base.DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/cdi_endpoint_temperature_rank_profile_clean')
base.MODEL_PATHS.update({
    "clean_pres_lambda1_eval_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080'),
    "clean_pres_lambda1_train_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_train_full80/checkpoints/update_0080'),
    "pres_lambda1_trainmode_confounded": _public_path('experiments/archive/functional_learning/data/preservation_lambda1p0_seed62064/checkpoints/update_0080'),
})

if __name__ == "__main__":
    base.main()
