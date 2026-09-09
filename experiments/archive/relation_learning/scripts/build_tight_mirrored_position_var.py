#!/usr/bin/env python3
"""research: build position-varied four-row family from the final tight relation-fact export.

This reuses the research mirrored construction but points it at the final tight
semantic export.  The resulting family is a mechanism substrate for testing
whether the research failure was limited by the number/diversity of instances.
The update sentences are controlled counterfactual augmentations, so this output
is not treated as a BabyLM submission stream.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
from pathlib import Path

ROOT = _public_path('.')
BASE_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/build_mirrored_four_row_family_position_var.py')
spec = importlib.util.spec_from_file_location("position_builder", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.MAPS_PATH = _public_path('experiments/archive/functional_learning/data/tight_semantic_assignment_reversal_export/a01_tight_semantic_assignment_reversal_operation_maps.jsonl')
mod.OUT_DIR = _public_path('experiments/archive/relation_learning/data/tight_mirrored_four_row_position_var')
mod.SEED = 95096

# research already supplies death_place, birthplace and founded_year frames.  The
# final tight export also contains birth_year, for which the answer type is a
# four-digit year.
mod.FRAMES.update({
    "birth_year": [
        ("f00_by", "train_seen", "The year when {entity} was born is ", "."),
        ("f01_by", "train_seen", "{entity} was born in ", "."),
        ("f02_by", "train_seen", "The birth year of {entity} is ", "."),
        ("f03_by", "train_seen", "When {entity} was born is ", "."),
        ("f04_by", "train_seen", "The year of {entity}'s birth is ", "."),
        ("f05_by", "eval_unseen", "According to records, {entity} was born in ", "."),
        ("f06_by", "eval_unseen", "{entity}'s birth year is ", "."),
        ("f07_by", "eval_unseen", "The recorded year of birth for {entity} is ", "."),
    ],
})

if __name__ == "__main__":
    mod.main()
