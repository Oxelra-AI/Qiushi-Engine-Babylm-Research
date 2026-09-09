#!/usr/bin/env python3
"""research: build a position-varied mirrored four-row family from the preferred high-confidence strict export.

This reuses the research four-row/assignment-position construction, but switches
from the original 120-map substrate to the research high-confidence strict
source/entity-disjoint map pool and adds frames for all seven relations.
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
spec = importlib.util.spec_from_file_location("builder", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.MAPS_PATH = _public_path('experiments/archive/functional_learning/data/highconfidence_strict_assignment_reversal_export/a01_highconf_strict_assignment_reversal_operation_maps.jsonl')
mod.OUT_DIR = _public_path('experiments/archive/relation_learning/data/highconf_mirrored_four_row_position_var')
mod.SEED = 92093

# Add relation-specific query frames.  Keep five train-seen frames and three eval-unseen
# frames so this pool can serve the same local-transfer role as the 120-map pilot.
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
    "located_in": [
        ("f00_li", "train_seen", "The place where {entity} is located is ", "."),
        ("f01_li", "train_seen", "{entity} is located in ", "."),
        ("f02_li", "train_seen", "The location of {entity} is ", "."),
        ("f03_li", "train_seen", "Where {entity} is located is ", "."),
        ("f04_li", "train_seen", "The area containing {entity} is ", "."),
        ("f05_li", "eval_unseen", "According to records, {entity} is located in ", "."),
        ("f06_li", "eval_unseen", "{entity}'s location is ", "."),
        ("f07_li", "eval_unseen", "The recorded place for {entity} is ", "."),
    ],
    "nationality": [
        ("f00_nat", "train_seen", "The nationality of {entity} is ", "."),
        ("f01_nat", "train_seen", "{entity} is ", "."),
        ("f02_nat", "train_seen", "The national background of {entity} is ", "."),
        ("f03_nat", "train_seen", "What nationality {entity} has is ", "."),
        ("f04_nat", "train_seen", "The recorded nationality for {entity} is ", "."),
        ("f05_nat", "eval_unseen", "According to records, {entity} is ", "."),
        ("f06_nat", "eval_unseen", "{entity}'s nationality is ", "."),
        ("f07_nat", "eval_unseen", "The nationality listed for {entity} is ", "."),
    ],
    "occupation": [
        ("f00_occ", "train_seen", "The occupation of {entity} is ", "."),
        ("f01_occ", "train_seen", "{entity} works as a ", "."),
        ("f02_occ", "train_seen", "The profession of {entity} is ", "."),
        ("f03_occ", "train_seen", "What {entity} does professionally is ", "."),
        ("f04_occ", "train_seen", "The recorded job of {entity} is ", "."),
        ("f05_occ", "eval_unseen", "According to records, {entity} works as a ", "."),
        ("f06_occ", "eval_unseen", "{entity}'s profession is ", "."),
        ("f07_occ", "eval_unseen", "The listed occupation for {entity} is ", "."),
    ],
})

if __name__ == "__main__":
    mod.main()
