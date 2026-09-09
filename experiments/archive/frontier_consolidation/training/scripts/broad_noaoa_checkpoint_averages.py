#!/usr/bin/env python3
"""research wrapper: broad no-AoA screen for newly created checkpoint averages.

Imports the research late-checkpoint evaluator and adds averaged-model targets.
Averages are arithmetic combinations of existing compact_view_reinvest checkpoints,
not additional training.  Use this only after the selected-checkpoint broad screen
shows a plausible measured tradeoff worth testing.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "experiments").exists():
    here = _public_path('experiments/archive/frontier_consolidation/training/scripts/broad_noaoa_checkpoint_averages.py')
    for parent in [here] + list(here.parents):
        if (parent / "experiments").exists():
            ROOT = parent
            break

BASE_PATH = ROOT / "experiments/archive/frontier_consolidation/training/scripts/broad_noaoa_late_checkpoints.py"
spec = importlib.util.spec_from_file_location("broad_noaoa_late_checkpoints", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.TARGETS.update({
    "avg43022_90_100": {
        "family": "compact_view_reinvest_checkpoint_average",
        "seed": "43022",
        "exposure_m": "avg_90_100",
        "model_path": ROOT / "experiments/archive/frontier_consolidation/data/checkpoint_averages/reinvest43022_avg_90_100",
        "reference_role": "Arithmetic weight average of seed43022 chck_90M and chck_100M; tests whether 90M Supplement/EWoK gains can retain 100M GlobalPIQA/Entity balance.",
    },
    "avg43122_80_100": {
        "family": "compact_view_reinvest_checkpoint_average",
        "seed": "43122",
        "exposure_m": "avg_80_100",
        "model_path": ROOT / "experiments/archive/frontier_consolidation/data/checkpoint_averages/reinvest43122_avg_80_100",
        "reference_role": "Arithmetic weight average of seed43122 chck_80M and chck_100M; tests whether 80M Supplement/GPIQA gains can retain 100M broad balance.",
    },
})

if __name__ == "__main__":
    base.main()
