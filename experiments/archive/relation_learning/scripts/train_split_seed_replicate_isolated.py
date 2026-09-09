#!/usr/bin/env python3
"""research isolated seed-replicate launcher for split controls.

Same as train_split_seed_replicate.py, but uses a preflight/result side
directory that is unique to (seed, arm). This allows VIEW_SPLIT seed43122 to run
while REPEAT_SPLIT seed43122 is already using the earlier shared research preflight
lock. The training stream, tokenizer, architecture, recipe, and default run-dir
naming are unchanged except for extra_init_seed/train_rng_seed and arm/seed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/train_split_seed_replicate_isolated.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import train_split_inwindow_control as base  # noqa: E402


def configure(seed: int, data_arm: str) -> None:
    if seed not in {43022, 43122, 43222}:
        raise SystemExit(f"unexpected seed {seed}; use one of 43022/43122/43222 for comparable DeBERTa coordinates")
    if data_arm not in {"view_split", "repeat_split"}:
        raise SystemExit(f"unexpected data arm {data_arm}")
    base.RECIPE = copy.deepcopy(base.RECIPE)
    base.RECIPE["extra_init_seed"] = seed
    base.RECIPE["train_rng_seed"] = seed + 1
    base.PREFLIGHT_DIR = base.WS / "data" / f"split_seed_replicate_preflight_seed{seed}_{data_arm}"

    def default_run_dir(data_arm_inner: str) -> pathlib.Path:
        short = "view" if data_arm_inner == "view_split" else "repeat"
        return base.RUNS_DIR / f"full_p2c_c2p_abs_{short}_split_dose2p64x_rowholdout_deberta100M_seed{seed}"

    base.default_run_dir = default_run_dir  # type: ignore[assignment]


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed-replicate", type=int, default=43122)
    ap.add_argument("--data-arm", choices=["view_split", "repeat_split"], required=True)
    known, remaining = ap.parse_known_args()
    configure(known.seed_replicate, known.data_arm)
    sys.argv = [sys.argv[0], "--data-arm", known.data_arm, *remaining]
    base.main()


if __name__ == "__main__":
    main()
