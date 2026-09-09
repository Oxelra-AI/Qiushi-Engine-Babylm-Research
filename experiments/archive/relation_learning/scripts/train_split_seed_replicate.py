#!/usr/bin/env python3
"""research: seed-replicate launcher for split in-window controls.

This wraps the audited research split-control trainer but changes only the model
initialization/training RNG coordinate and the output/preflight directories. The
split streams, tokenizer, DeBERTa architecture, MLM recipe, 100M exposure budget,
and research row-holdout split construction are unchanged.

Scientific purpose: test whether the research locality result is seed-stable. The
first replicate requested here is REPEAT_SPLIT at seed43122, comparable within
seed to existing original REPEAT/CLEAN/VIEW 43122 arms.
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

ROOT = _public_path('experiments/archive/relation_learning/scripts/train_split_seed_replicate.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import train_split_inwindow_control as base  # noqa: E402


def configure(seed: int) -> None:
    if seed not in {43022, 43122, 43222}:
        raise SystemExit(f"unexpected seed {seed}; use one of 43022/43122/43222 for comparable DeBERTa coordinates")
    base.RECIPE = copy.deepcopy(base.RECIPE)
    base.RECIPE["extra_init_seed"] = seed
    base.RECIPE["train_rng_seed"] = seed + 1
    base.PREFLIGHT_DIR = base.WS / "data" / "split_seed_replicate_preflight"

    def default_run_dir(data_arm: str) -> pathlib.Path:
        short = "view" if data_arm == "view_split" else "repeat"
        return base.RUNS_DIR / f"full_p2c_c2p_abs_{short}_split_dose2p64x_rowholdout_deberta100M_seed{seed}"

    base.default_run_dir = default_run_dir  # type: ignore[assignment]


def main() -> None:
    # Parse only the seed first, then let the base launcher parse its normal args.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seed-replicate", type=int, default=43122)
    known, remaining = ap.parse_known_args()
    configure(known.seed_replicate)
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
