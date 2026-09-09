#!/usr/bin/env python3
"""research: official-compatible evaluator for innovation-biased WWM endpoints.

Thin wrapper around research's hardened evaluation harness for the
innovation-biased WWM experiment.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import pathlib
import sys


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
SCRIPT_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_compliant_endpoint as ev  # noqa: E402

WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_RUN_DIR = WORKSPACE / "training/runs/innovation_biased_wwm_reinvest_seed43022_80M"
DEFAULT_OUT_ROOT = WORKSPACE / "data/innovation_biased_70_80M_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/innovation_biased_70_80M_collate"

DESCRIPTION = (
    "compact_view_reinvest with innovation-biased WWM (p_innov=0.5, p_copy=0.0) "
    "trained from random weights with the research legal 16k tokenizer; same model, "
    "optimizer, seeds, and exposure as the token-mean reinvest reference; only the "
    "WWM group selection on changed rows differs"
)
FAMILY = "end_to_end_legal16k_innovation_biased_wwm_density_reinvestment"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--endpoint", default="chck_80M")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    ev.ARM_DEFAULTS["innovation_biased"] = {
        "target": args.target,
        "run_dir": pathlib.Path(args.run_dir),
        "endpoint": args.endpoint,
        "description": DESCRIPTION,
        "family": FAMILY,
    }
    ns = argparse.Namespace(
        arm="innovation_biased",
        target=args.target,
        run_dir=args.run_dir,
        endpoint=args.endpoint,
        out_root=args.out_root,
        collate_root=args.collate_root,
        gpu=args.gpu,
        columns=args.columns,
        force=args.force,
        preflight_only=args.preflight_only,
    )
    ev.run_one(ns)


if __name__ == "__main__":
    main()
