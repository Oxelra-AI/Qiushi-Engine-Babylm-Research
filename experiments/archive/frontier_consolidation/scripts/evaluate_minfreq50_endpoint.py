#!/usr/bin/env python3
"""research: official-compatible cheap/full evaluator for minfreq50 endpoints.

This is a thin wrapper around research's hardened evaluation harness.  It reuses the
same official-coordinate task runners, EWoK coordinate, Reading, SuperGLUE, and
AoA machinery, but supplies a truthful family/description for the legal
support-floored minfreq50 tokenizer route.  Cheap screens should pass only the
zero-shot + Reading columns; full endpoint evaluation can later include
SuperGLUE/AoA if an 100M endpoint is justified.
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
DEFAULT_RUN_DIR = WORKSPACE / "training/runs/minfreq50_initmatched_reinvest_seed43022_80M"
DEFAULT_OUT_ROOT = WORKSPACE / "data/minfreq50_initmatched_70_80M_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/minfreq50_initmatched_70_80M_collate"

DESCRIPTION = (
    "compact_view_reinvest trained from random weights with a legal support-floored "
    "minfreq50 byte-level BPE tokenizer trained only on the exact 10M compact-view "
    "reinvest pool; same-shape contextual tensors are init-matched to a research legal16k "
    "reference while vocab-shaped tensors are tokenizer-specific"
)
FAMILY = "end_to_end_legal_minfreq50_supportfloor_initmatched_density_reinvestment"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--endpoint", default="chck_80M")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="*", default=None, help="Subset among zero-shot columns, Reading, SuperGLUE, AoA")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    ev.ARM_DEFAULTS["minfreq50_initmatched"] = {
        "target": args.target,
        "run_dir": pathlib.Path(args.run_dir),
        "endpoint": args.endpoint,
        "description": DESCRIPTION,
        "family": FAMILY,
    }
    ns = argparse.Namespace(
        arm="minfreq50_initmatched",
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
