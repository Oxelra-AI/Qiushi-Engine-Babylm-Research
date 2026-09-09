#!/usr/bin/env python3
"""research: score delivered RoBERTa REPEAT_SPLIT on existing relation probes.

This wrapper uses the research probe definitions with the research RoBERTa
checkpoint set (60M..100M) but scores only the newly delivered REPEAT_SPLIT
RoBERTa arm.  Integration against the already-scored RoBERTa C/R/V rows is done
in a separate CPU script so we do not duplicate model scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_roberta_repeat_split_probes.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

WS = _public_path('experiments/archive/relation_learning')
RUN = _public_path('experiments/archive/relation_learning/training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022')
OUT = _public_path('experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic')
CKS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
ARM = "RBT_RS_43022"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if not (_public_path('experiments/archive/relation_learning/training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022/hf_model/chck_100M')).exists():
        raise FileNotFoundError(_public_path('experiments/archive/relation_learning/training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022/hf_model/chck_100M'))
    base.ARM_CONFIGS.clear()
    base.ARM_CONFIGS[ARM] = RUN
    base.OUT_DEFAULT = OUT
    sys.argv = [
        "score_roberta_repeat_split_probes.py",
        "--arms", ARM,
        "--checkpoints", *CKS,
        "--gpu", str(args.gpu),
        "--batch-size", str(args.batch_size),
        "--out-dir", str(OUT),
    ]
    if args.plan_only:
        sys.argv.append("--plan-only")
    base.main()


if __name__ == "__main__":
    main()
