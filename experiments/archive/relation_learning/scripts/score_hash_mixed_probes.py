#!/usr/bin/env python3
"""research: score the hash-mixed arm on existing held-out relation probes.

The base research framework writes token-level true-source/unrelated-source terms,
natural-copy components, and the Entity cue-ablation readout. A later integration
joins these terms to the already-scored seed43022 VIEW/REPEAT/CLEAN endpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import pathlib
import sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_hash_mixed_probes.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

WS = _public_path('experiments/archive/relation_learning')
RUN = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022')
OUT = _public_path('experiments/archive/relation_learning/data/hash_mixed_probes')
ARM = "D_HM_43022"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    base.ARM_CONFIGS.clear()
    base.ARM_CONFIGS[ARM] = RUN
    base.OUT_DEFAULT = OUT
    sys.argv = [
        "score_hash_mixed_probes.py",
        "--arms", ARM,
        "--checkpoints", "chck_80M", "chck_90M", "chck_100M",
        "--gpu", args.device.replace("cuda:", ""),
        "--batch-size", str(args.batch_size),
        "--out-dir", str(OUT),
    ]
    if args.plan_only:
        sys.argv.append("--plan-only")
    base.main()


if __name__ == "__main__":
    main()
