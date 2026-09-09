#!/usr/bin/env python3
"""research: score split-control DeBERTa arms with the held-out relation probes.

Wrapper around research probe framework for REPEAT_SPLIT and/or VIEW_SPLIT arms
plus the original seed43022 CLEAN baseline. Produces the same output format so
results compare directly against notes/015_split_control_numeric_prestatement.md.

Usage:
  python score_split_arms.py --arm repeat_split [--device cuda:0]
  python score_split_arms.py --arm view_split [--device cuda:0]
  python score_split_arms.py --arm both [--device cuda:0]
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, pathlib, sys

ROOT = _public_path('experiments/archive/relation_learning/scripts/score_split_arms.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

REPRESENTATION_FRONTIER_STUDIES_RUNS = _public_path('experiments/archive/frontier_consolidation/training/runs')
FUNCTIONAL_RELATION_STUDIES_RUNS = _public_path('experiments/archive/relation_learning/training/runs')
OUT_BASE = _public_path('experiments/archive/relation_learning/data')

CLEAN_43022 = _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022')

SPLIT_ARMS = {
    "repeat_split": {
        "arm": "D_RS_43022",
        "path": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022'),
    },
    "view_split": {
        "arm": "D_VS_43022",
        "path": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022'),
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["repeat_split", "view_split", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    arms = list(SPLIT_ARMS.keys()) if args.arm == "both" else [args.arm]
    
    for arm_key in arms:
        cfg = SPLIT_ARMS[arm_key]
        if not (cfg["path"] / "hf_model" / "chck_100M").exists():
            print(f"[SKIP] {arm_key}: no chck_100M at {cfg['path']}", flush=True)
            continue
        
        print(f"\n=== Scoring {arm_key} ({cfg['arm']}) ===", flush=True)
        
        base.ARM_CONFIGS.clear()
        base.ARM_CONFIGS.update({
            cfg["arm"]: cfg["path"],
            "D_C_43022": CLEAN_43022,
        })
        base.OUT_DEFAULT = OUT_BASE / f"split_{arm_key}_probes"
        
        # Invoke with just the two arms
        sys.argv = [
            "score_split_arms.py",
            "--arms", cfg["arm"], "D_C_43022",
            "--checkpoints", "chck_80M", "chck_90M", "chck_100M",
            "--gpu", args.device.replace("cuda:", ""),
            "--batch-size", str(args.batch_size),
        ]
        base.main()
        
        print(f"[DONE] {arm_key} → {base.OUT_DEFAULT}", flush=True)


if __name__ == "__main__":
    main()
