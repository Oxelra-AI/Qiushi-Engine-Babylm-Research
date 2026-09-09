#!/usr/bin/env python3
"""research: Score SHUF and DUP seed43122 replication arms.

Uses the same probe families as score_paired_context_relation_design.py:
1. Compact T/U/N (overlap and nonoverlap)
2. Wikipedia T/U/N (overlap and nonoverlap)
3. Natural copy gain
4. Entity relevant-update stratification
5. Ordinary held-out MLM loss

OFF seed43122 serves as baseline; ALN seed43122 already scored at research.
"""

import os, sys, json, csv, glob
import numpy as np

WS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(WS, "data")
OUT_DIR = os.path.join(DATA, "shuf_dup_seed43122_probe")
os.makedirs(OUT_DIR, exist_ok=True)

# Checkpoint paths
SHUF_RUN = os.path.join(WS, "training", "runs", "qwen_shuffled_control_16k_seed43122")
DUP_RUN = os.path.join(WS, "training", "runs", "official_original_dup_16k_seed43122")
OFF_RUN = "experiments/archive/compact_experience/training/runs/official_only_16k_seed43122"

def find_late_checkpoints(run_dir, prefix="chck_"):
    """Find 80M, 90M, 100M checkpoints."""
    ckpts = {}
    for name in ["chck_80M", "chck_90M", "chck_100M"]:
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            ckpts[name] = path
    return ckpts

def check_readiness():
    """Check if training runs have completed."""
    results = {}
    for name, path in [("SHUF_s43122", SHUF_RUN), ("DUP_s43122", DUP_RUN), ("OFF_s43122", OFF_RUN)]:
        ckpts = find_late_checkpoints(path)
        results[name] = {
            "path": path,
            "exists": os.path.isdir(path),
            "checkpoints": list(ckpts.keys()),
            "ready": len(ckpts) >= 3,
        }
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Only check readiness")
    args = parser.parse_args()

    readiness = check_readiness()
    print(json.dumps(readiness, indent=2))

    if args.check_only:
        sys.exit(0)

    # Check if all arms are ready
    if not all(r["ready"] for r in readiness.values()):
        missing = [k for k, v in readiness.items() if not v["ready"]]
        print(f"Not ready yet: {missing}")
        sys.exit(1)

    print("All arms ready. Scoring would proceed here.")
    print("This script will be completed when training results are delivered.")
    # The full scoring implementation follows the same pattern as
    # score_paired_context_relation_design.py and replicate_paired_context_aln_off_seed43122.py
