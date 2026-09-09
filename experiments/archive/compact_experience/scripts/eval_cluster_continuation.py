#!/usr/bin/env python3
"""research: No-AoA trajectory evaluation for cluster continuation arms.

Evaluates all four continuation arms on 7 NLP columns + Reading,
ranks by equal7 score, and compares against clean-Qwen chck_100M reference.

Usage:
  python eval_cluster_continuation.py
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys

ROOT = _public_path('experiments/archive/compact_experience')
EVAL_SCRIPT = _public_path('experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py')
RUNS = _public_path('experiments/archive/compact_experience/training/runs')
OUT = _public_path('experiments/archive/compact_experience/data/cluster_continuation_eval')

ARMS = ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
CHECKPOINTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]

# Reference: clean-Qwen chck_100M equal7
CLEAN_QWEN_EQUAL7 = 43.112857142857145


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    args = ap.parse_args()
    
    gpus = args.gpu.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Build targets
    targets: list[dict] = []
    for arm in args.arms:
        run_dir = RUNS / f"continuation_{arm}"
        model_root = run_dir / "hf_model"
        if not model_root.exists():
            print(f"WARNING: {model_root} not found, skipping {arm}")
            continue
        for ckpt in args.checkpoints:
            ckpt_path = model_root / ckpt
            if ckpt_path.exists():
                targets.append({"arm": arm, "checkpoint": ckpt, "path": str(ckpt_path)})
    
    if not targets:
        print("No targets found. Training may not be complete.")
        return
    
    print(f"Evaluating {len(targets)} targets across {len(gpus)} GPUs")
    
    # Run evaluation for each arm using the trajectory evaluator
    results: list[dict] = []
    
    for arm in args.arms:
        run_dir = RUNS / f"continuation_{arm}"
        model_root = run_dir / "hf_model"
        if not model_root.exists():
            continue
        
        available_ckpts = [c for c in args.checkpoints if (model_root / c).exists()]
        if not available_ckpts:
            continue
        
        target_name = f"step043_{arm}"
        gpu = gpus[args.arms.index(arm) % len(gpus)]
        
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--target", target_name,
            "--model_root", str(model_root),
            "--checkpoints", ",".join(available_ckpts),
            "--out_root", str(OUT),
            "--gpu", gpu,
        ]
        
        print(f"  Evaluating {arm} ({len(available_ckpts)} checkpoints) on GPU {gpu}")
        rc = subprocess.run(cmd, capture_output=True, text=True)
        
        if rc.returncode != 0:
            print(f"  ERROR: {arm} eval failed (rc={rc.returncode})")
            print(f"  stderr: {rc.stderr[-500:]}")
            continue
        
        # Read trajectory summary
        summary_path = OUT / target_name / f"{target_name}_trajectory_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            best = summary.get("best", {})
            results.append({
                "arm": arm,
                "best_checkpoint": best.get("checkpoint", ""),
                "equal7": best.get("row", {}).get("equal7_full_eval", 0),
                "BLiMP": best.get("row", {}).get("BLiMP", 0),
                "Supplement": best.get("row", {}).get("Supplement", 0),
                "EWoK": best.get("row", {}).get("EWoK", 0),
                "Entity": best.get("row", {}).get("Entity", 0),
                "COMPS": best.get("row", {}).get("COMPS", 0),
                "GlobalPIQA": best.get("row", {}).get("GlobalPIQA", 0),
                "Reading": best.get("row", {}).get("Reading", 0),
            })
    
    # === Ranking and comparison ===
    results.sort(key=lambda r: r.get("equal7", 0), reverse=True)
    
    print("\n=== Cluster continuation no-AoA ranking ===")
    print(f"{'Arm':<25} {'Best_Ckpt':<12} {'Equal7':>8} {'vs_Clean':>9}")
    print("-" * 60)
    for r in results:
        delta = r["equal7"] - CLEAN_QWEN_EQUAL7
        print(f"{r['arm']:<25} {r['best_checkpoint']:<12} {r['equal7']:>8.3f} {delta:>+9.3f}")
    
    # Profile comparison
    print(f"\n{'Arm':<25} {'BLiMP':>7} {'Supp':>7} {'EWoK':>7} {'Ent':>7} {'COMPS':>7} {'GPIQA':>7} {'Read':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['arm']:<25} {r.get('BLiMP',0):>7.2f} {r.get('Supplement',0):>7.2f} "
              f"{r.get('EWoK',0):>7.2f} {r.get('Entity',0):>7.2f} {r.get('COMPS',0):>7.2f} "
              f"{r.get('GlobalPIQA',0):>7.2f} {r.get('Reading',0):>7.2f}")
    
    # Save
    ranking = {
        "status": "CLUSTER_CONTINUATION_NOAOA_RANKING",
        "reference_clean_qwen_equal7": CLEAN_QWEN_EQUAL7,
        "results": results,
        "promotion_criteria": {
            "E1_beats_E2": results[0]["arm"] == "E1_true_cluster" if results else False,
            "E1_beats_E3": any(r["arm"] == "E1_true_cluster" and r["equal7"] > 
                              next((x["equal7"] for x in results if x["arm"] == "E3_anchor_repeat"), 0)
                              for r in results),
            "E1_beats_E4": any(r["arm"] == "E1_true_cluster" and r["equal7"] > 
                              next((x["equal7"] for x in results if x["arm"] == "E4_untouched_tail"), 0)
                              for r in results),
        },
    }
    
    ranking_path = _public_path('experiments/archive/compact_experience/data/cluster_continuation_eval/cluster_continuation_ranking.json')
    ranking_path.write_text(json.dumps(ranking, indent=2) + "\n")
    print(f"\nSaved: {ranking_path}")
    print(json.dumps({"ranking": str(ranking_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
