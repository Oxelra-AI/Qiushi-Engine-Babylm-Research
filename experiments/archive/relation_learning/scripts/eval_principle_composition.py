#!/usr/bin/env python3
"""research: Materialize principle-composition endpoint and evaluate.

After training completes, this script:
1. Materializes the trained checkpoint as run-like endpoints at different alpha values
2. Runs cheap7 + Entity evaluation using the proven research/research pipeline
3. Parses Entity strata and runs the operation-content diagnostic

Usage:
  python eval_principle_composition.py [--alpha ALPHAS] [--gpu GPU]
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')

REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))

TRAIN_OUT = _public_path('experiments/archive/relation_learning/training/runs/principle_composition_coherent_plus_aln')
CHCK_82M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')

def materialize_alpha(train_dir: pathlib.Path, alpha: float, out_dir: pathlib.Path):
    """Create a run-like endpoint at a specific private_adapter_scale."""
    import materialize_private_scale as S133
    checkpoint = train_dir / "hf_model" / "final"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    run_dir = out_dir / f"alpha_{alpha:.2f}"
    model_dir = run_dir / "hf_model" / "final"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    S133.materialize(str(checkpoint), str(model_dir), alpha)
    
    # Read composition metadata
    comp_cfg = json.loads((train_dir / "composition_config.json").read_text()) if (train_dir / "composition_config.json").exists() else {}
    train_met = json.loads((train_dir / "scientific_metrics.json").read_text()) if (train_dir / "scientific_metrics.json").exists() else {}
    
    metrics = {
        "status": "PRINCIPLE_COMPOSITION_ENDPOINT",
        "composition": "coherent_replay_plus_aligned_restatement",
        "alpha": alpha,
        "pair_fraction": comp_cfg.get("pair_fraction", "?"),
        "pair_words": comp_cfg.get("pair_words", "?"),
        "ordinary_words": comp_cfg.get("ordinary_words", "?"),
        "initial_consumed_words": train_met.get("initial_consumed_words", 82012495),
        "tail_charged_words": train_met.get("tail_charged_words", "?"),
        "total_consumed_words": train_met.get("total_consumed_words", "?"),
        "total_params": train_met.get("total_params", 36458592),
        "private_params": train_met.get("private_params", 995584),
        "source_checkpoint": str(train_dir / "hf_model" / "final"),
    }
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return run_dir

def evaluate_endpoint(run_dir: pathlib.Path, label: str, gpu: int):
    """Run research cheap7 + Entity evaluation."""
    eval_script = _public_path('experiments/archive/relation_learning/scripts/frozen82_tail_eval_one.py')
    if not eval_script.exists():
        # Try alternate location
        eval_script = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')
    
    model_dir = run_dir / "hf_model" / "final"
    out = WS / f"data/eval_{label}"
    out.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-B", str(eval_script),
        "--model-dir", str(model_dir),
        "--label", label,
        "--output-dir", str(out),
        "--gpu", str(gpu),
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"Evaluating {label} on GPU {gpu}...", flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    
    if result.returncode != 0:
        print(f"EVAL FAILED: {result.stderr[-500:]}", flush=True)
        return None
    
    # Read results
    summary = None
    for candidate in [out / "summary.json", out / f"{label}_summary.json"]:
        if candidate.exists():
            summary = json.loads(candidate.read_text())
            break
    
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.50,0.75,1.00", help="Comma-separated alpha values")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="Skip materialization, just evaluate")
    args = ap.parse_args()
    
    alphas = [float(a) for a in args.alphas.split(",")]
    
    out_base = _public_path('experiments/archive/relation_learning/data/principle_composition_endpoints')
    out_base.mkdir(parents=True, exist_ok=True)
    
    if not args.eval_only:
        print("Materializing endpoints...", flush=True)
        for alpha in alphas:
            print(f"  Alpha {alpha:.2f}...", flush=True)
            materialize_alpha(TRAIN_OUT, alpha, out_base)
    
    print("\nEvaluating...", flush=True)
    results = {}
    for alpha in alphas:
        label = f"comp_alpha{alpha:.2f}".replace(".", "p")
        run_dir = out_base / f"alpha_{alpha:.2f}"
        summary = evaluate_endpoint(run_dir, label, args.gpu)
        if summary:
            results[f"alpha_{alpha:.2f}"] = summary
            print(f"  Alpha {alpha:.2f}: {json.dumps({k:v for k,v in summary.items() if k in ['cheap7','Entity','BLiMP','Supplement','EWoK','COMPS','GlobalPIQA','Reading']}, indent=2)}")
    
    # Save combined results
    (out_base / "combined_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved to {out_base / 'combined_results.json'}")

if __name__ == "__main__":
    main()
