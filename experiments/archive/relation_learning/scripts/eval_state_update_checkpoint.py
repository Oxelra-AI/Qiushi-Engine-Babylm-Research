#!/usr/bin/env python3
"""research: Evaluate a state-update-trained checkpoint through the frontier_consolidation cheap7 harness.

This reuses the validated research evaluation pipeline and research evaluation script
from frontier_consolidation. Run after a training completes to get cheap7 and projected Overall(AoA0).

Usage:
    python eval_state_update_checkpoint.py --run-dir <training_run_dir> --checkpoint chck_82M --gpu 0
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path.cwd()
EVAL_SCRIPT = ROOT / "experiments/archive/frontier_consolidation/scripts/eval_custom_checkpoint.py"
STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Training run directory")
    ap.add_argument("--checkpoint", default="chck_82M", help="Checkpoint name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cheap7-only", action="store_true")
    args = ap.parse_args()
    
    run_dir = pathlib.Path(args.run_dir)
    model_path = run_dir / "hf_model" / args.checkpoint
    
    if not model_path.exists():
        print(f"ERROR: checkpoint not found at {model_path}")
        # List available checkpoints
        hf_dir = run_dir / "hf_model"
        if hf_dir.exists():
            ckpts = sorted([d.name for d in hf_dir.iterdir() if d.is_dir()])
            print(f"Available: {ckpts}")
        raise SystemExit(1)
    
    out_dir = run_dir / "eval" / args.checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the frontier_consolidation evaluation script
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model-path", str(model_path),
        "--eval-root", str(STRICT),
        "--output-dir", str(out_dir),
        "--gpu", str(args.gpu),
    ]
    
    if args.cheap7_only:
        cmd.append("--cheap7-only")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print(f"Evaluating {model_path} -> {out_dir}", flush=True)
    proc = subprocess.run(cmd, env=env)
    
    # Read results
    summary_path = out_dir / "cheap7_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(f"No summary found at {summary_path}")
    
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
