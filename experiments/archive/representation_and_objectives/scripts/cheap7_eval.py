#!/usr/bin/env python3
"""research: General cheap7 evaluation wrapper.

Evaluates any DeBERTa-v2 checkpoint via the validated evaluator
for cheap7 columns (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading).

Usage:
  python3 cheap7_eval.py --checkpoint_dir <path> --output_dir <path> --gpu 0

The evaluator uses the official evaluation pipeline with the correct commit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, subprocess, sys, time
from statistics import mean


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
EVALUATOR = ROOT / "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py"
SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def evaluate_checkpoint(checkpoint_dir: pathlib.Path, output_dir: pathlib.Path,
                       target_name: str, gpu: int) -> dict:
    """Run the evaluator on a single checkpoint and return scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create proxy run directory expected by the evaluator
    proxy = output_dir / "proxy_run"
    proxy.mkdir(parents=True, exist_ok=True)
    hf_model = proxy / "hf_model"
    hf_model.mkdir(parents=True, exist_ok=True)
    
    # Symlink the checkpoint
    link = hf_model / "chck_100M"
    if link.exists():
        link.unlink()
    link.symlink_to(checkpoint_dir.resolve())
    
    # Write minimal scientific_metrics.json
    metrics = {
        "variant": target_name,
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "word_exposure": 100_000_000,
        "saved_checkpoints": [{"name": "chck_100M", "path": str(link)}],
    }
    (proxy / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2))
    
    cmd = [
        sys.executable, str(EVALUATOR),
        "--run_dir", str(proxy),
        "--checkpoints", "chck_100M",
        "--eval_columns", ",".join(SCORE_COLUMNS),
        "--output_dir", str(output_dir),
        "--gpu", str(gpu),
    ]
    
    print(json.dumps({"event": "eval_start", "checkpoint": str(checkpoint_dir),
                      "target": target_name, "gpu": gpu}), flush=True)
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(json.dumps({"event": "eval_error", "returncode": result.returncode,
                          "stderr_tail": result.stderr[-500:] if result.stderr else ""}), flush=True)
        return {"error": result.returncode, "stderr": result.stderr[-1000:]}
    
    # Parse scores from output
    scores = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "scores" in obj:
                scores = obj["scores"]
            elif "BLiMP" in obj or "Supplement" in obj:
                scores = obj
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Also check for results files
    for results_file in output_dir.glob("**/results*.json"):
        try:
            data = json.loads(results_file.read_text())
            if isinstance(data, dict) and "BLiMP" in data:
                scores.update(data)
        except Exception:
            pass
    
    if scores:
        cheap7_vals = [scores.get(k, 0) for k in SCORE_COLUMNS if k in scores]
        scores["cheap7"] = mean(cheap7_vals) if cheap7_vals else 0
    
    print(json.dumps({"event": "eval_done", "target": target_name,
                      "elapsed_sec": round(elapsed, 1), "scores": scores}), flush=True)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True, help="Path to HF checkpoint directory")
    p.add_argument("--output_dir", required=True, help="Where to save evaluation results")
    p.add_argument("--target_name", default="eval", help="Label for this evaluation")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_dir}", flush=True)
        sys.exit(1)
    
    scores = evaluate_checkpoint(checkpoint_dir, output_dir, args.target_name, args.gpu)
    
    # Save summary
    summary = {
        "status": "CHEAP7_EVAL",
        "checkpoint": str(checkpoint_dir),
        "target_name": args.target_name,
        "scores": scores,
    }
    summary_path = output_dir / "cheap7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "summary_saved", "path": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
