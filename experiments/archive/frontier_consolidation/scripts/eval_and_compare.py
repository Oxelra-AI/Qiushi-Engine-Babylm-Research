#!/usr/bin/env python3
"""research evaluation and comparison for the pathway-separation factorial.

Evaluates a completed research run on cheap7 columns and compares against
known references (mlm_only 39.787, research 39.664, broad-aligned 38.763).

Usage:
  python eval_and_compare.py --run_dir <path> --target <label> --gpu <id>
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/dualview_eval_one.py')

REFERENCES = {
    "reference_20M": {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73,
                   "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195,
                   "Reading": 8.67, "cheap7": 39.6636},
    "mlm_only_20M": {"BLiMP": 62.05, "Supplement": 58.43, "EWoK": 50.10,
                     "Entity": 18.36, "COMPS": 50.70, "GlobalPIQA": 32.67,
                     "Reading": 6.20, "cheap7": 39.787},
    "broad_aligned_20M": {"BLiMP": 56.72, "Supplement": 51.76, "EWoK": 51.11,
                          "Entity": 17.91, "COMPS": 50.17, "GlobalPIQA": 36.10,
                          "Reading": 7.57, "cheap7": 38.763},
}

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def evaluate(run_dir: str, target: str, gpu: int):
    """Run dualview_eval_one.py and return the summary dict."""
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--run_dir", str(run_dir),
        "--eval_checkpoint", "final",
        "--target", target,
        "--gpu", str(gpu),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(json.dumps({"event": "eval_start", "target": target, "gpu": gpu}), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"Eval failed: {result.stderr[-500:]}", flush=True)
        return None
    # Parse the last JSON line from stdout
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def compare(scores: dict, label: str):
    """Compare scores against all references."""
    comparisons = []
    for ref_name, ref_scores in REFERENCES.items():
        delta = {}
        for col in CHEAP_COLS:
            if col in scores and col in ref_scores:
                delta[col] = round(scores[col] - ref_scores[col], 4)
        cheap7_delta = round(scores.get("cheap7", 0) - ref_scores.get("cheap7", 0), 4)
        comparisons.append({
            "vs": ref_name,
            "cheap7_delta": cheap7_delta,
            "column_deltas": delta,
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "hf_model/final").exists():
        print(f"ERROR: {run_dir}/hf_model/final not found", flush=True)
        sys.exit(1)

    # Load training metrics
    metrics_path = run_dir / "scientific_metrics.json"
    train_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    # Evaluate
    eval_result = evaluate(str(run_dir), args.target, args.gpu)
    if eval_result is None:
        print("Evaluation failed", flush=True)
        sys.exit(1)

    # Extract scores from evaluation summary
    summary_path = eval_result.get("summary_json")
    if summary_path:
        summary = json.loads(Path(summary_path).read_text()) if Path(summary_path).exists() else {}
    else:
        summary = {}

    scores = {}
    official = summary.get("official_overall", {}).get("scores", {})
    for col in CHEAP_COLS:
        if col == "GlobalPIQA":
            gp = official.get("GlobalPIQA_parallel")
            gn = official.get("GlobalPIQA_nonparallel")
            if gp is not None and gn is not None:
                scores["GlobalPIQA"] = round((gp + gn) / 2, 4)
            elif "GlobalPIQA" in official and official["GlobalPIQA"] is not None:
                scores["GlobalPIQA"] = official["GlobalPIQA"]
        elif col in official and official[col] is not None:
            scores[col] = official[col]

    if scores:
        scores["cheap7"] = round(sum(scores.get(c, 0) for c in CHEAP_COLS) / 7, 4)

    # Compare
    comparisons = compare(scores, args.target)

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else (
        _public_path('experiments/archive/frontier_consolidation/data') / f"step130_{args.target}_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "EVAL_AND_COMPARE",
        "target": args.target,
        "scores": scores,
        "comparisons": comparisons,
        "train_metrics": {
            "mode": train_metrics.get("mode"),
            "pathway": train_metrics.get("pathway"),
            "updates": train_metrics.get("updates"),
            "total_charged_words": train_metrics.get("total_charged_words"),
            "mean_loss": train_metrics.get("mean_loss"),
            "mean_aux_loss": train_metrics.get("mean_aux_loss"),
            "mean_neutral_loss": train_metrics.get("mean_neutral_loss"),
        },
        "eval_result": eval_result,
    }

    out_json = out_dir / f"step130_{args.target}_eval.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "complete", "target": args.target, "scores": scores,
                      "comparisons": comparisons, "out_json": str(out_json)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
