#!/usr/bin/env python3
"""research: Evaluate formation experiment checkpoints at alpha=0.75.

Runs Cheap7 evaluation on all formation checkpoints using training evaluate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

REFERENCE_SCORES = {
    "coherent86_fast": 44.56428571428571,
    "coherent86_entity": 27.78,
    "coherent86_overall": 42.1210247099666,
    "coherent_replay_fast": 44.106,
    "spanbreak_fast": 43.121,
    "chck_82M_fast": 43.959,
}


def find_checkpoints(arm_dir: Path) -> list[Path]:
    """Find all HF model checkpoint directories."""
    hf = arm_dir / "hf_model"
    if not hf.exists():
        return []
    ckpts = sorted(
        [d for d in hf.iterdir() if d.is_dir() and (d / "model.safetensors").exists()],
        key=lambda d: d.name
    )
    return ckpts


def evaluate_checkpoint(ckpt_path: Path, eval_scale: float, gpu: int,
                        out_dir: Path) -> dict:
    """Run training evaluate on a single checkpoint."""
    ckpt_name = ckpt_path.name
    eval_out = out_dir / f"eval_{ckpt_name}_scale{eval_scale}"
    eval_out.mkdir(parents=True, exist_ok=True)

    # Build training evaluate command
    cmd = [
        "training", "evaluate",
        "--model", str(ckpt_path),
        "--output", str(eval_out),
        "--gpu", str(gpu),
        "--trust-remote-code",
        "--private-adapter-scale", str(eval_scale),
        "--fast",
    ]

    print(json.dumps({"event": "eval_start", "checkpoint": ckpt_name,
                      "scale": eval_scale, "cmd": " ".join(cmd)}), flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Try to find the eval results
            for f in eval_out.rglob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if "scores" in data or "BLiMP" in data:
                        return {"checkpoint": ckpt_name, "scale": eval_scale,
                                "status": "ok", "result": data, "path": str(f)}
                except Exception:
                    continue
        return {"checkpoint": ckpt_name, "scale": eval_scale,
                "status": "eval_failed", "returncode": result.returncode,
                "stderr": result.stderr[-500:] if result.stderr else ""}
    except Exception as e:
        return {"checkpoint": ckpt_name, "scale": eval_scale,
                "status": "error", "error": str(e)}


def cheap7_from_scores(scores: dict) -> dict:
    """Extract Cheap7 metrics from evaluation scores."""
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    vals = {}
    for k in keys:
        if k in scores:
            vals[k] = float(scores[k])
        elif k == "GlobalPIQA" and "GlobalPIQA_mean" in scores:
            vals[k] = float(scores["GlobalPIQA_mean"])
    if len(vals) >= 7:
        vals["cheap7_mean"] = sum(vals[k] for k in keys) / 7
        vals["ex_entity"] = (sum(vals[k] for k in keys if k != "Entity") / 6
                             if len([k for k in keys if k in vals and k != "Entity"]) >= 6 else None)
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["reference_4M", "full_18M", "both"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eval_scale", type=float, default=0.75)
    p.add_argument("--experiment_dir", default=str(_public_path('experiments/archive/functional_learning/data/formation_experiment')))
    args = p.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = exp_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [args.arm] if args.arm != "both" else ["reference_4M", "full_18M"]
    all_results = {}

    for arm in arms:
        arm_dir = exp_dir / arm
        if not arm_dir.exists():
            print(f"Arm directory not found: {arm_dir}", flush=True)
            continue

        ckpts = find_checkpoints(arm_dir)
        print(json.dumps({"event": "found_checkpoints", "arm": arm,
                          "count": len(ckpts),
                          "names": [c.name for c in ckpts]}), flush=True)

        arm_results = []
        for ckpt in ckpts:
            result = evaluate_checkpoint(ckpt, args.eval_scale, args.gpu, out_dir)
            if result.get("status") == "ok" and "result" in result:
                scores = result["result"].get("scores", result["result"])
                c7 = cheap7_from_scores(scores)
                result["cheap7"] = c7
                print(json.dumps({"event": "eval_done", "arm": arm,
                                  "checkpoint": result["checkpoint"],
                                  "cheap7_mean": c7.get("cheap7_mean"),
                                  "entity": c7.get("Entity"),
                                  "ex_entity": c7.get("ex_entity")}), flush=True)
            arm_results.append(result)

        all_results[arm] = arm_results

    # Summary
    summary = {
        "status": "FORMATION_EVAL_COMPLETE",
        "arms": {},
        "reference_scores": REFERENCE_SCORES,
    }
    for arm, results in all_results.items():
        best = None
        for r in results:
            c7 = r.get("cheap7", {})
            mean = c7.get("cheap7_mean")
            if mean is not None and (best is None or mean > best.get("cheap7", {}).get("cheap7_mean", 0)):
                best = r
        summary["arms"][arm] = {
            "checkpoints_evaluated": len(results),
            "best_checkpoint": best.get("checkpoint") if best else None,
            "best_cheap7_mean": best.get("cheap7", {}).get("cheap7_mean") if best else None,
            "best_entity": best.get("cheap7", {}).get("Entity") if best else None,
            "all_results": results,
        }

    summary_path = out_dir / "formation_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2), flush=True)
    for arm, data in summary.get("arms", {}).items():
        print(json.dumps({"arm": arm, "best_checkpoint": data.get("best_checkpoint"),
                          "best_cheap7_mean": data.get("best_cheap7_mean"),
                          "best_entity": data.get("best_entity")}), flush=True)


if __name__ == "__main__":
    main()
