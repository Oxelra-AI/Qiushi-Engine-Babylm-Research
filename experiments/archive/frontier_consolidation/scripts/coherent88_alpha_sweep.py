#!/usr/bin/env python3
"""research: Alpha-sweep evaluation for chck_84M coherent replay.

After coherent88 training completes, evaluates cheap7 columns at
alpha = 0.0 (pure 84M anchor), 0.5, 0.75, 1.0.

Does NOT submit to leaderboard, upload, or modify the anchor.
"""
import json, os, shutil, sys, time
from pathlib import Path

ROOT = Path("experiments/archive/frontier_consolidation")
EVAL_SCRIPT = ROOT / "scripts/eval_custom_checkpoint.py"
COHERENT88_DIR = ROOT / "training/runs/frozen84_coherent88_seed43022"
ANCHOR_DIR = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M"

# Reference values from coherent86 (82M anchor) for comparison
COHERENT86_REFERENCE = {
    "anchor_82M_cheap7": 43.9594,
    "alpha_0.50_cheap7": 44.1779,
    "alpha_0.75_cheap7": 44.1814,
    "alpha_1.00_cheap7": 44.1064,
    "alpha_0.75_projected_overall": 42.1210,
}


def check_training_complete():
    metrics_path = COHERENT88_DIR / "scientific_metrics.json"
    if not metrics_path.exists():
        print(json.dumps({"status": "TRAINING_NOT_COMPLETE", "path": str(metrics_path)}))
        return False
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") != "TRAINING_COMPLETE":
        print(json.dumps({"status": "TRAINING_INCOMPLETE", "actual_status": metrics.get("status")}))
        return False
    print(json.dumps({
        "status": "TRAINING_VERIFIED",
        "updates": metrics.get("updates"),
        "tail_charged_words": metrics.get("tail_charged_words"),
        "total_consumed_words": metrics.get("total_consumed_words"),
        "total_params": metrics.get("total_params"),
        "private_params": metrics.get("private_params"),
        "final_main_loss": metrics.get("final_main_loss"),
        "elapsed_sec": metrics.get("elapsed_sec"),
    }, indent=2))
    return True


def set_alpha(model_dir: Path, alpha: float):
    """Set private_adapter_scale in the model config."""
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["private_adapter_scale"] = alpha
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--alphas", default="0.0,0.5,0.75,1.0")
    p.add_argument("--gpu", default="0")
    args = p.parse_args()

    if not check_training_complete():
        return

    if args.check_only:
        return

    alphas = [float(a) for a in args.alphas.split(",")]
    out_dir = ROOT / "data/coherent88_alpha_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for alpha in alphas:
        print(f"\n--- Evaluating alpha={alpha} ---")
        # For alpha=0.0, evaluate the raw anchor
        if alpha == 0.0:
            eval_path = str(ANCHOR_DIR)
            tag = "anchor_84M"
        else:
            # Set alpha in the coherent88 model config
            set_alpha(COHERENT88_DIR / "hf_model", alpha)
            eval_path = str(COHERENT88_DIR / "hf_model")
            tag = f"alpha_{alpha:.2f}"

        result_dir = out_dir / f"eval_{tag}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Run cheap7 evaluation
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu} python3 {EVAL_SCRIPT} "
            f"--checkpoint {eval_path} --output-dir {result_dir} "
            f"--trust-remote-code"
        )
        print(f"Running: {cmd}")
        rc = os.system(cmd)
        
        # Read results
        summary_path = result_dir / "eval_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            cheap7 = summary.get("cheap7") or summary.get("equal7")
            results[tag] = {
                "alpha": alpha,
                "cheap7": cheap7,
                "summary": summary,
            }
            print(f"  {tag}: cheap7 = {cheap7}")
        else:
            print(f"  {tag}: evaluation failed (no summary)")

    # Save sweep results
    sweep_summary = {
        "status": "ALPHA_SWEEP_COMPLETE",
        "coherent88_training": str(COHERENT88_DIR),
        "reference_coherent86": COHERENT86_REFERENCE,
        "results": results,
    }
    (out_dir / "alpha_sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2) + "\n")
    print(json.dumps(sweep_summary, indent=2))


if __name__ == "__main__":
    main()
