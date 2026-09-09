#!/usr/bin/env python3
"""research: Evaluate the same-trainer reference tail on Cheap7 + Entity.

This evaluates the completed ordinary-tail reference continuation (354 updates,
13,994,705 ordinary words, zero relation words) using the trusted evaluator.
"""
import subprocess, json, sys, os
from pathlib import Path

STUDY = Path("experiments/archive/functional_learning")
EVAL_SCRIPT = STUDY / "scripts" / "bridge_eval.py"
REF_MODEL = STUDY / "data" / "reference_tail_corrected_ordinary" / "checkpoints" / "update_0354"
OUT_DIR = STUDY / "data" / "reference_tail_cheap7"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REF_MODEL.exists():
        print(f"ERROR: reference model not found: {REF_MODEL}", file=sys.stderr)
        sys.exit(1)

    # Run Entity evaluation
    entity_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model-path", str(REF_MODEL),
        "--out-dir", str(OUT_DIR),
        "--eval-cheap7",
        "--cheap7-columns", "Cheap7",
        "--gpu", str(args.gpu),
    ]
    if args.force:
        entity_cmd.append("--force")

    print(json.dumps({"event": "reference_tail_eval_start", "model": str(REF_MODEL), "gpu": args.gpu}), flush=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    proc = subprocess.run(entity_cmd, capture_output=True, text=True, env=env)
    
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    
    summary = {
        "status": "REFERENCE_TAIL_EVAL_DONE" if proc.returncode == 0 else "REFERENCE_TAIL_EVAL_FAILED",
        "model_path": str(REF_MODEL),
        "output_dir": str(OUT_DIR),
        "exit_code": proc.returncode,
    }
    
    # Try to find and include the eval result
    eval_json = OUT_DIR / "bridge_eval_summary.json"
    if eval_json.exists():
        with open(eval_json) as f:
            eval_data = json.loads(f.read())
        if "cheap7_scores" in eval_data:
            summary["cheap7_scores"] = eval_data["cheap7_scores"]
    
    with open(OUT_DIR / "reference_tail_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
