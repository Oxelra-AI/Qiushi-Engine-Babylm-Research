#!/usr/bin/env python3
"""research: extract AoA surprisals from calibration arm checkpoints.

Runs the official AoA surprisal extractor on each 1M checkpoint from the
calibration arms. Produces per-word mean surprisal files compatible with
the β-measurement analysis.

Usage: Run after calibration training completes.
  python extract_calibration_surprisals.py --arm schedule --gpu 0
  python extract_calibration_surprisals.py --arm enrichment --gpu 1
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, json, os, pathlib, subprocess, sys, time

ROOT = _public_path('.')
EVAL_DIR = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
AOA_RUNNER = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/AoA_word/run.py')
EVAL_DATA = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def extract_surprisals(checkpoint_dir: pathlib.Path, output_dir: pathlib.Path,
                       step_label: str, gpu: int):
    """Run the official AoA surprisal extractor on one checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    cmd = [
        sys.executable, "-B", str(AOA_RUNNER),
        "--model_path", str(checkpoint_dir),
        "--eval_data_path", str(EVAL_DATA),
        "--output_dir", str(output_dir),
        "--use_bos_only", "False",
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    elapsed = time.time() - t0
    
    return {
        "step": step_label,
        "checkpoint": rel(checkpoint_dir),
        "output": rel(output_dir),
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["schedule", "enrichment"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-checkpoint-m", type=int, default=30)
    args = parser.parse_args()
    
    if args.arm == "schedule":
        arm_dir = _public_path('experiments/archive/relation_learning/data/schedule_arm')
    else:
        arm_dir = _public_path('experiments/archive/relation_learning/data/enrichment_arm')
    
    out_base = arm_dir / "aoa_surprisals"
    
    # Find available checkpoints
    hf_dir = arm_dir / "hf_model"
    checkpoints = []
    for m in range(1, args.max_checkpoint_m + 1):
        cp = hf_dir / f"chck_{m}M"
        if cp.exists():
            checkpoints.append((m, cp))
    
    print(json.dumps({"event": "start", "arm": args.arm, "gpu": args.gpu,
                      "checkpoints": len(checkpoints)}), flush=True)
    
    results = []
    for m, cp in checkpoints:
        step_label = f"chck_{m}M"
        out_dir = out_base / step_label
        print(json.dumps({"event": "extracting", "checkpoint": step_label}), flush=True)
        r = extract_surprisals(cp, out_dir, step_label, args.gpu)
        results.append(r)
        print(json.dumps({"event": "extracted", **r}), flush=True)
    
    summary = {
        "status": f"SURPRISAL_EXTRACTION_{args.arm.upper()}_DONE",
        "created_utc": now(),
        "arm": args.arm,
        "n_checkpoints": len(results),
        "results": results
    }
    summary_path = out_base / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
