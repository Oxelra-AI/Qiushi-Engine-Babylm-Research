#!/usr/bin/env python3
"""research: Full parallel official evaluation for scale-1.75 100M endpoint.

Splits all evaluation tasks into independent jobs across both GPUs:
- GPU0: SuperGLUE subtasks (4 of 7)
- GPU1: SuperGLUE subtasks (3 of 7) + cheap zero-shot columns
- CPU: Reading
- After all: AoA (sequential through checkpoints, on GPU0)

Each job is a subprocess calling the evaluator with one column.
Output goes to isolated per-column directories to avoid JSON races.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')

EVAL = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
RUN_DIR_100M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
ENDPOINT = "chck_100M"

OUT_BASE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100M_full_eval')
COLLATE_BASE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100M_collate')

# Cheap zero-shot columns (each is independent)
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
              "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

# SuperGLUE subtasks are handled as one column by the evaluator
# AoA is one column

# GPU assignment plan for maximum parallelism:
# GPU0: SuperGLUE (7 subtasks, sequential within evaluator, ~60-90 min)
# GPU1: All cheap columns (each ~3-5 min, sequential, ~24-40 min) + Reading
# After cheap/SG: AoA on GPU0 (sequential through 19 checkpoints, ~30-60 min)

SUPERGLUE_PRIMARY_METRIC = {
    "boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1",
    "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_eval_column(col: str, gpu: int, target_suffix: str = "") -> dict[str, Any]:
    """Run one evaluation column via the evaluator."""
    target = f"scale1p75_100M_{col.lower().replace('_', '')}{target_suffix}"
    out_root = OUT_BASE / col
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-B", str(EVAL),
        "--arm", "reinvest",
        "--run-dir", str(RUN_DIR_100M),
        "--target", target,
        "--endpoint", ENDPOINT,
        "--out-root", str(out_root),
        "--collate-root", str(COLLATE_BASE / col),
        "--gpu", str(gpu),
        "--columns", col,
        "--force",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100M_full_eval/logs') / f"{col}_gpu{gpu}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(json.dumps({"event": "col_start", "col": col, "gpu": gpu, "target": target, "utc": now()}), flush=True)

    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    log.write_text(f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n[rc={proc.returncode} elapsed={elapsed:.1f}s]\n", encoding="utf-8")

    rec = {"col": col, "gpu": gpu, "target": target, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3)}
    if proc.returncode == 0:
        rec["status"] = "done"
        rec["stdout_tail"] = proc.stdout[-2000:]
    else:
        rec["status"] = "error"
        rec["stderr_tail"] = proc.stderr[-3000:]

    print(json.dumps({"event": "col_done", **{k: v for k, v in rec.items() if k != "stdout_tail" and k != "stderr_tail"}}), flush=True)
    return rec


def extract_score(col: str) -> float | None:
    """Extract score from evaluator output."""
    out_root = OUT_BASE / col
    for p in out_root.rglob("*.json"):
        if "per_target" in str(p):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                tasks = data.get("tasks", {})
                if col in tasks:
                    r = tasks[col]
                    if isinstance(r, dict) and r.get("score") is not None:
                        return float(r["score"])
                if col == "SuperGLUE" and "SuperGLUE" in tasks:
                    return tasks["SuperGLUE"].get("superglue_mean")
                if col == "Reading" and "Reading" in tasks:
                    r = tasks["Reading"]
                    if isinstance(r, dict):
                        if isinstance(r.get("scores"), dict):
                            return float(r["scores"].get("Reading", 0))
                        return float(r.get("score", 0))
                if col == "AoA" and "AoA" in tasks:
                    r = tasks["AoA"]
                    if isinstance(r, dict):
                        return float(r.get("aoa_leaderboard_score", r.get("aoa_for_provisional_overall", 0)))
            except Exception:
                continue
    return None


def check_training_done() -> bool:
    model_path = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M')
    return model_path.exists() and (model_path / "config.json").exists()


def wait_for_training(timeout_sec: int = 7200) -> bool:
    """Wait for training to finish."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if check_training_done():
            print(json.dumps({"event": "training_done", "endpoint": ENDPOINT, "utc": now()}), flush=True)
            return True
        # Check progress
        log_path = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/training_log.jsonl')
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            if lines:
                try:
                    last = json.loads(lines[-1])
                    words = last.get("cumulative_word_exposure", 0)
                    print(json.dumps({"event": "waiting", "words": words, "target": 100_000_000, "elapsed_wait": round(time.time() - t0, 1)}), flush=True)
                except Exception:
                    pass
        time.sleep(30)
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-wait", action="store_true", help="Skip waiting for training completion")
    p.add_argument("--skip-cheap", action="store_true", help="Skip cheap columns (already evaluated)")
    p.add_argument("--skip-superglue", action="store_true", help="Skip SuperGLUE")
    p.add_argument("--skip-aoa", action="store_true", help="Skip AoA")
    p.add_argument("--max-workers", type=int, default=6)
    args = p.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Wait for training if needed
    if not args.no_wait and not check_training_done():
        print(json.dumps({"event": "waiting_for_training", "run_dir": str(RUN_DIR_100M), "utc": now()}), flush=True)
        if not wait_for_training():
            print(json.dumps({"event": "training_timeout", "status": "error"}), flush=True)
            return

    if not check_training_done():
        print(json.dumps({"event": "training_not_done", "status": "error", "model_path": str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'))}), flush=True)
        return

    # Plan parallel execution
    jobs: list[tuple[str, int]] = []  # (column, gpu)

    if not args.skip_cheap:
        # Cheap columns on GPU1 (fast, ~3-5 min each)
        for col in CHEAP_COLS:
            jobs.append((col, 1))

    if not args.skip_superglue:
        # SuperGLUE on GPU0 (slow, 7 subtasks sequential ~60-90 min)
        jobs.append(("SuperGLUE", 0))

    # Run cheap + SuperGLUE in parallel
    results: list[dict[str, Any]] = []
    if jobs:
        print(json.dumps({"event": "parallel_launch", "jobs": len(jobs), "columns": [j[0] for j in jobs], "utc": now()}), flush=True)
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(run_eval_column, col, gpu): (col, gpu) for col, gpu in jobs}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    col, gpu = futs[fut]
                    results.append({"col": col, "gpu": gpu, "status": "exception", "error": str(e)})

    # AoA (needs sequential checkpoint access, run on GPU0)
    if not args.skip_aoa:
        aoa_rec = run_eval_column("AoA", 0)
        results.append(aoa_rec)

    # Extract all scores
    scores = {}
    for col in CHEAP_COLS + ["SuperGLUE", "AoA"]:
        scores[col] = extract_score(col)

    # Compute GlobalPIQA mean
    gp_par = scores.get("GlobalPIQA_parallel")
    gp_nonpar = scores.get("GlobalPIQA_nonparallel")
    gp_mean = mean([gp_par, gp_nonpar]) if gp_par is not None and gp_nonpar is not None else None

    # Compute cheap7
    cheap7_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
    cheap7_vals = [scores.get(c) for c in cheap7_cols] + [gp_mean]
    cheap7 = mean([v for v in cheap7_vals if v is not None]) if all(v is not None for v in cheap7_vals) else None

    # Compute Overall (9 columns)
    overall_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "Reading", "AoA"]
    overall_vals = [scores.get(c) for c in overall_cols] + [gp_mean]
    overall = mean([v for v in overall_vals if v is not None]) if all(v is not None for v in overall_vals) else None

    synthesis = {
        "status": "SCALE1P75_100M_FULL_EVAL",
        "created_utc": now(),
        "model": {
            "run_dir": str(RUN_DIR_100M),
            "endpoint": ENDPOINT,
            "architecture": "AdapterDebertaV2ForMaskedLM",
            "adapter_bottleneck": 128,
            "adapter_scale": 1.75,
        },
        "scores": {
            "BLiMP": scores.get("BLiMP"),
            "Supplement": scores.get("Supplement"),
            "EWoK": scores.get("EWoK"),
            "Entity": scores.get("Entity"),
            "COMPS": scores.get("COMPS"),
            "SuperGLUE": scores.get("SuperGLUE"),
            "GlobalPIQA": gp_mean,
            "GlobalPIQA_parallel": gp_par,
            "GlobalPIQA_nonparallel": gp_nonpar,
            "Reading": scores.get("Reading"),
            "AoA": scores.get("AoA"),
        },
        "cheap7": cheap7,
        "overall": overall,
        "frontier_comparison": {
            "leader_overall": 41.80,
            "this_overall": overall,
            "crosses_leader": overall is not None and overall >= 41.80,
        },
        "jobs": results,
    }

    out_path = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_100M_full_eval/scale1p75_100M_full_eval_synthesis.json')
    out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": synthesis["status"],
        "overall": overall,
        "cheap7": cheap7,
        "superglue": scores.get("SuperGLUE"),
        "aoa": scores.get("AoA"),
        "crosses_leader": synthesis["frontier_comparison"]["crosses_leader"],
        "synthesis": str(out_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
