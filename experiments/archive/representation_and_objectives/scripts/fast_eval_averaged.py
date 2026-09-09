#!/usr/bin/env python3
"""research: Fast zero-shot evaluation of averaged models on ALL BabyLM tasks.

Runs BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading on the
cross-seed averaged and tail-averaged reinvest models through the official
babylm-eval pipeline. This gives the full fast surface to decide whether
averaging produces a robust above-leader endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

USER_ROOT = pathlib.Path(".").resolve()
SESS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
OUT = SESS / "data" / "fast_eval_averaged"

# Babylm-eval strict directory (inherited from INITIAL_MODEL_STUDIES)
STRICT = USER_ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
PYTHON_EXE = sys.executable

# Tasks to run (same as research fast eval)
TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/full_eval/ewok_filtered", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("Entity_full", "entity_tracking", "evaluation_data/full_eval/entity_tracking", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COL = {c: (t, d, b) for c, t, d, b in TASKS}

# Averaged model paths (created by weight_avg_and_ewok_margins.py)
AVG_BASE = SESS / "data" / "weight_avg_and_margins" / "models"
TARGETS = {
    "cross_seed_avg": AVG_BASE / "cross_seed_avg_100M",
    "tail_avg_022": AVG_BASE / "tail_avg_seed43022_90_100M",
    "tail_avg_122": AVG_BASE / "tail_avg_seed43122_90_100M",
}

# Reference scores
REF = {
    "visible_leader_overall": 41.8,
    "reinvest_43022_overall": 42.0331,
    "reinvest_43122_overall": 41.2482,
    "reinvest_mean_overall": 41.6407,
}


def make_env(gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    tmp = SESS / "staging" / "hf_cache_step046"
    for key, sub in [("HF_HOME", "home"), ("HF_HUB_CACHE", "hub"),
                     ("HF_DATASETS_CACHE", "datasets"), ("TRANSFORMERS_CACHE", "transformers"),
                     ("HF_MODULES_CACHE", "modules"), ("TMPDIR", "tmp"),
                     ("NLTK_DATA", "nltk")]:
        env[key] = str(tmp / sub)
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def parse_score(text: str) -> Optional[float]:
    for pat in [
        r"### AVERAGE [A-Z_ '\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if -5.0 <= val <= 105.0:
                return val
    return None


def parse_reading(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"),
                       ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def latest_file(root: pathlib.Path, pattern: str) -> Optional[pathlib.Path]:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    for pat in ["best_temperature_report.txt", "*.txt"]:
        for p in sorted(task_out.rglob(pat), reverse=True):
            try:
                val = parse_score(p.read_text(encoding="utf-8", errors="replace"))
                if val is not None:
                    return val
            except Exception:
                pass
    return None


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path,
            timeout: int = 2400) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ "
                + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                          capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
        f.write(f"\n[returncode={proc.returncode}]\n")
    return proc


def eval_sentence(model_path: pathlib.Path, label: str, column: str,
                  env: Dict[str, str]) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT / "eval_outputs" / label / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{label}_{column}.log"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data_path,
        "--revision_name", f"step046_{label}_{column}",
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    proc = run_cmd(cmd, env, log, timeout=2400)
    score = read_report_score(task_out)
    rec: Dict[str, Any] = {
        "column": column, "task": task, "score": score,
        "returncode": proc.returncode, "elapsed_sec": round(time.time() - t0, 1),
    }
    if proc.returncode != 0:
        rec["error_tail"] = proc.stderr[-500:] if proc.stderr else ""
    return rec


def eval_reading(model_path: pathlib.Path, label: str, env: Dict[str, str]) -> Dict[str, Any]:
    task_out = OUT / "eval_outputs" / label / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{label}_Reading.log"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", f"step046_{label}_Reading",
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    proc = run_cmd(cmd, env, log, timeout=2400)
    scores = parse_reading(proc.stdout) if proc.stdout else {}
    return {"column": "Reading", "scores": scores,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - t0, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env(args.gpu)
    results = {"status": "FAST_EVAL_AVERAGED", "targets": {}}

    for label, model_path in TARGETS.items():
        # Wait up to 10 min for the model to appear (created by parallel margin script)
        waited = 0
        while not (model_path / "model.safetensors").exists() and waited < 600:
            time.sleep(10)
            waited += 10
            if waited % 60 == 0:
                print(f"  Waiting for {label} model... ({waited}s)", flush=True)
        if not (model_path / "model.safetensors").exists():
            print(f"SKIP {label}: model not created after {waited}s at {model_path}", flush=True)
            results["targets"][label] = {"skipped": True, "reason": "model_not_found"}
            continue

        print(f"\n=== Evaluating {label} ===", flush=True)
        t0 = time.time()
        scores = {}

        # Run sentence zero-shot tasks
        for col in TASK_BY_COL:
            rec = eval_sentence(model_path, label, col, env)
            scores[col] = rec
            s = rec.get("score")
            print(f"  {col}: {s}", flush=True)

        # Reading
        reading_rec = eval_reading(model_path, label, env)
        scores["Reading"] = reading_rec
        print(f"  Reading: {reading_rec.get('scores', {}).get('Reading')}", flush=True)

        # Compute fast surface
        col_vals = {}
        for col in ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS"]:
            col_vals[col] = scores.get(col, {}).get("score")
        gp = scores.get("GlobalPIQA_parallel", {}).get("score")
        gn = scores.get("GlobalPIQA_nonparallel", {}).get("score")
        if gp is not None and gn is not None:
            col_vals["GlobalPIQA_mean"] = (gp + gn) / 2
        rdg = scores.get("Reading", {}).get("scores", {}).get("Reading")
        col_vals["Reading"] = rdg

        # equal7 mean (without AoA and SuperGLUE)
        val_list = [v for v in col_vals.values() if v is not None]
        equal7 = sum(val_list) / len(val_list) if val_list else None

        elapsed = time.time() - t0
        summary = {
            "elapsed_sec": round(elapsed, 1),
            "column_scores": col_vals,
            "equal7_mean": round(equal7, 4) if equal7 else None,
            "all_task_records": scores,
        }
        results["targets"][label] = summary
        print(f"  equal7_mean: {equal7:.4f}" if equal7 else "  equal7_mean: None", flush=True)

    # Save
    out_json = OUT / "fast_eval_averaged_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print compact summary
    compact = {"status": results["status"]}
    for label, data in results["targets"].items():
        if isinstance(data, dict) and "equal7_mean" in data:
            compact[label] = {
                "equal7_mean": data["equal7_mean"],
                "columns": data.get("column_scores", {}),
            }
    print(json.dumps(compact, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
