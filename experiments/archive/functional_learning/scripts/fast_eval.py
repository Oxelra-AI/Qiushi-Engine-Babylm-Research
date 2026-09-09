#!/usr/bin/env python3
"""research fast evaluation for context-credit arms.

Runs BLiMP, BLiMP-Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading
on a trust_remote_code model checkpoint.  Reports equal7 (equal-weight average
of the seven columns) and per-column scores.  This is a research screen, not
a full official submission package — it omits SuperGLUE and AoA.

Usage:
  python fast_eval.py --model_path <hf_dir> --gpu 0 --tag standard
  python fast_eval.py --model_path <hf_dir> --gpu 1 --tag carrier_residual
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
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
OUT_DEFAULT = _public_path('experiments/archive/functional_learning/data/eval')

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 32),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 32),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 16),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 32),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 32),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 32),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 32),
]


def parse_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    for line in reversed(text.splitlines()):
        s = line.strip()
        m2 = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m2:
            val = float(m2.group(1))
            if -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def parse_reading(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def setup_env(out_root: pathlib.Path, tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    nltk_path = _public_path('experiments/archive/initial_model_studies/data/nltk_data')
    if nltk_path.exists():
        env["NLTK_DATA"] = str(nltk_path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 1800) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')), env=env, capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(p.stdout[-5000:] if len(p.stdout) > 5000 else p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr[-3000:] if len(p.stderr) > 3000 else p.stderr)
        f.write(f"\n[returncode={p.returncode}]\n")
    return p


def eval_sentence(model_path: str, col: str, task: str, data: str, batch: int,
                  tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task_out = out_root / "outputs" / tag / col
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / tag / f"{col}.log"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(pathlib.Path(model_path).resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data,
        "--revision_name", f"step025_{tag}_{col}",
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    try:
        p = run_cmd(cmd, env, log)
    except subprocess.TimeoutExpired:
        return {"column": col, "score": None, "error": "timeout"}

    score = None
    for txt_file in sorted(task_out.rglob("best_temperature_report.txt")):
        score = parse_score(txt_file.read_text(errors="replace"))
        if score is not None:
            break
    if score is None:
        for txt_file in sorted(task_out.rglob("*.txt")):
            score = parse_score(txt_file.read_text(errors="replace"))
            if score is not None:
                break

    return {
        "column": col,
        "score": score,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def eval_reading(model_path: str, tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task_out = out_root / "outputs" / tag / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / tag / "Reading.log"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(pathlib.Path(model_path).resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", f"step025_{tag}_reading",
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    try:
        p = run_cmd(cmd, env, log)
    except subprocess.TimeoutExpired:
        return {"Reading": None, "error": "timeout"}

    scores = {}
    for rpt in sorted(task_out.rglob("report.txt")):
        scores = parse_reading(rpt.read_text(errors="replace"))
        if scores:
            break

    return {
        **scores,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--out_root", default=str(OUT_DEFAULT))
    parser.add_argument("--skip_reading", action="store_true")
    args = parser.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    env = setup_env(out_root, args.tag, args.gpu)

    results: Dict[str, Any] = {"model_path": args.model_path, "tag": args.tag, "gpu": args.gpu}
    scores: Dict[str, Optional[float]] = {}

    for col, task, data, batch in TASKS:
        print(f"[{args.tag}] Evaluating {col}...", flush=True)
        r = eval_sentence(args.model_path, col, task, data, batch, args.tag, env, out_root)
        scores[col] = r.get("score")
        results[col] = r
        print(f"  {col} = {r.get('score')} ({r.get('elapsed_sec', '?')}s)", flush=True)

    # GlobalPIQA mean
    gp = scores.get("GlobalPIQA_parallel")
    gnp = scores.get("GlobalPIQA_nonparallel")
    if gp is not None and gnp is not None:
        scores["GlobalPIQA_mean"] = (gp + gnp) / 2.0
    else:
        scores["GlobalPIQA_mean"] = gp or gnp

    # Reading
    if not args.skip_reading:
        print(f"[{args.tag}] Evaluating Reading...", flush=True)
        rr = eval_reading(args.model_path, args.tag, env, out_root)
        scores["Reading"] = rr.get("Reading")
        scores["Reading_eye"] = rr.get("Reading_eye")
        scores["Reading_self_paced"] = rr.get("Reading_self_paced")
        results["Reading"] = rr
        print(f"  Reading = {rr.get('Reading')} ({rr.get('elapsed_sec', '?')}s)", flush=True)

    # equal7
    eq7_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    eq7_vals = [scores.get(k) for k in eq7_keys]
    valid = [v for v in eq7_vals if v is not None]
    if valid:
        scores["equal7"] = sum(valid) / len(valid)
        scores["equal7_n_valid"] = len(valid)

    results["scores"] = scores

    # Reference comparison
    ref = {
        "coherent86_alpha075": {
            "BLiMP": 67.20, "Supplement": 65.14, "EWoK": 55.35,
            "Entity": 27.92, "COMPS": 51.65, "GlobalPIQA_mean": 37.78,
            "Reading": 3.81,
            "note": "approximate from REPRESENTATION_FRONTIER_STUDIES records; exact values may differ"
        }
    }
    results["reference"] = ref

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.tag} evaluation summary")
    print(f"{'='*60}")
    for k in eq7_keys:
        v = scores.get(k)
        print(f"  {k:25s}: {v if v is not None else 'MISSING':>8}")
    print(f"  {'equal7':25s}: {scores.get('equal7', 'N/A'):>8}")
    print(f"{'='*60}\n")

    out_file = out_root / f"step025_{args.tag}_eval.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"status": "eval_complete", "tag": args.tag, "scores": scores,
                       "out": str(out_file)}))


if __name__ == "__main__":
    main()
