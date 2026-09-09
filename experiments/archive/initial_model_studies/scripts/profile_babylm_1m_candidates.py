#!/usr/bin/env python3
"""Run official BabyLM fast/local profile for selected 1M candidate checkpoints.

The script intentionally uses official repository modules/commands through subprocess
so outputs have the same structure as prior research evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

STUDY = pathlib.Path("experiments/archive/initial_model_studies")
STRICT_DIR = STUDY / "repos/babylm-eval/strict"
AI_RUNS = STUDY / "training/runs"
HF_MODULES_CACHE = STUDY / "training/hf_modules_cache"

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]

RUNS = {
    "dense6x256": "babylm_compare_dense6x256_1M",
    "sparse4x256": "babylm_compare_sparse4x256_1M",
    "morphside4x256": "babylm_compare_morphside4x256_1M_fix",
    "dense5x288": "babylm_compare_dense5x288_1M",
    "dense6x384": "babylm_compare_dense6x384_1M",
    "dense_untied4x256": "babylm_compare_dense_untied4x256_1M_control",
    "memory4x256_m64": "babylm_compare_memory4x256_m64_1M",
    "dense_untied4x256_s43": "babylm_compare_dense_untied4x256_1M_seed43",
    "memory4x256_m64_s43": "babylm_compare_memory4x256_m64_1M_seed43",
}


def run_cmd(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> None:
    print("CMD", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_avg(report: pathlib.Path) -> float | None:
    if not report.exists():
        return None
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    return float(m.group(1)) if m else None


def read_reading(report: pathlib.Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if not report.exists():
        return {"eye_tracking": None, "self_paced": None}
    txt = report.read_text(encoding="utf-8", errors="replace")
    for label, key in [("EYE TRACKING SCORE", "eye_tracking"), ("SELF-PACED READING SCORE", "self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        out[key] = float(m.group(1)) if m else None
    return out


def profile_run(alias: str, run_id: str, env: dict[str, str]) -> dict[str, Any]:
    run_dir = AI_RUNS / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_profile_1M").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"alias": alias, "run_id": run_id, "model_path": str(model_path), "outdir": str(outdir), "scores": {}, "reports": {}}
    for task_name, task, data_path in TASKS:
        dp = STRICT_DIR / data_path
        if not dp.exists():
            result["scores"][task_name] = None
            result["reports"][task_name] = f"MISSING_DATA:{dp}"
            print(f"SKIP {alias} {task_name}: missing {dp}", flush=True)
            continue
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--revision_name", "chck_1M",
            "--batch_size", "64",
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        subdir = outdir / "hf_model" / "chck_1M" / "zero_shot" / "causal" / task / task_name
        report = subdir / "best_temperature_report.txt"
        result["scores"][task_name] = read_avg(report)
        result["reports"][task_name] = str(report)
    # Reading
    reading_data = STRICT_DIR / "evaluation_data/fast_eval/reading/reading_data.csv"
    if reading_data.exists():
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
            "--revision_name", "chck_1M",
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "causal" / "reading" / "report.txt"
        result["scores"].update({f"reading_{k}": v for k, v in read_reading(rreport).items()})
        result["reports"]["reading"] = str(rreport)
    else:
        result["scores"].update({"reading_eye_tracking": None, "reading_self_paced": None})
        result["reports"]["reading"] = f"MISSING_DATA:{reading_data}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(RUNS.keys()), choices=list(RUNS.keys()))
    args = ap.parse_args()
    HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    all_results = []
    for alias in args.runs:
        all_results.append(profile_run(alias, RUNS[alias], env))
    out = STUDY / "data/profile_dense_sparse_1m.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out)
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
