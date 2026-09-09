#!/usr/bin/env python3
"""Profile multiple checkpoint revisions from one BabyLM local HF model directory."""
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
HF_MODULES_CACHE = STUDY / "training/hf_modules_cache"

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]


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
    if not report.exists():
        return {"eye_tracking": None, "self_paced": None}
    txt = report.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float | None] = {}
    for label, key in [("EYE TRACKING SCORE", "eye_tracking"), ("SELF-PACED READING SCORE", "self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        out[key] = float(m.group(1)) if m else None
    return out


def profile_revision(run_dir: pathlib.Path, revision: str, outdir: pathlib.Path, env: dict[str, str]) -> dict[str, Any]:
    # The BabyLM evaluator's `revision_name` is used for output labeling; for local
    # checkpoint trajectories we must pass the actual checkpoint directory as the
    # model path. Otherwise every revision silently profiles the root final model.
    model_path = (run_dir / "hf_model" / revision).resolve()
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "revision": revision,
        "outdir": str(outdir),
        "scores": {},
        "reports": {},
    }
    for task_name, task, data_path in TASKS:
        dp = STRICT_DIR / data_path
        if not dp.exists():
            result["scores"][task_name] = None
            result["reports"][task_name] = f"MISSING_DATA:{dp}"
            continue
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--revision_name", revision,
            "--batch_size", "64",
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        report = outdir / model_path.name / revision / "zero_shot" / "causal" / task / task_name / "best_temperature_report.txt"
        result["scores"][task_name] = read_avg(report)
        result["reports"][task_name] = str(report)
    reading_data = STRICT_DIR / "evaluation_data/fast_eval/reading/reading_data.csv"
    if reading_data.exists():
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
            "--revision_name", revision,
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        rreport = outdir / model_path.name / revision / "zero_shot" / "causal" / "reading" / "report.txt"
        result["scores"].update({f"reading_{k}": v for k, v in read_reading(rreport).items()})
        result["reports"]["reading"] = str(rreport)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    outdir = (run_dir / "eval_results_profile_curve").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for rev in args.revisions:
        if not (run_dir / "hf_model" / rev).exists():
            raise FileNotFoundError(f"checkpoint directory missing: {run_dir / 'hf_model' / rev}")
    results = [profile_revision(run_dir, rev, outdir, env) for rev in args.revisions]
    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out_json)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
