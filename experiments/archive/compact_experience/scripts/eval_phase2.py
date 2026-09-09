#!/usr/bin/env python3
"""research Phase 2 evaluation: evaluate both Phase 2 SOTA arms + baselines.

Evaluates:
- phase2_official: DeBERTa-v2 12×384, LAMB, 40k tok, official data
- phase2_mix25: DeBERTa-v2 12×384, LAMB, 40k tok, 25% aligned mix
- initial_model_baseline: inherited DeBERTa-v2 8×480, AdamW, 16k tok (40.7028 reference)
- leader_target: target scores from the leaderboard (41.8 Overall)

Uses the same proven evaluation pattern from research/research/research.
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
from typing import Any, Dict, Iterable, Optional, Tuple

ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
EVAL_REPO = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/phase2_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/22_phase2_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "phase2_eval_summary.json"

# Run name -> model path
RUNS: Dict[str, pathlib.Path] = {
    "phase2_official": RUN_BASE / "phase2_official_40k_12x384_seed43200/hf_model/chck_100M",
    "phase2_mix25": RUN_BASE / "phase2_mix25_40k_12x384_seed43200/hf_model/chck_100M",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M",
}

# Leader scores for comparison
LEADER_SCORES = {
    "BLiMP": 67.20, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
    "COMPS": 53.57, "GlobalPIQA": 39.67, "SuperGLUE": 69.79,
    "Reading": 5.42, "AoA": 0.0, "Overall": 41.80,
}

COLUMNS = [
    # (display_name, task_module, data_path, batch_size)
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/globalpiqa_parallel_fast", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/globalpiqa_nonparallel_fast", 128),
    ("Reading", "reading", "evaluation_data/fast_eval/reading_fast", 32),
]

TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean",
]


def model_path_for(target: str) -> pathlib.Path:
    return RUNS[target]


def run_evaluation(model_path: pathlib.Path, column: str, task: str,
                   data_path: str, batch_size: int, out_dir: pathlib.Path,
                   gpu: int) -> Dict[str, Any]:
    """Run a single evaluation column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_data = data_path.replace("/", "_")
    task_out = out_dir / safe_data
    task_out.mkdir(parents=True, exist_ok=True)
    log_file = task_out / "output.log"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TRANSFORMERS_CACHE"] = "/tmp/hf_cache_eval"
    env["HF_HOME"] = "/tmp/hf_home_eval"

    if task == "reading":
        cmd = [sys.executable, "-m", "evaluation_pipeline.reading.run",
               "--model", str(model_path), "--data_path", data_path,
               "--batch_size", str(batch_size)]
    else:
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model", str(model_path), "--task", task, "--data_path", data_path,
               "--batch_size", str(batch_size)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EVAL_REPO),
                           env=env, timeout=600)
    log_file.write_text(result.stderr + "\n" + result.stdout, encoding="utf-8")
    return {"column": column, "task": task, "data_path": data_path,
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr_tail": result.stderr[-500:] if result.stderr else ""}


def parse_score(text: str, column: str) -> Optional[float]:
    """Extract score from evaluation output."""
    if "reading" in column.lower() or column == "Reading":
        # Reading returns eye + self_paced
        eye = None; sp = None
        m = re.search(r"eye[_\s]*tracking[^:]*:\s*([0-9.]+)", text, re.IGNORECASE)
        if m: eye = float(m.group(1))
        m = re.search(r"self[_\s]*paced[^:]*:\s*([0-9.]+)", text, re.IGNORECASE)
        if m: sp = float(m.group(1))
        if eye is not None and sp is not None:
            return (eye, sp)
        # Fallback: single number
        m = re.search(r"AVERAGE[^:]*:\s*([0-9.]+)", text)
        if m: return float(m.group(1))
        return None

    # Standard: look for "AVERAGE ACCURACY" or last numeric line
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
    return None


def eval_target(target: str, gpu: int, columns: list, active_columns: list) -> Dict[str, Any]:
    """Evaluate all columns for one target."""
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = OUT_ROOT / "eval_outputs" / target
    scores = {}

    for col_name, task, data_path, batch_size in columns:
        if col_name not in active_columns:
            continue
        print(json.dumps({"event": "eval_start", "target": target, "column": col_name}), flush=True)
        result = run_evaluation(model_path, col_name, task, data_path, batch_size, out_dir, gpu)

        if result["returncode"] == 0:
            output_text = result["stdout"]
            score = parse_score(output_text, col_name)
            if isinstance(score, tuple):
                scores[f"{col_name}_eye"] = score[0]
                scores[f"{col_name}_self_paced"] = score[1]
                scores[col_name] = round((score[0] + score[1]) / 2, 3)
            elif score is not None:
                scores[col_name] = round(score, 2) if score > 1 else round(score * 100, 2)
            else:
                scores[col_name] = None
        else:
            scores[col_name] = None
        print(json.dumps({"event": "eval_done", "target": target, "column": col_name,
                          "score": scores.get(col_name), "rc": result["returncode"]}), flush=True)

    # Compute derived scores
    if scores.get("GlobalPIQA_parallel") is not None and scores.get("GlobalPIQA_nonparallel") is not None:
        scores["GlobalPIQA_mean"] = round(
            (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2, 3)

    # equal-7 mean (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_mean, Reading)
    eq7_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    eq7_vals = [scores.get(k) for k in eq7_keys if scores.get(k) is not None]
    if len(eq7_vals) == 7:
        scores["equal7_mean"] = round(sum(eq7_vals) / 7, 4)

    return scores


def build_summary(targets: list[str], all_scores: Dict[str, Dict]) -> Dict:
    """Build comparison summary with contrasts."""
    table = {t: all_scores[t] for t in targets if t in all_scores}

    # Contrasts vs baseline
    contrasts = {}
    baseline = table.get("initial_model_baseline", {})
    for t in targets:
        if t == "initial_model_baseline" or t not in table:
            continue
        delta = {}
        for k in TABLE_KEYS:
            bv = baseline.get(k)
            tv = table[t].get(k)
            if bv is not None and tv is not None:
                delta[k] = round(tv - bv, 4)
        contrasts[f"{t}_minus_initial_model_studies_baseline"] = delta

    # Contrast: phase2_mix25 vs phase2_official
    if "phase2_mix25" in table and "phase2_official" in table:
        delta = {}
        for k in TABLE_KEYS:
            mv = table["phase2_mix25"].get(k)
            ov = table["phase2_official"].get(k)
            if mv is not None and ov is not None:
                delta[k] = round(mv - ov, 4)
        contrasts["phase2_mix25_minus_phase2_official"] = delta

    return {
        "status": "PHASE2_EVAL_DONE",
        "table": table,
        "contrasts": contrasts,
        "leader_target": LEADER_SCORES,
        "best_by_equal7": max(
            [(t, s.get("equal7_mean", -999)) for t, s in table.items()],
            key=lambda x: x[1]
        ) if table else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+",
                        default=["phase2_official", "phase2_mix25", "initial_model_baseline"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--columns", nargs="+", default=None,
                        help="Specific columns to evaluate (default: all)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    active_columns = args.columns or [c[0] for c in COLUMNS]

    all_scores = {}
    for target in args.targets:
        if target not in RUNS:
            print(f"WARNING: unknown target {target}, skipping")
            continue
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        scores = eval_target(target, args.gpu, COLUMNS, active_columns)
        all_scores[target] = scores
        print(json.dumps({"event": "target_done", "target": target, "scores": scores}), flush=True)

    summary = build_summary(args.targets, all_scores)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write note
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# research — Phase 2 SOTA evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "## Table",
        "",
        "| target | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in args.targets:
        if t not in all_scores:
            continue
        s = all_scores[t]
        lines.append(f"| {t} | {s.get('BLiMP','-')} | {s.get('Supplement','-')} | "
                     f"{s.get('EWoK','-')} | {s.get('Entity','-')} | {s.get('COMPS','-')} | "
                     f"{s.get('GlobalPIQA_mean','-')} | {s.get('Reading','-')} | {s.get('equal7_mean','-')} |")
    lines.append("")
    lines.append("## Leader target: Overall 41.8")
    lines.append(f"Leader scores: {LEADER_SCORES}")
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
