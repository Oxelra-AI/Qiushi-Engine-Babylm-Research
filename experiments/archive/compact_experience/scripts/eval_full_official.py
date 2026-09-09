#!/usr/bin/env python3
"""research: Full nine-column official-style evaluation for both arms.

Evaluates official_only_16k_seed43022 and qwen_aligned_16k_seed43022.
Runs all nine official columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA,
(Super)GLUE, Reading, AoA.

The AoA evaluation requires chck_1M through chck_9M plus chck_10M..chck_100M.
With checkpoint_words=1000000, all required checkpoints exist.
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
import subprocess
import sys
import time
from typing import Any, Dict, Optional

ROOT = _public_path('experiments/archive/compact_experience')  # 
STRICT = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
sys.path.insert(0, str(STRICT.resolve()))

OUT_DIR = _public_path('experiments/archive/compact_experience/data/full_eval')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate
TARGETS = {
    "official_only": {
        "description": "Official-only baseline: DeBERTa-v2 8×480, WWM, 16k tok, 100M exposure",
        "family": "debertav2_8x480_16k",
        "run_dir": str(_public_path('experiments/archive/compact_experience/training/runs/official_only_16k_seed43022')),
        "model_root": str(_public_path('experiments/archive/compact_experience/training/runs/official_only_16k_seed43022/hf_model')),
    },
    "qwen_aligned": {
        "description": "Qwen-aligned: 25% legal Qwen rewrites + 75% official, same backbone",
        "family": "debertav2_8x480_16k",
        "run_dir": str(_public_path('experiments/archive/compact_experience/training/runs/qwen_aligned_16k_seed43022')),
        "model_root": str(_public_path('experiments/archive/compact_experience/training/runs/qwen_aligned_16k_seed43022/hf_model')),
    },
}

# Zero-shot tasks
ZERO_SHOT_TASKS = [
    ("BLiMP", "blimp", "evaluation_data/full_eval/blimp_filtered"),
    ("Supplement", "blimp", "evaluation_data/full_eval/supplement_filtered"),
    ("EWoK", "ewok", "evaluation_data/full_eval/ewok"),
    ("Entity", "entity_tracking", "evaluation_data/full_eval/entity_tracking"),
    ("COMPS", "comps", "evaluation_data/full_eval/comps"),
    ("GlobalPIQA_parallel", "vqa", "evaluation_data/full_eval/global_piqa/parallel"),
    ("GlobalPIQA_nonparallel", "vqa", "evaluation_data/full_eval/global_piqa/non_parallel"),
]

# SuperGLUE tasks
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]

# Reading
READING_DATA = "evaluation_data/full_eval/reading/reading_data.csv"

# AoA
AOA_WORD_PATH = "evaluation_data/full_eval/aoa/cdi_childes.json"
AOA_CDI_HUMAN = "evaluation_data/full_eval/aoa/cdi_human.csv"


def run_zero_shot(model_path: str, task: str, data_path: str, gpu: int) -> Optional[float]:
    """Run a single zero-shot evaluation task."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", model_path,
        "--backend", "mlm",
        "--task", task,
        "--data_path", str(STRICT / data_path),
        "--save_predictions",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STRICT), env=env, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR in {task}: {result.stderr[:200]}")
        return None
    
    # Parse score from stdout
    for line in result.stdout.strip().split("\n"):
        if "AVERAGE" in line:
            # Next line has the score
            continue
        try:
            score = float(line.strip())
            return score
        except ValueError:
            continue
    
    # Try to read from output files
    return None


def run_reading(model_path: str, gpu: int) -> Optional[Dict[str, float]]:
    """Run reading evaluation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", model_path,
        "--backend", "mlm",
        "--data_path", str(STRICT / READING_DATA),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STRICT), env=env, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR in Reading: {result.stderr[:200]}")
        return None
    
    scores = {}
    for line in result.stdout.strip().split("\n"):
        if "EYE-TRACKING SCORE" in line:
            try:
                scores["Reading_eye"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "SELF-PACED READING SCORE" in line:
            try:
                scores["Reading_self_paced"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    
    if "Reading_eye" in scores and "Reading_self_paced" in scores:
        scores["Reading"] = (scores["Reading_eye"] + scores["Reading_self_paced"]) / 2
    return scores


def run_superglue(model_path: str, gpu: int) -> Optional[float]:
    """Run all SuperGLUE finetuning tasks and return mean accuracy."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    task_scores = []
    for task in SUPERGLUE_TASKS:
        data_dir = STRICT / "evaluation_data" / "full_eval" / "glue_filtered" / task
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping {task}")
            continue
        
        cmd = [
            sys.executable, "-m", "evaluation_pipeline.finetune.run",
            "--model_path", model_path,
            "--task", task,
            "--train_data", str(data_dir / "train.jsonl"),
            "--valid_data", str(data_dir / "valid.jsonl"),
            "--predict_data", str(data_dir / "valid.jsonl"),
            "--learning_rate", "3e-5",
            "--batch_size", "32" if task not in ("mnli", "qqp") else "16",
            "--num_epochs", "30" if task == "wsc" else "10",
            "--sequence_length", "512",
            "--results_dir", "results",
            "--save",
            "--save_dir", "models",
            "--metrics", "accuracy",
            "--metric_for_valid", "accuracy",
            "--seed", "42",
            "--verbose",
            "--padding_side", "left",
            "--take_final",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STRICT), env=env, timeout=1800)
        if result.returncode != 0:
            print(f"  ERROR in SuperGLUE/{task}: {result.stderr[:200]}")
            continue
        
        # Parse accuracy from stdout
        for line in result.stdout.strip().split("\n"):
            if "accuracy" in line.lower() and ":" in line:
                try:
                    score = float(line.split(":")[-1].strip().rstrip("%"))
                    if score <= 1.0:
                        score *= 100  # Convert from 0-1 to percentage
                    task_scores.append({"task": task, "accuracy": score})
                    break
                except ValueError:
                    continue
    
    if task_scores:
        mean_acc = sum(t["accuracy"] for t in task_scores) / len(task_scores)
        return mean_acc
    return None


def run_aoa(model_root: str, gpu: int) -> Optional[float]:
    """Run AoA evaluation using checkpoint ladder."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Check if required checkpoints exist
    model_root_path = pathlib.Path(model_root)
    required_checkpoints = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 11)]
    
    missing = [c for c in required_checkpoints if not (model_root_path / c).exists()]
    if missing:
        print(f"  WARNING: Missing AoA checkpoints: {missing[:5]}...")
        return None
    
    # Use the INITIAL_MODEL_STUDIES AoA runner pattern
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.AoA_word.run",
        "--model_name", str(model_root_path / "chck_100M"),
        "--backend", "mlm",
        "--track_name", "strict-small",
        "--word_path", str(STRICT / AOA_WORD_PATH),
        "--output_dir", "results",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STRICT), env=env, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR in AoA: {result.stderr[:200]}")
        return None
    
    # Parse AoA score
    for line in result.stdout.strip().split("\n"):
        if "aoa" in line.lower() or "AoA" in line:
            try:
                score = float(line.split(":")[-1].strip())
                return score
            except ValueError:
                continue
    return None


def evaluate_target(target_name: str, target_info: dict, gpu: int) -> dict:
    """Run full evaluation for one target."""
    model_path = str(pathlib.Path(target_info["model_root"]) / "chck_100M")
    model_root = target_info["model_root"]
    
    result = {
        "target": target_name,
        "description": target_info["description"],
        "model_path": model_path,
        "model_root": model_root,
        "gpu": gpu,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scores": {},
    }
    
    # Zero-shot tasks
    for col_name, task, data_path in ZERO_SHOT_TASKS:
        print(f"  [{target_name}] {col_name}...", flush=True)
        score = run_zero_shot(model_path, task, data_path, gpu)
        if score is not None:
            result["scores"][col_name] = score
            print(f"    → {score:.2f}")
    
    # GlobalPIQA mean
    if "GlobalPIQA_parallel" in result["scores"] and "GlobalPIQA_nonparallel" in result["scores"]:
        result["scores"]["GlobalPIQA"] = (
            result["scores"]["GlobalPIQA_parallel"] + result["scores"]["GlobalPIQA_nonparallel"]
        ) / 2
    
    # Reading
    print(f"  [{target_name}] Reading...", flush=True)
    reading = run_reading(model_path, gpu)
    if reading:
        result["scores"].update(reading)
        print(f"    → {reading.get('Reading', 'N/A'):.2f}")
    
    # SuperGLUE
    print(f"  [{target_name}] SuperGLUE (7 tasks)...", flush=True)
    superglue = run_superglue(model_path, gpu)
    if superglue is not None:
        result["scores"]["SuperGLUE"] = superglue
        print(f"    → {superglue:.2f}")
    
    # AoA
    print(f"  [{target_name}] AoA...", flush=True)
    aoa = run_aoa(model_root, gpu)
    if aoa is not None:
        result["scores"]["AoA"] = aoa
        print(f"    → {aoa:.4f}")
    else:
        result["scores"]["AoA"] = 0.0
        result["aoa_status"] = "checkpoint_ladder_available"
    
    # Compute Overall
    overall_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", 
                    "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
    overall_values = [result["scores"].get(c, 0.0) for c in overall_cols]
    result["scores"]["Overall"] = sum(overall_values) / len(overall_values)
    
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result


def main():
    raise SystemExit("RETired unsafe research evaluator: it predates research AoA scaling and may compute Overall from raw AoA. Use full_overall_eval_runner.py / full_eval_runner.py, which import babylm_official_scoring.py.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=list(TARGETS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    print(f"research Full Official Evaluation")
    print(f"Targets: {args.targets}")
    
    all_results = {}
    for i, target_name in enumerate(args.targets):
        if target_name not in TARGETS:
            print(f"Unknown target: {target_name}")
            continue
        
        gpu = (args.gpu + i) % 2  # Alternate GPUs
        print(f"\n{'='*60}")
        print(f"Evaluating: {target_name} on GPU {gpu}")
        print(f"{'='*60}")
        
        result = evaluate_target(target_name, TARGETS[target_name], gpu)
        all_results[target_name] = result
        
        # Save per-target
        per_target_dir = _public_path('experiments/archive/compact_experience/data/full_eval/per_target')
        per_target_dir.mkdir(parents=True, exist_ok=True)
        (per_target_dir / f"{target_name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
    
    # Summary
    summary = {
        "status": "FULL_EVAL_DONE",
        "targets": list(all_results.keys()),
        "overall_scores": {t: r["scores"].get("Overall", 0.0) for t, r in all_results.items()},
        "leader_reference": {"Overall": 41.8},
        "inherited_reference": {"Overall": 40.7028},
    }
    
    # Compute delta
    if "official_only" in all_results and "qwen_aligned" in all_results:
        delta = {}
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", 
                    "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall"]:
            o = all_results["official_only"]["scores"].get(col, 0.0)
            a = all_results["qwen_aligned"]["scores"].get(col, 0.0)
            delta[col] = round(a - o, 4)
        summary["qwen_aligned_minus_official"] = delta
    
    (_public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
