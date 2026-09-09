#!/usr/bin/env python3
"""Full official-compatible evaluation of the scale-1.75 80M checkpoint.

Runs SuperGLUE (the missing piece for Overall projection) and records AoA status.
Cheap7 columns are already evaluated; we merge those scores
and produce the projected Overall.

SuperGLUE subtasks are parallelized across the available GPU using a ThreadPoolExecutor.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any, Dict

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')

# Official-compatible evaluator and base runner
A02_EVAL = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
COMPACT_EXPERIENCE_SCRIPTS = _public_path('experiments/archive/compact_experience/scripts')

# Scale 1.75 80M model
RUN_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022')
MODEL_ROOT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model')
ENDPOINT = "chck_80M"
MODEL_PATH = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M')

# Pristine eval data
PRISTINE = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
FULL_EVAL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')

# Output
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval')
COLLATE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_80M_collate')

# Existing cheap7 scores
A02_MERGED_80M = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/adapter128_scale1p75_chck_80M.json')

# SuperGLUE specs from the official evaluator
SUPERGLUE_TASKS = [
    {"task": "boolq", "metric": "accuracy"},
    {"task": "mnli", "metric": "accuracy"},
    {"task": "mrpc", "metric": "f1"},
    {"task": "multirc", "metric": "accuracy"},
    {"task": "qqp", "metric": "f1"},
    {"task": "rte", "metric": "accuracy"},
    {"task": "wsc", "metric": "accuracy"},
]

# Known cheap7 scores from the endpoint evaluation
KNOWN_CHEAP7 = {
    "BLiMP": 68.110,
    "Supplement": 62.620,
    "EWoK": 49.240,
    "Entity": 28.200,
    "COMPS": 52.110,
    "GlobalPIQA": 38.105,
    "Reading": 8.300,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_superglue_subtask(task: str, gpu: int, out_dir: pathlib.Path, force: bool) -> Dict[str, Any]:
    """Run one SuperGLUE subtask using the official finetuning pipeline."""
    data_dir = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / task
    if not data_dir.exists():
        return {"task": task, "status": "error", "error": f"Data dir not found: {data_dir}"}

    task_out = out_dir / "superglue" / task
    task_out.mkdir(parents=True, exist_ok=True)

    results_txt = None
    for p in task_out.rglob("results.txt"):
        results_txt = p
    if results_txt and not force:
        return {"task": task, "status": "skip_existing", "results_txt": str(results_txt)}

    # Use the babylm-eval finetuning script
    finetune_script = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/finetune/finetune_classification.py')
    if not finetune_script.exists():
        return {"task": task, "status": "error", "error": f"Finetune script not found: {finetune_script}"}

    cmd = [
        sys.executable, "-B", str(finetune_script),
        "--model_name_or_path", str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M')),
        "--task_name", task,
        "--validation_file", str((data_dir / "val.jsonl").resolve()) if (data_dir / "val.jsonl").exists() else str((data_dir / "validation.jsonl").resolve()),
        "--train_file", str((data_dir / "train.jsonl").resolve()),
        "--do_train",
        "--do_eval",
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "64",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "10",
        "--output_dir", str(task_out.resolve()),
        "--overwrite_output_dir",
        "--trust_remote_code",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path = out_dir / "logs" / f"superglue_{task}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(json.dumps({"event": "superglue_start", "task": task, "gpu": gpu, "utc": now()}), flush=True)

    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    log_path.write_text(f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n\n[rc={proc.returncode} elapsed={elapsed:.1f}s]\n", encoding="utf-8")

    rec = {"task": task, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3)}
    if proc.returncode != 0:
        rec["status"] = "error"
        rec["stderr_tail"] = proc.stderr[-2000:]
        print(json.dumps({"event": "superglue_error", **rec}), flush=True)
        return rec

    # Find results.txt
    results_txt = None
    for p in task_out.rglob("results.txt"):
        results_txt = p
    if results_txt:
        rec["results_txt"] = str(results_txt)
        rec["status"] = "done"
    else:
        # Try to find eval_results.json
        eval_json = None
        for p in task_out.rglob("eval_results.json"):
            eval_json = p
        if eval_json:
            rec["eval_results"] = str(eval_json)
            rec["status"] = "done_json_only"
        else:
            rec["status"] = "done_no_results"
    print(json.dumps({"event": "superglue_done", **rec}), flush=True)
    return rec


def run_superglue_via_evaluator(gpu: int, out_dir: pathlib.Path, force: bool) -> Dict[str, Any]:
    """Run SuperGLUE using the official-compatible evaluator wrapper."""
    target = "scale1p75_80M_superglue"
    eval_out = out_dir / "evaluator_superglue"
    eval_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-B", str(A02_EVAL),
        "--arm", "reinvest",
        "--run-dir", str(RUN_DIR),
        "--target", target,
        "--endpoint", ENDPOINT,
        "--out-root", str(eval_out),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", "SuperGLUE",
        "--force" if force else "",
    ]
    cmd = [c for c in cmd if c]  # remove empty strings

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path = out_dir / "logs" / "superglue_evaluator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(json.dumps({"event": "superglue_evaluator_start", "gpu": gpu, "target": target, "utc": now()}), flush=True)

    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    log_path.write_text(f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n\n[rc={proc.returncode} elapsed={elapsed:.1f}s]\n", encoding="utf-8")

    rec = {"target": target, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3)}
    if proc.returncode == 0:
        rec["status"] = "done"
        rec["stdout_tail"] = proc.stdout[-3000:]
    else:
        rec["status"] = "error"
        rec["stderr_tail"] = proc.stderr[-3000:]
        rec["stdout_tail"] = proc.stdout[-2000:]
    print(json.dumps({"event": "superglue_evaluator_done", **rec}), flush=True)
    return rec


def run_reading_via_evaluator(gpu: int, out_dir: pathlib.Path, force: bool) -> Dict[str, Any]:
    """Run Reading evaluation using the evaluator (cross-check)."""
    target = "scale1p75_80M_reading"
    eval_out = out_dir / "evaluator_reading"
    eval_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-B", str(A02_EVAL),
        "--arm", "reinvest",
        "--run-dir", str(RUN_DIR),
        "--target", target,
        "--endpoint", ENDPOINT,
        "--out-root", str(eval_out),
        "--collate-root", str(_public_path('experiments/archive/representation_and_objectives/data/scale1p75_80M_collate/reading')),
        "--gpu", str(gpu),
        "--columns", "Reading",
        "--force" if force else "",
    ]
    cmd = [c for c in cmd if c]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path = out_dir / "logs" / "reading_evaluator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    log_path.write_text(f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n\n[rc={proc.returncode} elapsed={elapsed:.1f}s]\n", encoding="utf-8")

    rec = {"target": target, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3)}
    if proc.returncode == 0:
        rec["status"] = "done"
        rec["stdout_tail"] = proc.stdout[-3000:]
    else:
        rec["status"] = "error"
        rec["stderr_tail"] = proc.stderr[-3000:]
    return rec


def extract_superglue_from_evaluator(out_dir: pathlib.Path) -> float | None:
    """Extract SuperGLUE score from the evaluator output."""
    target = "scale1p75_80M_superglue"
    payload_path = out_dir / "evaluator_superglue" / "per_target" / f"{target}.json"
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    return sg.get("superglue_mean")


def compute_overall(cheap7: Dict[str, float], superglue: float | None, aoa: float = 0.0) -> Dict[str, Any]:
    """Compute Overall from 9 columns."""
    scores = {
        "BLiMP": cheap7.get("BLiMP"),
        "Supplement": cheap7.get("Supplement"),
        "EWoK": cheap7.get("EWoK"),
        "Entity": cheap7.get("Entity"),
        "COMPS": cheap7.get("COMPS"),
        "SuperGLUE": superglue,
        "GlobalPIQA": cheap7.get("GlobalPIQA"),
        "Reading": cheap7.get("Reading"),
        "AoA": aoa,
    }
    vals = [v for v in scores.values() if v is not None]
    if len(vals) == 9:
        overall = mean(vals)
        complete = True
    else:
        overall = mean(vals) if vals else None
        complete = False
    return {
        "scores": scores,
        "overall": overall,
        "complete": complete,
        "n_columns": len(vals),
        "cheap7": mean([cheap7[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"] if c in cheap7]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", choices=["preflight", "superglue", "project", "all"], default="all")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Preflight
    print(json.dumps({
        "event": "preflight",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "config_exists": (_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/config.json')).exists() if MODEL_PATH.exists() else False,
        "custom_code_exists": (_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/adapter_scaled_modeling.py')).exists() if MODEL_PATH.exists() else False,
        "evaluator_exists": A02_EVAL.exists(),
        "a02_merged_80M_exists": A02_MERGED_80M.exists(),
        "known_cheap7": KNOWN_CHEAP7,
        "gpu": args.gpu,
        "utc": now(),
    }, indent=2), flush=True)

    if args.stage == "preflight":
        return

    if args.stage in ("superglue", "all"):
        # Run SuperGLUE via the official-compatible evaluator
        sg_result = run_superglue_via_evaluator(args.gpu, OUT_ROOT, args.force)

        # Extract SuperGLUE score
        sg_score = extract_superglue_from_evaluator(OUT_ROOT)
        print(json.dumps({"event": "superglue_extracted", "superglue_score": sg_score}), flush=True)

    if args.stage in ("project", "all"):
        sg_score = extract_superglue_from_evaluator(OUT_ROOT)

        # Compute Overall projection
        result = compute_overall(KNOWN_CHEAP7, sg_score, aoa=0.0)

        # Save synthesis
        synthesis = {
            "status": "SCALE1P75_80M_FULL_EVAL",
            "created_utc": now(),
            "model": {
                "run_dir": str(RUN_DIR),
                "endpoint": ENDPOINT,
                "architecture": "AdapterDebertaV2ForMaskedLM",
                "adapter_bottleneck": 128,
                "adapter_scale": 1.75,
                "param_count": 35463008,
                "tokenizer": "compliant16k_reinvest10M",
                "vocab_size": 16384,
                "word_exposure": 80034368,
            },
            "cheap7_source": "A02 research split eval",
            "cheap7_scores": KNOWN_CHEAP7,
            "cheap7": result["cheap7"],
            "superglue": sg_score,
            "superglue_source": "research evaluator on GPU" + str(args.gpu),
            "aoa": 0.0,
            "aoa_reason": "80M model missing chck_1M..chck_9M and chck_90M..chck_100M; AoA=0 by convention",
            "overall_projection": result,
            "frontier_comparison": {
                "leader_overall": 41.80,
                "leader_cheap7": 43.77,
                "this_overall": result["overall"],
                "this_cheap7": result["cheap7"],
                "crosses_leader": result["overall"] is not None and result["overall"] >= 41.80,
            },
            "note": "This is the 80M checkpoint evaluation. The 100M official-ladder run is still in progress on A02. Training gains need not be monotonic; 80M may exceed 100M on some axes.",
        }

        out_path = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval/scale1p75_80M_full_eval_synthesis.json')
        out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": synthesis["status"],
            "overall": result["overall"],
            "superglue": sg_score,
            "cheap7": result["cheap7"],
            "crosses_leader": synthesis["frontier_comparison"]["crosses_leader"],
            "synthesis": str(out_path),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
