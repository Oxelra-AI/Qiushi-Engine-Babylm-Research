#!/usr/bin/env python3
"""research: full official-style BabyLM Strict-Small Overall evaluator.

Runs the official strict evaluation code on one local MLM checkpoint:
  - full zero-shot BLiMP, Supplement, EWoK, Entity Tracking, COMPS,
    GlobalPIQA parallel/nonparallel, and Reading;
  - official glue_filtered finetuning tasks for the (Super)GLUE column;
  - official AoA only when the model_root contains the complete strict-small
    checkpoint ladder chck_1M..chck_9M and chck_10M..chck_100M.

The script writes per-task predictions and a per-target JSON under
data/full_overall_eval/.  If AoA checkpoints are missing, it
records that the artifact is not submit-ready for official AoA and uses AoA=0.0
only for an immediate leaderboard-style provisional Overall, matching the
leaderboard convention for missing AoA-like fields while not pretending this is
an official AoA measurement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from babylm_official_scoring import (
    OFFICIAL_OVERALL_KEYS,
    NLP_KEYS,
    normalize_aoa_record,
    target_scores_from_tasks,
    compute_overall_from_tasks,
)
WORKSPACE = _public_path('experiments/archive/compact_experience')
STUDY = _public_path('experiments/archive/compact_experience')
USER_ROOT = _public_path('.')
ROOT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/full_overall_eval')
PER_TARGET_DIR = _public_path('experiments/archive/compact_experience/data/full_overall_eval/per_target')

TARGETS: Dict[str, Dict[str, Any]] = {
    "phase2s_mix25": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/phase2s_mix25_40k_12x384_seed43200'),
        "endpoint": "chck_100M",
        "description": "Corrected Phase-2 stagewise 12x384/LAMB/40k/shared-tokenizer mix25 endpoint.",
        "family": "phase2_stagewise_40k",
    },
    "mix25_16k_seed43": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/mix_25pct_100M_seed43'),
        "endpoint": "chck_100M",
        "description": "Strongest known 16k mix25 endpoint from the research dose-response screen; exploratory model-init because extra_init_seed=-1.",
        "family": "debertav2_8x480_16k",
    },
    "phase2s_official": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/phase2s_official_40k_12x384_seed43200'),
        "endpoint": "chck_100M",
        "description": "Matched Phase-2 official-data control for phase2s_mix25.",
        "family": "phase2_stagewise_40k",
    },
    "mix25_16k_fixedinit": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/mix25_fixedinit16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "Causal fixed-initialization 16k mix25 endpoint.",
        "family": "debertav2_8x480_16k_fixedinit",
    },
    "official_16k_fixedinit": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/official_fixedinit16k_seed43022'),
        "endpoint": "chck_100M",
        "description": "Matched causal fixed-initialization 16k official-data control.",
        "family": "debertav2_8x480_16k_fixedinit",
    },
    "initial_model_studies_best_80M": {
        "run_dir": _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256'),
        "endpoint": "chck_80M",
        "description": "Inherited strongest internal coordinate endpoint from initial_model_studies research.",
        "family": "inherited_reference",
    },
}

ZERO_SHOT_TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": "evaluation_data/full_eval/blimp_filtered", "batch_size": 128},
    {"column": "Supplement", "task": "blimp", "data_path": "evaluation_data/full_eval/supplement_filtered", "batch_size": 128},
    {"column": "EWoK", "task": "ewok", "data_path": "evaluation_data/full_eval/ewok_filtered", "batch_size": 64},
    {"column": "Entity", "task": "entity_tracking", "data_path": "evaluation_data/full_eval/entity_tracking", "batch_size": 128},
    {"column": "COMPS", "task": "comps", "data_path": "evaluation_data/full_eval/comps", "batch_size": 128},
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": "evaluation_data/full_eval/global_piqa_parallel", "batch_size": 128},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": "evaluation_data/full_eval/global_piqa_nonparallel", "batch_size": 128},
]
ZERO_BY_COL = {t["column"]: t for t in ZERO_SHOT_TASKS}

SUPERGLUE_TASKS = [
    {"task": "boolq", "num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "multirc", "num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "rte", "num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "wsc", "num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 30},
    {"task": "mrpc", "num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "qqp", "num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "mnli", "num_labels": 3, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy"], "epochs": 10},
]

AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


def model_root_for(target: str) -> pathlib.Path:
    return pathlib.Path(TARGETS[target]["run_dir"]) / "hf_model"


def model_path_for(target: str) -> pathlib.Path:
    return model_root_for(target) / TARGETS[target]["endpoint"]


def per_target_path(target: str) -> pathlib.Path:
    return PER_TARGET_DIR / f"{target}.json"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_env(target: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = _public_path('experiments/archive/compact_experience/data/full_overall_eval/hf_cache') / target
    tmp = _public_path('experiments/archive/compact_experience/data/full_overall_eval/tmp') / target
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str(_public_path('experiments/archive/initial_model_studies/data/nltk_data'))
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TMPDIR"] = str(tmp.resolve())
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def append_log(log_path: pathlib.Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: Optional[int] = None) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_log(log_path, "\n[{}] $ {}".format(now_utc(), " ".join(cmd)))
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    elapsed = time.time() - t0
    append_log(log_path, f"[returncode={proc.returncode} elapsed_sec={elapsed:.2f}]")
    return {"returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "log": str(log_path)}


def parse_sentence_score(text: str) -> Optional[float]:
    """Parse official sentence zero-shot report averages without a loose fallback.

    Earlier evaluation code fell back to the last numeric token in the report.  That
    can silently treat a timestamp, count, temperature, or unrelated trailing
    number as the task score if the report format changes.  The full-evaluation
    chain now accepts only the explicit official average headers and a finite
    percentage-like value.
    """
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -5.0 <= val <= 105.0:
                return val
            return None
    return None


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        val = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if val is not None:
            return val
    return None


def parse_reading_scores(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_reading_report(task_out: pathlib.Path) -> Dict[str, float]:
    candidates = sorted(task_out.rglob("report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def latest_file(root: pathlib.Path, pattern: str) -> Optional[pathlib.Path]:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def load_or_new_payload(target: str, gpu: int, force: bool) -> Dict[str, Any]:
    p = per_target_path(target)
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))
    model_root = model_root_for(target)
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload: Dict[str, Any] = {
        "target": target,
        "description": TARGETS[target].get("description", ""),
        "family": TARGETS[target].get("family", ""),
        "run_dir": str(TARGETS[target]["run_dir"]),
        "model_root": str(model_root),
        "model_path": str(model_path),
        "endpoint": TARGETS[target]["endpoint"],
        "started_utc": now_utc(),
        "gpu": gpu,
        "tasks": {},
    }
    metrics = pathlib.Path(TARGETS[target]["run_dir"]) / "scientific_metrics.json"
    if metrics.exists():
        try:
            m = json.loads(metrics.read_text(encoding="utf-8"))
            payload["run_summary"] = {k: m.get(k) for k in [
                "variant", "word_exposure", "actual_training_steps", "parameter_count", "vocab_size",
                "tokenizer_label", "tokenizer_path", "loss_first", "loss_last", "seed", "extra_init_seed",
                "train_rng_seed", "mask_mode", "mask_switch_mode", "mask_switch_words", "seq_length",
                "batch_size", "optimizer", "learning_rate", "n_layer", "hidden_size", "n_head",
                "intermediate_size",
            ] if k in m}
            payload["run_summary"]["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
        except Exception as exc:
            payload["run_summary_error"] = repr(exc)
    save_payload(target, payload)
    return payload


def save_payload(target: str, payload: Dict[str, Any]) -> None:
    PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    per_target_path(target).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def task_done(payload: Dict[str, Any], name: str) -> bool:
    rec = payload.get("tasks", {}).get(name)
    if not isinstance(rec, dict):
        return False
    if name == "SuperGLUE":
        tasks = rec.get("tasks") or []
        return (
            isinstance(tasks, list)
            and len(tasks) == len(SUPERGLUE_TASKS)
            and all(isinstance(r, dict) and r.get("returncode") == 0 and r.get("accuracy") is not None for r in tasks)
            and rec.get("superglue_mean") is not None
        )
    if name == "AoA":
        return rec.get("returncode", 1) == 0 and (
            rec.get("status", "").startswith("not_official")
            or rec.get("status") == "official_aoa_done"
            or rec.get("aoa_official") is not None
            or rec.get("aoa_for_provisional_overall") is not None
        )
    return rec.get("returncode", 1) == 0 and (
        rec.get("score") is not None or rec.get("scores") or rec.get("status", "").startswith("not_official")
    )


def eval_zero_shot_column(target: str, payload: Dict[str, Any], column: str, gpu: int, force: bool) -> None:
    if task_done(payload, column) and not force:
        print(json.dumps({"event": "skip_existing", "target": target, "column": column}), flush=True)
        return
    spec = ZERO_BY_COL[column]
    env = setup_env(target, gpu)
    model_path = model_path_for(target).resolve()
    task_out = _public_path('experiments/archive/compact_experience/data/full_overall_eval/official_outputs') / target / column
    log = _public_path('experiments/archive/compact_experience/data/full_overall_eval/logs') / target / f"zero_shot_{column}.log"
    revision = f"full_{target}_{column}"
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--task", spec["task"],
        "--data_path", spec["data_path"],
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(spec["batch_size"]),
        "--non_causal_batch_size", "64",
        "--output_dir", str(task_out.resolve()),
    ]
    rec = {"column": column, "task": spec["task"], "data_path": spec["data_path"], "revision_name": revision, "output_dir": str(task_out)}
    try:
        rr = run_cmd(cmd, env, log, timeout=7200)
        rec.update(rr)
        score = read_report_score(task_out)
        rec["score"] = score
        pred = latest_file(task_out, "predictions.json")
        report = latest_file(task_out, "best_temperature_report.txt")
        if pred:
            rec["predictions"] = str(pred)
        if report:
            rec["report"] = str(report)
        if rr["returncode"] != 0:
            raise RuntimeError(f"zero-shot {column} failed rc={rr['returncode']}")
        if score is None:
            raise RuntimeError(f"zero-shot {column} produced no parseable score")
    except Exception as exc:
        rec["error"] = repr(exc)
        payload["tasks"][column] = rec
        save_payload(target, payload)
        raise
    payload["tasks"][column] = rec
    save_payload(target, payload)
    print(json.dumps({"event": "zero_shot_done", "target": target, "column": column, "score": rec.get("score"), "gpu": gpu}), flush=True)


def eval_reading(target: str, payload: Dict[str, Any], gpu: int, force: bool) -> None:
    column = "Reading"
    if task_done(payload, column) and not force:
        print(json.dumps({"event": "skip_existing", "target": target, "column": column}), flush=True)
        return
    env = setup_env(target, gpu)
    model_path = model_path_for(target).resolve()
    task_out = _public_path('experiments/archive/compact_experience/data/full_overall_eval/official_outputs') / target / column
    log = _public_path('experiments/archive/compact_experience/data/full_overall_eval/logs') / target / "zero_shot_Reading.log"
    revision = f"full_{target}_Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--data_path", "evaluation_data/full_eval/reading/reading_data.csv",
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    rec: Dict[str, Any] = {"column": column, "data_path": "evaluation_data/full_eval/reading/reading_data.csv", "revision_name": revision, "output_dir": str(task_out)}
    try:
        rr = run_cmd(cmd, env, log, timeout=7200)
        rec.update(rr)
        scores = read_reading_report(task_out)
        rec["scores"] = scores
        pred = latest_file(task_out, "predictions.json")
        report = latest_file(task_out, "report.txt")
        if pred:
            rec["predictions"] = str(pred)
        if report:
            rec["report"] = str(report)
        if rr["returncode"] != 0:
            raise RuntimeError(f"reading failed rc={rr['returncode']}")
        if not scores or "Reading" not in scores:
            raise RuntimeError("reading produced no parseable combined score")
    except Exception as exc:
        rec["error"] = repr(exc)
        payload["tasks"][column] = rec
        save_payload(target, payload)
        raise
    payload["tasks"][column] = rec
    save_payload(target, payload)
    print(json.dumps({"event": "reading_done", "target": target, "scores": scores, "gpu": gpu}), flush=True)


def labels_for_superglue(task: str) -> List[int]:
    data = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{task}.valid.jsonl"
    labels: List[int] = []
    for line in data.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(int(json.loads(line)["label"]))
    return labels


def score_superglue_predictions(task: str, pred_path: pathlib.Path) -> Dict[str, Any]:
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = [int(x["pred"]) for x in data[task]["predictions"]]
    labels = labels_for_superglue(task)
    if len(preds) != len(labels):
        raise RuntimeError(f"{task} predictions length {len(preds)} != labels length {len(labels)}")
    correct = sum(int(a == b) for a, b in zip(preds, labels))
    counts: Dict[str, int] = {}
    for p in preds:
        counts[str(p)] = counts.get(str(p), 0) + 1
    return {"task": task, "predictions": str(pred_path), "num_examples": len(labels), "correct": correct, "accuracy": correct / len(labels) * 100.0, "pred_counts": counts}


def run_superglue_task(target: str, model_path: pathlib.Path, spec: Dict[str, Any], env: Dict[str, str], gpu: int, force: bool) -> Dict[str, Any]:
    task = spec["task"]
    results_dir = _public_path('experiments/archive/compact_experience/data/full_overall_eval/superglue_results') / target / task
    save_dir = _public_path('experiments/archive/compact_experience/data/full_overall_eval/superglue_models') / target / task
    log = _public_path('experiments/archive/compact_experience/data/full_overall_eval/logs') / target / f"superglue_{task}.log"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(model_path.resolve()),
        "--train_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{task}.train.jsonl").resolve()),
        "--valid_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{task}.valid.jsonl").resolve()),
        "--predict_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{task}.valid.jsonl").resolve()),
        "--task", task,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", "3e-5",
        "--num_epochs", str(spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str(results_dir.resolve()),
        "--save",
        "--save_dir", str(save_dir.resolve()),
        "--metrics", *spec["metrics"],
        "--metric_for_valid", spec["metric_for_valid"],
        "--seed", "42",
        "--verbose",
        "--padding_side", "left",
        "--take_final",
    ]
    rr = run_cmd(cmd, env, log, timeout=14400)
    pred = latest_file(results_dir, f"finetune/{task}/predictions.json")
    rec: Dict[str, Any] = {
        "task": task,
        "num_labels": spec["num_labels"],
        "batch_size": spec["batch_size"],
        "epochs": spec["epochs"],
        "metric_for_valid": spec["metric_for_valid"],
        "results_dir": str(results_dir),
        "save_dir": str(save_dir),
        **rr,
    }
    if pred:
        rec.update(score_superglue_predictions(task, pred))
    if rr["returncode"] != 0:
        raise RuntimeError(f"SuperGLUE task {task} failed rc={rr['returncode']}")
    if "accuracy" not in rec:
        raise RuntimeError(f"SuperGLUE task {task} produced no predictions")
    print(json.dumps({"event": "superglue_task_done", "target": target, "task": task, "accuracy": rec["accuracy"], "gpu": gpu}), flush=True)
    return rec


def eval_superglue(target: str, payload: Dict[str, Any], gpu: int, force: bool) -> None:
    name = "SuperGLUE"
    if task_done(payload, name) and not force:
        print(json.dumps({"event": "skip_existing", "target": target, "column": name}), flush=True)
        return
    env = setup_env(target, gpu)
    model_path = model_path_for(target)
    old_rec = payload.get("tasks", {}).get(name)
    if isinstance(old_rec, dict) and not force:
        old_tasks = [r for r in old_rec.get("tasks", []) if isinstance(r, dict)]
        rec: Dict[str, Any] = {"column": name, "tasks": old_tasks, "started_utc": old_rec.get("started_utc", now_utc())}
    else:
        rec = {"column": name, "tasks": [], "started_utc": now_utc()}
    try:
        for spec in SUPERGLUE_TASKS:
            # Resume a partially completed SuperGLUE column when possible.
            existing = {r.get("task"): r for r in rec.get("tasks", []) if isinstance(r, dict)}
            if spec["task"] in existing and existing[spec["task"]].get("returncode") == 0 and existing[spec["task"]].get("accuracy") is not None and not force:
                print(json.dumps({"event": "skip_existing_superglue_task", "target": target, "task": spec["task"], "accuracy": existing[spec["task"]].get("accuracy"), "gpu": gpu}), flush=True)
                continue
            task_rec = run_superglue_task(target, model_path, spec, env, gpu, force)
            rec["tasks"] = [r for r in rec.get("tasks", []) if not (isinstance(r, dict) and r.get("task") == spec["task"])]
            rec["tasks"].append(task_rec)
            rec["superglue_mean"] = sum(float(r["accuracy"]) for r in rec["tasks"] if isinstance(r, dict) and r.get("accuracy") is not None) / len([r for r in rec["tasks"] if isinstance(r, dict) and r.get("accuracy") is not None])
            payload["tasks"][name] = rec
            save_payload(target, payload)
        if len(rec["tasks"]) != len(SUPERGLUE_TASKS):
            raise RuntimeError(f"SuperGLUE completed {len(rec['tasks'])}/{len(SUPERGLUE_TASKS)} tasks")
        rec["finished_utc"] = now_utc()
        rec["superglue_mean"] = sum(float(r["accuracy"]) for r in rec["tasks"]) / len(rec["tasks"])
    except Exception as exc:
        rec["error"] = repr(exc)
        payload["tasks"][name] = rec
        save_payload(target, payload)
        raise
    payload["tasks"][name] = rec
    save_payload(target, payload)
    print(json.dumps({"event": "superglue_done", "target": target, "superglue_mean": rec["superglue_mean"], "gpu": gpu}), flush=True)


def run_or_record_aoa(target: str, payload: Dict[str, Any], gpu: int, force: bool) -> None:
    name = "AoA"
    if task_done(payload, name) and not force:
        print(json.dumps({"event": "skip_existing", "target": target, "column": name}), flush=True)
        return
    model_root = model_root_for(target)
    available = {p.name for p in model_root.iterdir() if p.is_dir()} if model_root.exists() else set()
    missing = [x for x in AOA_STEPS if x not in available]
    rec: Dict[str, Any] = {
        "column": name,
        "required_strict_small_steps": AOA_STEPS,
        "available_checkpoint_count": len(available),
        "missing_required_steps": missing,
        "model_root": str(model_root),
    }
    if missing:
        rec.update({
            "status": "not_official_missing_checkpoints",
            "aoa_official": None,
            "aoa_raw_correlation": None,
            "aoa_leaderboard_score": 0.0,
            "aoa_for_provisional_overall": 0.0,
            "interpretation": "This artifact lacks at least one required strict-small AoA checkpoint (chck_1M..chck_9M and chck_10M..chck_100M). AoA is not official for this artifact without retraining/resaving; 0.0 is used only for immediate provisional Overall arithmetic.",
            "returncode": 0,
        })
        payload["tasks"][name] = rec
        save_payload(target, payload)
        print(json.dumps({"event": "aoa_unavailable", "target": target, "missing": missing[:5], "missing_count": len(missing)}), flush=True)
        return
    env = setup_env(target, gpu)
    log = _public_path('experiments/archive/compact_experience/data/full_overall_eval/logs') / target / "aoa.log"
    out_dir = _public_path('experiments/archive/compact_experience/data/full_overall_eval/aoa_outputs') / target
    out_json = _public_path('experiments/archive/compact_experience/data/full_overall_eval/aoa_outputs') / target / "aoa_local_ckpts.json"
    out_note = _public_path('experiments/archive/compact_experience/data/full_overall_eval/aoa_outputs') / target / "aoa_local_ckpts.md"
    # Prefer the repaired local AoA helper. The inherited helper's relative cache
    # paths fail when evaluation starts from the strict-evaluator directory;
    # the repaired helper derives absolute local paths.
    local_runner = _public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')
    runner = local_runner if local_runner.exists() else _public_path('experiments/archive/initial_model_studies/scripts/aoa_local_ckpts_for_model.py')
    cmd = [
        sys.executable, str(runner.resolve()),
        "--model_root", str(model_root.resolve()),
        "--out_dir", str(out_dir.resolve()),
        "--out_json", str(out_json.resolve()),
        "--out_note", str(out_note.resolve()),
        "--log", str(log.resolve()),
        "--gpu", str(gpu),
    ]
    try:
        rr = run_cmd(cmd, env, log, timeout=14400)
        rec.update(rr)
        if rr["returncode"] != 0:
            raise RuntimeError(f"AoA failed rc={rr['returncode']}")
        data = json.loads(out_json.read_text(encoding="utf-8"))
        curve = data.get("curve_fitness_record") if isinstance(data, dict) else None
        aoa_raw = float(data["aoa"])
        expected_steps = data.get("expected_steps")
        step_counts = data.get("step_counts") or {}
        row_count_values = data.get("row_count_values") or []
        validation_errors: List[str] = []
        if data.get("status") != "AOA_LOCAL_CKPTS_DONE":
            validation_errors.append(f"helper status {data.get('status')}")
        if data.get("num_steps") != len(AOA_STEPS):
            validation_errors.append(f"num_steps {data.get('num_steps')} != {len(AOA_STEPS)}")
        if expected_steps != AOA_STEPS:
            validation_errors.append("expected_steps mismatch")
        if data.get("missing_steps") not in ([], None):
            validation_errors.append(f"missing_steps {data.get('missing_steps')}")
        if data.get("unexpected_steps") not in ([], None):
            validation_errors.append(f"unexpected_steps {data.get('unexpected_steps')}")
        if sorted(step_counts.keys()) != sorted(AOA_STEPS):
            validation_errors.append("step_counts keys do not equal required AoA steps")
        if len(row_count_values) != 1 or int(row_count_values[0]) <= 0:
            validation_errors.append(f"row_count_values {row_count_values}")
        if data.get("finite_surprisals") is not True:
            validation_errors.append(f"finite_surprisals {data.get('finite_surprisals')}")
        if not math.isfinite(aoa_raw):
            validation_errors.append(f"nonfinite raw AoA {aoa_raw}")
        if validation_errors:
            raise RuntimeError({
                "error": "aoa_helper_output_validation_failed",
                "validation_errors": validation_errors,
                "out_json": str(out_json),
                "runner": str(runner.resolve()),
            })
        rec.update(normalize_aoa_record({
            "status": "official_aoa_done",
            "helper_status": data.get("status"),
            "aoa_helper_runner": str(runner.resolve()),
            "aoa_official": aoa_raw,
            "out_json": str(out_json),
            "out_note": str(out_note),
            "surprisal_path": data.get("surprisal_path"),
            "score_path": data.get("score_path"),
            "score_tokenizer_path": data.get("score_tokenizer_path"),
            "num_rows": data.get("num_rows"),
            "num_steps": data.get("num_steps"),
            "step_counts": step_counts,
            "expected_steps": expected_steps,
            "missing_steps": data.get("missing_steps"),
            "unexpected_steps": data.get("unexpected_steps"),
            "row_count_values": row_count_values,
            "finite_surprisals": data.get("finite_surprisals"),
            "curve_fitness_record": curve,
            "aoa_p_value": curve.get("p_value") if isinstance(curve, dict) else None,
            "aoa_n_words": curve.get("n_words") if isinstance(curve, dict) else None,
            "aoa_validation": "passed full 19-step ladder, finite surprisal, and finite raw-AoA checks",
        }))
    except Exception as exc:
        rec["error"] = repr(exc)
        payload["tasks"][name] = rec
        save_payload(target, payload)
        raise
    payload["tasks"][name] = rec
    save_payload(target, payload)
    print(json.dumps({"event": "aoa_done", "target": target, "aoa": rec.get("aoa_official"), "gpu": gpu}), flush=True)


def target_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return target_scores_from_tasks(payload.get("tasks", {}))


def compute_overall_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    out = compute_overall_from_tasks(tasks)
    aoa = tasks.get("AoA", {})
    out["aoa_status"] = aoa.get("status") if isinstance(aoa, dict) else None
    out["submit_ready_aoa"] = (isinstance(aoa, dict) and aoa.get("status") == "official_aoa_done")
    out["submit_ready_overall"] = bool(out.get("submit_ready_aoa") and out.get("complete_for_provisional_overall"))
    if isinstance(aoa, dict):
        out["aoa_raw_correlation"] = aoa.get("aoa_raw_correlation")
        out["aoa_leaderboard_score"] = aoa.get("aoa_leaderboard_score")
    return out


def finalize_payload(target: str, payload: Dict[str, Any]) -> None:
    payload["finished_utc"] = now_utc()
    payload["official_overall"] = compute_overall_fields(payload)
    save_payload(target, payload)
    print(json.dumps({"event": "target_finalized", "target": target, **payload["official_overall"]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Subset among zero-shot columns, Reading, SuperGLUE, AoA. Default: all official columns.")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = load_or_new_payload(args.target, args.gpu, args.force)
    payload.setdefault("tasks", {})
    columns = args.columns or [t["column"] for t in ZERO_SHOT_TASKS] + ["Reading", "SuperGLUE", "AoA"]
    print(json.dumps({"event": "target_start", "target": args.target, "gpu": args.gpu, "columns": columns, "model_path": str(model_path_for(args.target))}), flush=True)

    for col in columns:
        if col in ZERO_BY_COL:
            eval_zero_shot_column(args.target, payload, col, args.gpu, args.force)
        elif col == "Reading":
            eval_reading(args.target, payload, args.gpu, args.force)
        elif col == "SuperGLUE":
            eval_superglue(args.target, payload, args.gpu, args.force)
        elif col == "AoA":
            run_or_record_aoa(args.target, payload, args.gpu, args.force)
        else:
            raise ValueError(f"unknown column {col}")
    finalize_payload(args.target, payload)


if __name__ == "__main__":
    main()
