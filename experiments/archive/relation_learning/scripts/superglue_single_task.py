#!/usr/bin/env python3
"""research independent single-SuperGLUE task runner.

Runs exactly one BabyLM strict fine-tuning task for one repaired AutoModel
checkpoint, with isolated output/cache directories and a physical-GPU process
snapshot.  This avoids the old endpoint-level serial SuperGLUE loop and lets the
closing O/(M,S) controls use both H100s across independent subtasks.
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
from typing import Any

ROOT = _public_path('.')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/superglue_single_tasks')
TASKS: dict[str, dict[str, Any]] = {
    "boolq": {"num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    "multirc": {"num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    "rte": {"num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    "wsc": {"num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 30},
    "mrpc": {"num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    "qqp": {"num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    "mnli": {"num_labels": 3, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy"], "epochs": 10},
}
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_env(out_dir: pathlib.Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    # Physical GPU selection: do not combine with an external CUDA_VISIBLE_DEVICES
    # remapping.  The child receives the requested physical index directly.
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache = out_dir / "runtime_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "NLTK_DATA": _public_path('experiments/archive/initial_model_studies/data/nltk_data'),
        "TMPDIR": cache / "tmp",
    }
    for k, p in mapping.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    return env


def nvidia_snapshot(child_pid: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"utc": now(), "child_pid": child_pid}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
        out["gpus"] = q.stdout.strip().splitlines() if q.returncode == 0 else []
        if q.returncode != 0:
            out["gpu_query_stderr"] = q.stderr[-500:]
    except Exception as e:
        out["gpu_query_error"] = repr(e)
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
        lines = q.stdout.strip().splitlines() if q.returncode == 0 else []
        out["compute_apps"] = lines
        uuid_to_idx: dict[str, str] = {}
        for line in out.get("gpus", []):
            parts = [x.strip() for x in str(line).split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = parts[0]
        hits = []
        if child_pid is not None:
            for line in lines:
                parts = [x.strip() for x in line.split(",")]
                if parts and parts[0] == str(child_pid):
                    hits.append(parts)
        out["child_compute_apps"] = hits
        out["child_physical_gpu_indices"] = sorted({uuid_to_idx.get(h[1], h[1]) for h in hits if len(h) >= 2})
        if q.returncode != 0:
            out["apps_query_stderr"] = q.stderr[-500:]
    except Exception as e:
        out["apps_query_error"] = repr(e)
    return out


def parse_results_txt(path: pathlib.Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, val = line.partition(":")
        if sep:
            try:
                out[key.strip()] = float(val.strip()) * 100.0
            except ValueError:
                pass
    return out


def find_latest(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--gpu", type=int, required=True, help="Physical GPU index to expose to the official child process.")
    ap.add_argument("--finetune-seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--task-timeout", type=float, default=14400.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_dir = out_root / args.label / f"ftseed{int(args.finetune_seed)}" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(ckpt / "config.json")
    spec = TASKS[args.task]
    revision = f"step127_{args.label}_{args.task}_ftseed{int(args.finetune_seed)}"
    cmd = [
        sys.executable, "-B", "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(ckpt.resolve()),
        "--train_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{args.task}.train.jsonl").resolve()),
        "--valid_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{args.task}.valid.jsonl").resolve()),
        "--predict_data", str((_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/glue_filtered') / f"{args.task}.valid.jsonl").resolve()),
        "--task", args.task,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str((out_dir / "superglue_results").resolve()),
        "--save",
        "--save_dir", str((out_dir / "superglue_models").resolve()),
        "--metrics", *spec["metrics"],
        "--metric_for_valid", spec["metric_for_valid"],
        "--seed", str(args.finetune_seed),
        "--verbose",
        "--padding_side", "left",
        "--take_final",
        "--revision_name", revision,
    ]
    log_path = out_dir / f"official_finetune_{args.task}.log"
    pre = nvidia_snapshot(None)
    (out_dir / "gpu_snapshot_pre.json").write_text(json.dumps(pre, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "superglue_single_start", "label": args.label, "task": args.task, "requested_physical_gpu": int(args.gpu), "checkpoint": rel(ckpt), "pre_snapshot": pre, "cmd": cmd, "utc": now()}), flush=True)
    t0 = time.time()
    env = setup_env(out_dir, int(args.gpu))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(STRICT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        snap = None
        for _ in range(12):
            time.sleep(5)
            snap = nvidia_snapshot(proc.pid)
            if snap.get("child_physical_gpu_indices"):
                break
        (out_dir / "gpu_snapshot_child.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rc = proc.wait(timeout=float(args.task_timeout))
    elapsed = round(time.time() - t0, 3)
    if rc != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        payload = {"status": "SUPERGLUE_SINGLE_FAILED", "label": args.label, "task": args.task, "returncode": rc, "elapsed_sec": elapsed, "log": rel(log_path), "log_tail": tail, "child_snapshot": snap, "requested_physical_gpu": int(args.gpu)}
        (out_dir / "superglue_single_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(rc)
    results_txt = find_latest(out_dir / "superglue_results", "results.txt")
    predictions = find_latest(out_dir / "superglue_results", "predictions.json")
    if results_txt is None:
        raise FileNotFoundError(out_dir / "superglue_results")
    metrics = parse_results_txt(results_txt)
    primary_metric = PRIMARY[args.task]
    primary = metrics.get(primary_metric)
    if primary is None:
        raise RuntimeError(f"missing primary metric {primary_metric}: {metrics}")
    payload = {
        "status": "SUPERGLUE_SINGLE_DONE",
        "created_utc": now(),
        "label": args.label,
        "task": args.task,
        "checkpoint": rel(ckpt),
        "finetune_seed": int(args.finetune_seed),
        "requested_physical_gpu": int(args.gpu),
        "child_pid": proc.pid,
        "child_snapshot": snap,
        "metrics": metrics,
        "primary_metric": primary_metric,
        "primary_score": float(primary),
        "results_txt": rel(results_txt),
        "predictions": rel(predictions) if predictions else None,
        "log": rel(log_path),
        "elapsed_sec": elapsed,
        "revision_name": revision,
    }
    (out_dir / "superglue_single_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "label": args.label, "task": args.task, "score": primary, "gpu_indices": (snap or {}).get("child_physical_gpu_indices"), "out_json": rel(out_dir / "superglue_single_payload.json")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
