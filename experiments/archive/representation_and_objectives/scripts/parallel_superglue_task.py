#!/usr/bin/env python3
"""research: standalone single-SuperGLUE-task finetune runner for parallel completion.

The hardened evaluator runs the seven SuperGLUE finetune
tasks serially inside one --columns SuperGLUE call, which risks per-call timeouts
and wastes an idle GPU. This runner executes exactly ONE SuperGLUE subtask against
an explicit model_path on an explicit GPU, writing predictions.json + results.txt
into an official-compatible results tree so the existing collation/patch code can
read accuracy (and f1 for MRPC/QQP) unchanged.

It reuses the base finetune command exactly (same evaluation_pipeline.finetune.run
invocation, same hyperparameters, same STRICT data paths, seed 42) so that a task
completed here is bitwise-consistent with a task the hardened evaluator would have
produced. This only fills missing subtasks; it does not alter model weights, data,
or scoring definitions.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

STRICT = base.STRICT
SUPERGLUE_SPEC = {s["task"]: s for s in base.SUPERGLUE_TASKS}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def latest_file(root: pathlib.Path, name_suffix: str) -> pathlib.Path | None:
    hits = sorted(root.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    # name_suffix like "finetune/{task}/predictions.json"
    want = name_suffix.split("/")[-1]
    filtered = [h for h in hits if h.name == want and name_suffix in str(h)]
    return filtered[-1] if filtered else (hits[-1] if hits else None)


def attach_cache(env: dict, cache_root: pathlib.Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "TMPDIR": cache_root / "tmp",
    }
    for k, p in mapping.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    env["NLTK_DATA"] = str((USER_ROOT / "experiments/archive/initial_model_studies/data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(SUPERGLUE_SPEC))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--results-dir", required=True, help="superglue_results/<target>/<task> root")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    task = args.task
    spec = SUPERGLUE_SPEC[task]
    model_path = pathlib.Path(args.model_path).resolve()
    results_dir = pathlib.Path(args.results_dir).resolve()
    save_dir = pathlib.Path(args.save_dir).resolve()
    log = pathlib.Path(args.log).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(json.dumps({"event": "error", "reason": "model_path_missing", "model_path": str(model_path)}), flush=True)
        sys.exit(2)

    existing_pred = latest_file(results_dir, f"finetune/{task}/predictions.json")
    if existing_pred and not args.force:
        try:
            rec = base.score_superglue_predictions(task, existing_pred)
            print(json.dumps({"event": "skip_existing_superglue_task", "task": task,
                              "accuracy": rec["accuracy"], "predictions": str(existing_pred)}), flush=True)
            return
        except Exception:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    attach_cache(env, pathlib.Path(args.cache_root))

    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(model_path),
        "--train_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.train.jsonl").resolve()),
        "--valid_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.valid.jsonl").resolve()),
        "--predict_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.valid.jsonl").resolve()),
        "--task", task,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", "3e-5",
        "--num_epochs", str(spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str(results_dir),
        "--save",
        "--save_dir", str(save_dir),
        "--metrics", *spec["metrics"],
        "--metric_for_valid", spec["metric_for_valid"],
        "--seed", "42",
        "--verbose",
        "--padding_side", "left",
        "--take_final",
    ]
    print(json.dumps({"event": "superglue_task_start", "task": task, "gpu": args.gpu,
                      "model_path": str(model_path), "utc": now()}), flush=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] $ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                              stdout=f, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
        f.write(f"[returncode={proc.returncode} elapsed_sec={time.time()-t0:.2f}]\n")
    if proc.returncode != 0:
        print(json.dumps({"event": "error", "task": task, "returncode": proc.returncode, "log": str(log)}), flush=True)
        sys.exit(3)

    pred = latest_file(results_dir, f"finetune/{task}/predictions.json")
    if not pred:
        print(json.dumps({"event": "error", "task": task, "reason": "no_predictions", "log": str(log)}), flush=True)
        sys.exit(4)
    rec = base.score_superglue_predictions(task, pred)
    rt = pred.parent / "results.txt"
    print(json.dumps({"event": "superglue_task_done", "task": task, "gpu": args.gpu,
                      "accuracy": rec["accuracy"], "predictions": str(pred),
                      "results_txt": str(rt), "results_txt_exists": rt.exists(),
                      "elapsed_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
