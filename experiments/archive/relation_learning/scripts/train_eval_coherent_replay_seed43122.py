#!/usr/bin/env python3
"""research: train and cheap7-evaluate a second coherent replay private seed.

This is a reference-variance run for allocation/composition arms.  It repeats the
coherent86 private-only suffix replay from the same frozen chck_82M slow path and the
same legal suffix, changing only the training/masking RNG seed.  The inference alpha
is fixed at 0.75 to match coherent86 before scores are read.
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
from statistics import mean
from typing import Any

ROOT = _public_path('.')
TRAINER = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_fastpath_replay_trainer.py')
MATERIALIZER = _public_path('experiments/archive/frontier_consolidation/scripts/materialize_private_scale.py')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')
DEFAULT_TRAIN_DIR = _public_path('experiments/archive/relation_learning/training/runs/coherent_replay_seed43122')
DEFAULT_ALPHA_DIR = _public_path('experiments/archive/relation_learning/training/runs/coherent_replay_seed43122_alpha0p75')
DEFAULT_EVAL_ROOT = _public_path('experiments/archive/relation_learning/data/coherent_seed43122_eval')
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"event": "subprocess_start", "cmd": [str(x) for x in cmd], "utc": now()}), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    print(json.dumps({"event": "subprocess_done", "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-3000:], "utc": now()}), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}\nSTDOUT={proc.stdout[-4000:]}\nSTDERR={proc.stderr[-6000:]}")
    return proc


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=pathlib.Path, default=DEFAULT_TRAIN_DIR)
    ap.add_argument("--alpha-dir", type=pathlib.Path, default=DEFAULT_ALPHA_DIR)
    ap.add_argument("--eval-root", type=pathlib.Path, default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43122)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    train_dir = args.train_dir
    alpha_dir = args.alpha_dir
    eval_root = args.eval_root
    train_done = train_dir / "scientific_metrics.json"

    if args.force_train and train_dir.exists():
        import shutil
        shutil.rmtree(train_dir)
    if not train_done.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-B", str(TRAINER),
            "--output_dir", str(train_dir),
            "--replay_mode", "coherent_replay",
            "--train_rng_seed", str(args.seed),
            "--span_shuffle_seed", str(args.seed + 101),
            "--private_adapter_scale", "1.0",
            "--log_every", "25",
        ]
        run(cmd, timeout=1800)
    train_metrics = read_json(train_done)

    if args.force_train and alpha_dir.exists():
        import shutil
        shutil.rmtree(alpha_dir)
    endpoint = alpha_dir / "hf_model" / "final"
    if not (endpoint / "model.safetensors").exists():
        cmd = [sys.executable, "-B", str(MATERIALIZER), "--source", str(train_dir / "hf_model" / "final"), "--output", str(endpoint), "--scale", "0.75", "--force"]
        mat = run(cmd, timeout=900)
    else:
        mat = None
    metrics = dict(train_metrics)
    metrics.update({
        "status": "COHERENT_REPLAY_SEED43122_ALPHA075_ENDPOINT",
        "source_train_metrics_path": rel(train_done),
        "source_train_run_dir": rel(train_dir),
        "private_adapter_scale_materialized": 0.75,
        "reference_role": "second coherent private-replay seed; alpha fixed to coherent86 value before reading allocation-arm scores",
        "created_utc": now(),
    })
    write_json(alpha_dir / "scientific_metrics.json", metrics)

    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--run-dir", str(alpha_dir),
        "--target", "coherent_seed43122_alpha0p75",
        "--endpoint", "final",
        "--out-root", str(eval_root / "eval"),
        "--collate-root", str(eval_root / "collate"),
        "--summary-root", str(eval_root / "summary"),
        "--gpu", str(args.gpu),
        "--force" if args.force_eval else "",
    ]
    cmd = [x for x in cmd if x != ""]
    run(cmd, timeout=9000)
    summary_path = eval_root / "summary" / "coherent_seed43122_alpha0p75_summary.json"
    summary = read_json(summary_path)
    out = {
        "status": "COHERENT_REPLAY_SEED43122_TRAIN_EVAL_DONE",
        "created_utc": now(),
        "train_dir": rel(train_dir),
        "alpha_dir": rel(alpha_dir),
        "eval_summary": rel(summary_path),
        "train_tail_charged_words": train_metrics.get("tail_charged_words"),
        "train_total_consumed_words": train_metrics.get("total_consumed_words"),
        "scores": summary.get("scores"),
        "cheap7": summary.get("cheap7"),
        "cheap7_delta_vs_chck82": summary.get("cheap7_delta_vs_chck82"),
        "materializer_stdout_tail": None if mat is None else mat.stdout[-1000:],
    }
    write_json(eval_root / "summary.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
