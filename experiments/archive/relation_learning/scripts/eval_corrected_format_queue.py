#!/usr/bin/env python3
"""Evaluate corrected research format-replay endpoints when training completes.

This script is dependency-aware: it waits for each corrected trainer output to write
`summary.json` and an alpha endpoint, then materializes a run-like directory under the
evaluation output root and invokes the proven research cheap7 evaluator.  It does not
write into the training directories, so it can be launched while the training queues
are still running.
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
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
TRAIN_ROOT = _public_path('experiments/archive/relation_learning/data/format_replay_corrected')
OUT_ROOT = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected')
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')

INITIAL_CONSUMED_WORDS = 82_012_495
LR_TOTAL_STEPS = 455


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_path(path: pathlib.Path, timeout_sec: float, interval_sec: float) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start >= timeout_sec:
            raise TimeoutError(f"waited {timeout_sec}s for {path}")
        time.sleep(interval_sec)


def read_training_log(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def make_runlike(train_dir: pathlib.Path, label: str, alpha: float, eval_root: pathlib.Path) -> pathlib.Path:
    summary = read_json(train_dir / "summary.json")
    config = read_json(train_dir / "train_config.json") if (train_dir / "train_config.json").exists() else {}
    logs = read_training_log(train_dir / "training_log.jsonl")
    first = logs[0] if logs else (summary.get("first_update") or {})
    last = logs[-1] if logs else (summary.get("last_update") or {})
    alpha_src = train_dir / f"alpha_{alpha:.2f}"
    ckpt_src = train_dir / "checkpoint"
    model_src = alpha_src if (alpha_src / "model.safetensors").exists() else ckpt_src
    if not (model_src / "model.safetensors").exists():
        raise FileNotFoundError(f"No model.safetensors under {model_src}")

    runlike = eval_root / "runlike" / label
    hf_final = runlike / "hf_model" / "final"
    runlike.mkdir(parents=True, exist_ok=True)
    copy_tree(model_src, hf_final)
    total_words = int(summary.get("total_words") or config.get("total_words_scheduled") or config.get("total_words_loaded") or 3_992_800)
    metrics = {
        "status": "CORRECTED_FORMAT_REPLAY_ENDPOINT",
        "mode": "corrected_format_replay",
        "replay_mode": config.get("arm_label", label),
        "endpoint": rel(model_src),
        "initial_consumed_words": INITIAL_CONSUMED_WORDS,
        "skip_rows": 530944,
        "max_tail_charged_words": total_words,
        "tail_main_word_exposure": total_words,
        "tail_aux_word_exposure": 0,
        "tail_charged_words": total_words,
        "total_consumed_words": INITIAL_CONSUMED_WORDS + total_words,
        "updates": int(summary.get("updates") or config.get("macro_updates") or len(logs)),
        "schedule_total": LR_TOTAL_STEPS,
        "stopped_before_cap": False,
        "trainable": "private_adapter_only",
        "total_params": config.get("total_params"),
        "private_params": config.get("trainable_params"),
        "frozen_slow_params": None if config.get("total_params") is None or config.get("trainable_params") is None else int(config["total_params"]) - int(config["trainable_params"]),
        "first_main_loss": first.get("main_ce"),
        "final_main_loss": last.get("main_ce"),
        "mean_main_loss": mean([float(x["main_ce"]) for x in logs]) if logs else None,
        "main_loss_batches": len(logs),
        "first_neutral_loss": first.get("readout_neutral_kl"),
        "final_neutral_loss": last.get("readout_neutral_kl"),
        "mean_neutral_loss": mean([float(x.get("readout_neutral_kl", 0.0)) for x in logs]) if logs else None,
        "first_leash_neutral_loss": first.get("leash_neutral_kl"),
        "final_leash_neutral_loss": last.get("leash_neutral_kl"),
        "mean_leash_neutral_loss": mean([float(x.get("leash_neutral_kl", 0.0)) for x in logs]) if logs else None,
        "deterministic_neutrality_eval_mode": True,
        "first_target_ratio": first.get("target_ratio"),
        "final_target_ratio": last.get("target_ratio"),
        "mean_target_ratio": mean([float(x.get("target_ratio", 0.0)) for x in logs]) if logs else None,
        "first_update": first,
        "last_update": last,
        "source_train_dir": rel(train_dir),
        "created_utc": now(),
        "scientific_note": "corrected format-replay screen: token-start WWM, matched macro-updates/words, separate coherent leash and no-gradient readout sets",
    }
    (runlike / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runlike


def run_eval(train_dir: pathlib.Path, label: str, gpu: int, alpha: float, eval_root: pathlib.Path, force: bool) -> dict[str, Any]:
    wait_for_path(train_dir / "summary.json", args_wait_timeout, args_wait_interval)
    runlike = make_runlike(train_dir, label, alpha, eval_root)
    out_root = eval_root / "eval" / label
    collate_root = eval_root / "collate" / label
    summary_root = eval_root / "summary"
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--run-dir", str(runlike),
        "--target", label,
        "--endpoint", "final",
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--summary-root", str(summary_root),
        "--gpu", str(gpu),
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(json.dumps({"event": "eval_start", "label": label, "train_dir": rel(train_dir), "gpu": gpu, "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=7200)
    log_dir = eval_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}_stdout.log").write_text(p.stdout, encoding="utf-8")
    (log_dir / f"{label}_stderr.log").write_text(p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-3000:], flush=True)
        print(p.stderr[-3000:], flush=True)
        raise RuntimeError(f"research eval failed label={label} rc={p.returncode}")
    summary_path = summary_root / f"{label}_summary.json"
    payload_path = out_root / "per_target" / f"{label}.json"
    rec = {
        "label": label,
        "train_dir": rel(train_dir),
        "runlike": rel(runlike),
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
        "summary_path": rel(summary_path),
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / f"{label}_stdout.log"),
        "stderr_log": rel(log_dir / f"{label}_stderr.log"),
    }
    if summary_path.exists():
        s = read_json(summary_path)
        rec["cheap7"] = s.get("cheap7")
        rec["cheap7_delta_vs_chck82"] = s.get("cheap7_delta_vs_chck82")
        rec["scores"] = s.get("scores")
    print(json.dumps({"event": "eval_finished", **rec}, ensure_ascii=False), flush=True)
    return rec


# Globals used by run_eval wait helper; set in main.
args_wait_timeout = 0.0
args_wait_interval = 30.0


def main() -> None:
    global args_wait_timeout, args_wait_interval
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["isolated_all", "half_coherent_half_isolated", "coherent_unsplit_special"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--wait-timeout", type=float, default=14400.0)
    ap.add_argument("--wait-interval", type=float, default=60.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args_wait_timeout = float(args.wait_timeout)
    args_wait_interval = float(args.wait_interval)

    eval_root = OUT_ROOT / args.arm
    eval_root.mkdir(parents=True, exist_ok=True)
    # Wait for the whole arm queue to finish before starting any evaluation on the
    # same physical GPU.  Otherwise seed98097 evaluation can overlap seed98098
    # training and turn the screen into a mixed compute/schedule artifact.
    queue_summary = TRAIN_ROOT / args.arm / "queue_summary.json"
    wait_for_path(queue_summary, args_wait_timeout, args_wait_interval)
    qsum = read_json(queue_summary)
    if qsum.get("status") != "FORMAT_QUEUE_DONE":
        raise RuntimeError(f"training queue did not finish cleanly: {queue_summary} status={qsum.get('status')}")
    records = []
    for seed in args.seeds:
        train_dir = TRAIN_ROOT / args.arm / f"seed{seed}"
        label = f"step098_{args.arm}_seed{seed}_alpha0p75"
        records.append(run_eval(train_dir, label, args.gpu, args.alpha, eval_root, args.force))
    summary = {"status": "FORMAT_EVAL_QUEUE_DONE", "created_utc": now(), "arm": args.arm, "records": records}
    (eval_root / "queue_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
