#!/usr/bin/env python3
"""Launch one research density-overlay BabyLM training arm.

This wrapper keeps each H100 experiment independently recoverable while matching the
COMPACT_EXPERIENCE clean-Qwen 8x480/baseline16k/fixed-seq256 WWM training recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
WORKSPACE = ROOT
DATA_DIR = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard"
META = DATA_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"
RUNS_DIR = WORKSPACE / "training" / "runs"
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")

ARM_TO_FILE = {
    "cleanqwen_fineweb_repeat_near_core": "cleanqwen_fineweb_repeat_near_core_100M.jsonl",
    "cleanqwen_fineweb_near_view_core": "cleanqwen_fineweb_near_view_core_100M.jsonl",
    "cleanqwen_fineweb_repeat_compact_core_neutral": "cleanqwen_fineweb_repeat_compact_core_neutral_100M.jsonl",
    "cleanqwen_fineweb_compact_view_core_neutral": "cleanqwen_fineweb_compact_view_core_neutral_100M.jsonl",
    "cleanqwen_fineweb_compact_view_reinvest": "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "cleanqwen_fineweb_repeat_compact_reinvest": "cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl",
    "cleanqwen_lengthmatched_near_core": "cleanqwen_lengthmatched_near_core_100M.jsonl",
    "cleanqwen_lengthmatched_compact_core_neutral": "cleanqwen_lengthmatched_compact_core_neutral_100M.jsonl",
    "cleanqwen_lengthmatched_compact_reinvest": "cleanqwen_lengthmatched_compact_reinvest_100M.jsonl",
}

DEFAULT_RUN_PREFIX = {
    "cleanqwen_fineweb_repeat_near_core": "cleanqwen_fineweb_repeat_near_core_16k_seed43022",
    "cleanqwen_fineweb_near_view_core": "cleanqwen_fineweb_near_view_core_16k_seed43022",
    "cleanqwen_fineweb_repeat_compact_core_neutral": "cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022",
    "cleanqwen_fineweb_compact_view_core_neutral": "cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022",
    "cleanqwen_fineweb_compact_view_reinvest": "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
    "cleanqwen_fineweb_repeat_compact_reinvest": "cleanqwen_fineweb_repeat_compact_reinvest_16k_seed43022",
    "cleanqwen_lengthmatched_near_core": "cleanqwen_lengthmatched_near_core_16k_seed43022",
    "cleanqwen_lengthmatched_compact_core_neutral": "cleanqwen_lengthmatched_compact_core_neutral_16k_seed43022",
    "cleanqwen_lengthmatched_compact_reinvest": "cleanqwen_lengthmatched_compact_reinvest_16k_seed43022",
}

RECIPE = {
    "tokenizer_path": str(TOKENIZER),
    "tokenizer_label": "baseline16k",
    "model_family": "DeBERTa-v2 masked LM",
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "learning_rate": 0.001,
    "warmup_fraction": 0.06,
    "weight_decay": 0.01,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "checkpoint_words": 1_000_000,
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_meta() -> dict[str, Any]:
    if not META.exists():
        raise FileNotFoundError(META)
    return json.loads(META.read_text(encoding="utf-8"))


def build_command(arm: str, run_dir: pathlib.Path) -> list[str]:
    train_file = DATA_DIR / ARM_TO_FILE[arm]
    return [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(train_file),
        "--example_jsonl_label", arm,
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "baseline16k",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", "1000000",
        "--max_word_exposure", "100000000",
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]


def preflight(arm: str, run_dir: pathlib.Path, check_hash: bool = False) -> dict[str, Any]:
    meta = read_meta()
    train_file = DATA_DIR / ARM_TO_FILE[arm]
    if not TRAINER.exists():
        raise FileNotFoundError(TRAINER)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    expected_hash = meta.get("sha256", {}).get(train_file.name)
    hash_ok = None
    actual_hash = None
    if check_hash:
        actual_hash = sha256_file(train_file)
        hash_ok = (actual_hash == expected_hash)
        if not hash_ok:
            raise RuntimeError(f"training file hash mismatch for {train_file.name}: {actual_hash} != {expected_hash}")
    status = {
        "arm": arm,
        "train_file": str(train_file),
        "train_file_bytes": train_file.stat().st_size,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "hash_ok": hash_ok,
        "run_dir": str(run_dir),
        "metadata": str(META),
        "metadata_status": meta.get("status"),
        "all_exact_10M": bool(meta.get("all_exact_10M") or meta.get("audit", {}).get("all_exact_10M")),
        "write_training": bool(meta.get("write_training")),
        "recipe": RECIPE,
    }
    return status


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARM_TO_FILE))
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--run-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-hash", action="store_true")
    p.add_argument("--allow-nonempty", action="store_true")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / DEFAULT_RUN_PREFIX[args.arm]
    status = preflight(args.arm, run_dir, check_hash=args.check_hash)
    cmd = build_command(args.arm, run_dir)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    if args.dry_run:
        print(json.dumps({"status": "DENSITY_TRAIN_DRY_RUN_OK", **status}, indent=2, ensure_ascii=False))
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step016_{args.arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "arm": args.arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "DENSITY_TRAIN_FINISHED" if proc.returncode == 0 else "DENSITY_TRAIN_FAILED",
        "arm": args.arm,
        "gpu": args.gpu,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["metrics"] = {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "saved_checkpoints": len(metrics.get("saved_checkpoints", [])),
            "first_checkpoint": metrics.get("saved_checkpoints", [{}])[0].get("name") if metrics.get("saved_checkpoints") else None,
            "last_checkpoint": metrics.get("saved_checkpoints", [{}])[-1].get("name") if metrics.get("saved_checkpoints") else None,
        }
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
