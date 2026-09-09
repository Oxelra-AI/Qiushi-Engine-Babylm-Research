#!/usr/bin/env python3
"""Launch one 8x480/baseline16k WWM run with explicit batch/lr-total-step controls.

This is for mechanism experiments where row geometry changes the number of DataLoader
steps at a fixed 100M word budget. It exposes batch_size and lr_total_steps so a run can
hold optimizer-step/LR-time approximately constant across geometries. Uses the same base
trainer and tokenizer as the clean-Qwen/cap-120 experiments.
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
from datetime import datetime, timezone

ROOT = _public_path('experiments/archive/compact_experience')
TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr_total_steps", type=int, default=0)
    ap.add_argument("--extra_init_seed", type=int, default=43022)
    ap.add_argument("--train_rng_seed", type=int, default=43023)
    ap.add_argument("--max_word_exposure", type=int, default=100_000_000)
    ap.add_argument("--checkpoint_words", type=int, default=1_000_000)
    ap.add_argument("--fail_if_nonempty", action="store_true")
    args = ap.parse_args()
    train_file = pathlib.Path(args.train_file)
    run_dir = pathlib.Path(args.run_dir)
    if not train_file.exists():
        raise SystemExit(f"missing train file: {train_file}")
    if args.meta and not pathlib.Path(args.meta).exists():
        raise SystemExit(f"missing meta: {args.meta}")
    if args.fail_if_nonempty and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run_dir nonempty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TRAINER),
        "--example_jsonl", str(train_file),
        "--example_jsonl_label", args.label,
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "baseline16k",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", str(args.extra_init_seed),
        "--train_rng_seed", str(args.train_rng_seed),
        "--batch_size", str(args.batch_size),
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", str(args.checkpoint_words),
        "--max_word_exposure", str(args.max_word_exposure),
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]
    if args.meta:
        cmd.extend(["--example_jsonl_meta", args.meta])
    if args.lr_total_steps > 0:
        cmd.extend(["--lr_total_steps", str(args.lr_total_steps)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TMPDIR"] = f"/tmp/q_step038_{args.label[:24]}"
    (run_dir / "launch_command.json").write_text(json.dumps({"started_utc": now(), "cmd": cmd, "gpu": args.gpu, "custom_batch_note": "batch/lr_total_steps explicitly controlled", "env_subset": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "TMPDIR"]}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "launch", "run_dir": str(run_dir), "label": args.label, "gpu": args.gpu, "batch_size": args.batch_size, "lr_total_steps": args.lr_total_steps, "started_utc": now()}, indent=2), flush=True)
    with (run_dir / "train_stdout.log").open("w", encoding="utf-8") as out, (run_dir / "train_stderr.log").open("w", encoding="utf-8") as err:
        rc = subprocess.run(cmd, env=env, stdout=out, stderr=err).returncode
    payload = {"event": "finished", "run_dir": str(run_dir), "label": args.label, "gpu": args.gpu, "returncode": rc, "finished_utc": now()}
    mp = run_dir / "scientific_metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        payload.update({"word_exposure": m.get("word_exposure"), "actual_training_steps": m.get("actual_training_steps"), "loss_first": m.get("loss_first"), "loss_last": m.get("loss_last"), "saved_checkpoints": len(m.get("saved_checkpoints", []))})
    print(json.dumps(payload, indent=2), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
