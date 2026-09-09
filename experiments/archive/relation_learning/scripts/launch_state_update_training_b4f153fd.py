#!/usr/bin/env python3
"""research: Launch state-update intervention training using the exact REPRESENTATION_FRONTIER_STUDIES SOTA recipe.

Corrected from research launcher which had wrong tokenizer path and incompatible CLI flags.
Verified against:
  - frontier_consolidation notes/105_scale1p75_50m_prefix_and_80m_launch_note.md (exact command)
  - frontier_consolidation notes/scale1p75_endpoint_continuation_plan.md (exact command)
  - relation_learning research SHUF launch_command.json (verified trainer interface)

Uses adapter_scaled_trainer.py (adapter128, scale1.75) wrapping the
COMPACT_EXPERIENCE masking_curriculum_trainer.py base. Same model, optimizer, schedule,
tokenizer, and word-budget accounting as the compact-view-reinvest SOTA.
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

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"

# Verified components
TRAINER = ROOT / "experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py"
TOKENIZER = ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
# Stream path is set by argument; default points to the materialized output
DEFAULT_STREAM = STUDY / "data/state_use_generation/training_corpora/state_update_100M.jsonl"

RUN_BASE = STUDY / "training/runs"

# Verified seed convention: from research SHUF launch_command.json
# seed43022: --seed 43 --extra_init_seed 43022 --train_rng_seed 43023
# seed43122: --seed 43 --extra_init_seed 43122 --train_rng_seed 43123
SEED_MAP = {
    43022: {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023},
    43122: {"seed": 43, "extra_init_seed": 43122, "train_rng_seed": 43123},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=43022, choices=[43022, 43122])
    ap.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stream", type=str, default=str(DEFAULT_STREAM),
                    help="Path to the 100M-word training JSONL")
    ap.add_argument("--stream-meta", type=str, default="",
                    help="Path to the stream metadata JSON (optional)")
    ap.add_argument("--validate-only", action="store_true",
                    help="Print command and exit without training")
    ap.add_argument("--run-label", type=str, default="",
                    help="Override run directory label")
    args = ap.parse_args()

    stream_path = pathlib.Path(args.stream)
    seeds = SEED_MAP[args.seed]

    # Validate prerequisites
    errors = []
    if not stream_path.exists():
        errors.append(f"Stream not found: {stream_path}")
    if not TOKENIZER.exists():
        errors.append(f"Tokenizer not found: {TOKENIZER}")
    if not TRAINER.exists():
        errors.append(f"Trainer not found: {TRAINER}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", flush=True)
        raise SystemExit(1)

    # Validate stream word count
    total_words = 0
    n_rows = 0
    with stream_path.open() as f:
        for line in f:
            rec = json.loads(line)
            total_words += rec.get("words", 0)
            n_rows += 1

    print(f"Stream: {n_rows} rows, {total_words:,} words", flush=True)
    if total_words != 100_000_000:
        print(f"WARNING: Expected 100M words, got {total_words:,}", flush=True)

    # Run directory
    label = args.run_label or f"state_update_adapter128_scale1p75_seed{args.seed}"
    run_dir = RUN_BASE / label

    # Build the training command — exact match to verified REPRESENTATION_FRONTIER_STUDIES/COMPACT_EXPERIENCE interface
    cmd = [
        sys.executable, "-B",
        str(TRAINER),
        # Adapter parameters (captured by research before passing to base trainer)
        "--adapter_bottleneck", "128",
        "--adapter_enabled", "1",
        "--adapter_scale", "1.75",
        "--gpu", str(args.gpu),
        # Data
        "--example_jsonl", str(stream_path),
        "--example_jsonl_label", label,
        "--output_dir", str(run_dir),
        # Tokenizer (verified: direct path, no subdirectory)
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        # Model architecture: DeBERTa-v2 8x480
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        # Seeds (verified convention from research launch_command.json)
        "--seed", str(seeds["seed"]),
        "--extra_init_seed", str(seeds["extra_init_seed"]),
        "--train_rng_seed", str(seeds["train_rng_seed"]),
        # Training hyperparameters
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        # Masking
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        # Budget and checkpointing
        "--checkpoint_words", "1000000",
        "--max_word_exposure", "100000000",
        # Logging
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]

    if args.stream_meta:
        cmd.extend(["--example_jsonl_meta", args.stream_meta])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"

    launch_meta = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmd": cmd,
        "run_label": label,
        "run_dir": str(run_dir),
        "seed": args.seed,
        "seeds": seeds,
        "gpu": args.gpu,
        "stream": str(stream_path),
        "stream_words": total_words,
        "stream_rows": n_rows,
        "stream_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest() if total_words > 0 else "",
        "tokenizer": str(TOKENIZER),
        "trainer": str(TRAINER),
        "adapter_bottleneck": 128,
        "adapter_scale": 1.75,
        "intervention": "state_update_companion_pairs",
        "description": (
            "Entity-state-update three-part companion packets (source/update/use) "
            "replacing Qwen restatement pairs in the SOTA compact-view-reinvest recipe. "
            "Same originals subset, word budget, packed-row topology, tokenizer, model "
            "architecture, adapter128 scale1.75, optimizer, and schedule. "
            "Tests relation-typed readout principle on Entity column via state-update "
            "relations with updated-use and unchanged-distractor-use packets."
        ),
    }

    if args.validate_only:
        print("VALIDATE-ONLY — command generated but not launched:", flush=True)
        print(json.dumps(launch_meta, indent=2), flush=True)
        print("\nCommand:", flush=True)
        print(" ".join(cmd), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launch_command.json").write_text(
        json.dumps(launch_meta, indent=2, ensure_ascii=False))

    print(f"Launching: {label}", flush=True)
    print(f"  GPU: {args.gpu}", flush=True)
    print(f"  Seed: {args.seed} (init={seeds['extra_init_seed']}, rng={seeds['train_rng_seed']})", flush=True)
    print(f"  Stream: {total_words:,} words, {n_rows} rows", flush=True)
    print(f"  Adapter: bottleneck=128, scale=1.75", flush=True)

    proc = subprocess.run(cmd, env=env, capture_output=False)

    result = {
        "run_label": label,
        "exit_code": proc.returncode,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
