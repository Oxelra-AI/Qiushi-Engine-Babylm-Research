#!/usr/bin/env python3
"""research: Launch state-update intervention training using the exact REPRESENTATION_FRONTIER_STUDIES SOTA recipe.

Uses the validated state_update_100M.jsonl stream with the exact same:
- Tokenizer: compliant16k (frontier_consolidation)  
- Model: DeBERTa-v2 8x480, adapter128, scale1.75 (~35.5M params)
- Optimizer: AdamW, same LR/schedule
- Training: 100M words, same masking/curriculum
- Seeds: 43022 first, 43122 second (matched to baseline)

Comparison targets:
- chck_82M: Overall 41.94 (public submitted)
- chck_84M: projected Overall 42.02 (clean branch)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"

# The frontier_consolidation training script handles the exact adapter+scale1.75 recipe
TRAINER = ROOT / "experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py"
STREAM = STUDY / "data/state_update_materialization/training_corpora/state_update_100M.jsonl"
TOKENIZER = ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer/babylm_16k_compliant"

RUN_BASE = STUDY / "training/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=43022, choices=[43022, 43122])
    ap.add_argument("--gpu", type=int, default=1, choices=[0, 1])
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="Override total steps (0=full)")
    args = ap.parse_args()
    
    extra_init_seed = args.seed
    train_rng_seed = args.seed + 1
    
    run_id = f"state_update_adapter128_scale1p75_seed{args.seed}"
    run_dir = RUN_BASE / run_id
    
    if not STREAM.exists():
        print(f"ERROR: Training stream not found: {STREAM}", flush=True)
        print("Run validate_and_materialize_state_updates.py first.", flush=True)
        raise SystemExit(1)
    
    if not TOKENIZER.exists():
        print(f"ERROR: Tokenizer not found: {TOKENIZER}", flush=True)
        raise SystemExit(1)
    
    if not TRAINER.exists():
        print(f"ERROR: Trainer not found: {TRAINER}", flush=True)
        raise SystemExit(1)
    
    # Count stream words for validation
    total_words = 0
    n_rows = 0
    with STREAM.open() as f:
        for line in f:
            rec = json.loads(line)
            total_words += rec["words"]
            n_rows += 1
    
    print(f"Stream: {n_rows} rows, {total_words} words", flush=True)
    assert total_words == 100_000_000, f"Expected 100M words, got {total_words}"
    
    # Build the training command matching the exact frontier_consolidation recipe
    # The adapter_scaled_trainer uses the masking_curriculum_trainer as base
    cmd = [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(STREAM),
        "--tokenizer_path", str(TOKENIZER),
        "--output_dir", str(run_dir),
        # Model architecture: 8 layers, 480 hidden, adapter128, scale1.75
        "--n_layers", "8",
        "--d_model", "480",
        "--n_heads", "8",
        "--adapter_bottleneck", "128",
        "--adapter_scale", "1.75",
        # Training
        "--batch_size", "128",
        "--seq_len", "256",
        "--lr", "1e-3",
        "--warmup_frac", "0.06",
        "--weight_decay", "0.01",
        "--masking_rate", "0.15",
        "--wwm",
        "--curriculum", "fixed",
        "--extra_init_seed", str(extra_init_seed),
        "--train_rng_seed", str(train_rng_seed),
        # Checkpointing  
        "--save_every_m_words", "1000000",
        "--save_hf",
    ]
    
    if args.max_steps > 0:
        cmd.extend(["--max_steps", str(args.max_steps)])
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    
    launch_meta = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmd": cmd,
        "run_id": run_id,
        "seed": args.seed,
        "gpu": args.gpu,
        "stream": str(STREAM),
        "stream_words": total_words,
        "stream_rows": n_rows,
        "tokenizer": str(TOKENIZER),
        "trainer": str(TRAINER),
        "intervention": "state_update_companion_pairs",
        "description": "Entity-state-update companion pairs replacing Qwen restatement "
                       "pairs in the SOTA compact-view-reinvest recipe. Same originals, "
                       "word budget, packed-row topology, model architecture, optimizer. "
                       "Tests relation-typed readout principle on Entity column.",
    }
    
    if args.validate_only:
        print("VALIDATE-ONLY mode - not launching training", flush=True)
        print(json.dumps(launch_meta, indent=2), flush=True)
        return
    
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launch_command.json").write_text(
        json.dumps(launch_meta, indent=2, ensure_ascii=False))
    
    print(f"Launching training: {run_id}", flush=True)
    print(f"  GPU: {args.gpu}", flush=True)
    print(f"  Seed: {args.seed}", flush=True)
    print(f"  Stream: {total_words} words", flush=True)
    
    proc = subprocess.run(cmd, env=env, capture_output=False)
    
    result = {
        "run_id": run_id,
        "exit_code": proc.returncode,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
