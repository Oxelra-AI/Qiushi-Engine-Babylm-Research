#!/usr/bin/env python3
"""research: Launch full 100M spatial-repair training (contingent on 15M screen).

Only launch this if the 15M quick comparison shows clear EWoK improvement
without broad regression. Uses the same recipe as original compact_view_reinvest.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
WORKSPACE = ROOT
CORPUS_DIR = WORKSPACE / "data" / "spatial_repair_corpus"
TRAIN_FILE = CORPUS_DIR / "spatial_repair_100M.jsonl"
META_FILE = CORPUS_DIR / "spatial_repair_metadata.json"
RUNS_DIR = WORKSPACE / "training" / "runs"
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--seed", default="43022", choices=["43022", "43122"])
args = ap.parse_args()

SEED_MAP = {
    "43022": {"extra_init_seed": "43022", "train_rng_seed": "43023"},
    "43122": {"extra_init_seed": "43122", "train_rng_seed": "43123"},
}
seed_cfg = SEED_MAP[args.seed]
RUN_NAME = f"spatial_repair_100M_seed{args.seed}"

def main():
    run_dir = RUNS_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for p, label in [(TRAINER, "trainer"), (TOKENIZER, "tokenizer"), (TRAIN_FILE, "corpus")]:
        if not pathlib.Path(p).exists():
            print(f"ERROR: {label} not found: {p}"); sys.exit(1)
    
    # Pool has 9,999,984 words per pass. 10 passes = 99,999,840 words.
    MAX_EXPOSURE = 99999840  # 10 passes exactly
    
    cmd = [
        sys.executable, str(TRAINER),
        "--example_jsonl", str(TRAIN_FILE),
        "--example_jsonl_label", f"spatial_repair_reinvest_seed{args.seed}",
        "--example_jsonl_meta", str(META_FILE),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "baseline16k",
        "--hidden_size", "480", "--n_layer", "8", "--n_head", "8", "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", seed_cfg["extra_init_seed"],
        "--train_rng_seed", seed_cfg["train_rng_seed"],
        "--batch_size", "256", "--seq_length", "256", "--max_seq_length", "256",
        "--learning_rate", "0.001", "--warmup_fraction", "0.06", "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15", "--mask_prob_end", "0.15",
        "--checkpoint_words", "1000000",
        "--max_word_exposure", str(MAX_EXPOSURE),
        "--num_workers", "0", "--log_every", "50", "--dynamics_trace_every", "200",
    ]
    
    print(f"Launching: {RUN_NAME}")
    print(f"Output: {run_dir}")
    print(f"Exposure: {MAX_EXPOSURE} words (10 passes of 9,999,984)")
    
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    (run_dir / "stdout.log").write_text(proc.stdout)
    (run_dir / "stderr.log").write_text(proc.stderr)
    
    result = {
        "status": "COMPLETE" if proc.returncode == 0 else "FAILED",
        "run_name": RUN_NAME, "run_dir": str(run_dir),
        "returncode": proc.returncode, "elapsed_sec": round(elapsed, 1),
        "max_word_exposure": MAX_EXPOSURE, "seed": args.seed,
    }
    
    if proc.returncode == 0:
        # List all checkpoints
        hf_dir = run_dir / "hf_model"
        if hf_dir.exists():
            result["checkpoints"] = sorted([d.name for d in hf_dir.iterdir() if d.is_dir()])
    else:
        result["stderr_tail"] = proc.stderr[-2000:]
    
    (run_dir / "training_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
