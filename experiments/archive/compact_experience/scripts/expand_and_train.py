#!/usr/bin/env python3
"""research: Expand pool JSONL to 100M-word training file (10 shuffled epochs)
and launch training on the paired-alignment arms.

The full-cycle trainer's --example_jsonl loads examples in order without epoch
cycling. To get 100M word exposure from a ~10M pool, we pre-bake 10 shuffled
epochs into the training JSONL. Each epoch shuffles the example ORDER (not
within-example structure), so alignment/mismatched structure is preserved.

Usage:
  python expand_and_train.py --expand-only   # Just create training files
  python expand_and_train.py --train ARM     # Train a specific arm
  python expand_and_train.py --launch-all    # Launch all arms
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

POOL_DIR = Path("experiments/archive/compact_experience/data/paired_alignment")
TRAIN_DIR = Path("experiments/archive/compact_experience/data/paired_alignment/training_expanded")
RUN_DIR = Path("experiments/archive/compact_experience/training/runs")

TRAINER = "experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py"
TOKENIZER = "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"

SEED = 43
TARGET_EXPOSURE = 100_000_000  # 100M word exposure
WORDS_PER_EXAMPLE = 160

# Training recipe (matches INITIAL_MODEL_STUDIES research baseline)
TRAIN_ARGS = {
    "model_type": "deberta_v2",
    "n_layer": 8,
    "hidden_size": 480,
    "n_head": 8,
    "intermediate_size": 1920,
    "max_seq_length": 256,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "warmup_fraction": 0.05,
    "seed": SEED,
    "mask_prob": 0.15,
    "wwm": "true",
    "num_workers": 0,
    "checkpoint_words": "1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000,20000000,30000000,40000000,50000000,60000000,70000000,80000000,90000000,100000000",
}

ARMS = {
    "aligned": "aligned_pool.jsonl",
    "mismatched": "mismatched_pool.jsonl",
    "single_repeat": "single_repeat_pool.jsonl",
    "single_orig": "single_orig_pool.jsonl",
}


# ──────────────────────────────────────────────────────────────────────────────
# Epoch expansion
# ──────────────────────────────────────────────────────────────────────────────

def expand_pool(arm_name: str, pool_file: str) -> Path:
    """Expand a pool JSONL to ~100M words via shuffled epochs."""
    pool_path = POOL_DIR / pool_file
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool not found: {pool_path}")
    
    # Load pool
    examples = []
    with pool_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    pool_words = sum(e["words"] for e in examples)
    n_epochs_needed = (TARGET_EXPOSURE + pool_words - 1) // pool_words
    print(f"  {arm_name}: {len(examples)} examples, {pool_words:,} pool words, "
          f"need {n_epochs_needed} epochs for {TARGET_EXPOSURE:,} exposure")
    
    # Build training sequence: shuffled epochs
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAIN_DIR / f"{arm_name}_100M.jsonl"
    
    written_words = 0
    written_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        epoch = 0
        while written_words < TARGET_EXPOSURE:
            epoch_examples = list(examples)
            shuffle_seed = SEED + 1000003 * epoch
            random.Random(shuffle_seed).shuffle(epoch_examples)
            
            for ex in epoch_examples:
                if written_words >= TARGET_EXPOSURE:
                    break
                # Update source to include epoch info
                ex_out = dict(ex)
                ex_out["source"] = f"epoch{epoch+1}::{ex.get('source', 'unknown')}"
                
                if written_words + ex["words"] <= TARGET_EXPOSURE:
                    f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                    written_words += ex["words"]
                    written_rows += 1
                else:
                    # Partial last example to exactly hit target
                    take = TARGET_EXPOSURE - written_words
                    if take > 0:
                        partial_text = " ".join(ex["text"].split()[:take])
                        ex_out["text"] = partial_text
                        ex_out["words"] = take
                        f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                        written_words += take
                        written_rows += 1
                    break
            epoch += 1
    
    print(f"    → {out_path}: {written_rows} rows, {written_words:,} words, {epoch} epochs")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Training launcher
# ──────────────────────────────────────────────────────────────────────────────

def build_train_cmd(arm_name: str, expanded_path: Path, gpu: int) -> list[str]:
    """Build training command for one arm."""
    run_name = f"step017_{arm_name}_100M_seed43"
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pool_words = sum(1 for _ in open(expanded_path)) * WORDS_PER_EXAMPLE  # approximate
    
    cmd = [
        sys.executable, str(TRAINER),
        "--tokenizer_path", str(TOKENIZER),
        "--example_jsonl", str(expanded_path),
        "--example_jsonl_label", arm_name,
        "--max_word_exposure", str(TARGET_EXPOSURE),
        "--output_dir", str(output_dir),
    ]
    
    for key, val in TRAIN_ARGS.items():
        cmd.extend([f"--{key}", str(val)])
    
    return cmd, run_name, output_dir


def train_arm(arm_name: str, gpu: int = 0):
    """Train one arm on specified GPU."""
    expanded_path = TRAIN_DIR / f"{arm_name}_100M.jsonl"
    if not expanded_path.exists():
        print(f"Expanding {arm_name} pool first...")
        expanded_path = expand_pool(arm_name, ARMS[arm_name])
    
    cmd, run_name, output_dir = build_train_cmd(arm_name, expanded_path, gpu)
    
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu} TMPDIR=/tmp/q_compact_experience_{arm_name}"
    full_cmd = f"{env_prefix} {' '.join(cmd)}"
    
    print(f"\n  Training {arm_name} on GPU {gpu}")
    print(f"  Output: {output_dir}")
    print(f"  Command (abbreviated): CUDA_VISIBLE_DEVICES={gpu} python {TRAINER} --example_jsonl {expanded_path} ...")
    
    return full_cmd, run_name, output_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expand-only", action="store_true", help="Only create expanded training files")
    parser.add_argument("--train", type=str, help="Train a specific arm (aligned/mismatched/single_repeat/single_orig)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--launch-all", action="store_true", help="Print launch commands for all arms")
    args = parser.parse_args()
    
    if args.expand_only:
        print("Expanding all pools to 100M training files...")
        for arm_name, pool_file in ARMS.items():
            expand_pool(arm_name, pool_file)
        print("\nDone. Training files in:", TRAIN_DIR)
        return
    
    if args.train:
        if args.train not in ARMS:
            print(f"Unknown arm: {args.train}. Choose from: {list(ARMS.keys())}")
            return
        cmd, name, out = train_arm(args.train, args.gpu)
        print(f"\n  Full command:\n  {cmd}")
        return
    
    if args.launch_all:
        print("=" * 60)
        print("LAUNCH PLAN: 4-arm paired-alignment training")
        print("=" * 60)
        
        # Expand all
        print("\n--- Expanding pools ---")
        for arm_name, pool_file in ARMS.items():
            expand_pool(arm_name, pool_file)
        
        # Generate launch commands
        print("\n--- Training commands ---")
        print("\n# Wave 1: ALIGNED (GPU0) + MISMATCHED (GPU1)")
        cmd_a, _, _ = train_arm("aligned", 0)
        cmd_m, _, _ = train_arm("mismatched", 1)
        
        print(f"\n# Wave 2: SINGLE_REPEAT (GPU0) + SINGLE_ORIG (GPU1)")
        cmd_r, _, _ = train_arm("single_repeat", 0)
        cmd_o, _, _ = train_arm("single_orig", 1)
        
        # Write shell script
        script = POOL_DIR.parent.parent / "scripts" / "launch_training.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        with script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("# research: Launch paired-alignment training\n")
            f.write("# Wave 1: critical pair (ALIGNED + MISMATCHED)\n")
            f.write(f"mkdir -p /tmp/q_compact_experience_aligned /tmp/q_compact_experience_mismatched\n")
            f.write(f"{cmd_a} > {RUN_DIR}/aligned_100M_seed43/train_stdout.log 2>&1 &\n")
            f.write(f"PID1=$!\n")
            f.write(f"{cmd_m} > {RUN_DIR}/mismatched_100M_seed43/train_stdout.log 2>&1 &\n")
            f.write(f"PID2=$!\n")
            f.write(f"echo \"Wave 1 launched: ALIGNED pid=$PID1, MISMATCHED pid=$PID2\"\n")
            f.write(f"wait $PID1 $PID2\n")
            f.write(f"echo \"Wave 1 complete\"\n\n")
            f.write(f"# Wave 2: controls (SINGLE_REPEAT + SINGLE_ORIG)\n")
            f.write(f"mkdir -p /tmp/q_compact_experience_single_repeat /tmp/q_compact_experience_single_orig\n")
            f.write(f"{cmd_r} > {RUN_DIR}/single_repeat_100M_seed43/train_stdout.log 2>&1 &\n")
            f.write(f"PID3=$!\n")
            f.write(f"{cmd_o} > {RUN_DIR}/single_orig_100M_seed43/train_stdout.log 2>&1 &\n")
            f.write(f"PID4=$!\n")
            f.write(f"echo \"Wave 2 launched: SINGLE_REPEAT pid=$PID3, SINGLE_ORIG pid=$PID4\"\n")
            f.write(f"wait $PID3 $PID4\n")
            f.write(f"echo \"Wave 2 complete\"\n")
            f.write(f"echo \"All training done\"\n")
        script.chmod(0o755)
        print(f"\n  Launch script written: {script}")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
