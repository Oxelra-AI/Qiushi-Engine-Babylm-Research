#!/usr/bin/env python3
"""research: Launch spatial-repair compact_view_reinvest 20M pilot training.

Identical recipe to the original compact_view_reinvest (DeBERTa-v2 8×480,
baseline16k tokenizer, seq256, WWM 0.15, AdamW 0.001, batch 256, seed43022)
but trains only 20M words on the spatial-repair corpus to compare against
the existing original compact_view_reinvest chck_20M.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
WORKSPACE = ROOT
CORPUS_DIR = WORKSPACE / "data" / "spatial_repair_corpus"
TRAIN_FILE = CORPUS_DIR / "spatial_repair_100M.jsonl"
META_FILE = CORPUS_DIR / "spatial_repair_metadata.json"
RUNS_DIR = WORKSPACE / "training" / "runs"
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")

RUN_NAME = "spatial_repair_reinvest_2pass_seed43022"


def main():
    run_dir = RUNS_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    # Preflight checks
    for p, label in [(TRAINER, "trainer"), (TOKENIZER, "tokenizer"), (TRAIN_FILE, "corpus")]:
        if not pathlib.Path(p).exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    meta = json.loads(META_FILE.read_text())
    print(f"Corpus: {meta['corpus']['total_pool_words']} words/pass, "
          f"{meta['modification']['substituted_core_pairs']} substituted pairs, "
          f"spatial retention {meta['spatial_verification']['original_core_spatial_retention']:.3f} → "
          f"{meta['spatial_verification']['repaired_core_spatial_retention']:.3f}")

    cmd = [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(TRAIN_FILE),
        "--example_jsonl_label", "spatial_repair_reinvest",
        "--example_jsonl_meta", str(META_FILE),
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
        "--max_word_exposure", "19999968",  # 2 passes of 9,999,984-word pool
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]

    print(f"\nLaunching: {RUN_NAME}")
    print(f"Output: {run_dir}")
    print(f"Training: 20M words on spatial-repair corpus")
    print(f"Command: {' '.join(cmd[:6])} ...")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    # Save output
    (run_dir / "stdout.log").write_text(proc.stdout)
    (run_dir / "stderr.log").write_text(proc.stderr)

    result = {
        "status": "SPATIAL_REPAIR_TRAINING_COMPLETE" if proc.returncode == 0 else "SPATIAL_REPAIR_TRAINING_FAILED",
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "max_word_exposure": 19_999_968,
        "corpus_sha256": meta["files"]["train_100m_sha256"],
        "recipe": {
            "model": "DeBERTa-v2 8×480",
            "tokenizer": "baseline16k",
            "seq_length": 256,
            "masking": "wwm_fixed 0.15",
            "optimizer": "AdamW lr=0.001",
            "batch_size": 256,
            "seed": 43,
            "extra_init_seed": 43022,
            "train_rng_seed": 43023,
        },
    }

    if proc.returncode == 0:
        # Check checkpoint exists
        chck_20m = run_dir / "hf_model" / "chck_20M"
        if chck_20m.exists():
            result["checkpoint_20M"] = str(chck_20m)
            result["checkpoint_files"] = sorted(os.listdir(chck_20m))
        else:
            result["warning"] = "chck_20M not found after training"
    else:
        result["stderr_tail"] = proc.stderr[-2000:] if proc.stderr else ""

    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"\n{json.dumps(result, indent=2)}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
