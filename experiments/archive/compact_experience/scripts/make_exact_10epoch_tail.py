#!/usr/bin/env python3
"""research: create exact <=10-pass paired-alignment training files.

The Wave-1 ALIGNED/MISMATCHED runs are valid matched mechanism screens, but
100,000,000 words of exposure over a 9,999,840-word
pool is 10 full passes plus 1,600 words.  For official-candidate reruns we need
either exactly 10,000,000-word pools or an exposure cap equal to 10 * base_pool_words.

This script keeps the existing paired pools unchanged and writes exact 10-epoch
training files with 99,998,400 words each (10 full passes, no partial epoch).  It
also prints launch commands with --max_word_exposure 99998400.  The current trainer
will name the final milestone checkpoint by its requested 10M checkpoint grid; the
true final exposure is recorded in scientific_metrics.json and should be handled
explicitly in later official packaging.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

POOL_DIR = Path("experiments/archive/compact_experience/data/paired_alignment")
TRAIN_DIR = POOL_DIR / "training_exact10"
RUN_DIR = Path("experiments/archive/compact_experience/training/runs")
TRAINER = "experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py"
TOKENIZER = "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
SEED = 43
ARMS = ["aligned", "mismatched"]


def read_pool(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_exact_10_epoch(arm: str) -> dict[str, Any]:
    pool_path = POOL_DIR / f"{arm}_pool.jsonl"
    rows = read_pool(pool_path)
    pool_words = sum(int(r["words"]) for r in rows)
    assert pool_words == 9_999_840, (arm, pool_words)
    target = pool_words * 10
    out_dir = TRAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{arm}_exact10_99998400.jsonl"
    written_words = 0
    written_rows = 0
    with out_path.open("w", encoding="utf-8") as f:
        for epoch in range(10):
            epoch_rows = list(rows)
            random.Random(SEED + 1000003 * epoch).shuffle(epoch_rows)
            for row in epoch_rows:
                rec = dict(row)
                rec["source"] = f"epoch{epoch+1}::{row.get('source', 'unknown')}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written_rows += 1
                written_words += int(row["words"])
    assert written_words == target, (arm, written_words, target)
    run_name = f"step018_{arm}_exact10_99998400_seed43"
    output_dir = RUN_DIR / run_name
    cmd = [
        "CUDA_VISIBLE_DEVICES=0",
        "TMPDIR=/tmp/q_compact_experience_exact10",
        "python", TRAINER,
        "--tokenizer_path", TOKENIZER,
        "--example_jsonl", str(out_path),
        "--example_jsonl_label", f"{arm}_exact10",
        "--max_word_exposure", str(target),
        "--output_dir", str(output_dir),
        "--model_type", "deberta_v2",
        "--n_layer", "8",
        "--hidden_size", "480",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--max_seq_length", "256",
        "--seq_length", "256",
        "--batch_size", "256",
        "--learning_rate", "1e-3",
        "--weight_decay", "0.01",
        "--warmup_fraction", "0.05",
        "--seed", str(SEED),
        "--mask_prob", "0.15",
        "--mask_mode", "wwm",
        "--num_workers", "0",
        "--checkpoint_words", "10000000",
        "--log_every", "50",
    ]
    return {
        "arm": arm,
        "pool_path": str(pool_path),
        "pool_words": pool_words,
        "target_exposure": target,
        "expanded_path": str(out_path),
        "rows": written_rows,
        "words": written_words,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "command": " ".join(cmd),
    }


def main() -> None:
    payload = {
        "status": "EXACT_10EPOCH_TRAINING_FILES_DONE",
        "note": "Exact 10-pass training files for the existing 9,999,840-word paired pools. These fix the literal epoch issue without changing the pool text.",
        "arms": [write_exact_10_epoch(a) for a in ARMS],
    }
    out = POOL_DIR / "exact10_training_files.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
