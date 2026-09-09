#!/usr/bin/env python3
"""research: Train compact_view_reinvest with legal 40k byte-BPE tokenizer.

Single-factor vocabulary-capacity experiment. Changes ONLY the tokenizer from
the legal 16k to a legal 40k byte-BPE, both trained on the exact same 10M
pool. Architecture, optimizer, data stream, seeds, and all other hyperparameters
are identical to the research legal-16k retrains.

Model size increases by ~11M parameters (embedding 40k×480 vs 16k×480).
This is an unavoidable confound of the vocabulary-capacity test.

Usage:
  python3 legal40k_compact_reinvest_train.py            # both seeds
  python3 legal40k_compact_reinvest_train.py --seed 43022  # one seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
AI_RUN_ROOT = WS / "training/runs"
OUT_DIR = WS / "data/legal40k_training"

TOKENIZER_DIR = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"
TRAIN_100M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")
TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")

EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOK_SHA = "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758"

SEEDS = {
    "43022": {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023},
    "43122": {"seed": 43, "extra_init_seed": 43122, "train_rng_seed": 43123},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight() -> dict[str, Any]:
    """Verify all inputs before launching expensive GPU work."""
    errors = []

    # Tokenizer
    tok_json = TOKENIZER_DIR / "tokenizer.json"
    if not tok_json.exists():
        errors.append(f"tokenizer.json not found at {tok_json}")
    else:
        tok_sha = sha256_file(tok_json)
        if tok_sha != EXPECTED_TOK_SHA:
            errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOK_SHA}")

    # Training data
    if not TRAIN_100M.exists():
        errors.append(f"100M stream not found: {TRAIN_100M}")
    else:
        stream_sha = sha256_file(TRAIN_100M)
        if stream_sha != EXPECTED_100M_SHA:
            errors.append(f"100M stream SHA mismatch: {stream_sha} != {EXPECTED_100M_SHA}")

    # Metadata
    if not META.exists():
        errors.append(f"Overlay metadata not found: {META}")

    # Trainer
    if not TRAINER.exists():
        errors.append(f"Trainer not found: {TRAINER}")

    # Tokenizer functional check
    if not errors:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
            assert tok.vocab_size == 40000, f"Expected 40k vocab, got {tok.vocab_size}"
            assert tok.is_fast, "Not a fast tokenizer"
            enc = tok("test word", add_special_tokens=False)
            assert enc.word_ids() is not None, "word_ids() failed"
        except Exception as e:
            errors.append(f"Tokenizer functional check failed: {e}")

    return {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "train_file": str(TRAIN_100M),
        "trainer": str(TRAINER),
    }


def train_command(seed_key: str) -> tuple[list[str], pathlib.Path, dict]:
    s = SEEDS[seed_key]
    run_dir = AI_RUN_ROOT / f"legal40k_compact_view_reinvest_seed{seed_key}"
    cmd = [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(TRAIN_100M),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_legal40k",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", "legal_byte_bpe_40k",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", str(s["seed"]),
        "--extra_init_seed", str(s["extra_init_seed"]),
        "--train_rng_seed", str(s["train_rng_seed"]),
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
    manifest = {
        "status": "LEGAL40K_TRAIN_COMMAND",
        "seed_key": seed_key,
        "run_dir": str(run_dir),
        "train_file": str(TRAIN_100M),
        "train_file_sha256": EXPECTED_100M_SHA,
        "tokenizer_path": str(TOKENIZER_DIR),
        "tokenizer_sha256": EXPECTED_TOK_SHA,
        "changed_factor": "tokenizer only: legal 40k byte-BPE vs legal 16k byte-BPE on same 10M pool",
        "fixed_recipe": {
            "model_family": "DeBERTa-v2 masked LM",
            "hidden_size": 480,
            "n_layer": 8,
            "n_head": 8,
            "ffn_mult": 4,
            "batch_size": 256,
            "seq_length": 256,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "warmup_fraction": 0.06,
            "weight_decay": 0.01,
            "masking": "wwm_fixed at 0.15",
            "data_order_seed": 43,
            "exposure": "100M words (10 passes)",
            "checkpoints": "every 1M words",
        },
        "parameter_count_note": "~43M total (~19.2M embedding at 40k×480). Legal 16k was ~32M (~7.9M embedding).",
        "cmd": cmd,
    }
    return cmd, run_dir, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, choices=["43022", "43122"], default=None,
                        help="Train one seed. Default: both.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preflight
    pf = preflight()
    print(json.dumps(pf, indent=2), flush=True)
    if pf["status"] != "PREFLIGHT_OK":
        sys.exit(1)

    seed_keys = [args.seed] if args.seed else ["43022", "43122"]
    
    for sk in seed_keys:
        cmd, run_dir, manifest = train_command(sk)
        manifest_path = OUT_DIR / f"legal40k_train_manifest_seed{sk}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSeed {sk} manifest saved: {manifest_path}", flush=True)
        print(f"  run_dir: {run_dir}", flush=True)
        print(f"  cmd (first 6 args): {' '.join(str(c) for c in cmd[:6])}", flush=True)

    if args.dry_run:
        print("\n[DRY RUN] Preflight passed. Commands ready but not launched.")
        return

    # Launch training for each seed
    for sk in seed_keys:
        cmd, run_dir, manifest = train_command(sk)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}", flush=True)
        print(f"Launching seed {sk}...", flush=True)
        print(f"  run_dir: {run_dir}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        result = {
            "status": "TRAINING_COMPLETE" if proc.returncode == 0 else "TRAINING_FAILED",
            "seed_key": sk,
            "run_dir": str(run_dir),
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }
        result_path = OUT_DIR / f"legal40k_train_result_seed{sk}.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2), flush=True)
        if proc.returncode != 0:
            print(f"STDERR tail: {proc.stderr[-500:]}", flush=True)


if __name__ == "__main__":
    main()
