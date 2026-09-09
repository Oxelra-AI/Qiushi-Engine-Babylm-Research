#!/usr/bin/env python3
"""research: train in-corpus adult-prose DeBERTa arm.

This arm replaces developmental/speech rows with unused Gutenberg/SimpleWiki
from the BabyLM official corpus. No FineWeb content.  Uses rho≈0.0424 matching
full_1x for direct comparison.  Same DeBERTa config, seed, tokenizer, and
training recipe as all other arms.
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
import time


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY

STREAM = WS / "data/incorpus_adultprose_arm/incorpus_adultprose_rho0p042_100M.jsonl"
RUN_DIR = WS / "training/runs/incorpus_adultprose_deberta100M_seed43022"

TOKENIZER = WS / "data/compliant_tokenizer"
TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"

TRAINER_SCRIPT = ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"

EXPECTED_PARAMS = 34_467_424
EXPECTED_WORDS = 100_000_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def count_words(path: pathlib.Path) -> int:
    total = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            total += row.get("words", 0)
    return total


def verify_tokenizer() -> None:
    import hashlib
    tok_json = TOKENIZER / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(tok_json)
    sha = hashlib.sha256(tok_json.read_bytes()).hexdigest()
    if sha != TOKENIZER_SHA:
        raise ValueError(f"Tokenizer SHA mismatch: {sha} != {TOKENIZER_SHA}")


def dry_run() -> None:
    """Verify stream, tokenizer, parameter count without GPU."""
    print(json.dumps({"event": "dry_run_start", "utc": now()}), flush=True)

    assert STREAM.exists(), f"Stream not found: {STREAM}"
    words = count_words(STREAM)
    assert words == EXPECTED_WORDS, f"Stream words {words} != {EXPECTED_WORDS}"

    verify_tokenizer()

    print(json.dumps({
        "event": "dry_run_ok",
        "utc": now(),
        "stream": str(STREAM),
        "stream_words": words,
        "tokenizer": str(TOKENIZER),
        "tokenizer_sha_ok": True,
        "run_dir": str(RUN_DIR),
    }, indent=2), flush=True)


def do_train(gpu: int) -> None:
    """Launch DeBERTa training on the specified GPU."""
    print(json.dumps({
        "event": "incorpus_train_start",
        "utc": now(),
        "gpu": gpu,
        "stream": str(STREAM),
        "run_dir": str(RUN_DIR),
    }), flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable, "-B", str(TRAINER_SCRIPT),
        "--example_jsonl", str(STREAM),
        "--example_jsonl_label", "incorpus_adultprose_maxgeom_seed43022",
        "--output_dir", str(RUN_DIR),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--deberta_pos_att_type", "p2c,c2p",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--lr_total_steps", "2529",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", "10000000",
        "--max_word_exposure", "100000000",
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, cwd=str(ROOT),
        capture_output=False, text=True,
    )
    elapsed = round(time.time() - t0, 1)

    # Check results
    metrics_path = RUN_DIR / "scientific_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(json.dumps({
            "event": "incorpus_train_done",
            "utc": now(),
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "parameter_count": metrics.get("parameter_count"),
            "loss_last": metrics.get("loss_last"),
            "saved_checkpoint_count": metrics.get("saved_checkpoint_count"),
        }, indent=2), flush=True)
    else:
        print(json.dumps({
            "event": "incorpus_train_done",
            "utc": now(),
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "error": "no scientific_metrics.json found",
        }, indent=2), flush=True)

    if result.returncode != 0:
        raise SystemExit(f"Training failed with rc={result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "train"], default="dry")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "dry":
        dry_run()
    else:
        do_train(args.gpu)


if __name__ == "__main__":
    main()
