#!/usr/bin/env python3
"""research: FW compact-view vs whole-sentence source-breadth comparison.

Uses pre-built and verified compact-view/source-breadth data assets:
  - compact_view: FineWeb source + aligned compact rewrite (same propositions)
  - source_breadth_wholesentence: same word budget, same row structure,
    but companion words are coherent independent FineWeb sentences

Assets referenced (read-only):
  - 100M streams verified at research preflight
  - Shared 16k tokenizer (compliant, trained on ≤10M words)
  
Training recipe: identical to the legal-coordinate standard
  DeBERTa-v2 8×480, batch 256, seq 256, AdamW lr 0.001, warmup 0.06,
  WD 0.01, WWM fixed 0.15, 100M words (10 passes), checkpoints every 1M.

Decision rule:
  compact wins  → aligned compression creates more reusable knowledge;
                  compact reinvestment is the SOTA-facing substrate
  breadth wins  → data substrate should shift to broader source coverage
  similar       → source_repeat becomes the next discriminating test
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

# --- All paths ---
USER_ROOT = pathlib.Path(".").resolve()
A01_WS = pathlib.Path("experiments/archive/representation_and_objectives")
A02_WS = pathlib.Path("experiments/archive/frontier_consolidation")
AI_RUN_ROOT = A02_WS / "training/runs"
OUT_DIR = A02_WS / "data/fw_compact_vs_breadth_train"

# Verified compact-view/source-breadth data assets (read-only)
TOKENIZER_DIR = A01_WS / "data/shared_tokenizer/shared_16k_tokenizer"
EXPECTED_TOK_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"

# Use the COMPACT_EXPERIENCE base trainer shared with the reference runs
COMPACT_EXPERIENCE_TRAINER = pathlib.Path("experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py")

ARMS = {
    "compact_view": {
        "stream": A01_WS / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl",
        "pool": A01_WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl",
        "sha256": "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68",
        "pool_sha256": "ee08a1f8d974248aa9bdda14dfe48724b93873e6961db137d9bbdd078e5ef914",
        "run_dir": AI_RUN_ROOT / "fw_compact_view_shared16k_seed43022",
    },
    "source_breadth": {
        "stream": A01_WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_100M.jsonl",
        "pool": A01_WS / "data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_10M.jsonl",
        "sha256": "1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9c47d9b385731d",
        "pool_sha256": "166063a66b9cd8c7c899c1fe1a808966f768092e0f5b75cd8650b67d9cf0765f",
        "run_dir": AI_RUN_ROOT / "fw_source_breadth_shared16k_seed43022",
    },
}

SEED = {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023}
RECIPE = dict(
    hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
    batch_size=256, seq_length=256, max_seq_length=256,
    learning_rate=0.001, warmup_fraction=0.06, weight_decay=0.01,
    masking_curriculum="wwm_fixed", mask_prob_start=0.15, mask_prob_end=0.15,
    checkpoint_words=1_000_000, max_word_exposure=100_000_000,
)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight() -> dict[str, Any]:
    """Check all assets exist, SHA match, tokenizer loads."""
    errors: list[str] = []
    info: dict[str, Any] = {}
    
    # Tokenizer
    tok_json = TOKENIZER_DIR / "tokenizer.json"
    if not tok_json.exists():
        errors.append(f"tokenizer.json not found: {tok_json}")
    else:
        tok_sha = sha256_file(tok_json)
        info["tokenizer_sha256"] = tok_sha
        if tok_sha != EXPECTED_TOK_SHA:
            errors.append(f"tokenizer SHA mismatch: {tok_sha}")
    
    # Trainer
    if not COMPACT_EXPERIENCE_TRAINER.exists():
        errors.append(f"base trainer not found: {COMPACT_EXPERIENCE_TRAINER}")
    
    # Arms
    arm_info: dict[str, Any] = {}
    for name, a in ARMS.items():
        d: dict[str, Any] = {}
        for key in ("stream", "pool"):
            path = a[key]
            exists = path.exists()
            d[f"{key}_exists"] = exists
            d[f"{key}_path"] = str(path)
            if exists:
                d[f"{key}_bytes"] = path.stat().st_size
            else:
                errors.append(f"{name}: {key} missing at {path}")
        arm_info[name] = d
    
    # Run dir collisions
    rd = {name: str(a["run_dir"]) for name, a in ARMS.items()}
    if len(set(rd.values())) != len(rd):
        errors.append(f"run_dir collision: {rd}")
    info["run_dirs"] = rd
    
    # Tokenizer functional check
    if not errors:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
            assert tok.is_fast
            assert tok.vocab_size == 16384, f"vocab {tok.vocab_size}"
            enc = tok("The quick brown fox", add_special_tokens=False)
            assert enc.word_ids() is not None
            info["tokenizer_functional"] = {"vocab_size": tok.vocab_size, "is_fast": True}
        except Exception as e:
            errors.append(f"tokenizer functional check: {e}")
    
    return {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "arms": arm_info,
        "info": info,
        "recipe": RECIPE,
        "seed": SEED,
        "a01_data_provenance": {
            "compact_stream_sha256": ARMS["compact_view"]["sha256"],
            "breadth_stream_sha256": ARMS["source_breadth"]["sha256"],
            "tokenizer_sha256": EXPECTED_TOK_SHA,
            "source": "A01 Steps 101-104 verified assets",
        },
    }


def train_command(name: str, gpu: int) -> list[str]:
    a = ARMS[name]
    run_dir = a["run_dir"]
    
    cmd = [
        sys.executable, "-B", str(COMPACT_EXPERIENCE_TRAINER),
        "--example_jsonl", str(a["stream"]),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
        "--seed", str(SEED["seed"]),
        "--extra_init_seed", str(SEED["extra_init_seed"]),
        "--train_rng_seed", str(SEED["train_rng_seed"]),
        "--batch_size", str(RECIPE["batch_size"]),
        "--seq_length", str(RECIPE["seq_length"]),
        "--learning_rate", str(RECIPE["learning_rate"]),
        "--warmup_fraction", str(RECIPE["warmup_fraction"]),
        "--weight_decay", str(RECIPE["weight_decay"]),
        "--masking_curriculum", RECIPE["masking_curriculum"],
        "--mask_prob_start", str(RECIPE["mask_prob_start"]),
        "--mask_prob_end", str(RECIPE["mask_prob_end"]),
        "--checkpoint_words", str(RECIPE["checkpoint_words"]),
        "--max_word_exposure", str(RECIPE["max_word_exposure"]),
    ]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS.keys()), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    
    pf = preflight()
    print(json.dumps({k: pf[k] for k in ("status", "errors")}, indent=2), flush=True)
    
    if pf["status"] != "PREFLIGHT_OK":
        sys.exit(1)
    
    if args.preflight_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pf_path = OUT_DIR / "preflight.json"
        pf_path.write_text(json.dumps(pf, indent=2) + "\n")
        print(f"Preflight saved: {pf_path}", flush=True)
        print("Preflight OK, not launching.", flush=True)
        return
    
    cmd = train_command(args.arm, args.gpu)
    run_dir = ARMS[args.arm]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preflight to run_dir (writable) instead of shared OUT_DIR
    pf_path = run_dir / "preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2) + "\n")
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    print(f"\nLaunching FW comparison arm={args.arm} on GPU{args.gpu}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Stream: {ARMS[args.arm]['stream']}", flush=True)
    
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    
    result = {
        "status": "TRAINING_COMPLETE" if proc.returncode == 0 else "TRAINING_FAILED",
        "arm": args.arm,
        "gpu": args.gpu,
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
    }
    # Write result to run_dir (writable)
    (run_dir / f"train_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    
    if proc.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
