#!/usr/bin/env python3
"""research: train support-floored minfreq25 legal tokenizer on compact_view_reinvest.

Scientific purpose
------------------
The completed legal40k official evaluations showed a structured representation
trade-off: 40k recovered BLiMP/Supplement/EWoK relative to legal16k but damaged
GlobalPIQA/Entity, consistent with research token-support thinning.  This script
launches the next single general representation repair: a byte-level BPE trained
only on the allowed 10M pool with min_frequency=25, actual vocab 29,529.  It is
not another blind vocabulary-size sweep; it tests whether a support floor can keep
most of 40k's segmentation benefit while avoiding rare-token/support collapse.

Everything else is inherited from the completed research legal40k accumulated
recipe: same compact_view_reinvest 100M stream, same model shape, optimizer,
fixed WWM objective, data order, initialization seeds, training RNG seeds, and
full 100M exposure.  The accumulated trainer preserves the effective batch 256
update geometry using four 64-row microbatches.
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
OUT_DIR = WS / "data/minfreq25_supportfloor_training"

TOKENIZER_DIR = WS / "data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k_minfreq25"
TRAIN_100M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
TRAIN_10M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")
TRAINER = WS / "scripts/accumulated_masking_curriculum_trainer.py"
SUPPORT_SPECTRUM = WS / "data/tokenizer_support_spectrum/tokenizer_support_spectrum.json"
DECISION_NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/legal40k_decision_and_next_route.md')

EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_10M_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOK_SHA = "311e7a20cd8b20f512ab574b8d62b574b5106408e742c32b0bce152dfe8162e0"
EXPECTED_VOCAB = 29529
TOKENIZER_LABEL = "legal_byte_bpe_40k_minfreq25"

SEEDS = {
    "43022": {"seed": 43, "extra_init_seed": 43022, "train_rng_seed": 43023, "gpu": "0"},
    "43122": {"seed": 43, "extra_init_seed": 43122, "train_rng_seed": 43123, "gpu": "1"},
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def preflight() -> dict[str, Any]:
    errors: list[str] = []
    tok_json = TOKENIZER_DIR / "tokenizer.json"
    tok_sha = None
    if not tok_json.exists():
        errors.append(f"tokenizer.json not found at {tok_json}")
    else:
        tok_sha = sha256_file(tok_json)
        if tok_sha != EXPECTED_TOK_SHA:
            errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOK_SHA}")
    stream_sha = None
    if not TRAIN_100M.exists():
        errors.append(f"100M stream not found: {TRAIN_100M}")
    else:
        stream_sha = sha256_file(TRAIN_100M)
        if stream_sha != EXPECTED_100M_SHA:
            errors.append(f"100M stream SHA mismatch: {stream_sha} != {EXPECTED_100M_SHA}")
    pool_sha = None
    if not TRAIN_10M.exists():
        errors.append(f"10M pool not found: {TRAIN_10M}")
    else:
        pool_sha = sha256_file(TRAIN_10M)
        if pool_sha != EXPECTED_10M_SHA:
            errors.append(f"10M pool SHA mismatch: {pool_sha} != {EXPECTED_10M_SHA}")
    if not META.exists():
        errors.append(f"overlay metadata not found: {META}")
    if not TRAINER.exists():
        errors.append(f"accumulated trainer not found: {TRAINER}")
    if not SUPPORT_SPECTRUM.exists():
        errors.append(f"research support spectrum not found: {SUPPORT_SPECTRUM}")
    if not DECISION_NOTE.exists():
        errors.append(f"research decision note not found: {DECISION_NOTE}")
    tokenizer_info: dict[str, Any] = {}
    if not errors:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
            tokenizer_info = {
                "vocab_size_property": tok.vocab_size,
                "len_tokenizer": len(tok),
                "is_fast": tok.is_fast,
                "special_token_ids": {
                    "bos_token": tok.bos_token_id,
                    "eos_token": tok.eos_token_id,
                    "unk_token": tok.unk_token_id,
                    "pad_token": tok.pad_token_id,
                    "mask_token": tok.mask_token_id,
                },
                "word_ids_works": tok("test word", add_special_tokens=False).word_ids() is not None,
            }
            if tok.vocab_size != EXPECTED_VOCAB or len(tok) != EXPECTED_VOCAB:
                errors.append(f"Tokenizer vocab size {tok.vocab_size}/len {len(tok)} != {EXPECTED_VOCAB}")
            if not tok.is_fast:
                errors.append("Tokenizer is not fast")
            if tokenizer_info["word_ids_works"] is not True:
                errors.append("word_ids() failed")
            if tokenizer_info["special_token_ids"] != {"bos_token": 1, "eos_token": 2, "unk_token": 0, "pad_token": 3, "mask_token": 4}:
                errors.append(f"Unexpected special IDs: {tokenizer_info['special_token_ids']}")
        except Exception as e:
            errors.append(f"Tokenizer functional check failed: {e}")
    return {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "scientific_decision": "Test whether a legal support-floored tokenizer repairs the 16k/40k trade-off: retain Supplement/EWoK recovery while avoiding GlobalPIQA/Entity support collapse.",
        "minimum_reliable_cost": "Full 100M training plus official evaluation is required because research showed 10-20M and early cheap-column signals can have wrong sign for endpoint routes.",
        "tokenizer_dir": str(TOKENIZER_DIR),
        "tokenizer_sha256_expected": EXPECTED_TOK_SHA,
        "tokenizer_sha256_actual": tok_sha,
        "tokenizer_label": TOKENIZER_LABEL,
        "tokenizer_vocab_expected": EXPECTED_VOCAB,
        "tokenizer_info": tokenizer_info,
        "train_10m": str(TRAIN_10M),
        "train_10m_sha256_expected": EXPECTED_10M_SHA,
        "train_10m_sha256_actual": pool_sha,
        "train_100m": str(TRAIN_100M),
        "train_100m_sha256_expected": EXPECTED_100M_SHA,
        "train_100m_sha256_actual": stream_sha,
        "trainer": str(TRAINER),
        "changed_factor_relative_to_step061_legal40k": "support-floored minfreq25 tokenizer only; corpus/model/objective/seeds/schedule retained",
        "memory_repair": "effective batch 256 via micro_batch_size 64 and gradient accumulation 4, reused from legal40k",
        "input_records": {
            "tokenizer_json": file_record(tok_json),
            "support_spectrum": file_record(SUPPORT_SPECTRUM),
            "decision_note": file_record(DECISION_NOTE),
        },
    }


def train_command(seed_key: str) -> tuple[list[str], pathlib.Path, dict[str, Any]]:
    s = SEEDS[seed_key]
    run_dir = AI_RUN_ROOT / f"minfreq25_supportfloor_compact_view_reinvest_seed{seed_key}"
    cmd = [
        sys.executable,
        "-B",
        str(TRAINER),
        "--example_jsonl", str(TRAIN_100M),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_minfreq25_supportfloor_accum",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", TOKENIZER_LABEL,
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", str(s["seed"]),
        "--extra_init_seed", str(s["extra_init_seed"]),
        "--train_rng_seed", str(s["train_rng_seed"]),
        "--batch_size", "256",
        "--micro_batch_size", "64",
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
        "status": "MINFREQ25_SUPPORTFLOOR_TRAIN_COMMAND",
        "seed_key": seed_key,
        "run_dir": str(run_dir),
        "gpu_suggestion": s["gpu"],
        "train_file": str(TRAIN_100M),
        "train_file_sha256": EXPECTED_100M_SHA,
        "tokenizer_path": str(TOKENIZER_DIR),
        "tokenizer_sha256": EXPECTED_TOK_SHA,
        "tokenizer_label": TOKENIZER_LABEL,
        "tokenizer_actual_vocab_size": EXPECTED_VOCAB,
        "changed_factor_relative_to_step061_legal40k": "support-floored minfreq25 tokenizer only",
        "fixed_recipe": {
            "model_family": "DeBERTa-v2 masked LM",
            "hidden_size": 480,
            "n_layer": 8,
            "n_head": 8,
            "ffn_mult": 4,
            "effective_batch_size": 256,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "warmup_fraction": 0.06,
            "weight_decay": 0.01,
            "masking": "wwm_fixed at 0.15",
            "data_order_seed": 43,
            "exposure": "100M words (10 passes)",
            "checkpoints": "every 1M words",
        },
        "route_decision_evidence": {
            "legal40k_mean_overall": 40.78035474691613,
            "legal16k_mean_overall": 40.86397521241343,
            "legal40k_minus_legal16k": {
                "BLiMP": 1.3877258187260963,
                "Supplement": 2.6124868117669564,
                "EWoK": 1.1062469711029337,
                "Entity": -1.4142073537631958,
                "GlobalPIQA": -3.9417475728155296,
                "Overall": -0.0836204654972974,
            },
            "mechanism": "legal40k segmentation benefit cancelled by support-thinning damage; minfreq25 tests an intermediate support floor.",
            "decision_note": str(DECISION_NOTE),
        },
        "cmd": cmd,
    }
    return cmd, run_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, choices=["43022", "43122"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    preflight_path = OUT_DIR / "minfreq25_supportfloor_preflight.json"
    preflight_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(pf, indent=2), flush=True)
    if pf["status"] != "PREFLIGHT_OK":
        sys.exit(1)

    seed_keys = [args.seed] if args.seed else ["43022", "43122"]
    for sk in seed_keys:
        cmd, run_dir, manifest = train_command(sk)
        manifest_path = OUT_DIR / f"minfreq25_supportfloor_train_manifest_seed{sk}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSeed {sk} manifest saved: {manifest_path}", flush=True)
        print(f"  run_dir: {run_dir}", flush=True)
        print(f"  command_first_args: {' '.join(str(c) for c in cmd[:10])}", flush=True)

    if args.dry_run:
        print("\n[DRY RUN] minfreq25 support-floor preflight passed; commands ready but not launched.", flush=True)
        return

    for sk in seed_keys:
        cmd, run_dir, _manifest = train_command(sk)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}", flush=True)
        print(f"Launching minfreq25 support-floor seed {sk}...", flush=True)
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
            "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
        }
        result_path = OUT_DIR / f"minfreq25_supportfloor_train_result_seed{sk}.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2), flush=True)
        if proc.returncode != 0:
            print(f"STDERR tail: {proc.stderr[-1200:]}", flush=True)
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
