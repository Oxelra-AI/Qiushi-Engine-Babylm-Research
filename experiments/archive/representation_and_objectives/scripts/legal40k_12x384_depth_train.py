#!/usr/bin/env python3
"""research: launch true 12x384/1280 depth route on legal40k compact_view_reinvest.

This is a Lead-authorized new-factor route after legal16k and legal40k 8x480
fixed-WWM endpoints both missed the 41.8 frontier.  It is intentionally not a
vocabulary interpolation: it keeps the same legal40k tokenizer and exact 100M
compact_view_reinvest stream, changing only the DeBERTa-v2 geometry to the
leader-style 12 layers x hidden 384 x 12 heads x intermediate 1280 while keeping
current AdamW/cosine, fixed-WWM, batch accounting, data order, and seed identity.
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
OUT_DIR = WS / "data/legal40k_12x384_depth_training"

TOKENIZER_DIR = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"
TRAIN_100M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")
TRAINER = WS / "scripts/relation_bias_accumulated_trainer.py"

EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOK_SHA = "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758"
EXPECTED_PARAM_COUNT = 38_421_952
VISIBLE_LEADER = 41.80

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


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.relative_to(pathlib.Path(".").resolve()))
    except Exception:
        return str(p)


def preflight() -> dict[str, Any]:
    errors: list[str] = []
    tok_json = TOKENIZER_DIR / "tokenizer.json"
    tok_sha = None
    stream_sha = None
    trainer_sha = None
    if not tok_json.exists():
        errors.append(f"missing tokenizer.json: {tok_json}")
    else:
        tok_sha = sha256_file(tok_json)
        if tok_sha != EXPECTED_TOK_SHA:
            errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOK_SHA}")
    if not TRAIN_100M.exists():
        errors.append(f"missing 100M stream: {TRAIN_100M}")
    else:
        stream_sha = sha256_file(TRAIN_100M)
        if stream_sha != EXPECTED_100M_SHA:
            errors.append(f"100M stream SHA mismatch: {stream_sha} != {EXPECTED_100M_SHA}")
    if not META.exists():
        errors.append(f"missing metadata: {META}")
    if not TRAINER.exists():
        errors.append(f"missing trainer: {TRAINER}")
    else:
        trainer_sha = sha256_file(TRAINER)
    tokenizer_check: dict[str, Any] = {}
    if not errors:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
            enc = tok("The small model tracks the object in the box.", add_special_tokens=False)
            tokenizer_check = {
                "vocab_size": tok.vocab_size,
                "len_tokenizer": len(tok),
                "is_fast": bool(tok.is_fast),
                "unk_token_id": tok.unk_token_id,
                "bos_token_id": tok.bos_token_id,
                "eos_token_id": tok.eos_token_id,
                "pad_token_id": tok.pad_token_id,
                "mask_token_id": tok.mask_token_id,
                "word_ids_available": enc.word_ids() is not None,
            }
            if tok.vocab_size != 40000:
                errors.append(f"expected vocab_size 40000, got {tok.vocab_size}")
            if not tok.is_fast:
                errors.append("tokenizer is not fast")
            if enc.word_ids() is None:
                errors.append("tokenizer.word_ids unavailable")
        except Exception as e:
            errors.append(f"tokenizer functional check failed: {e}")
    return {
        "status": "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAILED",
        "errors": errors,
        "scientific_decision": (
            "Test whether true 12x384 depth-over-width leader geometry repairs "
            "legal40k compact_view_reinvest relational/stateful deficits without "
            "duplicating the support-floor tokenizer line."
        ),
        "visible_leader_overall": VISIBLE_LEADER,
        "changed_factor_relative_to_step061_legal40k": "architecture geometry only: 8x480/FFN1920 -> 12x384/FFN1280",
        "not_changed": [
            "legal40k tokenizer",
            "compact_view_reinvest 100M stream",
            "AdamW lr=0.001 cosine schedule and warmup_fraction=0.06",
            "fixed WWM mask probability 0.15",
            "effective batch 256 via microbatch 64",
            "sequence length 256",
            "data-order seed 43 and matched init/train RNG seeds",
        ],
        "expected_param_count": EXPECTED_PARAM_COUNT,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "tokenizer_sha256": tok_sha,
        "train_file": str(TRAIN_100M),
        "train_file_sha256": stream_sha,
        "metadata": str(META),
        "trainer": str(TRAINER),
        "trainer_sha256": trainer_sha,
        "tokenizer_check": tokenizer_check,
    }


def train_command(seed_key: str, *, max_word_exposure: int = 100_000_000, output_suffix: str = "") -> tuple[list[str], pathlib.Path, dict[str, Any]]:
    s = SEEDS[seed_key]
    suffix = output_suffix or ""
    run_dir = AI_RUN_ROOT / f"legal40k_12x384_depth_compact_view_reinvest_seed{seed_key}{suffix}"
    cmd = [
        sys.executable,
        "-B",
        str(TRAINER),
        "--example_jsonl", str(TRAIN_100M),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_legal40k_12x384_depth",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER_DIR),
        "--tokenizer_label", "legal_byte_bpe_40k",
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
        "--max_word_exposure", str(max_word_exposure),
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
        "--mask_stats_every", "200",
        "--hidden_size", "384",
        "--n_layer", "12",
        "--n_head", "12",
        "--intermediate_size", "1280",
        "--ffn_mult", "4",
    ]
    manifest = {
        "status": "LEGAL40K_12X384_DEPTH_TRAIN_COMMAND",
        "seed_key": seed_key,
        "run_dir": str(run_dir),
        "train_file": str(TRAIN_100M),
        "train_file_sha256": EXPECTED_100M_SHA,
        "tokenizer_path": str(TOKENIZER_DIR),
        "tokenizer_sha256": EXPECTED_TOK_SHA,
        "changed_factor_relative_to_step061_legal40k": (
            "true leader-style DeBERTa-v2 depth-over-width geometry: hidden_size=384, "
            "n_layer=12, n_head=12, intermediate_size=1280, expected 38,421,952 parameters"
        ),
        "fixed_recipe": {
            "model_family": "DeBERTa-v2 masked LM",
            "hidden_size": 384,
            "n_layer": 12,
            "n_head": 12,
            "intermediate_size": 1280,
            "expected_parameter_count": EXPECTED_PARAM_COUNT,
            "effective_batch_size": 256,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "warmup_fraction": 0.06,
            "weight_decay": 0.01,
            "masking": "wwm_fixed at 0.15",
            "sequence_length": 256,
            "data_order_seed": 43,
            "exposure": max_word_exposure,
            "checkpoints": "every 1M words",
        },
        "decision_conditions": [
            "If seed43022 deep-fixed coherently improves Entity/EWoK/COMPS/SuperGLUE over legal40k seed43022 without severe GlobalPIQA loss, train seed43122 for reproduction.",
            "If it nears or clears 41.8, protect and reproduce before adding further mechanisms.",
            "If flat/worse across relevant columns, deprioritize depth and choose from WWM-to-token evidence or A02 support-floor vector.",
        ],
        "cmd": cmd,
    }
    return cmd, run_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, choices=sorted(SEEDS), default="43022")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-words", type=int, default=0, help="If >0, run a short memory pilot with this max word exposure and _pilot suffix.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    pf_path = OUT_DIR / "legal40k_12x384_depth_preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in pf.items() if k not in ("not_changed", "tokenizer_check")}, indent=2), flush=True)
    if pf["status"] != "PREFLIGHT_OK":
        raise SystemExit(1)

    max_words = args.pilot_words if args.pilot_words > 0 else 100_000_000
    suffix = "_pilot" if args.pilot_words > 0 else ""
    cmd, run_dir, manifest = train_command(args.seed, max_word_exposure=max_words, output_suffix=suffix)
    manifest["pilot_words"] = args.pilot_words
    manifest["preflight_path"] = str(pf_path)
    manifest_path = OUT_DIR / f"legal40k_12x384_depth_train_manifest_seed{args.seed}{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "MANIFEST_READY",
        "seed": args.seed,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "pilot_words": args.pilot_words,
        "cmd_preview": cmd[:10],
    }, indent=2), flush=True)

    if args.dry_run:
        print("[DRY RUN] Depth preflight passed; command ready but not launched.", flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    result = {
        "status": "TRAINING_COMPLETE" if proc.returncode == 0 else "TRAINING_FAILED",
        "seed_key": args.seed,
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "pilot_words": args.pilot_words,
        "stdout_tail": proc.stdout[-8000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-8000:] if proc.stderr else "",
    }
    result_path = OUT_DIR / f"legal40k_12x384_depth_train_result_seed{args.seed}{suffix}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2), flush=True)
    if proc.returncode != 0:
        print("STDERR tail:\n" + result["stderr_tail"], flush=True)
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
