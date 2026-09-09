#!/usr/bin/env python3
"""research dormant launcher: legal minfreq50 support-floor tokenizer screen.

This is a prepared, not automatically selected, next-route asset.  It should only
be launched after the research/062 word-mean MLM screen is read.  If word-mean does
not materially improve the mature legal reinvest trajectory, this one-seed 80M
screen tests a genuinely distinct representation-layer repair while preserving the
validated compact-view reinvestment data mechanism.

Scientific factor isolated relative to the research legal 16k reinvest recipe:
  * tokenizer changes from research 16k to a legal support-floored minfreq50 BPE
    trained only on the exact compact-view reinvest 10M pool;
  * corpus, stream order, model shape, seeds, full-batch trainer, WWM objective,
    optimizer, LR schedule, seq length, and checkpoint cadence remain fixed.

The route is not a final endpoint.  Train seed43022 to 80M first and evaluate the
same cheap columns at 70M/80M before deciding whether to continue to 100M/full
nine-task official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
RUNS_DIR = WORKSPACE / "training/runs"
TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WORKSPACE / "data/supportfloor_tokenizers/legal_byte_bpe_40k_minfreq50"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
OUT_DIR = WORKSPACE / "data/minfreq50_supportfloor_training"
EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922"
EXPECTED_VOCAB = 19609
TOKENIZER_LABEL = "legal_byte_bpe_40k_minfreq50"
DEFAULT_RUN_NAME = "minfreq50_supportfloor_reinvest_seed43022_80M"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl_words(path: pathlib.Path) -> tuple[int, int]:
    rows = 0
    words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            w = int(obj.get("words", 0))
            actual = len(str(obj.get("text", "")).split())
            if w <= 0:
                w = actual
            if w != actual:
                raise RuntimeError(f"word-count mismatch at row {rows}: field {w}, actual {actual}")
            words += w
    return rows, words


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def tokenizer_function_check(errors: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
        info = {
            "vocab_size_property": tok.vocab_size,
            "len_tokenizer": len(tok),
            "is_fast": tok.is_fast,
            "special_token_ids": {
                "unk_token": tok.unk_token_id,
                "bos_token": tok.bos_token_id,
                "eos_token": tok.eos_token_id,
                "pad_token": tok.pad_token_id,
                "mask_token": tok.mask_token_id,
            },
            "word_ids_works": tok("test word", add_special_tokens=False).word_ids() is not None,
            "sample_tokens": tok.convert_ids_to_tokens(tok("A compact semantic view.", add_special_tokens=False)["input_ids"]),
        }
        if tok.vocab_size != EXPECTED_VOCAB or len(tok) != EXPECTED_VOCAB:
            errors.append(f"Tokenizer vocab size {tok.vocab_size}/len {len(tok)} != {EXPECTED_VOCAB}")
        if not tok.is_fast:
            errors.append("Tokenizer is not fast")
        if info["special_token_ids"] != {"unk_token": 0, "bos_token": 1, "eos_token": 2, "pad_token": 3, "mask_token": 4}:
            errors.append(f"Unexpected special IDs: {info['special_token_ids']}")
        if info["word_ids_works"] is not True:
            errors.append("word_ids() unavailable")
    except Exception as e:
        errors.append(f"Tokenizer functional check failed: {e}")
    return info


def preflight(run_dir: pathlib.Path, max_word_exposure: int, check_hash: bool, count_words: bool) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "trainer": TRAINER,
        "tokenizer_dir": TOKENIZER,
        "tokenizer_json": TOKENIZER / "tokenizer.json",
        "train_file": TRAIN_FILE,
        "pool_10m": POOL_10M,
        "metadata": META,
        "copy_metadata": TOKENIZER / "copy_metadata.json",
        "support_floor_metadata": TOKENIZER / "tokenizer_support_floor_metadata.json",
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"missing {name}: {path}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json") if (TOKENIZER / "tokenizer.json").exists() else None
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOKENIZER_SHA}")
    train_sha = sha256_file(TRAIN_FILE) if check_hash and TRAIN_FILE.exists() else None
    pool_sha = sha256_file(POOL_10M) if check_hash and POOL_10M.exists() else None
    if check_hash and train_sha != EXPECTED_TRAIN_SHA:
        errors.append(f"train SHA mismatch: {train_sha} != {EXPECTED_TRAIN_SHA}")
    if check_hash and pool_sha != EXPECTED_POOL_SHA:
        errors.append(f"pool SHA mismatch: {pool_sha} != {EXPECTED_POOL_SHA}")
    word_info = None
    if count_words and TRAIN_FILE.exists():
        rows, words = count_jsonl_words(TRAIN_FILE)
        word_info = {"rows": rows, "words": words, "exact_100M": words == 100_000_000}
        if words != 100_000_000:
            errors.append(f"training JSONL words != 100M: {word_info}")
    if max_word_exposure > 100_000_000:
        errors.append(f"max_word_exposure exceeds 100M: {max_word_exposure}")
    tokenizer_info = tokenizer_function_check(errors) if (TOKENIZER / "tokenizer.json").exists() else {}
    status = {
        "status": "MINFREQ50_PREFLIGHT_OK" if not errors else "MINFREQ50_PREFLIGHT_FAILED",
        "errors": errors,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_decision": "Dormant next-route asset: if word-mean MLM fails, train one 80M seed43022 support-floored minfreq50 tokenizer screen to test representation support/segmentation repair while preserving compact-view reinvestment.",
        "minimum_reliable_cost": "One seed to 80M plus cheap 70M/80M official-compatible columns first; continue to 100M/full evaluation only if the mature cheap surface improves the existing research legal reinvest trajectory plausibly enough to close the SOTA gap.",
        "run_dir": str(run_dir),
        "trainer": str(TRAINER),
        "train_file": str(TRAIN_FILE),
        "train_100m_sha256_expected": EXPECTED_TRAIN_SHA,
        "train_100m_sha256_actual": train_sha,
        "pool_10m": str(POOL_10M),
        "pool_10m_sha256_expected": EXPECTED_POOL_SHA,
        "pool_10m_sha256_actual": pool_sha,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_sha256_expected": EXPECTED_TOKENIZER_SHA,
        "tokenizer_sha256_actual": tok_sha,
        "tokenizer_label": TOKENIZER_LABEL,
        "tokenizer_vocab_expected": EXPECTED_VOCAB,
        "tokenizer_info": tokenizer_info,
        "word_info": word_info,
        "changed_factor_relative_to_step35_legal16k_reinvest": "support-floored minfreq50 tokenizer only",
        "fixed_recipe": {
            "model": "DeBERTa-v2 masked LM 8x480 n_head=8 ffn_mult=4",
            "seed": 43,
            "extra_init_seed": 43022,
            "train_rng_seed": 43023,
            "batch_size": 256,
            "seq_length": 256,
            "max_seq_length": 256,
            "learning_rate": 0.001,
            "warmup_fraction": 0.06,
            "weight_decay": 0.01,
            "masking_curriculum": "wwm_fixed",
            "mask_prob_start": 0.15,
            "mask_prob_end": 0.15,
            "checkpoint_words": 1000000,
            "max_word_exposure": max_word_exposure,
        },
        "input_records": {k: file_record(v) for k, v in required.items()},
    }
    return status


def build_command(run_dir: pathlib.Path, max_word_exposure: int) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(TRAINER),
        "--example_jsonl", str(TRAIN_FILE),
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_minfreq50_supportfloor",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", TOKENIZER_LABEL,
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
        "--max_word_exposure", str(max_word_exposure),
        "--num_workers", "0",
        "--log_every", "50",
        "--dynamics_trace_every", "200",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    ap.add_argument("--check-hash", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--max-word-exposure", type=int, default=80_000_000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / DEFAULT_RUN_NAME
    status = preflight(run_dir, args.max_word_exposure, args.check_hash, args.count_words)
    cmd = build_command(run_dir, args.max_word_exposure)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    manifest = OUT_DIR / "minfreq50_supportfloor_preflight.json"
    manifest.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    if status["status"] != "MINFREQ50_PREFLIGHT_OK":
        raise SystemExit(1)
    if args.dry_run:
        print(json.dumps({"status": "MINFREQ50_DRY_RUN_READY", "manifest": str(manifest), "run_dir": str(run_dir)}, indent=2), flush=True)
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step062_minfreq50_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "gpu": args.gpu, "tokenizer_label": TOKENIZER_LABEL}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "MINFREQ50_TRAIN_FINISHED" if proc.returncode == 0 else "MINFREQ50_TRAIN_FAILED",
        "gpu": args.gpu,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "max_word_exposure_requested": args.max_word_exposure,
        "tokenizer_label": TOKENIZER_LABEL,
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["metrics"] = {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "saved_checkpoints": len(metrics.get("saved_checkpoints", [])),
            "last_checkpoint": metrics.get("saved_checkpoints", [{}])[-1].get("name") if metrics.get("saved_checkpoints") else None,
        }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
