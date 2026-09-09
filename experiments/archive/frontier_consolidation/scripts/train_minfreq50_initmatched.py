#!/usr/bin/env python3
"""research dormant launcher: init-matched minfreq50 support-floor tokenizer screen.

Launch only if the research/062 word-mean screen fails to justify continuation.
This one-seed 80M screen preserves compact-view reinvestment and the frozen
training recipe while changing the tokenizer to the legal minfreq50 support-floor
inventory.  Unlike the research plain preflight launcher, it uses a wrapper trainer
that copies all same-shape tensors from a research legal16k reference initialization,
so the contextual body and prediction transform start identically and only
vocab-shaped tensors remain tokenizer-specific.
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
TRAINER = WORKSPACE / "scripts/minfreq50_initmatched_trainer.py"
TOKENIZER = WORKSPACE / "data/supportfloor_tokenizers/legal_byte_bpe_40k_minfreq50"
REF_TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
AUDIT_JSON = WORKSPACE / "data/supportfloor_factor_audit/supportfloor_factor_audit.json"
OUT_DIR = WORKSPACE / "data/minfreq50_initmatched_training"
EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922"
EXPECTED_REF_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_VOCAB = 19609
TOKENIZER_LABEL = "legal_byte_bpe_40k_minfreq50_initmatched"
DEFAULT_RUN_NAME = "minfreq50_initmatched_reinvest_seed43022_80M"


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
            text = str(obj.get("text", ""))
            w = int(obj.get("words", 0)) or len(text.split())
            actual = len(text.split())
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
        ref = AutoTokenizer.from_pretrained(str(REF_TOKENIZER), use_fast=True)
        info = {
            "vocab_size_property": tok.vocab_size,
            "len_tokenizer": len(tok),
            "ref_len_tokenizer": len(ref),
            "is_fast": tok.is_fast,
            "special_token_ids": {"unk_token": tok.unk_token_id, "bos_token": tok.bos_token_id, "eos_token": tok.eos_token_id, "pad_token": tok.pad_token_id, "mask_token": tok.mask_token_id},
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


def audit_check(errors: list[str]) -> dict[str, Any] | None:
    if not AUDIT_JSON.exists():
        errors.append(f"missing supportfloor factor audit: {AUDIT_JSON}")
        return None
    obj = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    try:
        init = obj["initialization_audit"]
        st = init["standard_same_seed_audit"]
        mt = init["init_matched_copy_simulation"]["postcopy_compare_to_step35"]
        if float(st["random_like_exact_numel_fraction"] or 0.0) > 0.01:
            errors.append("unexpected: standard same-seed already matches random-like tensors; audit/launcher mismatch")
        if float(mt["random_like_exact_numel_fraction"] or 0.0) < 0.999:
            errors.append("init-matched copy simulation did not preserve same-shape random tensors")
        comp = obj["token_accounting"]["comparisons"]["minfreq50_supportfloor_minus_step35_legal16k"]
        return {
            "audit_json": str(AUDIT_JSON),
            "standard_random_like_exact_numel_fraction": st["random_like_exact_numel_fraction"],
            "initmatched_random_like_exact_numel_fraction": mt["random_like_exact_numel_fraction"],
            "minfreq50_relative_expected_target_tokens": comp["relative_expected_target_tokens"],
            "minfreq50_relative_expected_selected_groups": comp["relative_expected_selected_groups"],
        }
    except Exception as e:
        errors.append(f"could not parse supportfloor factor audit: {e}")
        return None


def preflight(run_dir: pathlib.Path, max_word_exposure: int, check_hash: bool, count_words: bool) -> dict[str, Any]:
    errors: list[str] = []
    required = {
        "trainer_wrapper": TRAINER,
        "tokenizer_dir": TOKENIZER,
        "tokenizer_json": TOKENIZER / "tokenizer.json",
        "reference_step35_tokenizer": REF_TOKENIZER,
        "reference_step35_tokenizer_json": REF_TOKENIZER / "tokenizer.json",
        "train_file": TRAIN_FILE,
        "pool_10m": POOL_10M,
        "metadata": META,
        "supportfloor_factor_audit": AUDIT_JSON,
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"missing {name}: {path}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json") if (TOKENIZER / "tokenizer.json").exists() else None
    ref_tok_sha = sha256_file(REF_TOKENIZER / "tokenizer.json") if (REF_TOKENIZER / "tokenizer.json").exists() else None
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        errors.append(f"tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOKENIZER_SHA}")
    if ref_tok_sha != EXPECTED_REF_TOKENIZER_SHA:
        errors.append(f"reference tokenizer SHA mismatch: {ref_tok_sha} != {EXPECTED_REF_TOKENIZER_SHA}")
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
    audit_info = audit_check(errors)
    status = {
        "status": "MINFREQ50_INITMATCHED_PREFLIGHT_OK" if not errors else "MINFREQ50_INITMATCHED_PREFLIGHT_FAILED",
        "errors": errors,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_decision": "Dormant next-route asset: if word-mean MLM fails, train one 80M seed43022 init-matched support-floored minfreq50 tokenizer screen to test representation support/segmentation while preserving compact-view reinvestment and same-shape contextual initialization.",
        "minimum_reliable_cost": "One seed to 80M plus cheap 70M/80M official-compatible columns first; continue to 100M/full evaluation only if the mature cheap surface improves the existing research legal reinvest trajectory plausibly enough to close the SOTA gap.",
        "run_dir": str(run_dir),
        "trainer_wrapper": str(TRAINER),
        "base_training_loop": "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py via monkeypatched build_model only",
        "train_file": str(TRAIN_FILE),
        "train_100m_sha256_expected": EXPECTED_TRAIN_SHA,
        "train_100m_sha256_actual": train_sha,
        "pool_10m": str(POOL_10M),
        "pool_10m_sha256_expected": EXPECTED_POOL_SHA,
        "pool_10m_sha256_actual": pool_sha,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_sha256_expected": EXPECTED_TOKENIZER_SHA,
        "tokenizer_sha256_actual": tok_sha,
        "reference_tokenizer_dir": str(REF_TOKENIZER),
        "reference_tokenizer_sha256_expected": EXPECTED_REF_TOKENIZER_SHA,
        "reference_tokenizer_sha256_actual": ref_tok_sha,
        "tokenizer_label": TOKENIZER_LABEL,
        "tokenizer_vocab_expected": EXPECTED_VOCAB,
        "tokenizer_info": tokenizer_info,
        "word_info": word_info,
        "factor_audit_info": audit_info,
        "changed_factor_relative_to_step35_legal16k_reinvest": "support-floored minfreq50 tokenizer plus tokenizer-specific vocab-shaped tensors; same-shape contextual tensors are copied from a research legal16k random reference",
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
        "--example_jsonl_label", "cleanqwen_fineweb_compact_view_reinvest_minfreq50_supportfloor_initmatched",
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
    manifest = OUT_DIR / "minfreq50_initmatched_preflight.json"
    manifest.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    if status["status"] != "MINFREQ50_INITMATCHED_PREFLIGHT_OK":
        raise SystemExit(1)
    if args.dry_run:
        print(json.dumps({"status": "MINFREQ50_INITMATCHED_DRY_RUN_READY", "manifest": str(manifest), "run_dir": str(run_dir)}, indent=2), flush=True)
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step063_minfreq50_initmatched_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "gpu": args.gpu, "tokenizer_label": TOKENIZER_LABEL, "initmatched": True}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "MINFREQ50_INITMATCHED_TRAIN_FINISHED" if proc.returncode == 0 else "MINFREQ50_INITMATCHED_TRAIN_FAILED",
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
        "initmatched": True,
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
