#!/usr/bin/env python3
"""research launcher for whole-word-unit-normalized MLM credit.

Expensive-work purpose:
  Test one general legal optimization repair selected after the mature matched
  clean-vs-reinvest vector showed compact-view reinvestment survives the legal
  tokenizer coordinate.  The run keeps the research legal tokenizer, compact-view
  reinvest stream, DeBERTa-v2 8x480 architecture, seeds, optimizer, WWM sampling,
  and schedule fixed, changing only MLM loss normalization from selected-token
  mean to selected-word-group mean.

Decision role:
  Train seed43022 only to 80M first.  Cheap official-compatible evaluation at
  70M/80M decides whether to continue to a 100M full endpoint, change the route,
  or stop this intervention.  It is not a final endpoint by itself.
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
TRAINER = WORKSPACE / "scripts/wordmean_mlm_trainer.py"
RUNS_DIR = WORKSPACE / "training/runs"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
DEFAULT_RUN_NAME = "wordmean_mlm_complianttok_reinvest_seed43022_80M"


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
            if w <= 0:
                w = len(str(obj.get("text", "")).split())
            if w != len(str(obj.get("text", "")).split()):
                raise RuntimeError(f"word-count mismatch at row {rows}: field {w}")
            words += w
    return rows, words


def preflight(run_dir: pathlib.Path, max_word_exposure: int, check_hash: bool, count_words: bool) -> dict[str, Any]:
    required = {
        "trainer": TRAINER,
        "tokenizer_dir": TOKENIZER,
        "tokenizer_json": TOKENIZER / "tokenizer.json",
        "train_file": TRAIN_FILE,
        "pool_10m": POOL_10M,
        "metadata": META,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"research tokenizer SHA mismatch: {tok_sha} != {EXPECTED_TOKENIZER_SHA}")
    actual_train_sha = sha256_file(TRAIN_FILE) if check_hash else None
    actual_pool_sha = sha256_file(POOL_10M) if check_hash else None
    if check_hash and actual_train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {actual_train_sha} != {EXPECTED_TRAIN_SHA}")
    if check_hash and actual_pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {actual_pool_sha} != {EXPECTED_POOL_SHA}")
    word_info = None
    if count_words:
        rows, words = count_jsonl_words(TRAIN_FILE)
        word_info = {"rows": rows, "words": words, "exact_100M": words == 100_000_000}
        if words != 100_000_000:
            raise RuntimeError(f"training JSONL words != 100M: {word_info}")
    if max_word_exposure > 100_000_000:
        raise RuntimeError(f"max_word_exposure exceeds 100M: {max_word_exposure}")
    return {
        "status": "WORDMEAN_PREFLIGHT_OK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_decision": "80M single-seed screen of tokenizer-invariant whole-word MLM credit; continue to full endpoint only if 70/80M cheap columns materially improve over research legal reinvest",
        "intended_single_change": "loss_normalization token_mean -> selected_word_group_mean",
        "run_dir": str(run_dir),
        "trainer": str(TRAINER),
        "train_file": str(TRAIN_FILE),
        "pool_10m": str(POOL_10M),
        "metadata": str(META),
        "tokenizer_dir": str(TOKENIZER),
        "expected_train_sha256": EXPECTED_TRAIN_SHA,
        "actual_train_sha256": actual_train_sha,
        "expected_pool_sha256": EXPECTED_POOL_SHA,
        "actual_pool_sha256": actual_pool_sha,
        "expected_tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "actual_tokenizer_sha256": tok_sha,
        "word_info": word_info,
        "recipe": {
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
            "loss_normalization": "word_mean",
        },
    }


def build_command(run_dir: pathlib.Path, max_word_exposure: int, loss_normalization: str = "word_mean") -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(TRAIN_FILE),
        "--example_jsonl_label", "wordmean_complianttok_compact_view_reinvest",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--loss_normalization", loss_normalization,
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
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    ap.add_argument("--check-hash", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--max-word-exposure", type=int, default=80_000_000)
    ap.add_argument("--loss-normalization", choices=["word_mean", "token_mean"], default="word_mean")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / DEFAULT_RUN_NAME
    status = preflight(run_dir, args.max_word_exposure, args.check_hash, args.count_words)
    cmd = build_command(run_dir, args.max_word_exposure, args.loss_normalization)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    status["max_word_exposure_requested"] = args.max_word_exposure
    status["loss_normalization_requested"] = args.loss_normalization

    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step061_wordmean_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "gpu": args.gpu, "loss_normalization": args.loss_normalization}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "WORDMEAN_TRAIN_FINISHED" if proc.returncode == 0 else "WORDMEAN_TRAIN_FAILED",
        "gpu": args.gpu,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "max_word_exposure_requested": args.max_word_exposure,
        "loss_normalization": args.loss_normalization,
        "tokenizer_dir": str(TOKENIZER),
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
            "loss_normalization": metrics.get("loss_normalization"),
            "mean_tokens_per_selected_group_trace_mean": metrics.get("mean_tokens_per_selected_group_trace_mean"),
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
