#!/usr/bin/env python3
"""research: train frozen BabyLM models with a Strict-Small-compliant tokenizer.

Scientific purpose
------------------
The previous compact-view-reinvest 42.0331 endpoint used the inherited
baseline16k tokenizer whose provenance is not Strict-Small compliant.  This
launcher changes only the tokenizer path to the newly trained 16k BPE tokenizer
fit on the 10M compact_view_reinvest pool, while keeping the original training
JSONL, architecture, objective, optimizer, batch geometry, and seeds frozen.

It supports two matched arms:
  * reinvest: the current best compact_view_reinvest corpus, seed43022
  * clean_qwen: inherited clean-Qwen aligned corpus, seed43022

The clean_qwen arm uses the SAME tokenizer so the treatment effect can be read
like-for-like at fixed compliant tokenizer geometry.  Its role is a scientific
control, not a separate leaderboard submission candidate.
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
ROOT = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = ROOT
RUNS_DIR = WORKSPACE / "training" / "runs"
TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WORKSPACE / "data" / "compliant_tokenizer"

ARMS: dict[str, dict[str, Any]] = {
    "reinvest": {
        "train_file": WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
        "pool_10m": WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
        "meta": WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "density_cleanqwen_rowholdout_overlay_metadata.json",
        "expected_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
        "run_name": "complianttok_reinvest_seed43022",
        "example_label": "complianttok_cleanqwen_fineweb_compact_view_reinvest",
        "scientific_role": "submission-candidate endpoint recovery under the compliant tokenizer",
    },
    "clean_qwen": {
        "train_file": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl",
        "pool_10m": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
        "meta": USER_ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json",
        "expected_sha256": "728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345",
        "run_name": "complianttok_cleanqwen_seed43022",
        "example_label": "complianttok_qwen_clean_aligned",
        "scientific_role": "matched clean-Qwen control at the same compliant tokenizer",
    },
}

SEEDS = {
    "43022": {"seed": "43", "extra_init_seed": "43022", "train_rng_seed": "43023"},
}

RECIPE = {
    "model": "DeBERTa-v2 masked LM 8x480, n_head=8, ffn_mult=4",
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
    "checkpoint_words": 1_000_000,
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
}


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
            words += w
    return rows, words


def load_tokenizer_metadata() -> dict[str, Any]:
    meta_path = TOKENIZER / "tokenizer_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def preflight(arm: str, run_dir: pathlib.Path, check_hash: bool, count_words: bool) -> dict[str, Any]:
    cfg = ARMS[arm]
    required = {
        "trainer": TRAINER,
        "tokenizer_dir": TOKENIZER,
        "tokenizer_json": TOKENIZER / "tokenizer.json",
        "train_file": cfg["train_file"],
        "pool_10m": cfg["pool_10m"],
        "metadata": cfg["meta"],
    }
    missing = {k: str(v) for k, v in required.items() if not pathlib.Path(v).exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")

    tok_meta = load_tokenizer_metadata()
    if tok_meta.get("training_data", {}).get("pool_words") != 10_000_000:
        raise RuntimeError(f"Tokenizer metadata does not show 10M training data: {tok_meta}")
    if tok_meta.get("tokenizer", {}).get("vocab_size") != 16_384:
        raise RuntimeError(f"Tokenizer vocab not 16384: {tok_meta.get('tokenizer')}")

    actual_hash = None
    hash_ok = None
    if check_hash:
        actual_hash = sha256_file(pathlib.Path(cfg["train_file"]))
        hash_ok = actual_hash == cfg["expected_sha256"]
        if not hash_ok:
            raise RuntimeError(f"SHA256 mismatch for {cfg['train_file']}: {actual_hash} != {cfg['expected_sha256']}")

    word_info = None
    if count_words:
        rows, words = count_jsonl_words(pathlib.Path(cfg["train_file"]))
        word_info = {"rows": rows, "words": words, "exact_100M": words == 100_000_000}
        if words != 100_000_000:
            raise RuntimeError(f"Training JSONL words != 100M for {arm}: {word_info}")

    return {
        "status": "COMPLIANT_TRAIN_PREFLIGHT_OK",
        "arm": arm,
        "scientific_role": cfg["scientific_role"],
        "run_dir": str(run_dir),
        "train_file": str(cfg["train_file"]),
        "pool_10m": str(cfg["pool_10m"]),
        "metadata": str(cfg["meta"]),
        "expected_train_sha256": cfg["expected_sha256"],
        "actual_train_sha256": actual_hash,
        "hash_ok": hash_ok,
        "word_info": word_info,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_metadata": tok_meta,
        "recipe": RECIPE,
    }


def build_command(arm: str, run_dir: pathlib.Path, max_word_exposure: int) -> list[str]:
    cfg = ARMS[arm]
    seed_cfg = SEEDS["43022"]
    return [
        sys.executable,
        str(TRAINER),
        "--example_jsonl", str(cfg["train_file"]),
        "--example_jsonl_label", str(cfg["example_label"]),
        "--example_jsonl_meta", str(cfg["meta"]),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", seed_cfg["seed"],
        "--extra_init_seed", seed_cfg["extra_init_seed"],
        "--train_rng_seed", seed_cfg["train_rng_seed"],
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
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-hash", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    ap.add_argument("--max-word-exposure", type=int, default=100_000_000)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / ARMS[args.arm]["run_name"]
    status = preflight(args.arm, run_dir, check_hash=args.check_hash, count_words=args.count_words)
    cmd = build_command(args.arm, run_dir, args.max_word_exposure)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    status["max_word_exposure_requested"] = args.max_word_exposure

    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Allocator configuration only changes memory management; it does not change the
    # training recipe, seeds, data order, model, or objective. It protects the
    # compliance retrain from avoidable fragmentation failures on shared H100s.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step036_{args.arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "arm": args.arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "COMPLIANT_TRAIN_FINISHED" if proc.returncode == 0 else "COMPLIANT_TRAIN_FAILED",
        "arm": args.arm,
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
            "saved_checkpoints": len(metrics.get("saved_checkpoints", [])),
            "first_checkpoint": metrics.get("saved_checkpoints", [{}])[0].get("name") if metrics.get("saved_checkpoints") else None,
            "last_checkpoint": metrics.get("saved_checkpoints", [{}])[-1].get("name") if metrics.get("saved_checkpoints") else None,
        }
    if proc.returncode != 0:
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:] if stderr_path.exists() else ""
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
