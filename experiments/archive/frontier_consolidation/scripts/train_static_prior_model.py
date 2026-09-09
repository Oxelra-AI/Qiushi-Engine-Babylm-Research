#!/usr/bin/env python3
"""research: preflight/launch wrapper for a possible static-prior masking run.

This wrapper is intentionally conservative.  It permits launching a scientifically justified cheap screen or full run without reconstructing
paths, hashes, or the exact frozen recipe.  It should not be used until the
mature matched legal-tokenizer clean-control comparison indicates that a
learning-signal allocation repair is warranted.

The only intended training change relative to the research legal-tokenizer
compact-view-reinvest endpoint is:
  --masking_curriculum wwm_static_prior
  --static_prior_json <research corpus-only prior>
  --static_prior_scheme <relation_only_v1 or relation_info_v1>
All corpus, tokenizer, architecture, optimizer, batch geometry, seeds, and word
exposure accounting remain frozen.
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
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER = WORKSPACE / "scripts/masking_curriculum_trainer_static_prior.py"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
TRAIN_FILE = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
PRIOR = WORKSPACE / "data/static_token_mask_prior/static_token_mask_prior.json"
RUNS_DIR = WORKSPACE / "training/runs"
EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_SCHEMES = {"relation_only_v1", "relation_info_v1"}


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
            w = int(obj.get("words", len(str(obj.get("text", "")).split())))
            if w != len(str(obj.get("text", "")).split()):
                raise RuntimeError(f"word mismatch at row {rows}: field {w}")
            words += w
    return rows, words


def load_prior(scheme: str) -> dict[str, Any]:
    payload = json.loads(PRIOR.read_text(encoding="utf-8"))
    if payload.get("inputs", {}).get("pool_sha256") != EXPECTED_POOL_SHA:
        raise RuntimeError("static prior does not come from the frozen reinvest 10M pool")
    if payload.get("inputs", {}).get("tokenizer_json_sha256") != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError("static prior tokenizer SHA does not match the research tokenizer")
    schemes = payload.get("schemes") or {}
    if set(schemes) != EXPECTED_SCHEMES:
        raise RuntimeError(f"unexpected prior schemes: {sorted(schemes)}")
    rec = schemes.get(scheme)
    if not rec:
        raise RuntimeError(f"missing scheme {scheme}")
    weights = rec.get("weights")
    if not isinstance(weights, list) or len(weights) != 16_384:
        raise RuntimeError(f"scheme {scheme} has invalid weights")
    finite_weights = []
    for i, w in enumerate(weights):
        x = float(w)
        if not (x > 0.0 and x < 10.0):
            raise RuntimeError(f"scheme {scheme} has invalid weight at id {i}: {w}")
        if not (x == x):
            raise RuntimeError(f"scheme {scheme} has NaN weight at id {i}")
        finite_weights.append(x)
    return {"scheme": scheme, "prior_sha256": sha256_file(PRIOR), "occurrence_weighted_mean": rec.get("occurrence_weighted_mean"), "min": min(finite_weights), "max": max(finite_weights), "top_tokens": rec.get("top_tokens", [])[:20]}


def preflight(scheme: str, max_word_exposure: int, run_dir: pathlib.Path, do_hash: bool = True, do_count: bool = True) -> dict[str, Any]:
    required = {"trainer": TRAINER, "tokenizer_json": TOKENIZER / "tokenizer.json", "train_file": TRAIN_FILE, "pool_10m": POOL_10M, "metadata": META, "prior": PRIOR}
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")
    prior_info = load_prior(scheme)
    hashes: dict[str, str] = {
        "train_100m": sha256_file(TRAIN_FILE),
        "pool_10m": sha256_file(POOL_10M),
        "tokenizer_json": sha256_file(TOKENIZER / "tokenizer.json"),
        "trainer": sha256_file(TRAINER),
        "prior": sha256_file(PRIOR),
        "metadata": sha256_file(META),
    }
    if hashes["train_100m"] != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {hashes['train_100m']}")
    if hashes["pool_10m"] != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch: {hashes['pool_10m']}")
    if hashes["tokenizer_json"] != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {hashes['tokenizer_json']}")
    if hashes["prior"] != prior_info["prior_sha256"]:
        raise RuntimeError("prior SHA changed during preflight")
    rows, words = count_jsonl_words(TRAIN_FILE)
    word_info = {"rows": rows, "words": words, "exact_100M": words == 100_000_000}
    if words != 100_000_000:
        raise RuntimeError(f"training file does not contain exact 100M words: {word_info}")
    if not (1 <= max_word_exposure <= 100_000_000):
        raise RuntimeError(f"bad max_word_exposure: {max_word_exposure}")
    return {
        "status": "STATIC_PRIOR_LAUNCH_PREFLIGHT_OK",
        "scientific_role": "possible learning-signal allocation repair; not launched unless mature matched control selects it",
        "run_dir": str(run_dir),
        "train_file": str(TRAIN_FILE),
        "pool_10m": str(POOL_10M),
        "metadata": str(META),
        "tokenizer_dir": str(TOKENIZER),
        "trainer": str(TRAINER),
        "prior_json": str(PRIOR),
        "prior_info": prior_info,
        "hashes": hashes,
        "word_info": word_info,
        "recipe": {
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
            "masking_curriculum": "wwm_static_prior",
            "mask_prob_start": 0.15,
            "mask_prob_end": 0.15,
            "static_prior_scheme": scheme,
            "checkpoint_words": 1_000_000,
            "max_word_exposure": max_word_exposure,
        },
        "launch_guard": "Only launch after mature 70M/80M clean-vs-reinvest evidence warrants learning-signal repair; do not use this as a tokenizer or corpus change. Hash and word-count checks are mandatory for real launches.",
    }


def build_cmd(run_dir: pathlib.Path, scheme: str, max_word_exposure: int) -> list[str]:
    return [
        sys.executable, str(TRAINER),
        "--example_jsonl", str(TRAIN_FILE),
        "--example_jsonl_label", "staticprior_cleanqwen_fineweb_compact_view_reinvest",
        "--example_jsonl_meta", str(META),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M_staticprior",
        "--hidden_size", "480", "--n_layer", "8", "--n_head", "8", "--ffn_mult", "4",
        "--seed", "43", "--extra_init_seed", "43022", "--train_rng_seed", "43023",
        "--batch_size", "256", "--seq_length", "256", "--max_seq_length", "256",
        "--learning_rate", "0.001", "--warmup_fraction", "0.06", "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_static_prior", "--mask_prob_start", "0.15", "--mask_prob_end", "0.15",
        "--static_prior_json", str(PRIOR), "--static_prior_scheme", scheme,
        "--checkpoint_words", "1000000", "--max_word_exposure", str(max_word_exposure),
        "--num_workers", "0", "--log_every", "50", "--dynamics_trace_every", "200",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scheme", choices=sorted(EXPECTED_SCHEMES), default="relation_only_v1")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--max-word-exposure", type=int, default=20_000_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-hash", action="store_true", help="accepted for compatibility; hash checks are always enforced")
    ap.add_argument("--count-words", action="store_true", help="accepted for compatibility; 100M word count is always enforced")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_name = f"staticprior_{args.scheme}_reinvest_seed43022_{args.max_word_exposure//1_000_000}M"
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / run_name
    if not run_dir.is_absolute():
        run_dir = USER_ROOT / run_dir
    status = preflight(args.scheme, args.max_word_exposure, run_dir, True, True)
    cmd = build_cmd(run_dir, args.scheme, args.max_word_exposure)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step051_staticprior_{args.scheme}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "static_prior_launcher_start", "scheme": args.scheme, "max_word_exposure": args.max_word_exposure, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, env=env, stdout=out, stderr=err)
    result = {
        "status": "STATIC_PRIOR_TRAIN_FINISHED" if proc.returncode == 0 else "STATIC_PRIOR_TRAIN_FAILED",
        "returncode": proc.returncode,
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "scientific_metrics.json"),
        "metrics_exists": (run_dir / "scientific_metrics.json").exists(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
