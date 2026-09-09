#!/usr/bin/env python3
"""research: train the hash-mixed exact/rewrite in-window DeBERTa arm.

The stream is materialized by `materialize_hash_mixed_inwindow_arm.py`.
This wrapper holds the research/research DeBERTa MLM recipe fixed and changes only
the example stream and optional seed coordinate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/train_hash_mixed_inwindow_arm.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import train_split_inwindow_control as base  # noqa: E402

WS = _public_path('experiments/archive/relation_learning')
META_PATH = _public_path('experiments/archive/relation_learning/data/hash_mixed_inwindow_pools/hash_mixed_inwindow_metadata.json')
STREAM_KEY = "compact_hash_mixed_dose2p64x_100M"
STREAM_SHA_KEY = "compact_hash_mixed_dose2p64x_100M.jsonl"
STREAM_LABEL = "compact_hash_mixed_dose2p64x_matched_rowholdout"
PREFLIGHT_DIR = _public_path('experiments/archive/relation_learning/data/hash_mixed_train_preflight')
TOTAL_WORDS = 100_000_000
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure(seed: int) -> None:
    if seed not in {43022, 43122, 43222}:
        raise SystemExit(f"unexpected seed {seed}; use comparable DeBERTa coordinate 43022/43122/43222")
    base.RECIPE = copy.deepcopy(base.RECIPE)
    base.RECIPE["extra_init_seed"] = seed
    base.RECIPE["train_rng_seed"] = seed + 1


def default_run_dir(seed: int) -> pathlib.Path:
    return _public_path('experiments/archive/relation_learning/training/runs') / f"full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed{seed}"


def checkpoint_ok(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def summarize_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return {"metrics_exists": False, "ready": False}
    try:
        m = read_json(p)
    except Exception as exc:
        return {"metrics_exists": True, "ready": False, "error": repr(exc)}
    saved = m.get("saved_checkpoints") or []
    out = {
        "metrics_exists": True,
        "status": m.get("status"),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "saved_checkpoints": len(saved),
        "first_checkpoint": saved[0].get("name") if saved else None,
        "last_checkpoint": saved[-1].get("name") if saved else None,
        "all_expected_checkpoints_present": all(checkpoint_ok(run_dir, ck) for ck in CKS),
    }
    out["ready"] = (
        int(out.get("word_exposure") or 0) >= TOTAL_WORDS
        and int(out.get("parameter_count") or 0) == base.RECIPE["parameter_count_expected"]
        and int(out.get("vocab_size") or 0) == base.RECIPE["vocab_size_expected"]
        and out.get("tokenizer_label") == "compliant16k_reinvest10M"
        and out.get("all_expected_checkpoints_present") is True
    )
    return out


def build_command(run_dir: pathlib.Path, stream: pathlib.Path) -> list[str]:
    return [
        sys.executable, "-B", str(base.TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", STREAM_LABEL,
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(base.TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", str(base.RECIPE["hidden_size"]),
        "--n_layer", str(base.RECIPE["n_layer"]),
        "--n_head", str(base.RECIPE["n_head"]),
        "--ffn_mult", str(base.RECIPE["ffn_mult"]),
        "--deberta_pos_att_type", str(base.RECIPE["deberta_pos_att_type"]),
        "--seed", str(base.RECIPE["seed"]),
        "--extra_init_seed", str(base.RECIPE["extra_init_seed"]),
        "--train_rng_seed", str(base.RECIPE["train_rng_seed"]),
        "--batch_size", str(base.RECIPE["batch_size"]),
        "--seq_length", str(base.RECIPE["seq_length"]),
        "--max_seq_length", str(base.RECIPE["max_seq_length"]),
        "--learning_rate", str(base.RECIPE["learning_rate"]),
        "--warmup_fraction", str(base.RECIPE["warmup_fraction"]),
        "--weight_decay", str(base.RECIPE["weight_decay"]),
        "--lr_total_steps", str(base.RECIPE["lr_total_steps"]),
        "--masking_curriculum", str(base.RECIPE["masking_curriculum"]),
        "--mask_prob_start", str(base.RECIPE["mask_prob_start"]),
        "--mask_prob_end", str(base.RECIPE["mask_prob_end"]),
        "--checkpoint_words", str(base.RECIPE["checkpoint_words"]),
        "--max_word_exposure", str(base.RECIPE["max_word_exposure"]),
        "--num_workers", str(base.RECIPE["num_workers"]),
        "--log_every", str(base.RECIPE["log_every"]),
        "--dynamics_trace_every", str(base.RECIPE["dynamics_trace_every"]),
    ]


def preflight(run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    for p in [base.TRAINER, base.TOKENIZER, META_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
    meta = read_json(META_PATH)
    if meta.get("status") != "HASH_MIXED_INWINDOW_POOL_MATERIALIZED":
        raise RuntimeError(f"unexpected mixed metadata status {meta.get('status')}")
    audit = meta.get("audit") or {}
    required = {
        "all_sources_retained_once": True,
        "companion_assigned_once_per_pair": True,
        "same_window_source_companion_for_all_pairs": True,
        "row_length_sequence_matches_original_changed_block": True,
        "training_100M_written": True,
    }
    for k, v in required.items():
        if audit.get(k) != v:
            raise RuntimeError(f"mixed metadata invariant failed for {k}: {audit.get(k)}")
    if not audit.get("pool", {}).get("exact_10M"):
        raise RuntimeError(f"mixed pool not exact: {audit.get('pool')}")
    tok_sha = base.sha256_file(base.TOKENIZER / "tokenizer.json")
    if tok_sha != base.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    stream = pathlib.Path(meta["files"][STREAM_KEY])
    if not stream.is_absolute():
        stream = ROOT / stream
    if not stream.exists():
        raise FileNotFoundError(stream)
    stream_sha = base.sha256_file(stream)
    if stream_sha != meta["sha256"][STREAM_SHA_KEY]:
        raise RuntimeError(f"stream SHA mismatch: {stream_sha} != {meta['sha256'][STREAM_SHA_KEY]}")
    counts = base.count_jsonl(stream) if count_words else {"count_skipped": True, "exact_100M": True, "words": TOTAL_WORDS}
    if not counts.get("exact_100M"):
        raise RuntimeError(f"stream word count mismatch {counts}")
    mv = base.model_variant()
    if mv["parameter_count"] != base.RECIPE["parameter_count_expected"] or mv["vocab_size"] != base.RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {mv}")
    return {
        "status": "HASH_MIXED_TRAIN_PREFLIGHT_OK",
        "created_utc": now(),
        "run_dir": rel(run_dir),
        "stream": rel(stream),
        "stream_sha256": stream_sha,
        "stream_accounting": counts,
        "mixed_metadata": rel(META_PATH),
        "tokenizer_dir": rel(base.TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": mv,
        "recipe": base.RECIPE,
        "command": ["python3" if x == sys.executable else str(x) for x in build_command(run_dir, stream)],
        "scientific_purpose": "Graded in-window relation-mixture test under fixed 100M budget: half exact local recurrence, half compact restatement, all selected sources retained.",
        "pre_state_numeric_reading": rel(_public_path('research/notes/relation_learning/price_tradeoff.md')),
        "no_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-replicate", type=int, default=43022)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()
    configure(args.seed_replicate)
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.seed_replicate)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    info = preflight(run_dir, args.count_words)
    write_json(PREFLIGHT_DIR / f"hash_mixed_seed{args.seed_replicate}_preflight.json", info)
    if args.dry_run:
        print(json.dumps({
            "status": info["status"],
            "seed": args.seed_replicate,
            "run_dir": info["run_dir"],
            "stream_sha_prefix": info["stream_sha256"][:12],
            "tokenizer_sha_prefix": info["tokenizer_json_sha256"][:12],
            "words": info["stream_accounting"].get("words"),
            "parameter_count": info["model_variant"]["parameter_count"],
            "preflight_json": rel(PREFLIGHT_DIR / f"hash_mixed_seed{args.seed_replicate}_preflight.json"),
            "no_training_started": True,
        }, indent=2, ensure_ascii=False), flush=True)
        return
    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        print(json.dumps({"status": "HASH_MIXED_ALREADY_FINISHED", "run_dir": rel(run_dir), "metrics": existing}, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise SystemExit(f"run_dir exists and is non-empty but not complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "hash_mixed_train_preflight.json", info)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_relation_learning_step019_hash_mixed_seed{args.seed_replicate}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    stream = ROOT / info["stream"] if not pathlib.Path(info["stream"]).is_absolute() else pathlib.Path(info["stream"])
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    start = now()
    start_rec = {
        "status": "HASH_MIXED_TRAIN_STARTING",
        "seed": args.seed_replicate,
        "gpu": args.gpu,
        "run_dir": rel(run_dir),
        "created_utc": start,
        "scientific_purpose": info["scientific_purpose"],
        "pre_state_numeric_reading": info["pre_state_numeric_reading"],
        "no_leaderboard_submission": True,
    }
    print(json.dumps(start_rec, indent=2, ensure_ascii=False), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "hash_mixed_train_start", "utc": start, "seed": args.seed_replicate, "gpu": args.gpu, "recipe": base.RECIPE}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(build_command(run_dir, stream), cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    result = {
        "status": "HASH_MIXED_TRAIN_FINISHED" if proc.returncode == 0 else "HASH_MIXED_TRAIN_FAILED",
        "returncode": proc.returncode,
        "seed": args.seed_replicate,
        "gpu": args.gpu,
        "started_utc": start,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": summarize_metrics(run_dir),
        "no_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "hash_mixed_train_result.json", result)
    write_json(PREFLIGHT_DIR / f"hash_mixed_seed{args.seed_replicate}_train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
