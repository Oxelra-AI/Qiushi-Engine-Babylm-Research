#!/usr/bin/env python3
"""research: train the matched-MAX dose DeBERTa view/repeat arms.

This wrapper preserves the exact legal full-DeBERTa training coordinate used for
research/253, but replaces the example_jsonl with the research matched 2.64x dose
mechanism-instrument streams. It performs preflight checks on tokenizer, stream
hashes/accounting, architecture parameter count, and run directory before handing
control to the trusted masking_curriculum_trainer.py.

No evaluation, SuperGLUE, AoA, upload, or leaderboard action is performed here.
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

from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
POOL_DIR = WORKSPACE / "data/dose_2p64x_rowholdout_pools"
META_PATH = POOL_DIR / "dose2p64x_rowholdout_metadata.json"
RUNS_DIR = WORKSPACE / "training/runs"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000

RECIPE = {
    "parameter_count_expected": 34_467_424,
    "vocab_size_expected": 16_384,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "batch_size": 256,
    "seq_length": 256,
    "max_seq_length": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_fraction": 0.06,
    "lr_total_steps": 2529,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "checkpoint_words": 10_000_000,
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
    "deberta_pos_att_type": "p2c,c2p",
}

STREAM_LABELS = {
    "view": "compact_view_dose2p64x_matched_rowholdout",
    "repeat": "compact_repeat_dose2p64x_matched_rowholdout",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    by_source: dict[str, int] = {}
    first_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            text = str(obj.get("text") or "")
            field_words = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            if field_words != actual_words:
                raise RuntimeError(f"word mismatch {path} row {rows}: field={field_words} actual={actual_words}")
            words += field_words
            src = str(obj.get("source") or "")
            by_source[src] = by_source.get(src, 0) + field_words
            if len(first_rows) < 3:
                first_rows.append({"words": field_words, "example_id": obj.get("example_id"), "source": src})
    return {
        "rows": rows,
        "words": words,
        "exact_100M": words == TOTAL_WORDS,
        "first_rows": first_rows,
        "source_prefix_counts": {
            "compact_view": sum(v for k, v in by_source.items() if k.startswith("compact_view")),
            "compact_repeat": sum(v for k, v in by_source.items() if k.startswith("compact_repeat")),
            "neutral_topup": sum(v for k, v in by_source.items() if k.startswith("neutral_cleanqwen_topup")),
            "qwen_pair_packed": by_source.get("qwen_pair_packed", 0),
        },
    }


def stream_path(data_arm: str, meta: dict[str, Any]) -> pathlib.Path:
    key = "compact_view_dose2p64x" if data_arm == "view" else "compact_repeat_dose2p64x"
    p = pathlib.Path(meta["files"]["training"][key])
    return p if p.is_absolute() else USER_ROOT / p


def expected_stream_sha(data_arm: str, meta: dict[str, Any]) -> str:
    key = "compact_view_dose2p64x_100M.jsonl" if data_arm == "view" else "compact_repeat_dose2p64x_100M.jsonl"
    return str(meta["sha256"][key])


def default_run_dir(data_arm: str) -> pathlib.Path:
    return RUNS_DIR / f"full_p2c_c2p_abs_{data_arm}_dose2p64x_matched_rowholdout_deberta100M_seed43022"


def variant_param_count() -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    cfg = DebertaV2Config(
        vocab_size=len(tok),
        hidden_size=RECIPE["hidden_size"],
        num_hidden_layers=RECIPE["n_layer"],
        num_attention_heads=RECIPE["n_head"],
        intermediate_size=RECIPE["hidden_size"] * RECIPE["ffn_mult"],
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = DebertaV2ForMaskedLM(cfg)
    keys = list(model.state_dict().keys())
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tok),
        "has_pos_key_proj": any("pos_key_proj" in k for k in keys),
        "has_pos_query_proj": any("pos_query_proj" in k for k in keys),
        "has_encoder_rel_embeddings": any(k.endswith("encoder.rel_embeddings.weight") for k in keys),
    }


def build_command(data_arm: str, run_dir: pathlib.Path, stream: pathlib.Path) -> list[str]:
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", STREAM_LABELS[data_arm],
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
        "--deberta_pos_att_type", str(RECIPE["deberta_pos_att_type"]),
        "--seed", str(RECIPE["seed"]),
        "--extra_init_seed", str(RECIPE["extra_init_seed"]),
        "--train_rng_seed", str(RECIPE["train_rng_seed"]),
        "--batch_size", str(RECIPE["batch_size"]),
        "--seq_length", str(RECIPE["seq_length"]),
        "--max_seq_length", str(RECIPE["max_seq_length"]),
        "--learning_rate", str(RECIPE["learning_rate"]),
        "--warmup_fraction", str(RECIPE["warmup_fraction"]),
        "--weight_decay", str(RECIPE["weight_decay"]),
        "--lr_total_steps", str(RECIPE["lr_total_steps"]),
        "--masking_curriculum", str(RECIPE["masking_curriculum"]),
        "--mask_prob_start", str(RECIPE["mask_prob_start"]),
        "--mask_prob_end", str(RECIPE["mask_prob_end"]),
        "--checkpoint_words", str(RECIPE["checkpoint_words"]),
        "--max_word_exposure", str(RECIPE["max_word_exposure"]),
        "--num_workers", str(RECIPE["num_workers"]),
        "--log_every", str(RECIPE["log_every"]),
        "--dynamics_trace_every", str(RECIPE["dynamics_trace_every"]),
    ]


def summarize_metrics(metrics_path: pathlib.Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"metrics_exists": False}
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = m.get("saved_checkpoints", [])
    return {
        "metrics_exists": True,
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "mask_prob_start": m.get("mask_prob_start"),
        "mask_prob_end": m.get("mask_prob_end"),
        "seq_length": m.get("seq_length"),
        "max_seq_length": m.get("max_seq_length"),
        "saved_checkpoints": len(saved),
        "first_checkpoint": saved[0].get("name") if saved else None,
        "last_checkpoint": saved[-1].get("name") if saved else None,
    }


def preflight(args: argparse.Namespace, run_dir: pathlib.Path) -> dict[str, Any]:
    if not TRAINER.exists():
        raise FileNotFoundError(TRAINER)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)
    if not META_PATH.exists():
        raise FileNotFoundError(META_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected pool metadata status: {meta.get('status')}")
    if not meta.get("audit", {}).get("all_exact_10M") or not meta.get("audit", {}).get("row_length_sequence_identical_all_arms"):
        raise RuntimeError("pool metadata audit is not clean")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    stream = stream_path(args.data_arm, meta)
    if not stream.exists():
        raise FileNotFoundError(stream)
    sha = sha256_file(stream)
    expected_sha = expected_stream_sha(args.data_arm, meta)
    if sha != expected_sha:
        raise RuntimeError(f"stream SHA mismatch for {args.data_arm}: {sha} != {expected_sha}")
    count = count_jsonl(stream) if args.count_words else {"count_skipped": True, "exact_100M": True}
    if not count.get("exact_100M"):
        raise RuntimeError(f"stream accounting failed: {count}")
    model_variant = variant_param_count()
    if model_variant["parameter_count"] != RECIPE["parameter_count_expected"] or model_variant["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {model_variant}")
    cmd = build_command(args.data_arm, run_dir, stream)
    return {
        "status": "MATCHED_MAX_DOSE_TRAIN_PREFLIGHT_OK",
        "created_utc": now(),
        "scientific_purpose": "Train MAX dose full-DeBERTa arm to read exposure-ladder dose response of structured semantic compression versus hash-rotated repeat at fixed tokenizer and fixed 10M budget.",
        "data_arm": args.data_arm,
        "run_dir": str(run_dir),
        "pool_metadata": str(META_PATH),
        "dose": meta.get("dose"),
        "stream": str(stream),
        "stream_sha256": sha,
        "stream_accounting": count,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": model_variant,
        "recipe": RECIPE,
        "command": cmd,
        "cuda_visible_devices": str(args.gpu),
        "expensive_work_role": "This H100 run is justified because CPU audits show the MAX pool is nested and matched; it distinguishes dose-dependent semantic compression from flat compression term plus freed-budget/source-diversity effects over the 10M-100M ladder.",
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-arm", choices=["view", "repeat"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.data_arm)
    status = preflight(args, run_dir)
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step256_maxdose_{args.data_arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "utc": started, "data_arm": args.data_arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(status["command"], stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = now()
    metrics_path = run_dir / "scientific_metrics.json"
    result = {
        "status": "MATCHED_MAX_DOSE_TRAIN_FINISHED" if proc.returncode == 0 else "MATCHED_MAX_DOSE_TRAIN_FAILED",
        "returncode": proc.returncode,
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "metrics": summarize_metrics(metrics_path),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
