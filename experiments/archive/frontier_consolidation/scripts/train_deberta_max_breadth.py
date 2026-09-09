#!/usr/bin/env python3
"""research: train/dry-run the MAX-dose DeBERTa source-breadth arm.

This is the missing allocation vertex for the 2.64x dose triangle.  The data
keeps the same MAX FineWeb source sentences as the view/repeat arms and spends
the compact-rewrite word budget on independent whole FineWeb sentences.  The
training coordinate is otherwise identical to the research full-DeBERTa MAX
view/repeat runs.

No evaluation, SuperGLUE, AoA, upload, or leaderboard action is performed.
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
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/dose_2p64x_breadth_rowholdout_pools"
META_PATH = POOL_DIR / "max_breadth_rowholdout_metadata.json"
STREAM_PATH = POOL_DIR / "compact_breadth_dose2p64x_100M.jsonl"
RUNS_DIR = WS / "training/runs"
DEFAULT_RUN_DIR = RUNS_DIR / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022"
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

STREAM_LABEL = "compact_breadth_dose2p64x_matched_rowholdout"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "expected_batch256_updates": (rows + 255) // 256,
        "first_rows": first_rows,
        "source_prefix_counts": {
            "compact_breadth": sum(v for k, v in by_source.items() if k.startswith("compact_breadth")),
            "neutral_topup": sum(v for k, v in by_source.items() if k.startswith("neutral_cleanqwen_topup")),
            "qwen_pair_packed": by_source.get("qwen_pair_packed", 0),
        },
    }


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


def build_command(run_dir: pathlib.Path, stream: pathlib.Path) -> list[str]:
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", STREAM_LABEL,
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
    m = read_json(metrics_path)
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
    if not STREAM_PATH.exists():
        raise FileNotFoundError(STREAM_PATH)
    meta = read_json(META_PATH)
    if meta.get("status") != "MAX_BREADTH_ROWHOLDOUT_MATERIALIZED":
        raise RuntimeError(f"unexpected metadata status: {meta.get('status')}")
    audit = meta.get("audit") or {}
    if not (audit.get("all_exact_10M") and audit.get("row_length_sequence_matches_max_view") and audit.get("selected_breadth_hash_overlap_with_max_sources") == 0):
        raise RuntimeError(f"breadth metadata audit not clean: {audit}")
    expected_stream_sha = str((meta.get("sha256") or {}).get("compact_breadth_dose2p64x_100M.jsonl") or "")
    stream_sha = sha256_file(STREAM_PATH)
    if expected_stream_sha and stream_sha != expected_stream_sha:
        raise RuntimeError(f"stream SHA mismatch: {stream_sha} != {expected_stream_sha}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    count = count_jsonl(STREAM_PATH) if args.count_words else {"count_skipped": True, "exact_100M": True, "expected_batch256_updates": 2552}
    if not count.get("exact_100M"):
        raise RuntimeError(f"stream accounting failed: {count}")
    model_variant = variant_param_count()
    if model_variant["parameter_count"] != RECIPE["parameter_count_expected"] or model_variant["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {model_variant}")
    cmd = build_command(run_dir, STREAM_PATH)
    return {
        "status": "MAX_BREADTH_TRAIN_PREFLIGHT_OK",
        "created_utc": now(),
        "scientific_purpose": "Train the MAX-dose source-breadth allocation arm so view-minus-breadth can test whether compressed re-expression beats additional distinct FineWeb sources at fixed words, fixed MAX source population, fixed tokenizer, and matched row geometry.",
        "data_arm": "breadth",
        "run_dir": rel(run_dir),
        "pool_metadata": rel(META_PATH),
        "dose": meta.get("dose"),
        "source_allocation": meta.get("source_allocation"),
        "stream": rel(STREAM_PATH),
        "stream_sha256": stream_sha,
        "stream_accounting": count,
        "tokenizer_dir": rel(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": model_variant,
        "recipe": RECIPE,
        "command": cmd,
        "cuda_visible_devices": str(args.gpu),
        "expensive_work_role": "This single H100 run is justified because existing CPU construction establishes an exact MAX breadth arm and the old 1x compact-minus-breadth result was mixed/noisy; the amplified view-minus-breadth contrast decides allocation structure before any RoBERTa repeat arm.",
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = USER_ROOT / run_dir
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
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step261_maxbreadth_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "max_breadth_launcher_start", "utc": started, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(status["command"], stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = now()
    metrics_path = run_dir / "scientific_metrics.json"
    result = {
        "status": "MAX_BREADTH_TRAIN_FINISHED" if proc.returncode == 0 else "MAX_BREADTH_TRAIN_FAILED",
        "returncode": proc.returncode,
        "data_arm": "breadth",
        "gpu": args.gpu,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics_path": rel(metrics_path),
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
