#!/usr/bin/env python3
"""research: train/dry-run the intermediate 1.82x dose DeBERTa arms.

This launcher is deliberately parallel to research's MAX-dose launcher, but it
uses the research intermediate row-holdout streams.  The intended scientific use
is conditional: if the delivered MAX profile suggests turnover between 1x and
MAX, this midpoint can locate the restructuring optimum; if the MAX semantic leg
keeps growing, this prepared midpoint remains a bracket/control but a higher
dose may be more informative.

Dry-run performs tokenizer, stream accounting/hash, architecture, and command
checks only.  Actual training is H100-expensive and should be launched only after
reading the MAX dose ladder.
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
POOL_DIR = WORKSPACE / "data/dose_1p82x_rowholdout_pools"
META_PATH = POOL_DIR / "dose1p82_rowholdout_metadata.json"
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
    "view": "compact_view_dose1p82x_matched_rowholdout",
    "repeat": "compact_repeat_dose1p82x_matched_rowholdout",
}
META_TRAINING_KEYS = {
    "view": "compact_view_dose1p82x",
    "repeat": "compact_repeat_dose1p82x",
}
META_SHA_KEYS = {
    "view": "compact_view_dose1p82x_100M.jsonl",
    "repeat": "compact_repeat_dose1p82x_100M.jsonl",
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
    last_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            field_words = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            if field_words != actual_words:
                raise RuntimeError(f"word mismatch {path} row {rows}: field={field_words} actual={actual_words}")
            rows += 1
            words += field_words
            src = str(obj.get("source") or "")
            by_source[src] = by_source.get(src, 0) + field_words
            meta = {"row": rows - 1, "words": field_words, "example_id": obj.get("example_id"), "source": src}
            if len(first_rows) < 3:
                first_rows.append(meta)
            last_rows.append(meta)
            if len(last_rows) > 3:
                last_rows.pop(0)
    return {
        "rows": rows,
        "words": words,
        "exact_100M": words == TOTAL_WORDS,
        "expected_update_steps_at_batch256": (rows + RECIPE["batch_size"] - 1) // RECIPE["batch_size"],
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_prefix_counts": {
            "compact_view": sum(v for k, v in by_source.items() if k.startswith("compact_view")),
            "compact_repeat": sum(v for k, v in by_source.items() if k.startswith("compact_repeat")),
            "neutral_topup": sum(v for k, v in by_source.items() if k.startswith("neutral_cleanqwen_topup")),
            "qwen_pair_packed": by_source.get("qwen_pair_packed", 0),
        },
    }


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stream_path(data_arm: str, meta: dict[str, Any]) -> pathlib.Path:
    p = pathlib.Path(meta["files"]["training"][META_TRAINING_KEYS[data_arm]])
    return p if p.is_absolute() else USER_ROOT / p


def expected_stream_sha(data_arm: str, meta: dict[str, Any]) -> str:
    return str(meta["sha256"][META_SHA_KEYS[data_arm]])


def default_run_dir(data_arm: str) -> pathlib.Path:
    return RUNS_DIR / f"full_p2c_c2p_abs_{data_arm}_dose1p82x_matched_rowholdout_deberta100M_seed43022"


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
    meta = read_json(META_PATH)
    if meta.get("status") != "INTERMEDIATE_DOSE_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected pool metadata status: {meta.get('status')}")
    audit = meta.get("audit", {})
    if not (audit.get("all_exact_10M") and audit.get("row_length_sequence_identical_all_arms") and audit.get("view_repeat_suffix_identical_after_changed_block")):
        raise RuntimeError(f"pool metadata audit is not clean: {audit}")
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
        "status": "INTERMEDIATE_DOSE_TRAIN_PREFLIGHT_OK",
        "created_utc": now(),
        "scientific_purpose": "Train the 1.82x intermediate dose full-DeBERTa arm only if MAX-dose readout shows the fixed-budget restructuring optimum must be bracketed between 1x and MAX.",
        "data_arm": args.data_arm,
        "run_dir": str(run_dir.relative_to(USER_ROOT) if run_dir.is_relative_to(USER_ROOT) else run_dir),
        "pool_metadata": str(META_PATH.relative_to(USER_ROOT)),
        "dose": meta.get("dose"),
        "stream": str(stream.relative_to(USER_ROOT) if stream.is_relative_to(USER_ROOT) else stream),
        "stream_sha256": sha,
        "stream_accounting": count,
        "tokenizer_dir": str(TOKENIZER.relative_to(USER_ROOT)),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": model_variant,
        "recipe": RECIPE,
        "command": [sys.executable if x == sys.executable else str(x) for x in cmd],
        "expected_actual_updates_from_rows": count.get("expected_update_steps_at_batch256"),
        "lr_total_steps_intentionally_old_coordinate": RECIPE["lr_total_steps"],
        "expensive_work_role": "This H100 run would decide whether the semantic view-repeat leg turns over between 4.2352% and 11.1872% restructured budget; it should not be launched before the MAX ladder is read.",
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-arm", choices=["view", "repeat"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite-run-dir", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.data_arm)
    if not run_dir.is_absolute():
        run_dir = USER_ROOT / run_dir
    info = preflight(args, run_dir)
    if args.dry_run:
        out = POOL_DIR / f"train_{args.data_arm}_dryrun.json"
        out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": info["status"],
            "data_arm": args.data_arm,
            "rows": info["stream_accounting"].get("rows"),
            "words": info["stream_accounting"].get("words"),
            "expected_updates": info.get("expected_actual_updates_from_rows"),
            "parameter_count": info["model_variant"]["parameter_count"],
            "tokenizer_sha_prefix": info["tokenizer_json_sha256"][:12],
            "dose_multiple_vs_1x_budget": info["dose"].get("dose_multiple_vs_1x_budget"),
            "dryrun_json": str(out.relative_to(USER_ROOT)),
        }, indent=2), flush=True)
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_run_dir:
        # research wait wrappers legitimately create only wait/log records before
        # handing control to this launcher.  Preserve protection against partial
        # model outputs or stale training state, but allow those pre-launch files
        # so the queued midpoint jobs can start after the MAX H100 slots free.
        allowed_prelaunch = {
            "wait_and_train_intermediate_log.jsonl",
            "memory_wait_record.json",
            "wait_launcher_result.json",
        }
        unexpected = [p.name for p in run_dir.iterdir() if p.name not in allowed_prelaunch]
        if unexpected:
            raise SystemExit(f"run_dir already exists with unexpected non-prelaunch files: {run_dir}; unexpected={unexpected[:20]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "intermediate_dose_preflight.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = build_command(args.data_arm, run_dir, stream_path(args.data_arm, read_json(META_PATH)))
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(json.dumps({"status": "INTERMEDIATE_DOSE_TRAIN_STARTING", "data_arm": args.data_arm, "run_dir": str(run_dir), "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "command": info["command"]}, indent=2), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env)
    result = {"status": "INTERMEDIATE_DOSE_TRAIN_FINISHED", "data_arm": args.data_arm, "returncode": proc.returncode, "run_dir": str(run_dir), "metrics": summarize_metrics(run_dir / "scientific_metrics.json")}
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
