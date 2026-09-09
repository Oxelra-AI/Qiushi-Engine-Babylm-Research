#!/usr/bin/env python3
"""research: train MAX-dose DeBERTa view/repeat arms in the second basin.

Scientific role
---------------
The first-basin 2.64x MAX-dose curve is a single initialization/mask-stream draw.
This launcher repeats the already materialized legal MAX view/repeat streams with
only the basin seeds changed from 43/43022/43023 to 43/43122/43123.  The pair is
information-positive for every possible first-basin dose shape: it directly tests
whether the MAX semantic view-minus-repeat leg survives a different basin, and it
allows the same fixed clean reference to price MAX repeat-clean and view-clean
movements while the exact clean replication question remains explicit.

No evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action is
performed here.
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

# Same full-DeBERTa coordinate as research, but second-basin seeds matching the
# research/253 seed-spread pair: base seed retained, init/mask-stream changed.
RECIPE = {
    "parameter_count_expected": 34_467_424,
    "vocab_size_expected": 16_384,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "extra_init_seed": 43122,
    "train_rng_seed": 43123,
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
    "view": "compact_view_dose2p64x_matched_rowholdout_seed43122",
    "repeat": "compact_repeat_dose2p64x_matched_rowholdout_seed43122",
}
META_TRAINING_KEYS = {
    "view": "compact_view_dose2p64x",
    "repeat": "compact_repeat_dose2p64x",
}
META_SHA_KEYS = {
    "view": "compact_view_dose2p64x_100M.jsonl",
    "repeat": "compact_repeat_dose2p64x_100M.jsonl",
}


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
        "actual_updates_at_batch256": (rows + RECIPE["batch_size"] - 1) // RECIPE["batch_size"],
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_prefix_counts": {
            "compact_view": sum(v for k, v in by_source.items() if k.startswith("compact_view")),
            "compact_repeat": sum(v for k, v in by_source.items() if k.startswith("compact_repeat")),
            "neutral_topup": sum(v for k, v in by_source.items() if k.startswith("neutral_cleanqwen_topup")),
            "qwen_pair_packed": by_source.get("qwen_pair_packed", 0),
        },
    }


def stream_path(data_arm: str, meta: dict[str, Any]) -> pathlib.Path:
    p = pathlib.Path(meta["files"]["training"][META_TRAINING_KEYS[data_arm]])
    return p if p.is_absolute() else USER_ROOT / p


def expected_stream_sha(data_arm: str, meta: dict[str, Any]) -> str:
    return str(meta["sha256"][META_SHA_KEYS[data_arm]])


def default_run_dir(data_arm: str) -> pathlib.Path:
    return RUNS_DIR / f"full_p2c_c2p_abs_{data_arm}_dose2p64x_matched_rowholdout_deberta100M_seed43122"


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
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected pool metadata status: {meta.get('status')}")
    audit = meta.get("audit", {})
    if not (audit.get("all_exact_10M") and audit.get("row_length_sequence_identical_all_arms") and audit.get("view_repeat_suffix_identical_after_changed_block") and audit.get("training_order_seed_identical_for_view_and_repeat")):
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
    count = count_jsonl(stream) if args.count_words else {"count_skipped": True, "exact_100M": True, "actual_updates_at_batch256": 2552}
    if not count.get("exact_100M"):
        raise RuntimeError(f"stream accounting failed: {count}")
    model_variant = variant_param_count()
    if model_variant["parameter_count"] != RECIPE["parameter_count_expected"] or model_variant["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {model_variant}")
    cmd = build_command(args.data_arm, run_dir, stream)
    return {
        "status": "MAX_DOSE_SEED43122_PREFLIGHT_OK",
        "created_utc": now(),
        "scientific_purpose": "Cross-basin repeat of MAX 2.64x full-DeBERTa view/repeat arms to test whether the amplified compact-packet effect is larger and more reproducible than same-coordinate treatment noise.",
        "expensive_work_decision": "This run decides whether the MAX-dose semantic V-R movement survives a new initialization/mask-stream basin; together with the existing clean reference it also prices R-C and V-C under an explicit shared-clean condition. If both second-basin arms fail to support the first-basin carrier leg, the single-basin dose shape cannot be promoted into a general principle without further repair; if they reproduce it, the packet-allocation mechanism becomes substantially stronger.",
        "lowest_reliable_method": "Use the already materialized and audited research MAX streams, fixed research tokenizer SHA, same full p2c+c2p DeBERTa recipe, and change only extra_init_seed/train_rng_seed to 43122/43123. This is cheaper and cleaner than adding another construction vertex or retraining all doses.",
        "data_arm": args.data_arm,
        "run_dir": rel(run_dir),
        "pool_metadata": rel(META_PATH),
        "dose": meta.get("dose"),
        "stream": rel(stream),
        "stream_sha256": sha,
        "stream_accounting": count,
        "tokenizer_dir": rel(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": model_variant,
        "recipe": RECIPE,
        "command": [sys.executable if x == sys.executable else str(x) for x in cmd],
        "expected_actual_updates_from_rows": count.get("actual_updates_at_batch256"),
        "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-arm", choices=["view", "repeat"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.data_arm)
    if not run_dir.is_absolute():
        run_dir = USER_ROOT / run_dir
    info = preflight(args, run_dir)
    if args.dry_run:
        out = WORKSPACE / "data/max_seed43122_preflight" / f"{args.data_arm}_preflight.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": info["status"],
            "data_arm": args.data_arm,
            "run_dir": info["run_dir"],
            "rows": info["stream_accounting"].get("rows"),
            "words": info["stream_accounting"].get("words"),
            "expected_updates": info.get("expected_actual_updates_from_rows"),
            "parameter_count": info["model_variant"]["parameter_count"],
            "tokenizer_sha_prefix": info["tokenizer_json_sha256"][:12],
            "extra_init_seed": RECIPE["extra_init_seed"],
            "train_rng_seed": RECIPE["train_rng_seed"],
            "dryrun_json": rel(out),
            "no_training_started": True,
        }, indent=2), flush=True)
        return

    existing_metrics = summarize_metrics(run_dir / "scientific_metrics.json")
    if existing_metrics.get("metrics_exists") and int(existing_metrics.get("word_exposure") or 0) >= TOTAL_WORDS:
        result = {"status": "MAX_DOSE_SEED43122_ALREADY_FINISHED", "data_arm": args.data_arm, "run_dir": rel(run_dir), "metrics": existing_metrics}
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise SystemExit(f"run_dir already exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "max_seed43122_preflight.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = build_command(args.data_arm, run_dir, stream_path(args.data_arm, read_json(META_PATH)))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step264_max_seed43122_{args.data_arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    print(json.dumps({"status": "MAX_DOSE_SEED43122_TRAIN_STARTING", "data_arm": args.data_arm, "gpu": args.gpu, "run_dir": rel(run_dir), "created_utc": started}, indent=2), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "utc": started, "data_arm": args.data_arm, "gpu": args.gpu, "recipe": RECIPE}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, stdout=out, stderr=err, text=True)
    finished = now()
    metrics = summarize_metrics(run_dir / "scientific_metrics.json")
    result = {
        "status": "MAX_DOSE_SEED43122_TRAIN_FINISHED" if proc.returncode == 0 else "MAX_DOSE_SEED43122_TRAIN_FAILED",
        "returncode": proc.returncode,
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": metrics,
        "no_evaluation_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
