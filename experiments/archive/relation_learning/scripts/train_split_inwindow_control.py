#!/usr/bin/env python3
"""research: train split in-window controls for the relation-structure mechanism.

Scientific role
---------------
research materialized VIEW_SPLIT and REPEAT_SPLIT streams that preserve the
selected source text multiset, companion text multiset, suffix/filler budget, and
100M word exposure, but move each selected source and its companion into separate
training rows.  Training these arms in the same DeBERTa seed43022 coordinate as
research tests whether the previously measured copy benefit and nonidentical
source-use cost require same-window source/companion co-occurrence or can arise
from token exposure/repetition budget alone.

This launcher performs recipe and stream integrity checks, then runs only MLM
training.  It performs no benchmark evaluation, upload, or final expression.
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


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
REPRESENTATION_FRONTIER_STUDIES_WS = ROOT / "experiments/archive" / 'frontier_consolidation'
TRAINER = ROOT / "experiments/archive" / 'compact_experience' / "scripts" / "masking_curriculum_trainer.py"
TOKENIZER = REPRESENTATION_FRONTIER_STUDIES_WS / "data" / "compliant_tokenizer"
POOL_DIR = WS / "data" / "split_inwindow_rowholdout_pools"
META_PATH = POOL_DIR / "split_inwindow_rowholdout_metadata.json"
RUNS_DIR = WS / "training" / "runs"
PREFLIGHT_DIR = WS / "data" / "split_inwindow_train_preflight"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]

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

ARM_TO_STREAM_KEY = {
    "view_split": "compact_view_split_dose2p64x_100M",
    "repeat_split": "compact_repeat_split_dose2p64x_100M",
}

STREAM_LABELS = {
    "view_split": "compact_view_split_dose2p64x_rowholdout",
    "repeat_split": "compact_repeat_split_dose2p64x_rowholdout",
}


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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    first_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    min_words: int | None = None
    max_words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            wf = int(obj.get("words", len(text.split())))
            wa = len(text.split())
            if wf != wa:
                raise RuntimeError(f"word mismatch {path} row {rows}: field={wf} actual={wa}")
            rows += 1
            words += wf
            min_words = wf if min_words is None else min(min_words, wf)
            max_words = max(max_words, wf)
            src = str(obj.get("source") or "")
            by_source[src] = by_source.get(src, 0) + wf
            rec = {"row": rows - 1, "words": wf, "source": src, "example_id": obj.get("example_id")}
            if len(first_rows) < 3:
                first_rows.append(rec)
            last_rows.append(rec)
            if len(last_rows) > 3:
                last_rows.pop(0)
    return {
        "rows": rows,
        "words": words,
        "exact_100M": words == TOTAL_WORDS,
        "min_words": min_words,
        "max_words": max_words,
        "mean_words": words / rows if rows else 0,
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_word_prefix_counts": {
            "split_source": sum(v for k, v in by_source.items() if "split_source" in k or "source" in k),
            "split_companion": sum(v for k, v in by_source.items() if "companion" in k),
            "suffix_or_filler": sum(v for k, v in by_source.items() if "suffix" in k or "topup" in k or "filler" in k or "cleanqwen" in k),
        },
    }


def model_variant() -> dict[str, Any]:
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


def stream_for(data_arm: str, meta: dict[str, Any]) -> tuple[pathlib.Path, str]:
    key = ARM_TO_STREAM_KEY[data_arm]
    p = pathlib.Path(meta["files"][key])
    if not p.is_absolute():
        p = ROOT / p
    sha_key = f"{key}.jsonl"
    return p, str(meta["sha256"][sha_key])


def default_run_dir(data_arm: str) -> pathlib.Path:
    short = "view" if data_arm == "view_split" else "repeat"
    return RUNS_DIR / f"full_p2c_c2p_abs_{short}_split_dose2p64x_rowholdout_deberta100M_seed43022"


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
        and int(out.get("parameter_count") or 0) == RECIPE["parameter_count_expected"]
        and int(out.get("vocab_size") or 0) == RECIPE["vocab_size_expected"]
        and out.get("tokenizer_label") == "compliant16k_reinvest10M"
        and out.get("all_expected_checkpoints_present") is True
    )
    return out


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


def preflight(data_arm: str, run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    for p in [TRAINER, TOKENIZER, META_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
    meta = read_json(META_PATH)
    if meta.get("status") != "SPLIT_INWINDOW_CONTROLS_MATERIALIZED":
        raise RuntimeError(f"unexpected split metadata status {meta.get('status')}")
    audit = meta.get("audit", {})
    required = {
        "training_100M_written": True,
        "source_and_companion_never_same_row_by_construction": True,
        "split_arms_row_length_sequence_identical": True,
    }
    for k, v in required.items():
        if audit.get(k) != v:
            raise RuntimeError(f"split metadata audit failed for {k}: {audit.get(k)}")
    pool_key = "view_split_pool" if data_arm == "view_split" else "repeat_split_pool"
    if not audit.get(pool_key, {}).get("exact_10M"):
        raise RuntimeError(f"split 10M pool is not exact for {data_arm}: {audit.get(pool_key)}")
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    stream, expected_sha = stream_for(data_arm, meta)
    if not stream.exists():
        raise FileNotFoundError(stream)
    stream_sha = sha256_file(stream)
    if stream_sha != expected_sha:
        raise RuntimeError(f"stream SHA mismatch for {data_arm}: {stream_sha} != {expected_sha}")
    counts = count_jsonl(stream) if count_words else {"count_skipped": True, "exact_100M": True, "words": TOTAL_WORDS}
    if not counts.get("exact_100M"):
        raise RuntimeError(f"stream word count mismatch {counts}")
    mv = model_variant()
    if mv["parameter_count"] != RECIPE["parameter_count_expected"] or mv["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"model coordinate mismatch: {mv}")
    return {
        "status": "SPLIT_INWINDOW_TRAIN_PREFLIGHT_OK",
        "created_utc": now(),
        "data_arm": data_arm,
        "run_dir": rel(run_dir),
        "stream": rel(stream),
        "stream_sha256": stream_sha,
        "stream_accounting": counts,
        "split_metadata": rel(META_PATH),
        "tokenizer_dir": rel(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": mv,
        "recipe": RECIPE,
        "command": ["python3" if x == sys.executable else str(x) for x in build_command(data_arm, run_dir, stream)],
        "scientific_purpose": "Causal locality test for finite-experience relation structure: preserve text exposure and budget while removing same-window source/companion co-occurrence, then compare held-out copy and nonidentical-source conditioning against CLEAN seed43022.",
        "pre_state_numeric_interpretation": rel(WS / "notes" / "split_control_numeric_prestatement.md"),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-arm", choices=["view_split", "repeat_split"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.data_arm)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    info = preflight(args.data_arm, run_dir, args.count_words)
    write_json(PREFLIGHT_DIR / f"{args.data_arm}_preflight.json", info)
    if args.dry_run:
        print(json.dumps({
            "status": info["status"],
            "data_arm": args.data_arm,
            "run_dir": info["run_dir"],
            "stream_sha_prefix": info["stream_sha256"][:12],
            "tokenizer_sha_prefix": info["tokenizer_json_sha256"][:12],
            "words": info["stream_accounting"].get("words"),
            "parameter_count": info["model_variant"]["parameter_count"],
            "preflight_json": rel(PREFLIGHT_DIR / f"{args.data_arm}_preflight.json"),
            "no_training_started": True,
        }, indent=2, ensure_ascii=False), flush=True)
        return

    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        print(json.dumps({"status": "SPLIT_INWINDOW_ARM_ALREADY_FINISHED", "data_arm": args.data_arm, "run_dir": rel(run_dir), "metrics": existing}, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise SystemExit(f"run_dir exists and is non-empty but not complete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "split_inwindow_train_preflight.json", info)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_relation_learning_step015_{args.data_arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stream = pathlib.Path(info["stream"])
    if not stream.is_absolute():
        stream = ROOT / stream
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    start = now()
    start_rec = {
        "status": "SPLIT_INWINDOW_ARM_TRAIN_STARTING",
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "run_dir": rel(run_dir),
        "created_utc": start,
        "scientific_purpose": info["scientific_purpose"],
        "pre_state_numeric_interpretation": info["pre_state_numeric_interpretation"],
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(start_rec, indent=2, ensure_ascii=False), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "split_train_start", "utc": start, "data_arm": args.data_arm, "gpu": args.gpu, "recipe": RECIPE}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(build_command(args.data_arm, run_dir, stream), cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    result = {
        "status": "SPLIT_INWINDOW_ARM_TRAIN_FINISHED" if proc.returncode == 0 else "SPLIT_INWINDOW_ARM_TRAIN_FAILED",
        "returncode": proc.returncode,
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "started_utc": start,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": summarize_metrics(run_dir),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "split_inwindow_train_result.json", result)
    write_json(PREFLIGHT_DIR / f"{args.data_arm}_train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
