#!/usr/bin/env python3
"""research: launch paired DeBERTa positional-component ablation trainings.

This is an execution wrapper around the already-used research/COMPACT_EXPERIENCE
`masking_curriculum_trainer.py`. It changes only one DeBERTa positional-attention
configuration at a time while preserving the legal research tokenizer, compact or
repeat JSONL stream, WWM p=0.15, AdamW/LR/seed/seq/batch/exposure recipe, and
HuggingFace checkpoint format.

Primary first variant: `no_disentangle_abs` = relative_attention=True but
pos_att_type=[], with absolute input positions still enabled. In the Transformers
implementation this removes both c2p and p2c attention-score terms while leaving
DeBERTa's rest of architecture/head/objective/data coordinate unchanged.

The scientific estimand is compact-minus-repeat within the same variant at mature
checkpoints, not a single-arm gain and not a comparison against the old
baseline16k repeat run.
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
ROOT = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = ROOT
RUNS_DIR = WORKSPACE / "training" / "runs"
TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
TOKENIZER = WORKSPACE / "data" / "compliant_tokenizer"
STREAM_BASE = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard"

STREAMS = {
    "compact": {
        "path": STREAM_BASE / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
        "label": "legal_compact_view_reinvest",
        "expected_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    },
    "repeat": {
        "path": STREAM_BASE / "cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl",
        "label": "legal_repeat_compact_reinvest",
        "expected_sha256": "91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa",
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "full_p2c_c2p_abs": {
        "deberta_pos_att_type": "p2c,c2p",
        "run_stub": "full_p2c_c2p_abs",
        "relative_attention": True,
        "pos_att_type": ["p2c", "c2p"],
        "scientific_role": "stock DeBERTa full disentangled p2c+c2p score terms with absolute input positions",
    },
    "no_disentangle_abs": {
        "deberta_pos_att_type": "",
        "run_stub": "no_disentangle_abs",
        "relative_attention": True,
        "pos_att_type": [],
        "scientific_role": "remove both c2p and p2c disentangled attention-score terms while preserving legal data/objective/optimizer and absolute input positions",
    },
    "c2p_only_abs": {
        "deberta_pos_att_type": "c2p",
        "run_stub": "c2p_only_abs",
        "relative_attention": True,
        "pos_att_type": ["c2p"],
        "scientific_role": "retain content-to-position score term only; remove p2c",
    },
    "p2c_only_abs": {
        "deberta_pos_att_type": "p2c",
        "run_stub": "p2c_only_abs",
        "relative_attention": True,
        "pos_att_type": ["p2c"],
        "scientific_role": "retain position-to-content score term only; remove c2p",
    },
}

RECIPE = {
    "parameter_count_full_expected": 34_467_424,
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
}

EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"


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
    changed = 0
    unique_changed: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            text = str(obj.get("text", ""))
            field_words = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            if field_words != actual_words:
                raise RuntimeError(f"word mismatch {path} row {rows}: field={field_words} actual={actual_words}")
            words += field_words
            eid = obj.get("example_id")
            if isinstance(eid, int) and 950000 <= eid <= 953004:
                changed += 1
                unique_changed[eid] = unique_changed.get(eid, 0) + 1
    return {
        "rows": rows,
        "words": words,
        "changed_rows": changed,
        "unique_changed_ids": len(unique_changed),
        "bad_changed_repeat_counts": {str(k): v for k, v in unique_changed.items() if v != 10},
        "exact_100M": rows == 647400 and words == 100000000,
    }


def variant_param_count(variant: str) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    v = VARIANTS[variant]
    cfg = DebertaV2Config(
        vocab_size=len(tok),
        hidden_size=RECIPE["hidden_size"],
        num_hidden_layers=RECIPE["n_layer"],
        num_attention_heads=RECIPE["n_head"],
        intermediate_size=RECIPE["hidden_size"] * RECIPE["ffn_mult"],
        max_position_embeddings=512,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=v["relative_attention"],
        pos_att_type=v["pos_att_type"],
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
        "config": cfg.to_dict(),
        "has_encoder_rel_embeddings": any(k.endswith("encoder.rel_embeddings.weight") for k in keys),
        "has_pos_key_proj": any("pos_key_proj" in k for k in keys),
        "has_pos_query_proj": any("pos_query_proj" in k for k in keys),
    }


def default_run_dir(variant: str, data_arm: str) -> pathlib.Path:
    return RUNS_DIR / f"step242_{VARIANTS[variant]['run_stub']}_{data_arm}_deberta100M_seed43022"


def preflight(variant: str, data_arm: str, run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    if not TRAINER.exists():
        raise FileNotFoundError(TRAINER)
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)
    stream = STREAMS[data_arm]
    path = stream["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"legal tokenizer SHA mismatch: {tok_sha}")
    stream_sha = sha256_file(path)
    if stream_sha != stream["expected_sha256"]:
        raise RuntimeError(f"stream SHA mismatch for {data_arm}: {stream_sha} != {stream['expected_sha256']}")
    word_info = count_jsonl(path) if count_words else None
    if word_info is not None:
        if not word_info["exact_100M"] or word_info["changed_rows"] != 30050 or word_info["unique_changed_ids"] != 3005 or word_info["bad_changed_repeat_counts"]:
            raise RuntimeError(f"stream accounting failed for {data_arm}: {word_info}")
    return {
        "status": "DEBERTA_POSITIONAL_ABLATION_PREFLIGHT_OK",
        "variant": variant,
        "data_arm": data_arm,
        "scientific_role": VARIANTS[variant]["scientific_role"],
        "run_dir": str(run_dir),
        "trainer": str(TRAINER),
        "train_file": str(path),
        "train_file_sha256": stream_sha,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "word_info": word_info,
        "model_variant": variant_param_count(variant),
        "recipe": RECIPE,
        "scientific_estimand": "compact-minus-repeat within the same positional variant, read at mature official-compatible selected checkpoints; not a single-arm or loss-only result",
        "no_official_eval_upload_aoa_or_leaderboard": True,
    }


def build_command(variant: str, data_arm: str, run_dir: pathlib.Path) -> list[str]:
    v = VARIANTS[variant]
    stream = STREAMS[data_arm]
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(stream["path"]),
        "--example_jsonl_label", stream["label"],
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
        "--deberta_pos_att_type", str(v["deberta_pos_att_type"]),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--data-arm", required=True, choices=sorted(STREAMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.variant, args.data_arm)
    status = preflight(args.variant, args.data_arm, run_dir, count_words=args.count_words)
    cmd = build_command(args.variant, args.data_arm, run_dir)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)

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
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step242_{args.variant}_{args.data_arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "variant": args.variant, "data_arm": args.data_arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = now()

    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "DEBERTA_POSITIONAL_ABLATION_TRAIN_FINISHED" if proc.returncode == 0 else "DEBERTA_POSITIONAL_ABLATION_TRAIN_FAILED",
        "variant": args.variant,
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "no_official_eval_upload_aoa_or_leaderboard": True,
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved = metrics.get("saved_checkpoints", [])
        result["metrics"] = {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "mask_prob_start": metrics.get("mask_prob_start"),
            "mask_prob_end": metrics.get("mask_prob_end"),
            "saved_checkpoints": len(saved),
            "first_checkpoint": saved[0].get("name") if saved else None,
            "last_checkpoint": saved[-1].get("name") if saved else None,
        }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
