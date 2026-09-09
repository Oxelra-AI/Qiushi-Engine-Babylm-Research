#!/usr/bin/env python3
"""research: launch matched source-only extractive DeBERTa 100M trainings.

Runs the research stock DeBERTa-v2 masked-LM recipe on research source-only
extractive 100M streams. The two intended arms are:
  * extractive_balanced: density/token-mass closest to compact, source-only.
  * extractive_wide: coverage/content-maximized source-only.

The paired result spans the irreducible tradeoff found in research. Compact
superiority over either arm alone must not be read as isolating generated
re-expression; if both source-only arms lag compact, the evidence is a bundled
natural-compact deficit (fluency/source-absent/context distribution/token mass).

This launcher performs training only. It does not run official selected eval,
SuperGLUE, AoA, upload, or leaderboard submission.
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
STREAM_DIR = WORKSPACE / "data" / "extractive_view_100m_streams"
STREAM_MANIFEST = STREAM_DIR / "extractive_100m_streams_manifest.json"
PREFLIGHT = WORKSPACE / "data" / "extractive_view_pool_preflight" / "extractive_view_pool_preflight.json"

ARMS: dict[str, dict[str, Any]] = {
    "balanced": {
        "stream_name": "extractive_balanced_100M.jsonl",
        "run_name": "extractive_balanced_deberta100M_seed43022",
        "label": "extractive_balanced_source_only_reinvest",
        "scientific_role": "density/token-mass-nearest source-only compact proxy; one-sided positive source-selection sufficiency test",
    },
    "wide": {
        "stream_name": "extractive_wide_100M.jsonl",
        "run_name": "extractive_wide_deberta100M_seed43022",
        "label": "extractive_wide_source_only_reinvest",
        "scientific_role": "coverage/content-maximized source-only compact proxy spanning the research density-coverage tradeoff",
    },
}

RECIPE = {
    "model_family": "DebertaV2ForMaskedLM",
    "parameter_count_expected": 34467424,
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
    bad_counts = {str(e): c for e, c in unique_changed.items() if c != 10}
    return {
        "rows": rows,
        "words": words,
        "changed_rows": changed,
        "unique_changed_ids": len(unique_changed),
        "bad_changed_repeat_counts": bad_counts,
        "exact_100M": rows == 647400 and words == 100000000,
    }


def load_tokenizer_metadata() -> dict[str, Any]:
    meta_path = TOKENIZER / "tokenizer_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    meta["tokenizer_json_sha256_observed"] = tok_sha
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer sha mismatch: {tok_sha} != {EXPECTED_TOKENIZER_SHA}")
    if meta.get("training_data", {}).get("pool_words") != 10_000_000:
        raise RuntimeError(f"tokenizer metadata does not show 10M pool: {meta}")
    if meta.get("tokenizer", {}).get("vocab_size") != 16_384:
        raise RuntimeError(f"tokenizer vocab mismatch: {meta.get('tokenizer')}")
    return meta


def manifest_record_for(stream_name: str) -> dict[str, Any]:
    manifest = json.loads(STREAM_MANIFEST.read_text(encoding="utf-8"))
    for rec in manifest.get("variant_results", []):
        if pathlib.Path(rec.get("path", "")).name == stream_name:
            return rec
    raise KeyError(f"no manifest record for {stream_name}")


def preflight(arm: str, run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    cfg = ARMS[arm]
    train_file = STREAM_DIR / cfg["stream_name"]
    required = {
        "trainer": TRAINER,
        "tokenizer_dir": TOKENIZER,
        "tokenizer_json": TOKENIZER / "tokenizer.json",
        "train_file": train_file,
        "stream_manifest": STREAM_MANIFEST,
        "preflight": PREFLIGHT,
    }
    missing = {k: str(v) for k, v in required.items() if not pathlib.Path(v).exists()}
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    manifest_rec = manifest_record_for(cfg["stream_name"])
    actual_sha = sha256_file(train_file)
    if actual_sha != manifest_rec.get("sha256"):
        raise RuntimeError(f"stream sha mismatch for {arm}: {actual_sha} != {manifest_rec.get('sha256')}")
    if not manifest_rec.get("status_ok"):
        raise RuntimeError(f"stream manifest says not ok for {arm}: {manifest_rec}")
    word_info = None
    if count_words:
        word_info = count_jsonl(train_file)
        if not word_info["exact_100M"] or word_info["changed_rows"] != 30050 or word_info["unique_changed_ids"] != 3005 or word_info["bad_changed_repeat_counts"]:
            raise RuntimeError(f"stream word/count preflight failed for {arm}: {word_info}")
    tokenizer_meta = load_tokenizer_metadata()
    return {
        "status": "EXTRACTIVE_DEBERTA_PREFLIGHT_OK",
        "arm": arm,
        "scientific_role": cfg["scientific_role"],
        "run_dir": str(run_dir),
        "train_file": str(train_file),
        "train_file_sha256": actual_sha,
        "stream_manifest": str(STREAM_MANIFEST),
        "stream_manifest_record": manifest_rec,
        "preflight": str(PREFLIGHT),
        "word_info": word_info,
        "tokenizer_dir": str(TOKENIZER),
        "tokenizer_metadata": tokenizer_meta,
        "trainer": str(TRAINER),
        "recipe": RECIPE,
        "interpretation_boundary": {
            "paired_arms_required": "balanced and wide should be read together because source-only views cannot match compact simultaneously on density, source coverage, source-absent content, fluency, and token mass.",
            "negative_both_arms": "compact superiority over both extractive variants is bundled evidence for natural generated compact data; it does not isolate fluency, source-absent content, coverage, or token mass alone.",
            "positive_either_arm": "if a source-only variant approaches compact and beats repeat on stable late selected families, source selection is partly sufficient.",
        },
        "no_official_eval_upload_aoa_or_leaderboard": True,
    }


def build_command(arm: str, run_dir: pathlib.Path) -> list[str]:
    cfg = ARMS[arm]
    train_file = STREAM_DIR / cfg["stream_name"]
    return [
        sys.executable, "-B", str(TRAINER),
        "--example_jsonl", str(train_file),
        "--example_jsonl_label", cfg["label"],
        "--example_jsonl_meta", str(STREAM_MANIFEST),
        "--output_dir", str(run_dir),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", str(RECIPE["hidden_size"]),
        "--n_layer", str(RECIPE["n_layer"]),
        "--n_head", str(RECIPE["n_head"]),
        "--ffn_mult", str(RECIPE["ffn_mult"]),
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
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else RUNS_DIR / ARMS[args.arm]["run_name"]
    status = preflight(args.arm, run_dir, count_words=args.count_words)
    cmd = build_command(args.arm, run_dir)
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
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step225_extractive_{args.arm}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "launcher_start", "started_utc": started, "arm": args.arm, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = now()

    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "EXTRACTIVE_DEBERTA_TRAIN_FINISHED" if proc.returncode == 0 else "EXTRACTIVE_DEBERTA_TRAIN_FAILED",
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
        "tokenizer_dir": str(TOKENIZER),
        "no_official_eval_upload_aoa_or_leaderboard": True,
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
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
