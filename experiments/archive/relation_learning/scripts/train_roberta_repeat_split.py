#!/usr/bin/env python3
"""research: train RoBERTa REPEAT_SPLIT for architecture-transfer locality.

The existing RoBERTa seed43022 CLEAN/REPEAT/VIEW arms were trained in frontier_consolidation
with the same 16k tokenizer, 8x480 geometry, WWM p=0.15, 100M-word budget, and
seed43022 coordinate.  This launcher changes only the text stream to the
relation_learning row-split exact-recurrence stream, so the compact T/U readout can ask
whether removing same-window source/companion co-occurrence also removes the
recurrence cost in a second bidirectional MLM coordinate.

No official benchmark evaluation, upload, or final deliverable is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES_WS = ROOT / "experiments/archive/frontier_consolidation"
TRAINER = REPRESENTATION_FRONTIER_STUDIES_WS / "scripts/roberta_mlm_transfer_trainer.py"
TOKENIZER = REPRESENTATION_FRONTIER_STUDIES_WS / "data/compliant_tokenizer"
POOL_DIR = WS / "data/split_inwindow_rowholdout_pools"
META_PATH = POOL_DIR / "split_inwindow_rowholdout_metadata.json"
PREFLIGHT_DIR = WS / "data/roberta_repeat_split_train_preflight"
RUN_DIR_DEFAULT = WS / "training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOTAL_WORDS = 100_000_000
STREAM_KEY = "compact_repeat_split_dose2p64x_100M"
STREAM_SHA_KEY = "compact_repeat_split_dose2p64x_100M.jsonl"
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]

RECIPE = {
    "model_family": "roberta",
    "parameter_count_expected": 30_528_064,
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
    "mask_prob": 0.15,
    "checkpoint_words": 10_000_000,
    "max_word_exposure": 100_000_000,
    "lr_total_steps": 2529,
    "num_workers": 0,
    "log_every": 50,
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
    source_words: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            wf = int(obj.get("words", len(text.split())))
            wa = len(text.split())
            if wf != wa:
                raise RuntimeError(f"word-count mismatch row {rows}: field={wf} actual={wa}")
            rows += 1
            words += wf
            src = str(obj.get("source") or "")
            source_words[src] = source_words.get(src, 0) + wf
            rec = {"row": rows - 1, "words": wf, "source": src, "example_id": obj.get("example_id"), "text_prefix": text[:120]}
            if len(first_rows) < 3:
                first_rows.append(rec)
            last_rows.append(rec)
            if len(last_rows) > 3:
                last_rows.pop(0)
    return {
        "rows": rows,
        "words": words,
        "exact_100M": words == TOTAL_WORDS,
        "sha256": sha256_file(path),
        "first_rows": first_rows,
        "last_rows": last_rows,
        "source_words_top20": sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "expected_steps_at_batch256": math.ceil(rows / RECIPE["batch_size"]),
    }


def model_variant() -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    cfg = RobertaConfig(
        vocab_size=len(tok),
        hidden_size=RECIPE["hidden_size"],
        num_hidden_layers=RECIPE["n_layer"],
        num_attention_heads=RECIPE["n_head"],
        intermediate_size=RECIPE["hidden_size"] * RECIPE["ffn_mult"],
        max_position_embeddings=512,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        layer_norm_eps=1e-5,
    )
    model = RobertaForMaskedLM(cfg)
    return {"parameter_count": sum(p.numel() for p in model.parameters()), "vocab_size": len(tok), "model_type": cfg.model_type}


def stream_from_meta(meta: dict[str, Any]) -> tuple[pathlib.Path, str]:
    stream = pathlib.Path(meta["files"][STREAM_KEY])
    if not stream.is_absolute():
        stream = ROOT / stream
    expected_sha = str(meta["sha256"][STREAM_SHA_KEY])
    return stream, expected_sha


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
        "saved_checkpoints": [x.get("name") for x in saved if isinstance(x, dict)],
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


def build_command(run_dir: pathlib.Path, stream: pathlib.Path) -> list[str]:
    return [
        sys.executable, "-B", str(TRAINER),
        "--tokenizer_path", str(TOKENIZER),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--model_family", "roberta",
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
        "--weight_decay", str(RECIPE["weight_decay"]),
        "--warmup_fraction", str(RECIPE["warmup_fraction"]),
        "--mask_prob", str(RECIPE["mask_prob"]),
        "--max_word_exposure", str(RECIPE["max_word_exposure"]),
        "--checkpoint_words", str(RECIPE["checkpoint_words"]),
        "--lr_total_steps", str(RECIPE["lr_total_steps"]),
        "--num_workers", str(RECIPE["num_workers"]),
        "--log_every", str(RECIPE["log_every"]),
        "--example_jsonl", str(stream),
        "--example_jsonl_label", "roberta_repeat_split_dose2p64x_rowholdout",
        "--output_dir", str(run_dir),
    ]


def preflight(run_dir: pathlib.Path, count_words: bool) -> dict[str, Any]:
    for p in [TRAINER, TOKENIZER, META_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
    meta = read_json(META_PATH)
    if meta.get("status") != "SPLIT_INWINDOW_CONTROLS_MATERIALIZED":
        raise RuntimeError(f"unexpected split metadata status {meta.get('status')}")
    info = meta.get("audit", {})
    required_true = [
        "training_100M_written",
        "source_and_companion_never_same_row_by_construction",
        "split_arms_row_length_sequence_identical",
    ]
    for key in required_true:
        if info.get(key) is not True:
            raise RuntimeError(f"split metadata field not true: {key}={info.get(key)}")
    if not info.get("repeat_split_pool", {}).get("exact_10M"):
        raise RuntimeError("repeat_split 10M pool is not exact")
    stream, expected_sha = stream_from_meta(meta)
    if not stream.exists():
        raise FileNotFoundError(stream)
    tok_sha = sha256_file(TOKENIZER / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    counts = count_jsonl(stream) if count_words else {"count_skipped": True, "sha256": sha256_file(stream), "words": TOTAL_WORDS, "exact_100M": True}
    if counts.get("sha256") != expected_sha:
        raise RuntimeError(f"stream SHA mismatch: {counts.get('sha256')} != {expected_sha}")
    if not counts.get("exact_100M"):
        raise RuntimeError(f"stream word count mismatch: {counts}")
    mv = model_variant()
    if mv["parameter_count"] != RECIPE["parameter_count_expected"] or mv["vocab_size"] != RECIPE["vocab_size_expected"]:
        raise RuntimeError(f"RoBERTa coordinate mismatch: {mv}")
    payload = {
        "status": "ROBERTA_REPEAT_SPLIT_PREFLIGHT_OK",
        "created_utc": now(),
        "scientific_purpose": "Second-architecture locality test: train RoBERTa on the same row-split exact-recurrence stream and later compare compact T/U/N readout against existing RoBERTa CLEAN and REPEAT seed43022.",
        "stream": rel(stream),
        "stream_sha256": counts.get("sha256"),
        "stream_accounting": counts,
        "split_metadata": rel(META_PATH),
        "tokenizer_dir": rel(TOKENIZER),
        "tokenizer_json_sha256": tok_sha,
        "model_variant": mv,
        "recipe": RECIPE,
        "run_dir": rel(run_dir),
        "command": ["python3" if x == sys.executable else str(x) for x in build_command(run_dir, stream)],
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    write_json(PREFLIGHT_DIR / "roberta_repeat_split_preflight.json", payload)
    return payload


def query_gpu(gpu: int) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "stderr": proc.stderr[-1000:]}
    parts = [x.strip() for x in proc.stdout.strip().split(",")]
    return {"ok": True, "memory_used_mib": int(parts[0]), "memory_total_mib": int(parts[1]), "utilization_pct": int(parts[2])}


def wait_for_gpu(gpu: int, threshold_mib: int, util_threshold: int, interval_sec: int, max_wait_sec: int, log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        while True:
            q = query_gpu(gpu)
            q["utc"] = now(); q["elapsed_sec"] = round(time.time() - start, 1); q["threshold_mib"] = threshold_mib; q["util_threshold"] = util_threshold
            f.write(json.dumps(q) + "\n"); f.flush()
            if q.get("ok") and int(q.get("memory_used_mib", 10**9)) <= threshold_mib and int(q.get("utilization_pct", 100)) <= util_threshold:
                return
            if time.time() - start > max_wait_sec:
                raise TimeoutError(f"GPU {gpu} did not become free within {max_wait_sec}s; last={q}")
            time.sleep(interval_sec)


def copy_result_to_preflight(run_dir: pathlib.Path) -> None:
    result = run_dir / "roberta_repeat_split_train_result.json"
    if result.exists():
        shutil.copy2(result, PREFLIGHT_DIR / "roberta_repeat_split_train_result.json")


def launch(args: argparse.Namespace, run_dir: pathlib.Path, pre: dict[str, Any]) -> None:
    existing = summarize_metrics(run_dir)
    if existing.get("ready"):
        result = {"status": "ROBERTA_REPEAT_SPLIT_ALREADY_FINISHED", "run_dir": rel(run_dir), "metrics": existing}
        write_json(PREFLIGHT_DIR / "roberta_repeat_split_train_result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise SystemExit(f"run_dir exists and is non-empty but incomplete: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "roberta_repeat_split_preflight.json", pre)
    wait_log = run_dir / "gpu_wait_log.jsonl"
    if args.wait_for_gpu_free:
        wait_for_gpu(args.gpu, args.gpu_memory_threshold_mib, args.gpu_util_threshold_pct, args.gpu_wait_interval_sec, args.gpu_max_wait_sec, wait_log)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TMPDIR"] = f"/tmp/q_relation_learning_step022_roberta_repeat_split_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    stream = ROOT / pre["stream"] if not pathlib.Path(pre["stream"]).is_absolute() else pathlib.Path(pre["stream"])
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    start = now()
    print(json.dumps({"status": "ROBERTA_REPEAT_SPLIT_TRAIN_STARTING", "run_dir": rel(run_dir), "gpu": args.gpu, "started_utc": start, "wait_log": rel(wait_log)}, indent=2), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "roberta_repeat_split_train_start", "utc": start, "recipe": RECIPE, "gpu": args.gpu}) + "\n")
        out.flush()
        proc = subprocess.run(build_command(run_dir, stream), cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    result = {
        "status": "ROBERTA_REPEAT_SPLIT_TRAIN_FINISHED" if proc.returncode == 0 else "ROBERTA_REPEAT_SPLIT_TRAIN_FAILED",
        "returncode": proc.returncode,
        "gpu": args.gpu,
        "started_utc": start,
        "finished_utc": now(),
        "run_dir": rel(run_dir),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "wait_log": rel(wait_log),
        "metrics": summarize_metrics(run_dir),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
        "mechanism_instrument_not_leaderboard_submission": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(run_dir / "roberta_repeat_split_train_result.json", result)
    write_json(PREFLIGHT_DIR / "roberta_repeat_split_train_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--run-dir", default=str(RUN_DIR_DEFAULT))
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    ap.add_argument("--wait-for-gpu-free", action="store_true")
    ap.add_argument("--gpu-memory-threshold-mib", type=int, default=3000)
    ap.add_argument("--gpu-util-threshold-pct", type=int, default=25)
    ap.add_argument("--gpu-wait-interval-sec", type=int, default=60)
    ap.add_argument("--gpu-max-wait-sec", type=int, default=7200)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    pre = preflight(run_dir, args.count_words)
    if args.dry_run:
        print(json.dumps({
            "status": pre["status"],
            "run_dir": pre["run_dir"],
            "stream_sha_prefix": pre["stream_sha256"][:12],
            "tokenizer_sha_prefix": pre["tokenizer_json_sha256"][:12],
            "words": pre["stream_accounting"].get("words"),
            "rows": pre["stream_accounting"].get("rows"),
            "parameter_count": pre["model_variant"]["parameter_count"],
            "preflight_json": rel(PREFLIGHT_DIR / "roberta_repeat_split_preflight.json"),
            "no_training_started": True,
        }, indent=2, ensure_ascii=False), flush=True)
        return
    launch(args, run_dir, pre)


if __name__ == "__main__":
    main()
