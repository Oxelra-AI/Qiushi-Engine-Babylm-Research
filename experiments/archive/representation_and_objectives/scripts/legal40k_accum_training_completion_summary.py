#!/usr/bin/env python3
"""Post-completion verifier for research legal-40k accumulated trainings.

Run after both accumulated-batch training runs complete. The
script summarizes training completion, exact tokenizer/config identity, effective
batch/microbatch metadata, word exposure, checkpoint ladder, final loss, and
result-file status. It is CPU-only and does not evaluate BabyLM competence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WS / "data/legal40k_accum_training_completion"
NOTE = (ROOT / 'research/notes/representation_and_objectives/legal40k_accum_training_completion.md')
EXPECTED_SOURCE_TOKENIZER_SHA = "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758"
EXPECTED_SAVED_TOKENIZER_SHA = "e6c98b339c6583ea2d8721e6c736572d43b7f7b418203a5f3c84198698f9a0f3"
EXPECTED_TOKENIZER_VOCAB_HASH = "76e24ed084c191ed65091222465835080a3fd1a92a6127397dbb4f6f35aab7c2"
EXPECTED_STREAM_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
REQUIRED_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
SEEDS = {
    "43022": {
        "run_dir": WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022",
        "manifest": WS / "data/legal40k_accum_training/legal40k_accum_train_manifest_seed43022.json",
        "result": WS / "data/legal40k_accum_training/legal40k_accum_train_result_seed43022.json",
        "task_ref": "s61_t27_tool1",
    },
    "43122": {
        "run_dir": WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43122",
        "manifest": WS / "data/legal40k_accum_training/legal40k_accum_train_manifest_seed43122.json",
        "result": WS / "data/legal40k_accum_training/legal40k_accum_train_result_seed43122.json",
        "task_ref": "s61_t27_tool2",
    },
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_jsonable(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        return {
            "path": str(path),
            "exists": True,
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
            "vocab_size_property": tok.vocab_size,
            "len_tokenizer": len(tok),
            "is_fast": bool(tok.is_fast),
            "special_token_ids": {name: getattr(tok, name + "_id") for name in ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token"]},
            "vocab_hash": hash_jsonable(tok.get_vocab()),
        }
    except Exception as e:
        return {"path": str(path), "exists": True, "load_error": str(e), "tokenizer_json_sha256": sha256_file(path / "tokenizer.json")}


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {"_json_status": "empty_file", "path": str(path)}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_json_status": "invalid_json", "path": str(path), "error": str(e), "prefix": text[:200]}


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def last_jsonl(path: Path) -> tuple[int, Any]:
    n = 0
    last = None
    if not path.exists():
        return 0, None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                last = json.loads(line)
            except Exception:
                last = {"raw": line[-500:]}
    return n, last


def summarize_seed(seed: str, cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir: Path = cfg["run_dir"]
    model_root = run_dir / "hf_model"
    metrics = load_json(run_dir / "scientific_metrics.json")
    example_order = load_json(run_dir / "example_order_manifest.json")
    manifest = load_json(cfg["manifest"])
    result = load_json(cfg["result"])
    nlog, last_log = last_jsonl(run_dir / "training_log.jsonl")
    missing_steps = [s for s in REQUIRED_STEPS if not (model_root / s).exists()]
    tok_id_1m = tokenizer_identity(model_root / "chck_1M")
    tok_id_100m = tokenizer_identity(model_root / "chck_100M")
    tok_sha_1m = tok_id_1m.get("tokenizer_json_sha256")
    tok_sha_100m = tok_id_100m.get("tokenizer_json_sha256")
    cfg_1m = load_json(model_root / "chck_1M/config.json")
    cfg_100m = load_json(model_root / "chck_100M/config.json")
    errors: list[str] = []
    if not isinstance(result, dict):
        errors.append("missing training result json")
    elif result.get("status") != "TRAINING_COMPLETE" or result.get("returncode") != 0:
        errors.append(f"training result not complete: status={result.get('status')} returncode={result.get('returncode')}")
    if not isinstance(metrics, dict):
        errors.append("missing scientific_metrics.json")
    else:
        if metrics.get("word_exposure") != 100000000:
            errors.append(f"metrics word_exposure {metrics.get('word_exposure')} != 100000000")
        if metrics.get("vocab_size") != 40000:
            errors.append(f"metrics vocab_size {metrics.get('vocab_size')} != 40000")
        if metrics.get("effective_batch_size") != 256 or metrics.get("micro_batch_size") != 64:
            errors.append("metrics effective/micro batch metadata not 256/64")
        if metrics.get("gradient_accumulation_steps") != 4:
            errors.append("metrics gradient_accumulation_steps != 4")
    if not isinstance(example_order, dict):
        errors.append("missing example_order_manifest.json")
    else:
        if example_order.get("selected_for_training_words") != 100000000:
            errors.append(f"example_order selected words {example_order.get('selected_for_training_words')} != 100000000")
        if example_order.get("seed") != 43:
            errors.append(f"example_order seed {example_order.get('seed')} != 43")
    if missing_steps:
        errors.append(f"missing checkpoint ladder steps: {missing_steps}")
    for label, tok_id in [("chck_1M", tok_id_1m), ("chck_100M", tok_id_100m)]:
        if not tok_id.get("exists"):
            errors.append(f"missing {label} tokenizer directory")
            continue
        if tok_id.get("vocab_size_property") != 40000 or tok_id.get("len_tokenizer") != 40000 or not tok_id.get("is_fast"):
            errors.append(f"{label} tokenizer functional properties invalid: vocab_size={tok_id.get('vocab_size_property')} len={tok_id.get('len_tokenizer')} is_fast={tok_id.get('is_fast')}")
        if tok_id.get("special_token_ids") != {"bos_token": 1, "eos_token": 2, "unk_token": 0, "pad_token": 3, "mask_token": 4}:
            errors.append(f"{label} tokenizer special IDs changed: {tok_id.get('special_token_ids')}")
        if tok_id.get("vocab_hash") != EXPECTED_TOKENIZER_VOCAB_HASH and tok_id.get("tokenizer_json_sha256") not in (EXPECTED_SOURCE_TOKENIZER_SHA, EXPECTED_SAVED_TOKENIZER_SHA):
            errors.append(f"{label} tokenizer does not match expected legal40k by vocab hash or accepted tokenizer SHA")
    for label, c in [("chck_1M", cfg_1m), ("chck_100M", cfg_100m)]:
        if not isinstance(c, dict):
            errors.append(f"missing {label}/config.json")
            continue
        if c.get("vocab_size") != 40000:
            errors.append(f"{label} config vocab_size {c.get('vocab_size')} != 40000")
        if c.get("hidden_size") != 480:
            errors.append(f"{label} config hidden_size {c.get('hidden_size')} != 480")
        if c.get("num_hidden_layers") != 8:
            errors.append(f"{label} config num_hidden_layers {c.get('num_hidden_layers')} != 8")
    manifest_text = json.dumps(manifest) if isinstance(manifest, dict) else ""
    if EXPECTED_STREAM_SHA not in manifest_text or EXPECTED_SOURCE_TOKENIZER_SHA not in manifest_text:
        errors.append("manifest missing expected stream or source tokenizer sha")
    return {
        "seed": seed,
        "task_ref": cfg["task_ref"],
        "run_dir": file_record(run_dir),
        "manifest": file_record(cfg["manifest"]),
        "result": file_record(cfg["result"]),
        "result_json": result,
        "training_log_records": nlog,
        "last_training_log_record": last_log,
        "scientific_metrics": metrics,
        "example_order_manifest_excerpt": {
            k: example_order.get(k) for k in ["seed", "selected_for_training_words", "num_consumed_examples", "effective_batch_size", "micro_batch_size", "gradient_accumulation_steps", "example_jsonl_label"]
        } if isinstance(example_order, dict) else None,
        "checkpoint_missing_steps": missing_steps,
        "tokenizer_sha_chck_1M": tok_sha_1m,
        "tokenizer_sha_chck_100M": tok_sha_100m,
        "tokenizer_identity_chck_1M": tok_id_1m,
        "tokenizer_identity_chck_100M": tok_id_100m,
        "config_chck_1M": {k: cfg_1m.get(k) for k in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]} if isinstance(cfg_1m, dict) else None,
        "config_chck_100M": {k: cfg_100m.get(k) for k in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]} if isinstance(cfg_100m, dict) else None,
        "errors": errors,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_summaries = {seed: summarize_seed(seed, cfg) for seed, cfg in SEEDS.items()}
    all_errors = {seed: s["errors"] for seed, s in seed_summaries.items() if s["errors"]}
    complete = not all_errors
    payload = {
        "status": "LEGAL40K_ACCUM_TRAINING_COMPLETE" if complete else "LEGAL40K_ACCUM_TRAINING_INCOMPLETE_OR_ERROR",
        "complete": complete,
        "expected": {
            "source_tokenizer_sha256": EXPECTED_SOURCE_TOKENIZER_SHA,
            "saved_checkpoint_tokenizer_sha256": EXPECTED_SAVED_TOKENIZER_SHA,
            "tokenizer_vocab_hash": EXPECTED_TOKENIZER_VOCAB_HASH,
            "train_stream_sha256": EXPECTED_STREAM_SHA,
            "vocab_size": 40000,
            "effective_batch_size": 256,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "word_exposure": 100000000,
        },
        "seeds": seed_summaries,
        "errors_by_seed": all_errors,
    }
    out_json = OUT_DIR / "legal40k_accum_training_completion_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research legal-40k accumulated training completion", ""]
    lines.append(f"Status: `{payload['status']}`")
    lines.append("")
    for seed, summary in seed_summaries.items():
        metrics = summary.get("scientific_metrics") or {}
        result = summary.get("result_json") or {}
        lines.append(f"## Seed {seed}")
        lines.append(f"- task: `{summary['task_ref']}`")
        lines.append(f"- result status: `{result.get('status')}` returncode `{result.get('returncode')}`")
        lines.append(f"- word exposure: `{metrics.get('word_exposure')}`")
        lines.append(f"- final loss: `{metrics.get('loss_last')}`")
        lines.append(f"- vocab/hidden/layers at chck_100M: `{summary.get('config_chck_100M')}`")
        if summary["errors"]:
            lines.append(f"- errors: `{summary['errors']}`")
        lines.append("")
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "complete": complete, "errors_by_seed": all_errors, "out_json": str(out_json), "note": str(NOTE)}, indent=2))


if __name__ == "__main__":
    main()
