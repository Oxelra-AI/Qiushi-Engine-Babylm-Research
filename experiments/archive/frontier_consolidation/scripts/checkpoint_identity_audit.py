#!/usr/bin/env python3
"""research: CPU/file identity audit for a selected DeBERTa checkpoint.

Records legal exposure, file hashes, trusted-code metadata, and a small CPU loadability
check for a candidate checkpoint (e.g. reference chck_84M). It is endpoint/provenance
evidence only and performs no official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DEFAULT_RUN_DIR = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def checkpoint_rec(metrics: dict[str, Any], name: str) -> dict[str, Any] | None:
    for rec in metrics.get("saved_checkpoints", []):
        if isinstance(rec, dict) and rec.get("name") == name:
            return dict(rec)
    return None


def loadability(model_dir: pathlib.Path) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        adapter_param_count = sum(p.numel() for n, p in model.named_parameters() if "adapter" in n)
        enc = tok("A small child sees a dog.", return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        finite = bool(torch.isfinite(out.logits).all().item())
        return {
            "attempted": True,
            "ok": True,
            "model_class": type(model).__name__,
            "tokenizer_class": type(tok).__name__,
            "param_count": int(param_count),
            "adapter_param_count": int(adapter_param_count),
            "vocab_size_config": int(getattr(model.config, "vocab_size", -1)),
            "vocab_size_tokenizer": int(len(tok)),
            "adapter_scale": getattr(model.config, "adapter_scale", None),
            "adapter_bottleneck": getattr(model.config, "adapter_bottleneck", None),
            "auto_map": getattr(model.config, "auto_map", None),
            "logits_shape": list(out.logits.shape),
            "logits_all_finite": finite,
        }
    except Exception as e:
        return {"attempted": True, "ok": False, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--reference-endpoint", default="chck_82M")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--cpu-load", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "scientific_metrics.json"
    metrics = read_json(metrics_path)
    endpoint_dir = args.run_dir / "hf_model" / args.endpoint
    ref_dir = args.run_dir / "hf_model" / args.reference_endpoint
    endpoint_metrics = checkpoint_rec(metrics, args.endpoint)
    ref_metrics = checkpoint_rec(metrics, args.reference_endpoint)

    files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_scaled_modeling.py"]
    endpoint_files = {f: file_record(endpoint_dir / f) for f in files}
    ref_files = {f: file_record(ref_dir / f) for f in files}

    same_static_files = {}
    for f in files:
        if f == "model.safetensors":
            continue
        same_static_files[f] = endpoint_files[f].get("sha256") == ref_files[f].get("sha256") and endpoint_files[f].get("sha256") is not None

    cfg = read_json(endpoint_dir / "config.json") if (endpoint_dir / "config.json").exists() else {}
    result = {
        "status": "CHECKPOINT_IDENTITY_AUDIT",
        "created_utc": now(),
        "run_dir": rel(args.run_dir),
        "endpoint": args.endpoint,
        "reference_endpoint": args.reference_endpoint,
        "metrics_path": rel(metrics_path),
        "run_summary": {
            "variant": metrics.get("variant"),
            "backend": metrics.get("backend"),
            "model_family": metrics.get("model_family"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "seq_length": metrics.get("seq_length"),
            "masking_curriculum": metrics.get("masking_curriculum"),
            "mask_prob_start": metrics.get("mask_prob_start"),
            "mask_prob_end": metrics.get("mask_prob_end"),
            "seed": metrics.get("seed"),
        },
        "endpoint_checkpoint_record": endpoint_metrics,
        "reference_checkpoint_record": ref_metrics,
        "endpoint_epochs_against_10M_pool": (float(endpoint_metrics.get("actual_cumulative_word_exposure")) / 10_000_000.0 if endpoint_metrics and endpoint_metrics.get("actual_cumulative_word_exposure") is not None else None),
        "reference_epochs_against_10M_pool": (float(ref_metrics.get("actual_cumulative_word_exposure")) / 10_000_000.0 if ref_metrics and ref_metrics.get("actual_cumulative_word_exposure") is not None else None),
        "within_10_epoch_limit": (endpoint_metrics is not None and float(endpoint_metrics.get("actual_cumulative_word_exposure", 1e30)) <= 100_000_000),
        "endpoint_files": endpoint_files,
        "reference_files": ref_files,
        "static_files_same_as_reference_except_model": same_static_files,
        "all_static_files_same_as_reference_except_model": all(same_static_files.values()) if same_static_files else False,
        "config_core": {
            "architectures": cfg.get("architectures"),
            "auto_map": cfg.get("auto_map"),
            "adapter_enabled": cfg.get("adapter_enabled"),
            "adapter_scale": cfg.get("adapter_scale"),
            "adapter_bottleneck": cfg.get("adapter_bottleneck"),
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "vocab_size": cfg.get("vocab_size"),
        },
        "cpu_loadability": loadability(endpoint_dir) if args.cpu_load else {"attempted": False},
        "scientific_reading": "The audited checkpoint is legally inside the same 100M/10-epoch trajectory and shares tokenizer/code/config coordinate with the reference if all static file hashes match. This is provenance/loadability evidence, not a score result.",
    }
    out_json = args.out_dir / f"{args.endpoint}_identity_audit.json"
    out_md = args.out_dir / f"{args.endpoint}_identity_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# research checkpoint identity audit: {args.endpoint}\n\n"]
    lines.append(f"Run: `{rel(args.run_dir)}`\n\n")
    lines.append(f"Endpoint actual words: `{result['endpoint_checkpoint_record'].get('actual_cumulative_word_exposure') if result['endpoint_checkpoint_record'] else None}`; epochs against 10M pool: `{result['endpoint_epochs_against_10M_pool']}`.\n\n")
    lines.append(f"Model SHA: `{result['endpoint_files']['model.safetensors']['sha256']}`; size `{result['endpoint_files']['model.safetensors']['size_bytes']}` bytes.\n\n")
    lines.append(f"Static files same as {args.reference_endpoint} except model: `{result['all_static_files_same_as_reference_except_model']}`.\n\n")
    if result["cpu_loadability"].get("attempted"):
        lines.append(f"CPU trusted-code loadability: `{result['cpu_loadability'].get('ok')}`; class `{result['cpu_loadability'].get('model_class')}`; params `{result['cpu_loadability'].get('param_count')}`.\n\n")
    lines.append(f"JSON: `{rel(out_json)}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "endpoint": args.endpoint, "within_10_epoch_limit": result["within_10_epoch_limit"], "model_sha256": result["endpoint_files"]["model.safetensors"]["sha256"], "static_same_as_reference": result["all_static_files_same_as_reference_except_model"], "cpu_load_ok": result["cpu_loadability"].get("ok"), "out_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
