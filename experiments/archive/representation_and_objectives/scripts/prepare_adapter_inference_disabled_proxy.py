#!/usr/bin/env python3
"""Prepare an inference-disabled proxy checkpoint for the live adapter128 20M model.

The proxy does not change weights.  It copies only config.json with
adapter_enabled=false and symlinks the remaining checkpoint files, so the
validated official-compatible evaluator can load the same trained live-adapter
checkpoint with the adapter branch disabled at inference.  This separates direct
adapter output from stock-backbone drift in the matched-horizon result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create adapter-disabled inference proxy run")
    p.add_argument("--source_ckpt", required=True)
    p.add_argument("--dest_run", required=True)
    p.add_argument("--target_name", default="adapter128_live20M_inference_disabled_proxy")
    p.add_argument("--word_exposure", type=int, default=20_008_711)
    return p.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source = Path(args.source_ckpt)
    dest_run = Path(args.dest_run)
    dest_ckpt = dest_run / "hf_model" / "chck_20M"
    if not source.exists():
        raise FileNotFoundError(source)
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_modeling.py"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError({"source": str(source), "missing": missing})
    dest_ckpt.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if cfg.get("architectures") != ["AdapterDebertaV2ForMaskedLM"]:
        raise RuntimeError({"unexpected_architectures": cfg.get("architectures")})
    cfg["adapter_enabled"] = False
    cfg["inference_ablation"] = {
        "source_checkpoint": str(source),
        "changed_only_config_field": "adapter_enabled true->false",
        "purpose": "disable trained live adapter branch at inference to separate direct adapter output from stock-backbone drift",
    }
    (dest_ckpt / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in required:
        if name == "config.json":
            continue
        link_or_copy(source / name, dest_ckpt / name)

    metrics = {
        "variant": args.target_name,
        "backend": "mlm",
        "model_family": "AdapterDebertaV2ForMaskedLM",
        "word_exposure": args.word_exposure,
        "source_live_adapter_checkpoint": str(source),
        "inference_only_change": "adapter_enabled=false in config; weights symlinked unchanged",
        "saved_checkpoints": [
            {"name": "chck_20M", "path": str(dest_ckpt), "actual_cumulative_word_exposure": args.word_exposure}
        ],
    }
    (dest_run / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "status": "ADAPTER_INFERENCE_DISABLED_PROXY_READY",
        "target_name": args.target_name,
        "source_ckpt": str(source),
        "dest_run": str(dest_run),
        "dest_ckpt": str(dest_ckpt),
        "config_adapter_enabled": cfg.get("adapter_enabled"),
        "symlinked_files": [name for name in required if name != "config.json"],
        "scientific_question": "If cheap7/EWoK/GlobalPIQA recover when the live adapter branch is disabled at inference, A02 research damage came from direct adapter output; if not, it came mainly from the altered stock-backbone trajectory.",
    }
    out = dest_run / "adapter_inference_disabled_proxy_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "proxy_ready", "manifest": str(out), "dest_ckpt": str(dest_ckpt)}), flush=True)


if __name__ == "__main__":
    main()
