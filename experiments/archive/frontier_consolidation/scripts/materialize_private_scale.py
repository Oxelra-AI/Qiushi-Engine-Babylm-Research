#!/usr/bin/env python3
"""Materialize inference-time private-adapter scale variants for frozen-82M tails.

The frozen-tail model keeps the verified scale1.75 chck_82M function as the slow
path and stores a fresh private residual under `.private_adapter.*`.  The private
branch scale is a config-time scalar, so changing `private_adapter_scale` creates a
reversible endpoint without training.  This utility hard-links large weight files when
possible, updates config.json, and can optionally CPU-check that scale=0 matches the
original chck_82M function on a deterministic small input.

No evaluation or training is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any

ROOT = pathlib.Path(".").resolve()
DEFAULT_REFERENCE = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def materialize(source: pathlib.Path, output: pathlib.Path, scale: float, enabled: bool, force: bool) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not (source / "config.json").exists():
        raise FileNotFoundError(source / "config.json")
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Output already exists and is nonempty: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for src in sorted(source.iterdir()):
        dst = output / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied[src.name] = "copytree"
        elif src.name == "config.json":
            continue
        elif src.suffix in {".safetensors", ".bin"}:
            copied[src.name] = link_or_copy(src, dst)
        else:
            shutil.copy2(src, dst)
            copied[src.name] = "copy"
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    old_scale = cfg.get("private_adapter_scale")
    old_enabled = cfg.get("private_adapter_enabled")
    cfg["private_adapter_scale"] = float(scale)
    cfg["private_adapter_enabled"] = bool(enabled)
    cfg.setdefault("architectures", ["FrozenSlowPrivateDebertaV2ForMaskedLM"])
    cfg.setdefault("auto_map", {"AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"})
    (output / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "PRIVATE_SCALE_MATERIALIZED",
        "created_utc": now(),
        "source": rel(source),
        "output": rel(output),
        "new_private_adapter_scale": float(scale),
        "new_private_adapter_enabled": bool(enabled),
        "old_private_adapter_scale": old_scale,
        "old_private_adapter_enabled": old_enabled,
        "copied": copied,
        "files": {},
    }
    for p in sorted(output.iterdir()):
        if p.is_file():
            manifest["files"][p.name] = {"size_bytes": p.stat().st_size, "sha256": sha256(p) if p.name in {"config.json", "model.safetensors", "pytorch_model.bin"} else None}
    (output / "private_scale_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def cpu_equivalence(reference: pathlib.Path, scaled: pathlib.Path) -> dict[str, Any]:
    # Import only when requested so ordinary materialization is light.
    cache = scaled / "hf_cache_cpu_check"
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(reference), local_files_only=True, use_fast=True)
    text = "The child put the cup on the table and smiled."
    batch = tok(text, return_tensors="pt")
    m_ref = AutoModelForMaskedLM.from_pretrained(str(reference), trust_remote_code=True, local_files_only=True)
    m_scaled = AutoModelForMaskedLM.from_pretrained(str(scaled), trust_remote_code=True, local_files_only=True)
    m_ref.eval(); m_scaled.eval()
    with torch.no_grad():
        a = m_ref(**batch).logits
        b = m_scaled(**batch).logits
    diff = (a - b).abs()
    return {
        "reference": rel(reference),
        "scaled": rel(scaled),
        "input_text": text,
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "ref_class": m_ref.__class__.__name__,
        "scaled_class": m_scaled.__class__.__name__,
        "scaled_config_private_adapter_scale": getattr(m_scaled.config, "private_adapter_scale", None),
        "scaled_config_private_adapter_enabled": getattr(m_scaled.config, "private_adapter_enabled", None),
        "allclose_1e_7": bool(torch.allclose(a, b, atol=1e-7, rtol=0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--enabled", action="store_true", default=True)
    ap.add_argument("--disabled", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu-equivalence-reference", default="")
    args = ap.parse_args()
    enabled = bool(args.enabled) and not bool(args.disabled)
    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output)
    manifest = materialize(source, output, args.scale, enabled, args.force)
    if args.cpu_equivalence_reference:
        eq = cpu_equivalence(pathlib.Path(args.cpu_equivalence_reference), output)
        manifest["cpu_equivalence"] = eq
        (output / "private_scale_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
