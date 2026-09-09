#!/usr/bin/env python3
"""Materialize evaluation-only private-scale variants for shrinkage-null tests.

The dense/preservation checkpoints store the private adapter weights and execute the private
path with `config.private_adapter_scale`. To test whether preservation is only a smaller
effective private update, this script creates lightweight HF directories that symlink the
large binary/tokenizer files from a repaired dense checkpoint and write a modified
`config.json` with the requested private scale. No training is performed.
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
import shutil
import time
from typing import Any

ROOT = _public_path('.')
DEFAULT_SOURCE = _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/shrinkage_scale_variants')
DEFAULT_SCALES = [0.55, 0.57, 0.58, 0.60]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path, block: int = 1 << 20) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> dict[str, Any]:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
        mode = "symlink"
    except Exception:
        shutil.copy2(src, dst)
        mode = "copy"
    return {"source": rel(src), "target": rel(dst), "mode": mode, "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size}


def materialize(source: pathlib.Path, out_root: pathlib.Path, scale: float) -> dict[str, Any]:
    label = f"dense64_private_scale_{scale:.2f}".replace(".", "p")
    dst = out_root / label
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    old_scale = cfg.get("private_adapter_scale")
    cfg["private_adapter_scale"] = float(scale)
    cfg["private_adapter_enabled"] = True
    cfg.setdefault("auto_map", {
        "AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM",
        "AutoModel": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model",
    })
    if "AutoModel" not in cfg["auto_map"]:
        cfg["auto_map"]["AutoModel"] = "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files: dict[str, Any] = {"config.json": {"sha256": sha256_file(dst / "config.json"), "old_scale": old_scale, "new_scale": float(scale)}}
    for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "frozen82_private_modeling.py"]:
        sp = source / name
        if sp.exists():
            files[name] = link_or_copy(sp, dst / name)
    # Copy small ancillary files when present; they are useful provenance and harmless.
    for name in ["bridge_metadata.json", "generation_config.json", "vocab.json", "merges.txt", "added_tokens.json"]:
        sp = source / name
        if sp.exists() and name not in files:
            files[name] = link_or_copy(sp, dst / name)
    return {"label": label, "path": rel(dst), "source": rel(source), "scale": float(scale), "files": files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--scales", default=",".join(str(x) for x in DEFAULT_SCALES))
    args = ap.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    records = [materialize(source, out_root, s) for s in scales]
    endpoints = {r["label"]: r["path"] for r in records}
    (out_root / "dense64_scale_variant_paths_for_kl.json").write_text(json.dumps(endpoints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_root / "dense64_scale_variant_manifest.json").write_text(json.dumps({
        "status": "DENSE64_SCALE_VARIANTS_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Evaluation-only shrinkage-null variants; no training, only config private_adapter_scale changes.",
        "source": rel(source),
        "variants": records,
        "endpoints_json": rel(out_root / "dense64_scale_variant_paths_for_kl.json"),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DENSE64_SCALE_VARIANTS_DONE", "out_root": rel(out_root), "endpoints_json": rel(out_root / "dense64_scale_variant_paths_for_kl.json"), "n": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
