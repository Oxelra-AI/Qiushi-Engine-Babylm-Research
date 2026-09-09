#!/usr/bin/env python3
"""Create inference-scale shadow run dirs for any trained research-style adapter checkpoint.

Only checkpoint-local config.json and modeling source are edited/copied. The
model.safetensors and tokenizer files are hard-linked/copied unchanged. This is a
post-hoc endpoint sensitivity diagnostic for trained adapter checkpoints, not a
submission-time scale selector.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

USER_ROOT = _public_path('.')
DEFAULT_MODELING = _public_path('experiments/archive/frontier_consolidation/scripts/adapter_scaled_modeling.py')


def tag(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p").replace("-", "m")


def link_or_copy(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rel_to_user(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(path)


def make_shadow(src_run: Path, endpoint: str, out_root: Path, scale: float, prefix: str, force: bool, modeling_src: Path) -> Dict[str, Any]:
    src_ckpt = src_run / "hf_model" / endpoint
    model_file = src_ckpt / "model.safetensors"
    cfg_file = src_ckpt / "config.json"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    if not cfg_file.exists():
        raise FileNotFoundError(cfg_file)
    run_dir = out_root / f"{prefix}_scale_{tag(scale)}_from_{endpoint}"
    ckpt = run_dir / "hf_model" / endpoint
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = src_ckpt / name
        dst = ckpt / name
        if src.exists():
            if name == "model.safetensors":
                link_or_copy(src, dst, force)
            else:
                shutil.copy2(src, dst)
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg["adapter_scale"] = float(scale)
    cfg["adapter_enabled"] = True
    cfg["architectures"] = ["AdapterDebertaV2ForMaskedLM"]
    cfg["auto_map"] = {"AutoModelForMaskedLM": "adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM"}
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(modeling_src, ckpt / "adapter_scaled_modeling.py")

    src_metrics = src_run / "scientific_metrics.json"
    metrics = json.loads(src_metrics.read_text(encoding="utf-8")) if src_metrics.exists() else {}
    metrics.update({
        "shadow_source_run": rel_to_user(src_run),
        "shadow_source_checkpoint": rel_to_user(src_ckpt),
        "adapter_scale": float(scale),
        "variant": f"inference_scaled_adapter_scale_{tag(scale)}_from_{endpoint}",
    })
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"scale": float(scale), "tag": tag(scale), "run_dir": rel_to_user(run_dir), "checkpoint": rel_to_user(ckpt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", required=True, help="Run dir containing hf_model/<endpoint>")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--scales", nargs="+", type=float, required=True)
    ap.add_argument("--modeling-src", default=str(DEFAULT_MODELING))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    src_run = USER_ROOT / args.src_run if not Path(args.src_run).is_absolute() else Path(args.src_run)
    out_root = USER_ROOT / args.out_root if not Path(args.out_root).is_absolute() else Path(args.out_root)
    modeling_src = USER_ROOT / args.modeling_src if not Path(args.modeling_src).is_absolute() else Path(args.modeling_src)
    out_root.mkdir(parents=True, exist_ok=True)
    records = [make_shadow(src_run, args.endpoint, out_root, float(s), args.prefix, args.force, modeling_src) for s in args.scales]
    out_json = out_root / f"{args.prefix}_{args.endpoint}_scale_shadows.json"
    out_json.write_text(json.dumps({
        "status": "ADAPTER_SCALE_SHADOWS_GENERIC",
        "source_run": rel_to_user(src_run),
        "endpoint": args.endpoint,
        "records": records,
        "note": "Only adapter_scale in config/modeling source changes; weights/tokenizer are unchanged. Use as a diagnostic, not as benchmark-selected submission scale.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ADAPTER_SCALE_SHADOWS_GENERIC", "out_json": rel_to_user(out_json), "n": len(records), "records": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
