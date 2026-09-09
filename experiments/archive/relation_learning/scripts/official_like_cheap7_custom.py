#!/usr/bin/env python3
"""research official-like cheap7 scorer for an arbitrary repaired checkpoint.

The implementation reuses the repaired local cheap7/Reading scorer from research but
allows a caller-supplied checkpoint path.  The intended use here is the evaluation-
only dense64 private-scale0.60 matched-KL shrinkage control.  `with_special` is the
official-like input form; `no_special` remains available only for local mechanism
comparisons.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import torch

ROOT = _public_path('.')
PATH = _public_path('experiments/archive/relation_learning/scripts/no_boundary_cheap7_diagnostic.py')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/shrinkage_dense64_scale0p60_cheap7')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def import_step105(out_dir: pathlib.Path):
    # research sets writable HF caches from sys.argv before importing Transformers.
    old_argv = list(sys.argv)
    sys.argv = [str(PATH), "--out-dir", str(out_dir)]
    try:
        spec = importlib.util.spec_from_file_location("no_boundary_cheap7_for_step121", PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["no_boundary_cheap7_for_step121"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--input-forms", nargs="*", default=["with_special"], choices=["with_special", "no_special"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--non-causal-batch-size", type=int, default=128)
    ap.add_argument("--max-items-per-file", type=int, default=0)
    ap.add_argument("--max-reading-rows", type=int, default=0)
    ap.add_argument("--skip-reading", action="store_true")
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config not found: {ckpt / 'config.json'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = import_step105(out_dir)
    mod.ENDPOINTS = {
        args.label: {
            "model_path": ckpt,
            "known_official_scores": {},
        }
    }
    ns = argparse.Namespace(
        out_dir=str(out_dir),
        endpoints=[args.label],
        input_forms=list(args.input_forms),
        batch_size=int(args.batch_size),
        non_causal_batch_size=int(args.non_causal_batch_size),
        max_items_per_file=int(args.max_items_per_file),
        max_reading_rows=int(args.max_reading_rows),
        skip_reading=bool(args.skip_reading),
        no_predictions=bool(args.no_predictions),
        cpu=bool(args.cpu),
    )
    mod.setup_cache(out_dir)
    torch.manual_seed(0)
    result = mod.evaluate_endpoint(args.label, list(args.input_forms), ns)
    summary = mod.build_overall_summary({args.label: result}, out_dir)
    payload: dict[str, Any] = {
        "status": "OFFICIAL_LIKE_CHEAP7_CUSTOM_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(_public_path('experiments/archive/relation_learning/scripts/official_like_cheap7_custom.py')),
        "label": args.label,
        "checkpoint": rel(ckpt),
        "input_forms": list(args.input_forms),
        "summary_json": rel(out_dir / "summary.json"),
        "summary_md": rel(out_dir / "summary.md"),
        "score_rows_csv": rel(out_dir / "score_rows.csv"),
        "model_identity": result.get("model_identity"),
        "scores": {form: data.get("scores") for form, data in result.get("forms", {}).items()},
        "cheap7": {form: data.get("cheap7") for form, data in result.get("forms", {}).items()},
        "note": "with_special is the official-like local masked-LM/Reading surface used for the shrinkage practical control; final official platform packaging remains separate.",
    }
    (out_dir / "custom_cheap7_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "payload": payload, "summary_status": summary.get("status")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
