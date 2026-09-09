#!/usr/bin/env python3
"""research arbitrary-endpoint AoA extraction wrapper.

Imports the repaired batched AoA extractor, patches its ENDPOINTS table with one
repaired checkpoint, and runs only the endpoint step.  The shared 1M..80M
ancestry extraction already exists and is reused later by assembly/refit.  This
wrapper records a physical-GPU snapshot before extraction so we can verify that
CUDA_VISIBLE_DEVICES selects the intended H100.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = _public_path('.')
research = _public_path('experiments/archive/functional_learning/scripts/batched_aoa_extractor.py')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_endpoint_extract')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_step084():
    spec = importlib.util.spec_from_file_location("batched_aoa_extractor_for_step127", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["batched_aoa_extractor_for_step127"] = mod
    spec.loader.exec_module(mod)
    return mod


def snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
        out["gpus"] = q.stdout.strip().splitlines() if q.returncode == 0 else []
        if q.returncode != 0:
            out["gpu_query_stderr"] = q.stderr[-500:]
    except Exception as e:
        out["gpu_query_error"] = repr(e)
    try:
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["torch_visible_device_count"] = int(torch.cuda.device_count())
            out["torch_current_device"] = int(torch.cuda.current_device())
            out["torch_current_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception as e:
        out["torch_query_error"] = repr(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--words", type=int, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu", type=int, default=0, help="Visible CUDA index inside the current CUDA_VISIBLE_DEVICES setting.")
    ap.add_argument("--device", default="", help="Optional device string, e.g. cuda:0.")
    ap.add_argument("--word-start", type=int, default=0)
    ap.add_argument("--word-end", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(ckpt / "config.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = snapshot()
    (out_dir / "physical_gpu_snapshot_pre.json").write_text(json.dumps(pre, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "aoa_endpoint_wrapper_start", "label": args.label, "checkpoint": rel(ckpt), "words": args.words, "snapshot": pre, "utc": now()}), flush=True)

    mod = import_step084()
    mod.ENDPOINTS = {args.label: {"path": ckpt, "words": int(args.words)}}
    ns = argparse.Namespace(
        mode="endpoint",
        target=args.label,
        out_dir=out_dir,
        word_start=int(args.word_start),
        word_end=int(args.word_end),
        max_words=int(args.max_words),
        ancestral_limit=0,
        batch_size=int(args.batch_size),
        gpu=int(args.gpu),
        device=args.device,
        use_bos_only=False,
        progress_every=int(args.progress_every),
    )
    manifest = mod.run_extract(ns)
    post = snapshot()
    (out_dir / "physical_gpu_snapshot_post.json").write_text(json.dumps(post, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AOA_ARBITRARY_ENDPOINT_EXTRACT_DONE", "label": args.label, "out_dir": rel(out_dir), "manifest": rel(out_dir / "manifest.json"), "surprisal": rel(out_dir / "surprisal.json"), "status_inner": manifest.get("status"), "pre_snapshot": pre, "post_snapshot": post}, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
