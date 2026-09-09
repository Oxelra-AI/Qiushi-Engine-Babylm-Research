#!/usr/bin/env python3
"""Build a frozen-endpoint strict-small checkpoint ladder for local evaluation.

For a tail endpoint before 100M, the official strict-small AoA reader still looks
for standard names chck_1M..chck_9M and chck_10M..chck_100M.  This utility builds
a separate evaluation model root in which all names after the chosen endpoint
point to the endpoint weights, never to later-trained weights.  Nonstandard
endpoints such as chck_85M and chck_95M are also exposed under their own names so
SuperGLUE can be run on exactly the selected checkpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import re
import shutil
import time
from typing import Any

WORKSPACE = _public_path('experiments/archive/compact_experience')
SOURCE_RUN = _public_path('experiments/archive/compact_experience/training/runs/clean_tail_restart_ladder_seed43044')
PARENT = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model')
OUT_BASE = _public_path('experiments/archive/compact_experience/data/tail_endpoint_ladders')
REQUIRED = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10 * i}M" for i in range(1, 11)]
PARENT_LIMIT_M = 80


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_m(name: str) -> int:
    m = re.fullmatch(r"chck_(\d+)M", name)
    if not m:
        raise ValueError(f"not a chck_*M name: {name}")
    return int(m.group(1))


def rel_symlink(src: pathlib.Path, dst: pathlib.Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.exists() and dst.resolve() == src:
            return "already_correct"
        return "preexisting_not_changed"
    dst.symlink_to(os.path.relpath(src, dst.parent.resolve()), target_is_directory=True)
    return "symlinked"


def copy_or_link(src: pathlib.Path, dst: pathlib.Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.exists() and dst.resolve() == src:
            return "already_correct"
        raise RuntimeError(f"Refusing to reuse existing checkpoint path with different source: {dst} -> {dst.resolve() if dst.exists() else 'broken'}; expected {src}")
    if copy:
        shutil.copytree(src, dst, symlinks=True)
        if not dst.exists() or not dst.is_dir():
            raise RuntimeError(f"copy failed for {dst}")
        return "copied"
    action = rel_symlink(src, dst)
    if not dst.exists() or not dst.resolve().is_dir() or dst.resolve() != src:
        raise RuntimeError(f"symlink validation failed for {dst}; expected {src}, got {dst.resolve() if dst.exists() else 'missing'}")
    return action


def choose_source(required_name: str, endpoint: str, source_model_root: pathlib.Path, parent_root: pathlib.Path) -> tuple[pathlib.Path, str]:
    req_m = parse_m(required_name)
    end_m = parse_m(endpoint)
    endpoint_path = source_model_root / endpoint
    if req_m <= PARENT_LIMIT_M:
        return parent_root / required_name, "parent_before_restart"
    if req_m <= end_m and (source_model_root / required_name).exists():
        return source_model_root / required_name, "tail_before_or_at_endpoint"
    return endpoint_path, "endpoint_plateau_no_future_exposure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", default=str(SOURCE_RUN))
    ap.add_argument("--parent_model_root", default=str(PARENT))
    ap.add_argument("--endpoint", required=True, choices=["chck_85M", "chck_90M", "chck_95M", "chck_100M"])
    ap.add_argument("--out_run_dir", default="")
    ap.add_argument("--copy", action="store_true", help="Copy checkpoint directories instead of symlinking them.")
    args = ap.parse_args()

    source_run = pathlib.Path(args.source_run)
    source_model_root = source_run / "hf_model"
    parent_root = pathlib.Path(args.parent_model_root)
    endpoint_path = source_model_root / args.endpoint
    if not endpoint_path.exists():
        raise FileNotFoundError(endpoint_path)
    if not parent_root.exists():
        raise FileNotFoundError(parent_root)
    end_m = parse_m(args.endpoint)
    out_run = pathlib.Path(args.out_run_dir) if args.out_run_dir else (OUT_BASE / f"tail_restart_seed43044_endpoint_{end_m}M")
    model_root = out_run / "hf_model"
    records: list[dict[str, Any]] = []

    for name in REQUIRED:
        src, role = choose_source(name, args.endpoint, source_model_root, parent_root)
        if not src.exists():
            raise FileNotFoundError(src)
        action = copy_or_link(src, model_root / name, args.copy)
        records.append({"name": name, "source": str(src), "role": role, "path": str(model_root / name), "action": action})

    if args.endpoint not in REQUIRED:
        action = copy_or_link(endpoint_path, model_root / args.endpoint, args.copy)
        records.append({"name": args.endpoint, "source": str(endpoint_path), "role": "selected_nonstandard_endpoint_for_main_model", "path": str(model_root / args.endpoint), "action": action})

    source_metrics_path = source_run / "scientific_metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8")) if source_metrics_path.exists() else {}
    available = sorted(p.name for p in model_root.iterdir() if (p.is_dir() or p.is_symlink()) and p.exists())
    missing = [name for name in REQUIRED if name not in set(available)]
    payload = {
        "status": "ENDPOINT_CONSISTENT_LADDER_READY",
        "created_utc": now(),
        "source_run": str(source_run),
        "source_model_root": str(source_model_root),
        "parent_model_root": str(parent_root),
        "endpoint": args.endpoint,
        "endpoint_million_words_label": end_m,
        "out_run_dir": str(out_run),
        "model_root": str(model_root),
        "selected_endpoint_path": str(model_root / args.endpoint),
        "required_strict_small_steps": REQUIRED,
        "missing_required_steps": missing,
        "available_checkpoints": available,
        "lineage_policy": "Parent checkpoints are used through chck_80M; trained tail checkpoints up to the endpoint are used when their standard names exist; all standard names after the endpoint point to the endpoint weights so no later exposure contributes to an earlier endpoint's AoA curve.",
        "records": records,
        "source_training_summary": {k: source_metrics.get(k) for k in ["variant", "mechanism", "start_word_exposure", "continuation_word_exposure", "actual_total_word_exposure", "actual_training_steps", "train_rng_seed", "continuation_lr"]},
        "ready_by_required_names": not missing,
        "non_leakage_statement": "This evaluation root only rearranges already-trained clean-parent/tail checkpoints for frozen endpoint measurement; no downstream labels/items, AoA/CDI words, child curves, or leaderboard feedback are used.",
    }
    out_run.mkdir(parents=True, exist_ok=True)
    (out_run / "endpoint_ladder_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # A compact metrics file lets the common full-eval wrapper preserve ancestry in its per-target payload.
    (out_run / "scientific_metrics.json").write_text(json.dumps({
        "variant": f"tail_restart_seed43044_endpoint_{end_m}M_evalroot",
        "source_run": str(source_run),
        "endpoint": args.endpoint,
        "endpoint_million_words_label": end_m,
        "selected_endpoint_path": str(model_root / args.endpoint),
        "parent_model_root": str(parent_root),
        "lineage_policy": payload["lineage_policy"],
        "ready_by_required_names": not missing,
        "source_training_summary": payload["source_training_summary"],
        "created_utc": payload["created_utc"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_run_dir": str(out_run), "model_root": str(model_root), "endpoint": args.endpoint, "ready": not missing, "missing": missing}, indent=2), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
