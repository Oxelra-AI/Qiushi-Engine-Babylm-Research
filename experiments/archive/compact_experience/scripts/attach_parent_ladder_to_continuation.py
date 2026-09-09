#!/usr/bin/env python3
"""Attach the clean-Qwen parent AoA ladder to a research continuation run.

Continuation runs trained from clean-Qwen chck_80M save only their new tail
checkpoints. For official-style AoA evaluation, the model root must also expose
chck_1M..chck_9M and chck_10M..chck_80M from the true parent. This script links or
copies those parent checkpoint directories into the continuation's hf_model root
without changing the newly trained tail checkpoints.
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
import shutil
import time

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_PARENT = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model')
REQUIRED_PARENT = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 9)]
REQUIRED_FULL = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def link_or_copy(src: pathlib.Path, dst: pathlib.Path, copy_parent: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "already_correct"
        return "preexisting_not_changed"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_parent:
        shutil.copytree(src, dst, symlinks=True)
        return "copied"
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()), target_is_directory=True)
    return "symlinked"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--parent_model_root", default=str(DEFAULT_PARENT))
    ap.add_argument("--copy_parent_ladder", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    model_root = run_dir / "hf_model"
    parent = pathlib.Path(args.parent_model_root)
    if not model_root.exists():
        raise FileNotFoundError(model_root)
    if not parent.exists():
        raise FileNotFoundError(parent)
    records = []
    for name in REQUIRED_PARENT:
        src = parent / name
        dst = model_root / name
        if not src.exists():
            raise FileNotFoundError(src)
        records.append({"name": name, "source": str(src), "path": str(dst), "action": link_or_copy(src, dst, args.copy_parent_ladder)})
    available = sorted([p.name for p in model_root.iterdir() if p.is_dir() or p.is_symlink()])
    missing_full = [name for name in REQUIRED_FULL if name not in set(available)]
    metrics_path = run_dir / "scientific_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["attached_parent_ladder"] = {
            "created_utc": now(),
            "parent_model_root": str(parent),
            "records": records,
            "required_full_aoa_steps": REQUIRED_FULL,
            "missing_full_aoa_steps_after_attach": missing_full,
            "aoa_ladder_ready_by_names": not missing_full,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "status": "PARENT_LADDER_ATTACHED",
        "created_utc": now(),
        "run_dir": str(run_dir),
        "model_root": str(model_root),
        "parent_model_root": str(parent),
        "records": records,
        "available_checkpoints": available,
        "missing_full_aoa_steps_after_attach": missing_full,
        "aoa_ladder_ready_by_names": not missing_full,
    }
    out = pathlib.Path(args.out) if args.out else (run_dir / "parent_ladder_attachment.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "ready": not missing_full, "missing": missing_full}, indent=2), flush=True)
    if missing_full:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
