#!/usr/bin/env python3
"""Build an endpoint-consistent evaluation ladder for any mask continuation run.

Used for second-seed replication runs whose names differ from the research/046
hardcoded seed43022 paths. It links parent checkpoints through chck_80M, then
links trained continuation checkpoints up to the selected endpoint, and for
<100M endpoints plateaus later standard AoA names at the endpoint weights.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import time
from typing import Any

REQUIRED = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
PARENT_LIMIT_M = 80


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_m(name: str) -> int:
    m = re.fullmatch(r"chck_(\d+)M", name)
    if not m:
        raise ValueError(f"Bad checkpoint label: {name}")
    return int(m.group(1))


def copy_or_link(src: pathlib.Path, dst: pathlib.Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.exists() and dst.resolve() == src:
            return "already_correct"
        raise RuntimeError(f"Refusing to reuse existing checkpoint path with different source: {dst} -> {dst.resolve() if dst.exists() else 'broken'}; expected {src}")
    if copy:
        shutil.copytree(src, dst, symlinks=True)
        return "copied"
    dst.symlink_to(os.path.relpath(src, dst.parent.resolve()), target_is_directory=True)
    if not dst.exists() or dst.resolve() != src:
        raise RuntimeError(f"symlink validation failed for {dst}: expected {src}, got {dst.resolve() if dst.exists() else 'missing'}")
    return "symlinked"


def choose_source(req_name: str, endpoint: str, source_model_root: pathlib.Path, parent_root: pathlib.Path) -> tuple[pathlib.Path, str]:
    req_m = parse_m(req_name)
    end_m = parse_m(endpoint)
    endpoint_path = source_model_root / endpoint
    if req_m <= PARENT_LIMIT_M:
        return parent_root / req_name, "parent_before_continuation"
    if req_m <= end_m and (source_model_root / req_name).exists():
        return source_model_root / req_name, "continuation_before_or_at_endpoint"
    return endpoint_path, "endpoint_plateau_no_future_exposure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run", required=True, help="Continuation run dir containing hf_model checkpoints and scientific_metrics.json.")
    ap.add_argument("--parent_model_root", required=True, help="Clean parent hf_model root with chck_1M..chck_80M.")
    ap.add_argument("--endpoint", required=True, help="Selected endpoint checkpoint, e.g. chck_95M or chck_100M.")
    ap.add_argument("--out_run_dir", required=True, help="New evaluation root to create/reuse.")
    ap.add_argument("--label", default="custom_mask_endpoint_ladder")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    source_run = pathlib.Path(args.source_run)
    source_model_root = source_run / "hf_model"
    parent_root = pathlib.Path(args.parent_model_root)
    out_run = pathlib.Path(args.out_run_dir)
    model_root = out_run / "hf_model"
    endpoint_path = source_model_root / args.endpoint
    if not source_model_root.exists():
        raise FileNotFoundError(source_model_root)
    if not parent_root.exists():
        raise FileNotFoundError(parent_root)
    if not endpoint_path.exists():
        raise FileNotFoundError(endpoint_path)
    end_m = parse_m(args.endpoint)

    records: list[dict[str, Any]] = []
    for name in REQUIRED:
        src, role = choose_source(name, args.endpoint, source_model_root, parent_root)
        if not src.exists():
            raise FileNotFoundError(src)
        records.append({"name": name, "source": str(src), "role": role, "path": str(model_root / name), "action": copy_or_link(src, model_root / name, args.copy)})
    if args.endpoint not in REQUIRED:
        records.append({"name": args.endpoint, "source": str(endpoint_path), "role": "selected_nonstandard_endpoint_for_main_model", "path": str(model_root / args.endpoint), "action": copy_or_link(endpoint_path, model_root / args.endpoint, args.copy)})

    metrics_path = source_run / "scientific_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    available = sorted(p.name for p in model_root.iterdir() if (p.is_dir() or p.is_symlink()) and p.exists())
    missing = [n for n in REQUIRED if n not in set(available)]
    payload = {
        "status": "CUSTOM_MASK_ENDPOINT_LADDER_READY",
        "created_utc": now(),
        "label": args.label,
        "source_run": str(source_run),
        "source_model_root": str(source_model_root),
        "parent_model_root": str(parent_root),
        "endpoint": args.endpoint,
        "endpoint_million_words_label": end_m,
        "out_run_dir": str(out_run),
        "model_root": str(model_root),
        "required_strict_small_steps": REQUIRED,
        "missing_required_steps": missing,
        "available_checkpoints": available,
        "lineage_policy": "Parent checkpoints are used through chck_80M; continuation checkpoints up to the endpoint are used when available; later strict-small AoA names point to the endpoint weights for <100M frozen endpoint measurement so no future exposure enters the endpoint curve.",
        "source_training_summary": {k: metrics.get(k) for k in ["variant", "mask_mode", "mask_budget", "parent_start_word_exposure", "continuation_word_exposure", "actual_total_word_exposure", "actual_training_steps", "train_rng_seed", "continuation_lr", "masked_token_budget_ratio", "high_priority_fraction_among_selected_words", "loss_last", "train_file_sha256"]},
        "records": records,
        "ready_by_required_names": not missing,
        "non_leakage_statement": "This evaluation root only rearranges already-trained legal parent/continuation checkpoints for post-training measurement; it does not use downstream labels/items, AoA/CDI words, child curves, or leaderboard feedback.",
    }
    out_run.mkdir(parents=True, exist_ok=True)
    (out_run / "endpoint_ladder_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_run / "scientific_metrics.json").write_text(json.dumps({
        "variant": f"{args.label}_evalroot",
        "source_run": str(source_run),
        "endpoint": args.endpoint,
        "endpoint_million_words_label": end_m,
        "selected_endpoint_path": str(model_root / args.endpoint),
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
