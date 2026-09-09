#!/usr/bin/env python3
"""Compare research clean-tail ladder and mask-uniform control after training.

The two implementations should use the same clean parent, same seed, same WWM
sampling and corruption on the same clean pool. This script measures whether they
are exact duplicates or merely close stochastic repeats, and also summarizes the
mask-arm metrics needed to interpret target-selection effects.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUNS = _public_path('experiments/archive/compact_experience/training/runs')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/run_equivalence/tail_vs_uniform_and_mask_metrics.json')
TAIL = _public_path('experiments/archive/compact_experience/training/runs/clean_tail_restart_ladder_seed43044')
UNIFORM = _public_path('experiments/archive/compact_experience/training/runs/mask_uniform_control')
MASK_ARMS = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]


def sha256(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tail_log_rows(run: pathlib.Path) -> list[dict[str, Any]]:
    p = run / "training_log.jsonl"
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tensor_diff(a_path: pathlib.Path, b_path: pathlib.Path) -> dict[str, Any]:
    a_file = a_path / "model.safetensors"
    b_file = b_path / "model.safetensors"
    if not a_file.exists() or not b_file.exists():
        return {"available": False, "reason": "missing model.safetensors"}
    ah = sha256(a_file)
    bh = sha256(b_file)
    if ah == bh:
        return {"available": True, "exact_file_hash_equal": True, "sha256_a": ah, "sha256_b": bh}
    a = safe_load_file(str(a_file), device="cpu")
    b = safe_load_file(str(b_file), device="cpu")
    keys = sorted(set(a.keys()) & set(b.keys()))
    if set(a.keys()) != set(b.keys()):
        return {"available": True, "exact_file_hash_equal": False, "key_mismatch": True, "only_a": sorted(set(a)-set(b))[:20], "only_b": sorted(set(b)-set(a))[:20]}
    max_abs = 0.0
    l2_num = 0.0
    l2_den = 0.0
    nonzero_tensors = 0
    for k in keys:
        ta = a[k].float()
        tb = b[k].float()
        d = ta - tb
        ma = float(d.abs().max().item()) if d.numel() else 0.0
        max_abs = max(max_abs, ma)
        l2_num += float((d * d).sum().item())
        l2_den += float((ta * ta).sum().item())
        if ma != 0.0:
            nonzero_tensors += 1
    return {
        "available": True,
        "exact_file_hash_equal": False,
        "sha256_a": ah,
        "sha256_b": bh,
        "num_tensors": len(keys),
        "nonzero_diff_tensors": nonzero_tensors,
        "max_abs_diff": max_abs,
        "relative_l2_diff": math.sqrt(l2_num / max(l2_den, 1e-30)),
    }


def metric_view(run: pathlib.Path) -> dict[str, Any] | None:
    m = read_json(run / "scientific_metrics.json")
    if m is None:
        return None
    return {
        "path": str(run / "scientific_metrics.json"),
        "variant": m.get("variant"),
        "mask_mode": m.get("mask_mode"),
        "init_checkpoint": m.get("init_checkpoint"),
        "train_file_sha256": m.get("train_file_sha256"),
        "start_word_exposure": m.get("start_word_exposure", m.get("parent_start_word_exposure")),
        "continuation_word_exposure": m.get("continuation_word_exposure", m.get("word_exposure")),
        "actual_total_word_exposure": m.get("actual_total_word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "loss_mean": m.get("loss_mean"),
        "continuation_lr": m.get("continuation_lr"),
        "masked_token_budget_ratio": m.get("masked_token_budget_ratio"),
        "high_priority_fraction_among_selected_words": m.get("high_priority_fraction_among_selected_words"),
        "saved_checkpoint_names": [x.get("name") for x in m.get("saved_checkpoints", [])],
        "new_tail_checkpoint_names": [x.get("name") for x in m.get("new_tail_checkpoints", [])],
        "aoa_ladder_ready": m.get("aoa_ladder_ready"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", default=str(TAIL))
    ap.add_argument("--uniform", default=str(UNIFORM))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--checkpoint", default="chck_100M")
    ap.add_argument("--skip_tensor_diff", action="store_true")
    args = ap.parse_args()
    tail_run = pathlib.Path(args.tail)
    uniform_run = pathlib.Path(args.uniform)
    payload: dict[str, Any] = {
        "status": "TAIL_UNIFORM_RUN_COMPARISON",
        "tail_run": str(tail_run),
        "uniform_run": str(uniform_run),
        "checkpoint": args.checkpoint,
        "tail_metrics": metric_view(tail_run),
        "uniform_metrics": metric_view(uniform_run),
        "mask_arm_metrics": {},
    }
    for arm in MASK_ARMS:
        payload["mask_arm_metrics"][arm] = metric_view(RUNS / f"mask_{arm}")
    tail_logs = tail_log_rows(tail_run)
    uniform_logs = tail_log_rows(uniform_run)
    payload["log_comparison"] = {
        "tail_rows": len(tail_logs),
        "uniform_rows": len(uniform_logs),
        "tail_first_log": tail_logs[0] if tail_logs else None,
        "uniform_first_log": uniform_logs[0] if uniform_logs else None,
        "tail_last_log": tail_logs[-1] if tail_logs else None,
        "uniform_last_log": uniform_logs[-1] if uniform_logs else None,
        "loss_last_diff_tail_minus_uniform": (tail_logs[-1].get("loss") - uniform_logs[-1].get("loss")) if tail_logs and uniform_logs and tail_logs[-1].get("loss") is not None and uniform_logs[-1].get("loss") is not None else None,
    }
    if not args.skip_tensor_diff:
        payload["checkpoint_tensor_difference"] = tensor_diff(tail_run / "hf_model" / args.checkpoint, uniform_run / "hf_model" / args.checkpoint)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "tail_metrics_present": payload["tail_metrics"] is not None, "uniform_metrics_present": payload["uniform_metrics"] is not None, "tensor_diff": payload.get("checkpoint_tensor_difference")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
