#!/usr/bin/env python3
"""Prefill one research mask full-eval payload from the research mask no-AoA trajectory."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402
import mask_full_eval as mask_wrap  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
NOAOA_ROOT = _public_path('experiments/archive/compact_experience/data/mask_eval')
VALID = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=VALID, required=True)
    ap.add_argument("--endpoint", default="chck_100M")
    args = ap.parse_args()
    target_noaoa = f"mask_{args.arm}"
    noaoa_path = NOAOA_ROOT / f"{target_noaoa}_trajectory_summary.json"
    if not noaoa_path.exists():
        nested = NOAOA_ROOT / target_noaoa / f"{target_noaoa}_trajectory_summary.json"
        if not nested.exists():
            raise FileNotFoundError(noaoa_path)
        noaoa_path = nested
    obj = json.loads(noaoa_path.read_text(encoding="utf-8"))
    row = (obj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No {args.endpoint} row in {noaoa_path}")

    base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/mask_full_eval')
    base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
    base.TARGETS = mask_wrap.make_targets(args.arm)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    target = f"mask_{args.arm}_100M"
    spec = base.TARGETS[target]
    model_root = pathlib.Path(spec["run_dir"]) / "hf_model"
    model_path = model_root / spec["endpoint"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = base.per_target_path(target)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": target,
            "description": spec.get("description", ""),
            "family": spec.get("family", ""),
            "run_dir": str(spec["run_dir"]),
            "model_root": str(model_root),
            "model_path": str(model_path),
            "endpoint": spec["endpoint"],
            "started_utc": base.now_utc(),
            "gpu": None,
            "tasks": {},
        }
    payload.setdefault("tasks", {})
    payload["prefill_source"] = {
        "trajectory_summary": str(noaoa_path),
        "target": target_noaoa,
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from local post-training no-AoA evaluation, not training or selection signals.",
    }
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in no-AoA row")
        payload["tasks"][col] = {"column": col, "score": float(row[col]), "returncode": 0, "status": "prefilled_from_step045_mask_noaoa_trajectory", "source_trajectory_summary": str(noaoa_path), "checkpoint": args.endpoint}
    payload["tasks"]["Reading"] = {"column": "Reading", "scores": {"Reading_eye": row.get("Reading_eye"), "Reading_self_paced": row.get("Reading_self_paced"), "Reading": row.get("Reading")}, "returncode": 0, "status": "prefilled_from_step045_mask_noaoa_trajectory", "source_trajectory_summary": str(noaoa_path), "checkpoint": args.endpoint}
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload": str(p), "arm": args.arm, "endpoint": args.endpoint, "equal7": row.get("equal7_full_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
