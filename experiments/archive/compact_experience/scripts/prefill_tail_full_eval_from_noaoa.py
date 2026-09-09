#!/usr/bin/env python3
"""Prefill research tail full-eval payload from tail no-AoA trajectory results."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402
import tail_restart_full_eval as tail_wrap  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
NOAOA = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/clean_tail_restart_ladder_trajectory_summary.json')
TARGET = "clean_tail_restart_ladder_100M"
ENDPOINT = "chck_100M"

# Align base paths with the tail wrapper.
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_restart_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
base.TARGETS = tail_wrap.TARGETS
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def main() -> None:
    if not NOAOA.exists():
        raise FileNotFoundError(NOAOA)
    obj = json.loads(NOAOA.read_text(encoding="utf-8"))
    row = (obj.get("table") or {}).get(ENDPOINT)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No {ENDPOINT} no-AoA row in {NOAOA}")
    spec = tail_wrap.TARGETS[TARGET]
    model_root = pathlib.Path(spec["run_dir"]) / "hf_model"
    model_path = model_root / spec["endpoint"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = base.per_target_path(TARGET)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": TARGET,
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
        "trajectory_summary": str(NOAOA),
        "target": "clean_tail_restart_ladder",
        "endpoint": ENDPOINT,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from local post-training no-AoA evaluation, not training or selection signals.",
    }
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in no-AoA row")
        payload["tasks"][col] = {
            "column": col,
            "score": float(row[col]),
            "returncode": 0,
            "status": "prefilled_from_step045_tail_noaoa_trajectory",
            "source_trajectory_summary": str(NOAOA),
            "checkpoint": ENDPOINT,
        }
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "scores": {
            "Reading_eye": row.get("Reading_eye"),
            "Reading_self_paced": row.get("Reading_self_paced"),
            "Reading": row.get("Reading"),
        },
        "returncode": 0,
        "status": "prefilled_from_step045_tail_noaoa_trajectory",
        "source_trajectory_summary": str(NOAOA),
        "checkpoint": ENDPOINT,
    }
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload": str(p), "prefilled_endpoint": ENDPOINT, "equal7": row.get("equal7_full_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
