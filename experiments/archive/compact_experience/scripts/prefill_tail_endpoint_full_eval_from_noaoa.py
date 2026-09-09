#!/usr/bin/env python3
"""Prefill endpoint-aware tail full-eval payload from the no-AoA trajectory."""
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
import tail_endpoint_full_eval as wrap  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
NOAOA = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/clean_tail_restart_ladder_trajectory_summary.json')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, choices=wrap.VALID_ENDPOINTS)
    args = ap.parse_args()
    if not NOAOA.exists():
        raise FileNotFoundError(NOAOA)
    obj = json.loads(NOAOA.read_text(encoding="utf-8"))
    row = (obj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No {args.endpoint} row in {NOAOA}")

    base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_endpoint_full_eval')
    base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
    base.TARGETS = wrap.make_targets([args.endpoint])
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    target = wrap.target_name(args.endpoint)
    spec = base.TARGETS[target]
    model_root = pathlib.Path(spec["run_dir"]) / "hf_model"
    model_path = model_root / spec["endpoint"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    manifest_path = pathlib.Path(spec["run_dir"]) / "endpoint_ladder_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
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
    payload["endpoint_ladder_manifest"] = str(manifest_path) if manifest_path.exists() else None
    payload["endpoint_lineage_policy"] = manifest.get("lineage_policy") if isinstance(manifest, dict) else None
    payload["prefill_source"] = {
        "trajectory_summary": str(NOAOA),
        "target": "clean_tail_restart_ladder",
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from post-training no-AoA evaluation of the same frozen endpoint. It is not a training signal, but these component columns are explicitly used for endpoint selection and must be reported as post-selection evidence.",
    }
    for col in COLUMNS:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in no-AoA row")
        payload["tasks"][col] = {
            "column": col,
            "score": float(row[col]),
            "returncode": 0,
            "status": "prefilled_from_step046_tail_endpoint_noaoa_trajectory",
            "source_trajectory_summary": str(NOAOA),
            "checkpoint": args.endpoint,
        }
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "scores": {"Reading_eye": row.get("Reading_eye"), "Reading_self_paced": row.get("Reading_self_paced"), "Reading": row.get("Reading")},
        "returncode": 0,
        "status": "prefilled_from_step046_tail_endpoint_noaoa_trajectory",
        "source_trajectory_summary": str(NOAOA),
        "checkpoint": args.endpoint,
    }
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload": str(p), "target": target, "endpoint": args.endpoint, "equal7": row.get("equal7_full_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
