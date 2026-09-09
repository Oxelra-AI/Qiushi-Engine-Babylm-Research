#!/usr/bin/env python3
"""Prefill endpoint-aware mask full-eval payload from a mask no-AoA trajectory."""
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
import mask_endpoint_full_eval as wrap  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
STUDY = _public_path('experiments/archive/compact_experience')
DEFAULT_NOAOA_ROOTS = [
    _public_path('data/external/mask_noaoa_eval'),
    _public_path('experiments/archive/compact_experience/data/mask_eval'),
]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def find_traj(root: pathlib.Path, arm: str) -> pathlib.Path:
    target = f"mask_{arm}"
    candidates = [root / f"{target}_trajectory_summary.json", root / target / f"{target}_trajectory_summary.json"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(candidates[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=wrap.VALID_ARMS, required=True)
    ap.add_argument("--endpoint", choices=wrap.VALID_ENDPOINTS, required=True)
    ap.add_argument("--noaoa_root", default="")
    args = ap.parse_args()
    roots = [pathlib.Path(args.noaoa_root)] if args.noaoa_root else DEFAULT_NOAOA_ROOTS
    traj = None
    for root in roots:
        try:
            traj = find_traj(root, args.arm)
            break
        except FileNotFoundError:
            continue
    if traj is None:
        raise FileNotFoundError(f"No trajectory found for {args.arm} in {roots}")
    obj = json.loads(traj.read_text(encoding="utf-8"))
    row = (obj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No {args.endpoint} row in {traj}")

    base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval')
    base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"
    base.TARGETS = wrap.make_targets(args.arm, args.endpoint)
    base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}
    target = wrap.target_name(args.arm, args.endpoint)
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
        "trajectory_summary": str(traj),
        "target": f"mask_{args.arm}",
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from post-training no-AoA evaluation of the same frozen endpoint. It is not a training signal, but these component columns are explicitly used for arm/endpoint selection and must be reported as post-selection evidence.",
    }
    for col in COLUMNS:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in {traj}")
        payload["tasks"][col] = {"column": col, "score": float(row[col]), "returncode": 0, "status": "prefilled_from_step046_mask_endpoint_noaoa_trajectory", "source_trajectory_summary": str(traj), "checkpoint": args.endpoint}
    payload["tasks"]["Reading"] = {"column": "Reading", "scores": {"Reading_eye": row.get("Reading_eye"), "Reading_self_paced": row.get("Reading_self_paced"), "Reading": row.get("Reading")}, "returncode": 0, "status": "prefilled_from_step046_mask_endpoint_noaoa_trajectory", "source_trajectory_summary": str(traj), "checkpoint": args.endpoint}
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload": str(p), "target": target, "arm": args.arm, "endpoint": args.endpoint, "equal7": row.get("equal7_full_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
