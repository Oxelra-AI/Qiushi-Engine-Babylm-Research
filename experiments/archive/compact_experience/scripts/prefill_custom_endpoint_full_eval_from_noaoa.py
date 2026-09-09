#!/usr/bin/env python3
"""Prefill dynamic endpoint full-eval payload from a no-AoA trajectory summary."""
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

WORKSPACE = _public_path('experiments/archive/compact_experience')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_target", required=True)
    ap.add_argument("--run_dir", required=True, help="Evaluation root containing hf_model/<endpoint>.")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--trajectory_summary", required=True)
    ap.add_argument("--family", default="custom_endpoint")
    ap.add_argument("--description", default="Dynamic endpoint-consistent full-eval payload prefilled from no-AoA measurement.")
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    model_root = run_dir / "hf_model"
    model_path = model_root / args.endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    traj = pathlib.Path(args.trajectory_summary)
    obj = json.loads(traj.read_text(encoding="utf-8"))
    row = (obj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No {args.endpoint} row with equal7_full_eval in {traj}")
    base.OUT_ROOT = OUT_ROOT
    base.PER_TARGET_DIR = _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval/per_target')
    p = base.per_target_path(args.eval_target)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": args.eval_target,
            "description": args.description,
            "family": args.family,
            "run_dir": str(run_dir),
            "model_root": str(model_root),
            "model_path": str(model_path),
            "endpoint": args.endpoint,
            "started_utc": base.now_utc(),
            "gpu": None,
            "tasks": {},
        }
    payload.setdefault("tasks", {})
    payload["description"] = args.description
    payload["family"] = args.family
    payload["run_dir"] = str(run_dir)
    payload["model_root"] = str(model_root)
    payload["model_path"] = str(model_path)
    payload["endpoint"] = args.endpoint
    payload["prefill_source"] = {
        "trajectory_summary": str(traj),
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from post-training no-AoA evaluation of the same frozen endpoint; SuperGLUE and AoA are not used for endpoint choice.",
    }
    manifest = run_dir / "endpoint_ladder_manifest.json"
    if manifest.exists():
        payload["endpoint_ladder_manifest"] = str(manifest)
        try:
            payload["endpoint_lineage_policy"] = json.loads(manifest.read_text(encoding="utf-8")).get("lineage_policy")
        except Exception as exc:
            payload["endpoint_ladder_manifest_error"] = repr(exc)
    for col in COLUMNS:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in {traj}")
        payload["tasks"][col] = {"column": col, "score": float(row[col]), "returncode": 0, "status": "prefilled_from_noaoa_trajectory", "source_trajectory_summary": str(traj), "checkpoint": args.endpoint}
    payload["tasks"]["Reading"] = {"column": "Reading", "scores": {"Reading_eye": row.get("Reading_eye"), "Reading_self_paced": row.get("Reading_self_paced"), "Reading": row.get("Reading")}, "returncode": 0, "status": "prefilled_from_noaoa_trajectory", "source_trajectory_summary": str(traj), "checkpoint": args.endpoint}
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"payload": str(p), "eval_target": args.eval_target, "endpoint": args.endpoint, "equal7": row.get("equal7_full_eval")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
