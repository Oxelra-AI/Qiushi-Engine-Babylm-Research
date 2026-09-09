#!/usr/bin/env python3
"""Prefill semantic-view full-eval payload from no-AoA trajectory results.

The no-AoA trajectory evaluates BLiMP, Supplement, EWoK, Entity, COMPS,
GlobalPIQA parallel/nonparallel, and Reading for every 10M checkpoint.  After a
scientific endpoint is selected from that frozen post-training trajectory, this
script writes the corresponding per-target payload for the full official-style
runner so the next GPU work can run only SuperGLUE and AoA without recomputing
already measured columns.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/training/scripts')
COMPACT_EXPERIENCE_SCRIPTS = pathlib.Path("experiments/archive/compact_experience/scripts")
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS.resolve()))
import full_overall_eval_runner as base  # noqa: E402

OUT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/semantic_view_full_eval")
NOAOA_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval")
RUNS = pathlib.Path("experiments/archive/representation_and_objectives/training/runs")
RUN_DIRS = {
    "semantic_view_treatment": RUNS / "semantic_view_treatment_8x480_16k_wwm_seed43022",
    "original_packet_local": RUNS / "original_packet_local_8x480_16k_wwm_seed43022",
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def load_traj(noaoa_root: pathlib.Path, target: str) -> dict[str, Any]:
    p = noaoa_root / f"{target}_trajectory_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--endpoint", required=True, help="checkpoint name such as chck_90M")
    ap.add_argument("--noaoa-root", default=str(NOAOA_ROOT))
    ap.add_argument("--eval-target", default="", help="payload target name; default target__endpoint")
    ap.add_argument("--description", default="Semantic-view contrast endpoint selected from matched no-AoA trajectory; zero-shot/Reading prefilled from frozen checkpoint, SuperGLUE/AoA to be run by full evaluator.")
    args = ap.parse_args()

    noaoa_root = pathlib.Path(args.noaoa_root)
    traj = load_traj(noaoa_root, args.target)
    row = (traj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No complete row for {args.target} {args.endpoint} in {noaoa_root}")

    run_dir = RUN_DIRS[args.target]
    model_root = run_dir / "hf_model"
    model_path = model_root / args.endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    eval_target = args.eval_target or f"{args.target}__{args.endpoint}"

    base.OUT_ROOT = OUT_ROOT
    base.PER_TARGET_DIR = OUT_ROOT / "per_target"
    p = base.PER_TARGET_DIR / f"{eval_target}.json"
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": eval_target,
            "description": args.description,
            "family": "REPRESENTATION_FRONTIER_STUDIES_semantic_view_packet_local_contrast",
            "run_dir": str(run_dir),
            "model_root": str(model_root),
            "model_path": str(model_path),
            "endpoint": args.endpoint,
            "started_utc": base.now_utc(),
            "gpu": None,
            "tasks": {},
        }
    payload.setdefault("tasks", {})
    payload.update({
        "target": eval_target,
        "description": args.description,
        "family": "REPRESENTATION_FRONTIER_STUDIES_semantic_view_packet_local_contrast",
        "run_dir": str(run_dir),
        "model_root": str(model_root),
        "model_path": str(model_path),
        "endpoint": args.endpoint,
        "prefill_source": {
            "trajectory_summary": str(noaoa_root / f"{args.target}_trajectory_summary.json"),
            "target": args.target,
            "endpoint": args.endpoint,
            "equal7_full_eval": row.get("equal7_full_eval"),
            "non_leakage_statement": "No-AoA zero-shot/Reading prefill comes from post-training evaluation of the same frozen checkpoint. Endpoint choice must be reported as post-selection evidence; SuperGLUE and AoA are not used by this prefill script.",
        },
    })
    for col in COLUMNS:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in row for {args.target} {args.endpoint}")
        payload["tasks"][col] = {
            "column": col,
            "score": float(row[col]),
            "returncode": 0,
            "status": "prefilled_from_semantic_view_noaoa_trajectory",
            "source_trajectory_summary": str(noaoa_root / f"{args.target}_trajectory_summary.json"),
            "checkpoint": args.endpoint,
        }
    for k in ["Reading_eye", "Reading_self_paced", "Reading"]:
        if row.get(k) is None:
            raise RuntimeError(f"Missing {k} in row for {args.target} {args.endpoint}")
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "scores": {"Reading_eye": row.get("Reading_eye"), "Reading_self_paced": row.get("Reading_self_paced"), "Reading": row.get("Reading")},
        "returncode": 0,
        "status": "prefilled_from_semantic_view_noaoa_trajectory",
        "source_trajectory_summary": str(noaoa_root / f"{args.target}_trajectory_summary.json"),
        "checkpoint": args.endpoint,
    }

    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "SEMANTIC_VIEW_FULL_EVAL_PREFILLED",
        "payload": str(p),
        "eval_target": eval_target,
        "source_target": args.target,
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "model_path": str(model_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
