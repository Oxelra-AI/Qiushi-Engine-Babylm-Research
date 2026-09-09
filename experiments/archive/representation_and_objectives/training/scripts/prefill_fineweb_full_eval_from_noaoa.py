#!/usr/bin/env python3
"""Prefill FineWeb seqsafe96 full-eval payloads from repaired no-AoA trajectories.

The repaired trajectory evaluates the seven zero-shot/Reading columns at each 10M
checkpoint.  This script writes a per-target payload compatible with the inherited
full official-style evaluator so a later launcher can run only SuperGLUE and AoA for
selected frozen endpoints.  It is intentionally target-agnostic over the repaired
FineWeb treatment/control target names.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

COMPACT_EXPERIENCE_SCRIPTS = pathlib.Path("experiments/archive/compact_experience/scripts")
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS.resolve()))
import full_overall_eval_runner as base  # noqa: E402

OUT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval")
DEFAULT_NOAOA_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval_repairseq")
RUNS = pathlib.Path("experiments/archive/representation_and_objectives/training/runs")
RUN_DIRS = {
    "fineweb_seqsafe96_treatment_repairseq": RUNS / "cleanqwen_fineweb_seqsafe96_repairseq_8x480_16k_wwm_seed43022",
    "fineweb_seqsafe96_control_repairseq": RUNS / "cleanqwen_official_seqsafe96_control_repairseq_8x480_16k_wwm_seed43022",
}
NOAOA_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def load_traj(noaoa_root: pathlib.Path, target: str) -> dict[str, Any]:
    p = noaoa_root / f"{target}_trajectory_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-target", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--eval-target", default="", help="payload target name; default source-target__endpoint")
    ap.add_argument("--noaoa-root", default=str(DEFAULT_NOAOA_ROOT))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--description", default="FineWeb seqsafe96 source-breadth endpoint selected from repaired no-AoA trajectory; zero-shot/Reading prefilled from the frozen checkpoint, SuperGLUE/AoA to be run separately.")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    noaoa_root = pathlib.Path(args.noaoa_root)
    traj = load_traj(noaoa_root, args.source_target)
    row = (traj.get("table") or {}).get(args.endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No complete no-AoA row for {args.source_target} {args.endpoint} in {noaoa_root}")

    run_dir = RUN_DIRS[args.source_target]
    model_root = run_dir / "hf_model"
    model_path = model_root / args.endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    eval_target = args.eval_target or f"{args.source_target}__{args.endpoint}"

    base.OUT_ROOT = out_root
    base.PER_TARGET_DIR = out_root / "per_target"
    payload_path = base.PER_TARGET_DIR / f"{eval_target}.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": eval_target,
            "description": args.description,
            "family": "REPRESENTATION_FRONTIER_STUDIES_fineweb_seqsafe96_source_breadth_repairseq",
            "source_target": args.source_target,
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
        "family": "REPRESENTATION_FRONTIER_STUDIES_fineweb_seqsafe96_source_breadth_repairseq",
        "source_target": args.source_target,
        "run_dir": str(run_dir),
        "model_root": str(model_root),
        "model_path": str(model_path),
        "endpoint": args.endpoint,
        "prefill_source": {
            "trajectory_summary": str(noaoa_root / f"{args.source_target}_trajectory_summary.json"),
            "source_target": args.source_target,
            "endpoint": args.endpoint,
            "equal7_full_eval": row.get("equal7_full_eval"),
            "note": "No-AoA zero-shot/Reading prefill comes from post-training evaluation of this frozen checkpoint. SuperGLUE and AoA are not used in endpoint selection by this script.",
        },
    })

    for col in NOAOA_COLUMNS:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in no-AoA row for {args.source_target} {args.endpoint}")
        payload["tasks"][col] = {
            "column": col,
            "score": float(row[col]),
            "returncode": 0,
            "status": "prefilled_from_fineweb_seqsafe96_repaired_noaoa_trajectory",
            "source_trajectory_summary": str(noaoa_root / f"{args.source_target}_trajectory_summary.json"),
            "checkpoint": args.endpoint,
        }
    for k in ["Reading_eye", "Reading_self_paced", "Reading"]:
        if row.get(k) is None:
            raise RuntimeError(f"Missing {k} in no-AoA row for {args.source_target} {args.endpoint}")
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "scores": {
            "Reading_eye": row.get("Reading_eye"),
            "Reading_self_paced": row.get("Reading_self_paced"),
            "Reading": row.get("Reading"),
        },
        "returncode": 0,
        "status": "prefilled_from_fineweb_seqsafe96_repaired_noaoa_trajectory",
        "source_trajectory_summary": str(noaoa_root / f"{args.source_target}_trajectory_summary.json"),
        "checkpoint": args.endpoint,
    }

    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "FINEWEB_FULL_EVAL_PREFILLED",
        "payload": str(payload_path),
        "eval_target": eval_target,
        "source_target": args.source_target,
        "endpoint": args.endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "model_path": str(model_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
