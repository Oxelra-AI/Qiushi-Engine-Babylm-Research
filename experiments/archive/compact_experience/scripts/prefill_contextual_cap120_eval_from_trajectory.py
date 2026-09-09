#!/usr/bin/env python3
"""Prefill contextual cap-120 full-eval payloads from no-AoA trajectory records.

Copies BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading outputs from the
contextual cap-120 trajectory screen so the corrected full-eval runner only needs to
add SuperGLUE and AoA.  The source trajectory screen contains no AoA/CDI information.
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
from typing import Any, Dict, Tuple

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_eval_candidates as cand  # noqa: E402
import full_overall_eval_runner as base0  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
TRAJ_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_screen')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/full_eval_candidates')
PER_TARGET = _public_path('experiments/archive/compact_experience/data/full_eval_candidates/per_target')
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]

TARGET_TO_TRAJ: Dict[str, Tuple[str, str]] = {}
for m in [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100]:
    TARGET_TO_TRAJ[f"context_cap120_treat_seed43022_{m}M"] = ("context_cap120_treat_seed43022", f"chck_{m}M")
    TARGET_TO_TRAJ[f"context_cap120_control_seed43022_{m}M"] = ("context_cap120_control_seed43022", f"chck_{m}M")


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_checkpoint_record(traj_target: str, ckpt: str, col: str) -> Dict[str, Any]:
    p = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_screen/per_checkpoint') / traj_target / ckpt / f"{col}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_json(p)


def prefill(target: str, gpu: int) -> Dict[str, Any]:
    if target not in cand.CANDIDATES:
        raise KeyError(f"unknown candidate {target}")
    if target not in TARGET_TO_TRAJ:
        raise KeyError(f"target is not a contextual trajectory-prefill target: {target}")
    traj_target, ckpt = TARGET_TO_TRAJ[target]
    target_cfg = cand.CANDIDATES[target]
    model_root = pathlib.Path(target_cfg["run_dir"]) / "hf_model"
    model_path = model_root / str(target_cfg["endpoint"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    summary = read_json(TRAJ_ROOT / f"{traj_target}_trajectory_summary.json")
    row = summary.get("table", {}).get(ckpt, {})
    if row.get("equal7_full_eval") is None:
        raise RuntimeError(f"trajectory row incomplete for {traj_target} {ckpt}: {row}")
    payload: Dict[str, Any] = {
        "target": target,
        "description": str(target_cfg.get("description", "")) + " Prefilled from research contextual cap-120 no-AoA trajectory records; SuperGLUE/AoA run separately.",
        "family": target_cfg.get("family", ""),
        "run_dir": str(target_cfg["run_dir"]),
        "model_root": str(model_root),
        "model_path": str(model_path),
        "endpoint": str(target_cfg["endpoint"]),
        "prefilled_from_trajectory": {"trajectory_target": traj_target, "checkpoint": ckpt, "summary": str(TRAJ_ROOT / f"{traj_target}_trajectory_summary.json")},
        "gpu": gpu,
        "tasks": {},
    }
    for col in ZERO_COLS:
        rec = per_checkpoint_record(traj_target, ckpt, col)
        score = rec.get("score")
        if score is None:
            raise RuntimeError(f"missing score in {traj_target} {ckpt} {col}")
        payload["tasks"][col] = {
            "column": col,
            "task": rec.get("task"),
            "data_path": rec.get("data_path"),
            "revision_name": rec.get("revision_name"),
            "output_dir": rec.get("output_dir"),
            "returncode": 0,
            "score": float(score),
            "predictions": rec.get("predictions"),
            "report": rec.get("report"),
            "prefilled_from_trajectory": True,
        }
    rrec = per_checkpoint_record(traj_target, ckpt, "Reading")
    scores = rrec.get("scores") or {}
    if scores.get("Reading") is None:
        raise RuntimeError(f"missing Reading score in {traj_target} {ckpt}")
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "data_path": rrec.get("data_path"),
        "revision_name": rrec.get("revision_name"),
        "output_dir": rrec.get("output_dir"),
        "returncode": 0,
        "scores": scores,
        "predictions": rrec.get("predictions"),
        "report": rrec.get("report"),
        "prefilled_from_trajectory": True,
    }
    payload["trajectory_equal7_full_eval"] = row.get("equal7_full_eval")
    payload["trajectory_row"] = row
    payload["official_overall"] = base0.compute_overall_fields(payload)
    PER_TARGET.mkdir(parents=True, exist_ok=True)
    out = PER_TARGET / f"{target}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"target": target, "out": str(out), "trajectory_target": traj_target, "checkpoint": ckpt, "equal7": row.get("equal7_full_eval"), "complete_now": payload["official_overall"].get("complete_for_provisional_overall")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps({"status": "prefilled", "targets": [prefill(t, args.gpu) for t in args.targets]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
