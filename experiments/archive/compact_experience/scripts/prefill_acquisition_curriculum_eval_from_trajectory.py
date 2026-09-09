#!/usr/bin/env python3
"""Prefill acquisition-curriculum full-eval payloads from no-AoA trajectory records."""
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
import full_eval_acquisition_curriculum_candidates as cand  # noqa: E402
import full_overall_eval_runner as base0  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
TRAJ_ROOT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_screen')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_candidates')
PER_TARGET = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_candidates/per_target')
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
TARGET_TO_TRAJ: Dict[str, Tuple[str, str]] = {}
for family in ["acq_A0_baseline_order", "acq_A0p_repeated_random", "acq_A1_frequency_only", "acq_A2_full", "acq_A3_random_label", "acq_A4_reverse"]:
    for m in [10,20,30,40,50,60,70,75,80,85,90,95,100]:
        TARGET_TO_TRAJ[f"{family}_{m}M"] = (family, f"chck_{m}M")


def read_json(p: pathlib.Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def per_checkpoint_record(traj_target: str, ckpt: str, col: str) -> Dict[str, Any]:
    p = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_screen/per_checkpoint') / traj_target / ckpt / f"{col}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return read_json(p)


def prefill(target: str, gpu: int) -> Dict[str, Any]:
    if target not in cand.CANDIDATES:
        raise KeyError(f"unknown target {target}")
    traj_target, ckpt = TARGET_TO_TRAJ[target]
    cfg = cand.CANDIDATES[target]
    model_root = pathlib.Path(cfg["run_dir"]) / "hf_model"
    model_path = model_root / str(cfg["endpoint"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    summary = read_json(TRAJ_ROOT / f"{traj_target}_trajectory_summary.json")
    row = summary.get("table", {}).get(ckpt, {})
    if row.get("equal7_full_eval") is None:
        raise RuntimeError(f"trajectory row incomplete for {traj_target} {ckpt}: {row}")
    payload: Dict[str, Any] = {"target":target,"description":str(cfg.get("description", ""))+" Prefilled from acquisition no-AoA trajectory; SuperGLUE/AoA run separately.","family":cfg.get("family", ""),"run_dir":str(cfg["run_dir"]),"model_root":str(model_root),"model_path":str(model_path),"endpoint":str(cfg["endpoint"]),"prefilled_from_trajectory":{"trajectory_target":traj_target,"checkpoint":ckpt,"summary":str(TRAJ_ROOT / f"{traj_target}_trajectory_summary.json")},"gpu":gpu,"tasks":{}}
    for col in ZERO_COLS:
        rec = per_checkpoint_record(traj_target, ckpt, col)
        payload["tasks"][col] = {"column":col,"task":rec.get("task"),"data_path":rec.get("data_path"),"revision_name":rec.get("revision_name"),"output_dir":rec.get("output_dir"),"returncode":0,"score":float(rec.get("score")),"predictions":rec.get("predictions"),"report":rec.get("report"),"prefilled_from_trajectory":True}
    rrec = per_checkpoint_record(traj_target, ckpt, "Reading")
    payload["tasks"]["Reading"] = {"column":"Reading","data_path":rrec.get("data_path"),"revision_name":rrec.get("revision_name"),"output_dir":rrec.get("output_dir"),"returncode":0,"scores":rrec.get("scores"),"predictions":rrec.get("predictions"),"report":rrec.get("report"),"prefilled_from_trajectory":True}
    payload["trajectory_equal7_full_eval"] = row.get("equal7_full_eval")
    payload["trajectory_row"] = row
    payload["official_overall"] = base0.compute_overall_fields(payload)
    PER_TARGET.mkdir(parents=True, exist_ok=True)
    out = PER_TARGET / f"{target}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return {"target":target,"out":str(out),"trajectory_target":traj_target,"checkpoint":ckpt,"equal7":row.get("equal7_full_eval"),"complete_now":payload["official_overall"].get("complete_for_provisional_overall")}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("targets", nargs="+"); ap.add_argument("--gpu", type=int, default=0); args=ap.parse_args()
    print(json.dumps({"status":"prefilled","targets":[prefill(t,args.gpu) for t in args.targets]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
