#!/usr/bin/env python3
"""Full nine-column wrapper for research/044 cluster-tail candidates.

The no-AoA trajectory screen found that the strongest cluster-test artifacts are
not the true-cluster arm but the controls:
  * E4 untouched tail, chck_90M: best no-AoA equal7
  * E3 anchor repeat, chck_99M: close second no-AoA equal7

This wrapper registers those endpoints in the corrected research full evaluator,
prefills the already-computed official-style zero-shot/Reading columns from the
research no-AoA trajectory outputs, then runs only the missing SuperGLUE and AoA
columns. AoA is measured only if the model_root has the official strict-small
checkpoint ladder; otherwise the corrected evaluator records AoA=0.0 as a
provisional leaderboard-unit value and explicitly marks the artifact as not
AoA-submit-ready.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
from typing import Any, Dict

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import full_overall_eval_runner as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_BASE = _public_path('experiments/archive/compact_experience/training/runs')
NOAOA = _public_path('experiments/archive/compact_experience/data/cluster_continuation_eval')
base.OUT_ROOT = _public_path('experiments/archive/compact_experience/data/cluster_tail_full_eval')
base.PER_TARGET_DIR = base.OUT_ROOT / "per_target"

CANDIDATES: Dict[str, Dict[str, Any]] = {
    "cluster_tail_E4_untouched_90M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/continuation_E4_untouched_tail'),
        "endpoint": "chck_90M",
        "arm": "E4_untouched_tail",
        "description": "research E4 untouched clean-Qwen 80M tail continuation restart; no cluster intervention; best no-AoA equal7 in cluster screen.",
        "family": "tail_continuation_control",
    },
    "cluster_tail_E3_repeat_99M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/continuation_E3_anchor_repeat'),
        "endpoint": "chck_99M",
        "arm": "E3_anchor_repeat",
        "description": "research E3 anchor-repeat control; close no-AoA equal7 competitor to E4.",
        "family": "anchor_repeat_control",
    },
    "cluster_tail_E1_true_90M": {
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/continuation_E1_true_cluster'),
        "endpoint": "chck_90M",
        "arm": "E1_true_cluster",
        "description": "research E1 true natural-cluster arm; retained only for mechanism comparison, not the no-AoA frontier winner.",
        "family": "true_cluster",
    },
}

base.TARGETS = CANDIDATES
base.ZERO_BY_COL = {t["column"]: t for t in base.ZERO_SHOT_TASKS}


def trajectory_path(arm: str) -> pathlib.Path:
    return NOAOA / f"step043_{arm}_trajectory_summary.json"


def load_noaoa_row(arm: str, endpoint: str) -> dict[str, Any]:
    path = trajectory_path(arm)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    table = obj.get("table", {})
    row = table.get(endpoint)
    if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
        raise RuntimeError(f"No no-AoA row for {arm}/{endpoint} in {path}")
    return row


def prefill_target(target: str) -> None:
    spec = CANDIDATES[target]
    endpoint = spec["endpoint"]
    arm = spec["arm"]
    model_root = pathlib.Path(spec["run_dir"]) / "hf_model"
    model_path = model_root / endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    row = load_noaoa_row(arm, endpoint)

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
            "endpoint": endpoint,
            "started_utc": base.now_utc(),
            "gpu": None,
            "tasks": {},
        }
    payload.setdefault("tasks", {})
    payload["prefill_source"] = {
        "trajectory_summary": str(trajectory_path(arm)),
        "arm": arm,
        "endpoint": endpoint,
        "equal7_full_eval": row.get("equal7_full_eval"),
        "non_leakage_statement": "Zero-shot/Reading prefill comes from local post-training no-AoA evaluation, not training or selection signals.",
    }

    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if row.get(col) is None:
            raise RuntimeError(f"Missing {col} in {arm}/{endpoint} no-AoA row")
        payload["tasks"][col] = {
            "column": col,
            "score": float(row[col]),
            "returncode": 0,
            "status": "prefilled_from_step044_noaoa_trajectory",
            "source_trajectory_summary": str(trajectory_path(arm)),
            "checkpoint": endpoint,
        }
    payload["tasks"]["Reading"] = {
        "column": "Reading",
        "scores": {
            "Reading_eye": row.get("Reading_eye"),
            "Reading_self_paced": row.get("Reading_self_paced"),
            "Reading": row.get("Reading"),
        },
        "returncode": 0,
        "status": "prefilled_from_step044_noaoa_trajectory",
        "source_trajectory_summary": str(trajectory_path(arm)),
        "checkpoint": endpoint,
    }
    base.PER_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prefill_all() -> None:
    for target in CANDIDATES:
        prefill_target(target)


def main() -> None:
    if "--list-candidates" in sys.argv:
        print(json.dumps({k: {kk: str(vv) for kk, vv in v.items()} for k, v in sorted(CANDIDATES.items())}, indent=2))
        return
    prefill_all()
    base.main()


if __name__ == "__main__":
    main()
