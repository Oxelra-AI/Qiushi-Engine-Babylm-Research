#!/usr/bin/env python3
"""Regression tests for research compliant-evaluation continuation policy.

The tests use tiny synthetic per-target JSON files.  They do not train, evaluate,
or inspect active retrain directories.  They check that:
1. a non-official/missing AoA placeholder is not counted as a real zero;
2. missing-column evaluation continues for the reinvest endpoint unless hard
   upper-bound arithmetic proves the decision target unreachable;
3. the hard-stop path triggers only when even missing=100 cannot reach target.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import subprocess
import sys
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
PROJECTOR = STUDY / "scripts/project_compliant_eval_continuation.py"
OUT = STUDY / "data/projection_policy_regression"


def write_case(name: str, tasks: dict[str, Any]) -> pathlib.Path:
    p = OUT / f"{name}.json"
    payload = {
        "target": "complianttok_reinvest_seed43022",
        "family": "end_to_end_compliant_tokenizer_density_reinvestment",
        "description": "synthetic regression payload",
        "budget_semantics": {"reinvest": "synthetic submission-relevant endpoint"},
        "tasks": tasks,
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def run_case(p: pathlib.Path, target: float = 41.8) -> dict[str, Any]:
    out = p.with_name(p.stem + "_policy.json")
    cmd = [sys.executable, str(PROJECTOR), "--per-target-json", str(p), "--decision-target", str(target), "--out", str(out)]
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError({"cmd": cmd, "stdout": proc.stdout, "stderr": proc.stderr})
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}

    # Seven cheap columns known at roughly plausible values; SuperGLUE missing;
    # AoA has a missing-checkpoint placeholder and must remain missing, not zero.
    placeholder_aoa_tasks = {
        "BLiMP": {"score": 66.0},
        "Supplement": {"score": 63.0},
        "EWoK": {"score": 53.0},
        "Entity": {"score": 27.0},
        "COMPS": {"score": 52.0},
        "GlobalPIQA_parallel": {"score": 35.0},
        "GlobalPIQA_nonparallel": {"score": 36.0},
        "Reading": {"scores": {"Reading": 8.0}},
        "AoA": {"status": "not_official_missing_checkpoints", "aoa_leaderboard_score": 0.0},
    }
    cases["placeholder_aoa"] = run_case(write_case("placeholder_aoa", placeholder_aoa_tasks))

    # Hard impossible: one known zero-shot score of zero with eight missing columns
    # and decision target 90 means max overall=(0+8*100)/9 < 90, so stop.
    impossible_tasks = {"BLiMP": {"score": 0.0}}
    cases["hard_impossible"] = run_case(write_case("hard_impossible", impossible_tasks), target=90.0)

    # Complete official AoA zero should count as known when row count and finite
    # checks are present.
    complete_tasks = {
        "BLiMP": {"score": 66.0},
        "Supplement": {"score": 63.0},
        "EWoK": {"score": 53.0},
        "Entity": {"score": 27.0},
        "COMPS": {"score": 52.0},
        "SuperGLUE": {"superglue_mean": 71.0},
        "GlobalPIQA_parallel": {"score": 35.0},
        "GlobalPIQA_nonparallel": {"score": 36.0},
        "Reading": {"scores": {"Reading": 8.0}},
        "AoA": {"status": "official_aoa_done", "row_count_values": [8005], "finite_surprisals": True, "aoa_leaderboard_score": 0.0},
    }
    cases["complete_official_aoa_zero"] = run_case(write_case("complete_official_aoa_zero", complete_tasks))

    checks = {
        "placeholder_aoa_kept_missing": "AoA" in cases["placeholder_aoa"]["missing_keys"],
        "placeholder_recommends_continue": cases["placeholder_aoa"]["policy"]["recommended_action"] == "continue_remaining_official_evaluation_for_submission_relevant_endpoint",
        "hard_impossible_stops": cases["hard_impossible"]["policy"]["hard_stop_remaining_evaluation"] is True,
        "complete_aoa_counted_known": "AoA" in cases["complete_official_aoa_zero"]["known_keys"] and not cases["complete_official_aoa_zero"]["missing_keys"],
    }
    payload = {
        "status": "PROJECTION_POLICY_REGRESSION",
        "checks": checks,
        "pass": all(checks.values()),
        "case_outputs": {k: str((OUT / f"{k}_policy.json")) for k in cases},
    }
    out = OUT / "projection_policy_regression.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
