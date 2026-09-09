#!/usr/bin/env python3
"""Regression for the partial-surface projection policy.

Confirms that projection helpers are scheduling aids only, keep missing AoA
missing, and expose a hard-upper-bound feasibility check. This guards against
reusing a partial seven-column surface as a false endpoint after the corrected
Strict-Small-tokenizer retrains finish.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/projection_policy_regression.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT = _public_path('experiments/archive/representation_and_objectives/data/projection_policy_regression/projection_policy_regression.json')
PROJ_PATH = _public_path('experiments/archive/representation_and_objectives/scripts/projection_after_partial_surface.py')

spec = importlib.util.spec_from_file_location("proj", PROJ_PATH)
proj = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(proj)  # type: ignore


def case(name: str, scores: dict[str, float]) -> dict[str, Any]:
    r = proj.compute_projection(scores)  # type: ignore[attr-defined]
    return {"name": name, "scores": scores, "projection": r}


def main() -> None:
    old430_full = dict(proj.REFERENCE["old_inherited_tokenizer_seed43022_official"])  # type: ignore[attr-defined]
    old430_partial = {k: v for k, v in old430_full.items() if k not in {"SuperGLUE", "AoA", "Overall"}}
    impossible_partial = {"BLiMP": 0.0, "Supplement": 0.0, "EWoK": 0.0, "Entity": 0.0, "COMPS": 0.0, "GlobalPIQA": 0.0, "Reading": 0.0}
    full_with_official_aoa_zero = {k: v for k, v in old430_full.items() if k != "Overall"}

    cases = {
        "old430_partial_missing_sg_aoa": case("old430_partial_missing_sg_aoa", old430_partial),
        "hard_impossible_surface": case("hard_impossible_surface", impossible_partial),
        "complete_official_aoa_zero": case("complete_official_aoa_zero", full_with_official_aoa_zero),
    }

    checks = {
        "partial_keeps_superglue_and_aoa_missing": set(cases["old430_partial_missing_sg_aoa"]["projection"]["missing_columns"]) == {"SuperGLUE", "AoA"},
        "partial_reports_full_eval_required": cases["old430_partial_missing_sg_aoa"]["projection"].get("full_eval_required_for_two_seed_measurement") is True,
        "partial_upper_bound_can_reach": cases["old430_partial_missing_sg_aoa"]["projection"].get("hard_upper_bound_can_still_reach_41p8") is True,
        "hard_impossible_upper_bound_stops": cases["hard_impossible_surface"]["projection"].get("hard_upper_bound_can_still_reach_41p8") is False,
        "complete_aoa_zero_counted_known": cases["complete_official_aoa_zero"]["projection"].get("missing_columns") == [] and math.isclose(cases["complete_official_aoa_zero"]["projection"].get("overall_if_complete"), old430_full["Overall"], rel_tol=0, abs_tol=1e-12),
    }
    payload = {
        "status": "PROJECTION_POLICY_REGRESSION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "projection_helper": str(PROJ_PATH.relative_to(USER_ROOT)),
        "checks": checks,
        "pass": all(checks.values()),
        "cases": cases,
    }
    _public_path('experiments/archive/representation_and_objectives/data/projection_policy_regression').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT.relative_to(USER_ROOT)), "pass": payload["pass"], "checks": checks}, indent=2), flush=True)
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
