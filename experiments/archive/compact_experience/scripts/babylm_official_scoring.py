#!/usr/bin/env python3
"""Canonical local BabyLM 2026 leaderboard-style scoring helpers.

Important measurement correction (research): the official BabyLM leaderboard loader
(`read_evals.py` / `_get_benchmark_score`) reports raw benchmark outputs after a
uniform *100 conversion to percentage-like leaderboard units. The AoA evaluator
returns a raw curve-fitness/correlation value, so local ledgers must store both:

  * aoa_raw_correlation: the direct evaluator output (e.g. 0.229)
  * aoa_leaderboard_score: 100 * raw (e.g. 22.9)

The Overall column is the mean of nine leaderboard-unit columns:
BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

OFFICIAL_OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]
HUMAN_KEYS = ["Reading", "AoA"]
AOA_SCALE = 100.0


def f_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def aoa_leaderboard_from_raw(raw: Any) -> Optional[float]:
    v = f_or_none(raw)
    return None if v is None else AOA_SCALE * v


def normalize_aoa_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of an AoA task record with explicit raw and leaderboard units.

    Older experimental records used `aoa_official` and `aoa_for_provisional_overall`
    for the raw AoA correlation. We preserve those legacy keys but add corrected
    names and set `aoa_for_provisional_overall` to the leaderboard-unit value so
    old target-score code no longer silently underweights AoA.
    """
    out = dict(rec)
    # Missing/unofficial AoA is a leaderboard zero, not a raw correlation of 0.0.
    # Check this before looking at legacy `aoa_for_provisional_overall` fields.
    if str(out.get("status", "")).startswith("not_official"):
        lb = 0.0
        out["aoa_raw_correlation"] = None
        out["aoa_official"] = None
        out["aoa_leaderboard_score"] = lb
        out["aoa_for_provisional_overall"] = lb
        out["aoa_unit_correction"] = "AoA unavailable; leaderboard convention uses 0.0."
        return out
    raw = None
    for key in ["aoa_raw_correlation", "aoa_official", "aoa"]:
        if out.get(key) is not None:
            raw = f_or_none(out.get(key))
            break
    if raw is None:
        if out.get("aoa_leaderboard_score") is not None:
            lb = f_or_none(out.get("aoa_leaderboard_score"))
            raw = None if lb is None else lb / AOA_SCALE
        elif out.get("aoa_for_provisional_overall") is not None:
            # Legacy records used this field for raw AoA; corrected records use it for
            # leaderboard units.  If this is the only surviving field, infer from
            # magnitude: raw correlations live in [-1, 1], while leaderboard units may
            # be much larger.  Zero is invariant under either convention.
            v = f_or_none(out.get("aoa_for_provisional_overall"))
            if v is None:
                return out
            if abs(v) > 1.0:
                lb = v
                raw = v / AOA_SCALE
                out["aoa_unit_inferred_from_legacy_field"] = "leaderboard_units"
            else:
                raw = v
                out["aoa_unit_inferred_from_legacy_field"] = "raw_correlation"
        else:
            return out
    lb = aoa_leaderboard_from_raw(raw) if 'lb' not in locals() else lb
    out["aoa_raw_correlation"] = raw
    out["aoa_official"] = raw  # legacy alias retained for provenance
    out["aoa_leaderboard_score"] = lb
    out["aoa_for_provisional_overall"] = lb
    out["aoa_unit_correction"] = "AoA raw correlation converted to leaderboard score by multiplying by 100 before Overall arithmetic."
    return out


def target_scores_from_tasks(tasks: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in OFFICIAL_OVERALL_KEYS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            out[col] = float(rec["score"])
    gp = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp.append(float(rec["score"]))
    if len(gp) == 2:
        out["GlobalPIQA"] = sum(gp) / 2.0
    rec = tasks.get("Reading")
    if isinstance(rec, dict):
        scores = rec.get("scores") or {}
        if scores.get("Reading") is not None:
            out["Reading"] = float(scores["Reading"])
    rec = tasks.get("SuperGLUE")
    if isinstance(rec, dict) and rec.get("superglue_mean") is not None:
        out["SuperGLUE"] = float(rec["superglue_mean"])
    rec = tasks.get("AoA")
    if isinstance(rec, dict):
        nrec = normalize_aoa_record(rec)
        if nrec.get("aoa_leaderboard_score") is not None:
            out["AoA"] = float(nrec["aoa_leaderboard_score"])
        elif nrec.get("aoa_for_provisional_overall") is not None:
            out["AoA"] = float(nrec["aoa_for_provisional_overall"])
    return out


def compute_overall_from_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    complete = all(scores.get(k) is not None for k in OFFICIAL_OVERALL_KEYS)
    out: Dict[str, Any] = {"scores": {k: scores.get(k) for k in OFFICIAL_OVERALL_KEYS}, "complete_for_provisional_overall": complete}
    if complete:
        vals = [float(scores[k]) for k in OFFICIAL_OVERALL_KEYS]
        nlp = [float(scores[k]) for k in NLP_KEYS]
        human = [float(scores[k]) for k in HUMAN_KEYS]
        out.update({
            "Overall": sum(vals) / len(vals),
            "NLP_average": sum(nlp) / len(nlp),
            "Human_like_average": sum(human) / len(human),
            "official_like_arithmetic": "mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA_leaderboard_score)",
            "aoa_unit": "leaderboard_score=100*raw_correlation",
        })
    return out


def compute_overall_from_tasks(tasks: Dict[str, Any]) -> Dict[str, Any]:
    return compute_overall_from_scores(target_scores_from_tasks(tasks))


def patch_payload_official_overall(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Patch one per-target payload in memory and return it."""
    tasks = payload.setdefault("tasks", {})
    aoa = tasks.get("AoA")
    if isinstance(aoa, dict):
        tasks["AoA"] = normalize_aoa_record(aoa)
    oo = compute_overall_from_tasks(tasks)
    aoa_rec = tasks.get("AoA", {})
    if isinstance(aoa_rec, dict):
        oo["aoa_status"] = aoa_rec.get("status")
        oo["submit_ready_aoa"] = aoa_rec.get("status") == "official_aoa_done"
        oo["aoa_raw_correlation"] = aoa_rec.get("aoa_raw_correlation")
        oo["aoa_leaderboard_score"] = aoa_rec.get("aoa_leaderboard_score")
    else:
        oo["aoa_status"] = None
        oo["submit_ready_aoa"] = False
    oo["submit_ready_overall"] = bool(oo.get("submit_ready_aoa") and oo.get("complete_for_provisional_overall"))
    payload["official_overall"] = oo
    payload["measurement_correction"] = "research corrected AoA unit: raw AoA correlation is multiplied by 100 for leaderboard Overall."
    return payload


def leaderboard_overall(row: Dict[str, Any]) -> Optional[float]:
    mapping = {
        "BLiMP": "BLiMP",
        "Supplement": "BLiMP Supplement",
        "EWoK": "EWoK",
        "Entity": "Entity Tracking",
        "COMPS": "COMPS",
        "GlobalPIQA": "GlobalPIQA",
        "SuperGLUE": "(Super)GLUE",
        "Reading": "Reading",
        "AoA": "AoA",
    }
    vals = []
    for k in OFFICIAL_OVERALL_KEYS:
        v = f_or_none(row.get(mapping[k], row.get(k)))
        if v is None:
            return None
        vals.append(v)
    return sum(vals) / len(vals)


def self_test() -> Dict[str, Any]:
    """Regression tests for the research AoA unit correction."""
    row = {
        "BLiMP": 67.95,
        "BLiMP Supplement": 52.99,
        "EWoK": 51.25,
        "Entity Tracking": 19.66,
        "COMPS": 51.77,
        "GlobalPIQA": 32.64,
        "(Super)GLUE": 64.95,
        "Reading": 1.49,
        "AoA": 22.9,
        "Overall Average": 40.62,
    }
    calc = leaderboard_overall(row)
    if calc is None or abs(calc - float(row["Overall Average"])) > 0.01:
        raise AssertionError(f"Leaderboard AoA regression failed: calc={calc} expected={row['Overall Average']}")
    raw = 0.229
    if abs(aoa_leaderboard_from_raw(raw) - 22.9) > 1e-9:
        raise AssertionError("AoA raw-to-leaderboard scaling failed")
    return {"leaderboard_row_calc": calc, "expected": row["Overall Average"], "raw_0p229_scaled": aoa_leaderboard_from_raw(raw)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
