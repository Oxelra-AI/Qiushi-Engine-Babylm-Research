#!/usr/bin/env python3
"""research: score calibration for deciding whether a cheap screen can justify 100M/full work.

This CPU-only script reads already completed endpoint scores and cheap 70/80M
reference surfaces.  It computes how much improvement in the seven cheap columns
(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading) is needed to close
the current fully legal gap to the 41.8 target under different assumptions about
SuperGLUE and AoA.  It does not train, evaluate, or change any artifact.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
ENDPOINT_SUMMARY = STUDY / "data/compliant_endpoint_results/compliant_endpoint_results_summary.json"
TRAJ050 = STUDY / "data/legal_treatment_trajectory/legal_treatment_trajectory_comparison.json"
OUT_DIR = STUDY / "data/expensive_work_score_thresholds"
NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/expensive_work_score_thresholds.md')
CHEAP7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FULL9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
TARGET = 41.8


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return sum(float(v) for v in vals) / len(vals)


def score_mean(scores: dict[str, Any], cols: list[str]) -> float:
    return mean([float(scores[c]) for c in cols])


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return f"{float(x):.{nd}f}"
    return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    endpoint_summary = load(ENDPOINT_SUMMARY)
    traj = load(TRAJ050)
    current = next(r for r in endpoint_summary["records"] if r["name"] == "same_pool_tokenizer")
    old = endpoint_summary["old_nonsubmittable_reference"]
    scores = current["scores"]
    current_overall = float(current["overall"])
    cheap7_mean_100m = score_mean(scores, CHEAP7)
    cheap7_sum_100m = sum(float(scores[c]) for c in CHEAP7)
    full_sum_current = sum(float(scores[c]) for c in FULL9)
    needed_full_sum_gain = TARGET * 9.0 - full_sum_current
    assert abs(current_overall - full_sum_current / 9.0) < 1e-9

    assumptions = []
    for sg_delta in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]:
        for aoa_delta in [0.0, 2.0, 5.0]:
            cheap7_sum_gain_needed = needed_full_sum_gain - sg_delta - aoa_delta
            assumptions.append({
                "superglue_delta_assumed": sg_delta,
                "aoa_delta_assumed": aoa_delta,
                "cheap7_sum_gain_needed": cheap7_sum_gain_needed,
                "cheap7_mean_gain_needed": cheap7_sum_gain_needed / 7.0,
                "target_cheap7_mean_endpoint": cheap7_mean_100m + cheap7_sum_gain_needed / 7.0,
            })

    rows = []
    for row in traj.get("rows", []):
        m = int(row["exposure_m"])
        ref_mean7 = row.get("reinvest_mean7")
        clean_mean7 = row.get("clean_mean7")
        rec = {
            "exposure_m": m,
            "tokenmean_reinvest_mean7": ref_mean7,
            "clean_control_mean7": clean_mean7,
            "tokenmean_minus_clean_mean7": row.get("delta_mean7"),
        }
        if m == 80 and ref_mean7 is not None:
            rec["endpoint100m_minus_80m_tokenmean_mean7"] = cheap7_mean_100m - float(ref_mean7)
        if m == 70 and ref_mean7 is not None:
            rec["endpoint100m_minus_70m_tokenmean_mean7"] = cheap7_mean_100m - float(ref_mean7)
        rows.append(rec)

    # Translate cheap 80M deltas against the token-mean reference to projected complete scores
    # if the endpoint carries the same delta at 100M and SuperGLUE/AoA stay as current.
    delta_table = []
    for cheap_delta in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0]:
        projected = current_overall + (7.0 * cheap_delta) / 9.0
        delta_table.append({
            "cheap7_mean_delta_vs_step35_endpoint_assumed": cheap_delta,
            "projected_overall_if_superglue_aoa_flat": projected,
            "projected_margin_vs_41p8": projected - TARGET,
            "superglue_delta_needed_if_aoa_flat": TARGET * 9.0 - (full_sum_current + 7.0 * cheap_delta),
        })

    result = {
        "status": "EXPENSIVE_WORK_SCORE_THRESHOLDS",
        "created_utc": now_utc(),
        "scope": "CPU-only calibration using completed legal endpoint and cheap 70/80M reference scores; no training/evaluation.",
        "target_overall": TARGET,
        "current_best_legal": {
            "name": current["name"],
            "overall": current_overall,
            "scores": scores,
            "cheap7_mean_100m": cheap7_mean_100m,
            "cheap7_sum_100m": cheap7_sum_100m,
            "full9_sum": full_sum_current,
            "margin_vs_target": current_overall - TARGET,
        },
        "old_non_submittable_reference": {
            "overall": old["Overall"],
            "cheap7_mean": score_mean(old["scores"], CHEAP7),
            "scores": old["scores"],
        },
        "needed_full_sum_gain": needed_full_sum_gain,
        "needed_overall_gain": TARGET - current_overall,
        "cheap7_gain_needed_by_superglue_aoa_assumption": assumptions,
        "reference_mature_rows": rows,
        "projected_overall_from_cheap7_delta": delta_table,
        "decision_reading": [
            "If SuperGLUE and AoA remain at the current legal endpoint values, the seven cheap columns need about +0.697 mean points at the 100M endpoint to reach 41.8.",
            "A cheap-screen 80M gain much below +0.5 mean7 would require a large unmeasured SuperGLUE/AoA gain and should not by itself justify a 100M/full continuation.",
            "A continuation is stronger when the gain is broad and includes BLiMP/Supplement/EWoK without sacrificing Entity or GlobalPIQA, because prior larger-vocabulary evidence improved some language columns while harming those columns.",
            "AoA has repeatedly scored 0.0 or a negative thresholded value in this line; do not rely on AoA to close the gap unless a full trajectory proves it.",
        ],
    }
    out_json = OUT_DIR / "expensive_work_score_thresholds.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research score calibration for expensive route continuation")
    lines.append("")
    lines.append(result["scope"])
    lines.append("")
    lines.append("## Current legal score gap")
    lines.append(f"- Current best fully legal endpoint: Overall {fmt(current_overall, 6)}; target {fmt(TARGET, 3)}; needed Overall gain {fmt(TARGET - current_overall, 6)}.")
    lines.append(f"- Current cheap7 endpoint mean: {fmt(cheap7_mean_100m, 6)}; SuperGLUE {fmt(scores['SuperGLUE'], 6)}; AoA {fmt(scores['AoA'], 6)}.")
    lines.append(f"- Required total nine-column score gain: {fmt(needed_full_sum_gain, 6)}.")
    lines.append("")
    lines.append("## Cheap7 gain needed at 100M")
    lines.append("| assumed SuperGLUE gain | assumed AoA gain | needed cheap7 mean gain | target cheap7 endpoint mean |")
    lines.append("|---:|---:|---:|---:|")
    for rec in assumptions:
        if rec["aoa_delta_assumed"] in (0.0, 5.0):
            lines.append(f"| {fmt(rec['superglue_delta_assumed'],1)} | {fmt(rec['aoa_delta_assumed'],1)} | {fmt(rec['cheap7_mean_gain_needed'],4)} | {fmt(rec['target_cheap7_mean_endpoint'],4)} |")
    lines.append("")
    lines.append("## Current token-mean maturity reference")
    lines.append("| exposure | token-mean reinvest mean7 | clean mean7 | treatment Δ mean7 | endpoint100M - exposure mean7 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for rec in rows:
        d = rec.get("endpoint100m_minus_80m_tokenmean_mean7", rec.get("endpoint100m_minus_70m_tokenmean_mean7"))
        lines.append(f"| {rec['exposure_m']} | {fmt(rec['tokenmean_reinvest_mean7'],4)} | {fmt(rec['clean_control_mean7'],4)} | {fmt(rec['tokenmean_minus_clean_mean7'],4)} | {fmt(d,4)} |")
    lines.append("")
    lines.append("## Interpreting an 80M cheap-screen delta if SuperGLUE/AoA are flat")
    lines.append("| cheap7 mean gain assumed | projected Overall | projected margin vs 41.8 | SuperGLUE gain needed if AoA flat |")
    lines.append("|---:|---:|---:|---:|")
    for rec in delta_table:
        lines.append(f"| {fmt(rec['cheap7_mean_delta_vs_step35_endpoint_assumed'],2)} | {fmt(rec['projected_overall_if_superglue_aoa_flat'],4)} | {fmt(rec['projected_margin_vs_41p8'],4)} | {fmt(rec['superglue_delta_needed_if_aoa_flat'],4)} |")
    lines.append("")
    lines.append("## Scientific reading")
    for s in result["decision_reading"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_note": str(NOTE),
        "current_overall": current_overall,
        "needed_overall_gain": result["needed_overall_gain"],
        "cheap7_gain_needed_if_sg_aoa_flat": next(a["cheap7_mean_gain_needed"] for a in assumptions if a["superglue_delta_assumed"] == 0.0 and a["aoa_delta_assumed"] == 0.0),
        "current_80m_to_100m_mean7_delta": next((r.get("endpoint100m_minus_80m_tokenmean_mean7") for r in rows if r["exposure_m"] == 80), None),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
