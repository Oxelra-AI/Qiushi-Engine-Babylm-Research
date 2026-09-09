#!/usr/bin/env python3
"""research: arithmetic bridge from cheap columns to complete Overall for FW arms.

The BabyLM Overall score is a mean over nine columns in the local official collation
(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA).  Cheap
screening usually measures seven columns (all except SuperGLUE and AoA).  This script
computes the required SuperGLUE+AoA or required cheap7 for a target Overall and compares
available FW cheap files with known compliant endpoints.  It is a CPU-only decision aid;
it does not evaluate models or choose submissions.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
OUT = A01_WS / "data/fw_sota_plausibility"
NOTE = A01_WS / "notes/fw_sota_plausibility_calculator.md"
CHEAP_STATUS = A01_WS / "data/fw_peer_status/fw_peer_status_and_cheap_synthesis.json"

TARGET_OVERALL = 41.80
# Visible public leader vector from research/goal context.
VISIBLE_LEADER = {
    "name": "visible_41p80_leader_go76dof_wwm_curriculum_simplification_40k",
    "Overall": 41.80,
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.0,
}
# Completed local endpoints with reliable vectors from long memory. These are used only
# to bound SuperGLUE/AoA plausibility, not as a complete archival parser.
KNOWN_ENDPOINTS = [
    {
        "name": "legal16k_seed43022", "Overall": 40.703956400082454,
        "BLiMP": None, "Supplement": None, "EWoK": None, "Entity": None,
        "COMPS": None, "GlobalPIQA": None, "SuperGLUE": None, "Reading": None, "AoA": 0.0,
        "note": "full vector not embedded here; kept out of range summary",
    },
    {
        "name": "legal16k_seed43122", "Overall": 41.023994024744404,
        "BLiMP": None, "Supplement": None, "EWoK": None, "Entity": None,
        "COMPS": None, "GlobalPIQA": None, "SuperGLUE": None, "Reading": None, "AoA": 0.0,
        "note": "full vector not embedded here; kept out of range summary",
    },
    {
        "name": "legal40k_seed43022", "Overall": 41.140577774478444,
        "BLiMP": None, "Supplement": None, "EWoK": None, "Entity": None,
        "COMPS": None, "GlobalPIQA": None, "SuperGLUE": None, "Reading": None, "AoA": 0.0,
        "note": "full vector not embedded here; kept out of range summary",
    },
    {
        "name": "legal40k_depth_12x384_seed43022", "Overall": 41.02759583135309,
        "BLiMP": 67.4751, "Supplement": 60.1395, "EWoK": 50.5472, "Entity": 27.1714,
        "COMPS": 52.7023, "GlobalPIQA": 35.6359, "SuperGLUE": 68.2275, "Reading": 7.3493, "AoA": 0.0,
        "note": "research complete endpoint",
    },
    {
        "name": "legal40k_sgcr_K50d64_seed43022", "Overall": 40.331709413463635,
        "BLiMP": 67.6905, "Supplement": 57.7493, "EWoK": 50.5275, "Entity": 25.4409,
        "COMPS": 52.8968, "GlobalPIQA": 34.1505, "SuperGLUE": 67.0843, "Reading": 7.4457, "AoA": 0.0,
        "note": "research complete endpoint",
    },
    {
        "name": "noncompliant_inherited_tokenizer_compact_seed43022", "Overall": 42.0331347900748,
        "BLiMP": 66.87, "Supplement": 63.28, "EWoK": 53.54, "Entity": 27.75,
        "COMPS": 51.97, "GlobalPIQA": 35.62, "SuperGLUE": 71.04, "Reading": 8.24, "AoA": 0.0,
        "note": "mechanism evidence only; tokenizer noncompliant",
    },
]
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def f(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def cheap7(vec: dict[str, Any]) -> float | None:
    vals = [f(vec.get(c)) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return sum(vals) / 7.0  # type: ignore[arg-type]


def required_superglue_plus_aoa(target_overall: float, cheap7_value: float) -> float:
    return 9.0 * target_overall - 7.0 * cheap7_value


def required_cheap7(target_overall: float, superglue_plus_aoa: float) -> float:
    return (9.0 * target_overall - superglue_plus_aoa) / 7.0


def load_status_rows() -> list[dict[str, Any]]:
    rec = json.loads(CHEAP_STATUS.read_text(encoding="utf-8")) if CHEAP_STATUS.exists() else {}
    return rec.get("cheap_eval_rows", []) if isinstance(rec, dict) else []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status_rows = load_status_rows()
    endpoint_rows = []
    for vec in [VISIBLE_LEADER, *KNOWN_ENDPOINTS]:
        ch = cheap7(vec)
        sg = f(vec.get("SuperGLUE"))
        aoa = f(vec.get("AoA"))
        endpoint_rows.append({
            "name": vec.get("name"), "Overall": f(vec.get("Overall")), "cheap7": ch,
            "SuperGLUE": sg, "AoA": aoa, "SuperGLUE_plus_AoA": None if sg is None or aoa is None else sg + aoa,
            "note": vec.get("note", ""),
        })
    sg_plus_values = [r["SuperGLUE_plus_AoA"] for r in endpoint_rows if r.get("SuperGLUE_plus_AoA") is not None]
    sg_plus_values = [float(v) for v in sg_plus_values]
    sg_plus_summary = {
        "n": len(sg_plus_values),
        "min": min(sg_plus_values),
        "median": statistics.median(sg_plus_values),
        "max": max(sg_plus_values),
        "known_values": sorted(sg_plus_values),
    } if sg_plus_values else {"n": 0}
    # thresholds for several plausible SuperGLUE+AoA values.
    threshold_rows = []
    for sg_plus in [67.0, 68.0, 69.79, 70.0, 71.04, 72.0, 75.0, 80.0]:
        threshold_rows.append({
            "target_overall": TARGET_OVERALL,
            "assumed_SuperGLUE_plus_AoA": sg_plus,
            "required_cheap7": required_cheap7(TARGET_OVERALL, sg_plus),
        })
    fw_rows = []
    for row in status_rows:
        ch = f(row.get("cheap7"))
        rec = {
            "arm": row.get("arm"), "checkpoint": row.get("checkpoint"),
            "cheap7_complete": bool(row.get("cheap7_complete")), "cheap7": ch,
            "scores_available": row.get("scores_available"),
            "needed_SuperGLUE_plus_AoA_for_41p80": None if ch is None else required_superglue_plus_aoa(TARGET_OVERALL, ch),
            "gap_to_leader_cheap7": None if ch is None or cheap7(VISIBLE_LEADER) is None else ch - cheap7(VISIBLE_LEADER),
        }
        for c in CHEAP_COLS:
            rec[c] = row.get(c)
        fw_rows.append(rec)
    # For partial 70M breadth, compute the required sum of missing cheap columns under historical SG+AoA.
    partial_requirements = []
    for row in status_rows:
        vals = [f(row.get(c)) for c in CHEAP_COLS]
        n = sum(v is not None for v in vals)
        if 0 < n < 7:
            known_sum = sum(v for v in vals if v is not None)
            missing_cols = [c for c, v in zip(CHEAP_COLS, vals) if v is None]
            for sg_plus in [69.79, 71.04, 72.0]:
                req_total_cheap_sum = 9.0 * TARGET_OVERALL - sg_plus
                req_missing_sum = req_total_cheap_sum - known_sum
                partial_requirements.append({
                    "arm": row.get("arm"), "checkpoint": row.get("checkpoint"),
                    "known_columns": n, "missing_columns": len(missing_cols), "missing_column_names": missing_cols,
                    "known_sum": known_sum,
                    "assumed_SuperGLUE_plus_AoA": sg_plus,
                    "required_missing_sum": req_missing_sum,
                    "required_missing_mean": req_missing_sum / max(1, len(missing_cols)),
                })
    out = {
        "status": "FW_SOTA_PLAUSIBILITY_CALCULATED",
        "created_utc": now(),
        "target_overall": TARGET_OVERALL,
        "visible_leader_cheap7": cheap7(VISIBLE_LEADER),
        "superglue_plus_aoa_summary_from_known_vectors": sg_plus_summary,
        "known_endpoint_rows": endpoint_rows,
        "threshold_rows": threshold_rows,
        "fw_available_rows": fw_rows,
        "partial_requirements": partial_requirements,
        "interpretation": {
            "cheap7_needed_if_superglue_like_visible_leader": required_cheap7(TARGET_OVERALL, 69.79),
            "cheap7_needed_if_superglue_like_noncompliant_best": required_cheap7(TARGET_OVERALL, 71.04),
            "compact_70M_needed_sg_plus_aoa": next((r["needed_SuperGLUE_plus_AoA_for_41p80"] for r in fw_rows if r["arm"] == "compact_view" and r["checkpoint"] == "70M"), None),
            "warning": "This is arithmetic, not a model evaluation. It should reduce unnecessary full evaluation only after a 100M cheap7 value exists or cheap evidence is plainly far below the needed range.",
        },
    }
    json_path = OUT / "fw_sota_plausibility_calculator.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "fw_sota_thresholds.csv").open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["target_overall", "assumed_SuperGLUE_plus_AoA", "required_cheap7"])
        w.writeheader(); w.writerows(threshold_rows)
    with (OUT / "fw_available_plausibility_rows.csv").open("w", newline="", encoding="utf-8") as fcsv:
        fields = ["arm", "checkpoint", "cheap7_complete", "scores_available", "cheap7", "needed_SuperGLUE_plus_AoA_for_41p80", "gap_to_leader_cheap7", *CHEAP_COLS]
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in fw_rows])
    with (OUT / "fw_partial_missing_requirements.csv").open("w", newline="", encoding="utf-8") as fcsv:
        fields = ["arm", "checkpoint", "known_columns", "missing_columns", "missing_column_names", "known_sum", "assumed_SuperGLUE_plus_AoA", "required_missing_sum", "required_missing_mean"]
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in partial_requirements])
    lines = []
    lines.append("# research — FW cheap7 to Overall plausibility\n")
    lines.append("This CPU-only calculator translates available cheap-column screens into the SuperGLUE+AoA or cheap7 burden required to reach Overall 41.80. It does not evaluate models.\n")
    lines.append("## Required cheap7 for Overall 41.80\n")
    for r in threshold_rows:
        lines.append(f"- If SuperGLUE+AoA = {r['assumed_SuperGLUE_plus_AoA']:.2f}, required cheap7 = {r['required_cheap7']:.4f}.")
    lines.append(f"\nVisible leader cheap7 = {cheap7(VISIBLE_LEADER):.4f} with SuperGLUE+AoA = {VISIBLE_LEADER['SuperGLUE'] + VISIBLE_LEADER['AoA']:.2f}. Known complete local vectors with full columns have SuperGLUE+AoA range {sg_plus_summary.get('min'):.2f}–{sg_plus_summary.get('max'):.2f}.\n")
    lines.append("## Available FW rows\n")
    for r in fw_rows:
        ch = r["cheap7"]
        if ch is None:
            lines.append(f"- {r['checkpoint']} {r['arm']}: only {r['scores_available']}/7 cheap columns; no cheap7 burden yet.")
        else:
            lines.append(f"- {r['checkpoint']} {r['arm']}: cheap7={ch:.4f}; would need SuperGLUE+AoA={r['needed_SuperGLUE_plus_AoA_for_41p80']:.2f} to reach 41.80 at that cheap7.")
    if partial_requirements:
        lines.append("\n## Partial breadth 70M arithmetic\n")
        for r in partial_requirements:
            if r["arm"] == "source_breadth_rowblock" and r["checkpoint"] == "70M" and abs(r["assumed_SuperGLUE_plus_AoA"] - 69.79) < 1e-6:
                lines.append(f"- With BLiMP+Supplement+EWoK already known and SuperGLUE+AoA like the visible leader, the missing four cheap columns would need mean {r['required_missing_mean']:.3f}; compare compact 70M's four-column sum Entity+COMPS+GlobalPIQA+Reading = {27.74+51.76+38.605+7.515:.3f}, mean {(27.74+51.76+38.605+7.515)/4:.3f}.")
    lines.append("\n## Use\n")
    lines.append("Do not launch full official evaluation merely from a partial 70M advantage. Once complete paired 80M or 100M cheap7 exists, use this file to decide whether full official evaluation/readouts can plausibly change the SOTA judgment. If 100M cheap7 is far below about 43.5–43.8 under plausible SuperGLUE+AoA, the arm cannot reach Overall 41.80 without an historically large SuperGLUE jump.")
    lines.append("\n## Evidence files\n")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- Threshold CSV: `{OUT / 'fw_sota_thresholds.csv'}`")
    lines.append(f"- FW rows CSV: `{OUT / 'fw_available_plausibility_rows.csv'}`")
    lines.append(f"- Partial requirements CSV: `{OUT / 'fw_partial_missing_requirements.csv'}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "leader_cheap7": out["visible_leader_cheap7"], "cheap7_needed_visible_sg": out["interpretation"]["cheap7_needed_if_superglue_like_visible_leader"], "json": str(json_path), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
