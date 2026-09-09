#!/usr/bin/env python3
"""research: update seed43122 projection using delivered official-coordinate EWoK and AoA repairs.

This does NOT treat the full seed43122 vector as delivered. It uses:
  - seed43122 fast/no-AoA seven columns from research;
  - delivered official 7618-row EWoK re-eval for seed43122;
  - delivered official min_context=0 AoA=0 for seed43122, independently confirmed.

Purpose: correct stale fast-based projections and quantify what remains unknown before the full
official seed43122 vector arrives.
"""
from __future__ import annotations

import json
import pathlib
import statistics

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = STUDY / "data/seed43122_official_repairs_projection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VISIBLE_LEADER = 41.8
COLS9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
COLS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

# Known official coordinates / baselines.
CLEAN_43122 = {
    "BLiMP": 66.02, "Supplement": 61.51, "EWoK": 50.43, "Entity": 25.26,
    "COMPS": 52.06, "SuperGLUE": 68.74546760701206, "GlobalPIQA": 34.62,
    "Reading": 7.205, "AoA": 0.0,
}
CLEAN_43022 = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "SuperGLUE": 70.30861598316157, "GlobalPIQA": 36.62,
    "Reading": 7.76, "AoA": 0.0,
}
REINVEST_43022 = {
    "BLiMP": 66.87232315173485, "Supplement": 63.27576417952158,
    "EWoK": 53.536575594886855, "Entity": 27.745741097952372,
    "COMPS": 51.968828052457084, "SuperGLUE": 71.03604952825312,
    "GlobalPIQA": 35.62135922330097, "Reading": 8.241572282566393,
    "AoA": 0.0,
}
# research fast values for seed43122 reinvest; official EWoK/AoA repaired below.
REINVEST_43122_FAST = {
    "BLiMP": 65.75,
    "Supplement": 63.2,
    "EWoK": 49.36,
    "Entity": 26.29,
    "COMPS": 51.54,
    "GlobalPIQA": 35.135,
    "Reading": 8.865,
}

REPAIRS = ROOT / "experiments/archive/representation_and_objectives/data/seed43122_official_coordinate_repairs/seed43122_official_coordinate_repairs_summary.json"

def mean(vals):
    return statistics.mean(vals)

def main():
    repairs = json.loads(REPAIRS.read_text(encoding="utf-8")) if REPAIRS.exists() else {}
    ewok_official = float(repairs.get("outputs", {}).get("ewok", {}).get("score", 51.89))
    aoa_score = float(repairs.get("outputs", {}).get("aoa", {}).get("aoa_leaderboard_score", 0.0))
    aoa_rows = repairs.get("outputs", {}).get("aoa", {}).get("row_count_values")
    ewok_items = repairs.get("outputs", {}).get("ewok", {}).get("prediction_item_total")

    hybrid = dict(REINVEST_43122_FAST)
    hybrid["EWoK"] = ewok_official
    hybrid["AoA"] = aoa_score
    seven_sum = sum(hybrid[c] for c in COLS7)
    required_sg_for_41p8 = 9 * VISIBLE_LEADER - seven_sum - aoa_score
    scenarios = {}
    sg_refs = {
        "clean_seed43122_superglue": CLEAN_43122["SuperGLUE"],
        "clean_seed43022_superglue": CLEAN_43022["SuperGLUE"],
        "reinvest_seed43022_superglue": REINVEST_43022["SuperGLUE"],
        "compact_repeat_core_superglue": 70.62444886622056,
        "compact_view_core_superglue": 68.9012,
        "visible_leader_superglue": 69.79,
        "required_for_41p8": required_sg_for_41p8,
    }
    for name, sg in sg_refs.items():
        full = dict(hybrid)
        full["SuperGLUE"] = sg
        scenarios[name] = {
            "SuperGLUE": sg,
            "Overall": mean([full[c] for c in COLS9]),
            "margin_over_41p8": mean([full[c] for c in COLS9]) - VISIBLE_LEADER,
            "treatment_effect_vs_clean43122": mean([full[c] for c in COLS9]) - mean([CLEAN_43122[c] for c in COLS9]),
        }

    # EWoK DiD with official repair.
    ewok_te430 = REINVEST_43022["EWoK"] - CLEAN_43022["EWoK"]
    ewok_te431_repaired = ewok_official - CLEAN_43122["EWoK"]
    ewok_did_repaired = ewok_te431_repaired - ewok_te430
    ewok_fast_did_old = (REINVEST_43122_FAST["EWoK"] - CLEAN_43122["EWoK"]) - (53.09 - CLEAN_43022["EWoK"])

    # Unknown budget: if all fast non-EWoK cols hold and official EWoK/AoA fixed, what extra sum in BLiMP/Supp/Entity/COMPS/GPIQA/Reading/SG is needed?
    result = {
        "status": "SEED43122_OFFICIAL_REPAIRS_PROJECTION",
        "purpose": "Update seed43122 projection after official EWoK=7618-row and AoA=8005-row repairs, without inferring the full locked vector.",
        "sources": {
            "fast_projection": "experiments/archive/frontier_consolidation/data/seed43122_fast_projection/seed43122_fast_projection.json",
            "a01_step43_repairs": str(REPAIRS),
        },
        "delivered_official_repairs": {
            "EWoK": ewok_official,
            "EWoK_prediction_item_total": ewok_items,
            "AoA": aoa_score,
            "AoA_row_count_values": aoa_rows,
        },
        "seed43122_hybrid_vector_fast_with_official_EWoK_AoA": hybrid,
        "seven_sum_excluding_superglue_but_including_repaired_EWoK_excluding_AoA": seven_sum,
        "required_superglue_for_absolute_41p8_if_other_fast_columns_hold": required_sg_for_41p8,
        "superglue_scenarios": scenarios,
        "official_EWoK_treatment_effect_and_did": {
            "TE43022_EWoK_full_official": ewok_te430,
            "TE43122_EWoK_with_official_repair": ewok_te431_repaired,
            "DiD_EWoK_with_official_repair": ewok_did_repaired,
            "old_fast_based_DiD_EWoK": ewok_fast_did_old,
            "interpretation": "The official seed43122 EWoK repair materially weakens the apparent treatment-specific EWoK loss relative to the research fast estimate, but EWoK still does not replicate the large seed43022 treatment gain.",
        },
        "treatment_effect_thresholds": {
            "clean43122_overall": mean([CLEAN_43122[c] for c in COLS9]),
            "TE43022_overall_full_official": mean([REINVEST_43022[c] for c in COLS9]) - mean([CLEAN_43022[c] for c in COLS9]),
            "reinvest43122_if_TE43022_replicates": mean([CLEAN_43122[c] for c in COLS9]) + (mean([REINVEST_43022[c] for c in COLS9]) - mean([CLEAN_43022[c] for c in COLS9])),
            "TE43122_required_for_absolute_41p8": VISIBLE_LEADER - mean([CLEAN_43122[c] for c in COLS9]),
            "extra_TE_above_TE43022_required_for_absolute_41p8": (VISIBLE_LEADER - mean([CLEAN_43122[c] for c in COLS9])) - (mean([REINVEST_43022[c] for c in COLS9]) - mean([CLEAN_43022[c] for c in COLS9])),
        },
        "not_yet_known": [
            "seed43122 full official BLiMP/Supplement/Entity/COMPS/GlobalPIQA/Reading after the locked full run is staged",
            "seed43122 full official SuperGLUE primary-metric mean",
            "full-surface TE43122 and DiD over all nine columns",
        ],
    }
    out_json = OUT_DIR / "seed43122_official_repairs_projection.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = []
    lines.append("# research — seed43122 projection after official EWoK/AoA repairs\n\n")
    lines.append("This does not treat the full seed43122 official vector as delivered. It updates only the pieces that are now official-coordinate: EWoK and AoA.\n\n")
    lines.append("## Delivered official-coordinate repairs\n")
    lines.append(f"- EWoK: {ewok_official:.2f} over {ewok_items} items\n")
    lines.append(f"- AoA: {aoa_score:.2f}, row_count_values={aoa_rows}\n\n")
    lines.append("## Absolute 41.8 projection with fast non-EWoK columns held fixed\n")
    lines.append(f"- Seven-column sum (BLiMP/Supp/EWoK/Entity/COMPS/GPIQA/Reading): {seven_sum:.3f}\n")
    lines.append(f"- SuperGLUE required for Overall 41.8 if AoA=0: {required_sg_for_41p8:.3f}\n")
    for name, rec in scenarios.items():
        lines.append(f"  - {name}: SG={rec['SuperGLUE']:.3f}, Overall={rec['Overall']:.4f}, margin={rec['margin_over_41p8']:+.4f}, TE_vs_clean43122={rec['treatment_effect_vs_clean43122']:+.4f}\n")
    lines.append("\n## EWoK treatment-effect update\n")
    lines.append(f"- TE43022 EWoK full official: {ewok_te430:+.3f}\n")
    lines.append(f"- TE43122 EWoK with official repair: {ewok_te431_repaired:+.3f}\n")
    lines.append(f"- DiD EWoK with official repair: {ewok_did_repaired:+.3f}\n")
    lines.append(f"- Old fast-based EWoK DiD: {ewok_fast_did_old:+.3f}\n")
    lines.append("\nThe EWoK official repair narrows the apparent seed43122 treatment-specific EWoK loss, but it remains the main non-replicating column in the preliminary treatment-effect picture.\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "seed43122_official_repairs_projection.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "ewok_official": ewok_official,
        "aoa": aoa_score,
        "required_superglue_for_41p8_if_fast_cols_hold": required_sg_for_41p8,
        "ewok_did_repaired": ewok_did_repaired,
        "old_fast_ewok_did": ewok_fast_did_old,
        "out_json": str(out_json),
    }, indent=2))

if __name__ == "__main__":
    main()
