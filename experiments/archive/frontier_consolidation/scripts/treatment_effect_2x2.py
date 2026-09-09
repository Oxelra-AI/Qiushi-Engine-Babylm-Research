#!/usr/bin/env python3
"""research: full-surface 2x2 treatment (compact_view_reinvest vs clean-Qwen) x seed (43022/43122)
difference-in-differences framework.

do not equate seed43122 failing to clear 41.8 with failure of the
learning principle. Compare reinvestment against clean-Qwen *within each matched seed* across the
full surface. The absolute score measures endpoint robustness; the within-seed treatment effect
(reinvest - clean, same seed) reveals whether compact views + reinvested diversity replicate
despite inherited seed variance.

This script builds the 2x2 with:
  cell A = clean-Qwen seed43022      (KNOWN full official surface)
  cell B = clean-Qwen seed43122      (KNOWN full official surface)
  cell C = reinvest seed43022        (KNOWN official coordinate)
  cell D = reinvest seed43122        (PENDING full official vector; fillable parameter)

Within-seed treatment effects:
  TE(43022) = C - A   (KNOWN)
  TE(43122) = D - B   (partly known via fast columns; full via D when delivered)

DiD:
  DiD = TE(43122) - TE(43022) = (D - B) - (C - A)
  DiD ~ 0  => treatment replicates across seeds (learning principle robust)
  DiD < 0  => treatment effect is smaller/negative at seed43122 (seed-specific fragility)

We also provide a fast-consistent preliminary TE(43122) using the reinvest fast vectors for
both seeds, so an early replication read is possible before the full seed43122 vector arrives.
"""
from __future__ import annotations

import json
import pathlib
import statistics

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = STUDY / "data" / "treatment_effect_2x2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Official nine-column set. Overall is the mean of these nine.
COLS9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
COLS7_NOAOA_NOSG = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def overall9(v: dict) -> float:
    return statistics.mean([v[c] for c in COLS9])


# ---- KNOWN FULL OFFICIAL SURFACES ------------------------------------------------
# clean-Qwen full official 100M (from COMPACT_EXPERIENCE research official_100M_surface, echoed in
# clean_full_trajectory_seed_control.json).
CLEAN_43022 = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "SuperGLUE": 70.30861598316157, "GlobalPIQA": 36.62,
    "Reading": 7.76, "AoA": 0.0,
}
CLEAN_43122 = {
    "BLiMP": 66.02, "Supplement": 61.51, "EWoK": 50.43, "Entity": 25.26,
    "COMPS": 52.06, "SuperGLUE": 68.74546760701206, "GlobalPIQA": 34.62,
    "Reading": 7.205, "AoA": 0.0,
}
# reinvest seed43022 official coordinate (pristine collation).
REINVEST_43022 = {
    "BLiMP": 66.87232315173485, "Supplement": 63.27576417952158,
    "EWoK": 53.536575594886855, "Entity": 27.745741097952372,
    "COMPS": 51.968828052457084, "SuperGLUE": 71.03604952825312,
    "GlobalPIQA": 35.62135922330097, "Reading": 8.241572282566393, "AoA": 0.0,
}

# ---- FAST VECTORS (for preliminary, fast-consistent TE(43122)) -------------------
# reinvest fast vectors (7-col, no SuperGLUE/AoA). SuperGLUE not in fast screen.
REINVEST_43022_FAST = {
    "BLiMP": 66.63, "Supplement": 66.4, "EWoK": 53.09, "Entity": 27.75,
    "COMPS": 51.97, "GlobalPIQA": 35.62, "Reading": 8.24,
}
REINVEST_43122_FAST = {
    "BLiMP": 65.75, "Supplement": 63.2, "EWoK": 49.36, "Entity": 26.29,
    "COMPS": 51.54, "GlobalPIQA": 35.135, "Reading": 8.865,
}
# clean-Qwen fast vectors are not directly stored per column here; the COMPACT_EXPERIENCE full-surface
# clean vectors are the closest available for the same seeds. We use full clean vectors for
# clean cells and clearly mark the fast/full mismatch for reinvest cells in the preliminary read.

# Optional override: supply the reinvest seed43122 full official vector when available.
REINVEST_43122_FULL = None  # e.g. {"BLiMP":..., "SuperGLUE":..., "AoA":0.0, ...}


def within_seed_te(reinvest: dict, clean: dict, cols: list[str]) -> dict:
    return {c: reinvest[c] - clean[c] for c in cols if c in reinvest and c in clean}


def main() -> None:
    result: dict = {
        "status": "TREATMENT_EFFECT_2x2",
        "purpose": "Full-surface 2x2 treatment x seed difference-in-differences for compact_view_reinvest vs clean-Qwen.",
        "official_nine_columns": COLS9,
        "cells": {
            "clean_seed43022": CLEAN_43022,
            "clean_seed43122": CLEAN_43122,
            "reinvest_seed43022": REINVEST_43022,
            "reinvest_seed43122_full": REINVEST_43122_FULL,
        },
        "overall": {
            "clean_seed43022": overall9(CLEAN_43022),
            "clean_seed43122": overall9(CLEAN_43122),
            "reinvest_seed43022": overall9(REINVEST_43022),
        },
    }

    # Known within-seed treatment effect at seed43022 (full official).
    te_43022 = within_seed_te(REINVEST_43022, CLEAN_43022, COLS9)
    te_43022_overall = overall9(REINVEST_43022) - overall9(CLEAN_43022)
    result["treatment_effect_seed43022_full"] = {
        "per_column": te_43022,
        "overall": te_43022_overall,
        "note": "reinvest_seed43022 minus clean_seed43022, full official surface. This is the replicated treatment effect at the reference seed.",
    }

    # Inherited seed spread under clean (base recipe).
    clean_seed_gap = {c: CLEAN_43122[c] - CLEAN_43022[c] for c in COLS9}
    result["clean_seed_gap_43122_minus_43022"] = {
        "per_column": clean_seed_gap,
        "overall": overall9(CLEAN_43122) - overall9(CLEAN_43022),
        "note": "Inherited base-recipe seed spread. Broad negative spread means part of any reinvest seed43122 shortfall is not caused by the treatment.",
    }

    # Reinvest seed spread under treatment (fast columns only, until full arrives).
    reinvest_seed_gap_fast = {c: REINVEST_43122_FAST[c] - REINVEST_43022_FAST[c] for c in COLS7_NOAOA_NOSG}
    result["reinvest_seed_gap_43122_minus_43022_fast"] = {
        "per_column": reinvest_seed_gap_fast,
        "mean_7col": statistics.mean(reinvest_seed_gap_fast.values()),
        "note": "Reinvest seed spread on the 7-col fast screen (no SuperGLUE/AoA). SuperGLUE/AoA seed spread requires the full seed43122 vector.",
    }

    # Preliminary within-seed TE(43122) using fast reinvest vs full clean (same seed).
    # Mark the fast/full caveat: reinvest cells are fast, clean cells are full.
    te_43122_prelim = {c: REINVEST_43122_FAST[c] - CLEAN_43122[c] for c in COLS7_NOAOA_NOSG}
    te_43022_fastclean = {c: REINVEST_43022_FAST[c] - CLEAN_43022[c] for c in COLS7_NOAOA_NOSG}
    did_prelim = {c: te_43122_prelim[c] - te_43022_fastclean[c] for c in COLS7_NOAOA_NOSG}
    result["preliminary_did_7col_fast_reinvest_vs_full_clean"] = {
        "te_seed43022_7col": te_43022_fastclean,
        "te_seed43122_7col": te_43122_prelim,
        "did_seed43122_minus_seed43022": did_prelim,
        "did_mean_7col": statistics.mean(did_prelim.values()),
        "caveat": "reinvest cells use fast screen, clean cells use full official surface; treat as preliminary. The fast/full gap is column-dependent (e.g. Supplement fast overshoots full).",
    }

    # If full seed43122 reinvest vector is provided, compute exact full-surface DiD.
    if REINVEST_43122_FULL is not None:
        te_43122_full = within_seed_te(REINVEST_43122_FULL, CLEAN_43122, COLS9)
        te_43122_full_overall = overall9(REINVEST_43122_FULL) - overall9(CLEAN_43122)
        did_full = {c: te_43122_full[c] - te_43022[c] for c in COLS9}
        result["treatment_effect_seed43122_full"] = {
            "per_column": te_43122_full,
            "overall": te_43122_full_overall,
        }
        result["did_full_surface"] = {
            "per_column": did_full,
            "overall": te_43122_full_overall - te_43022_overall,
            "interpretation": {
                "replicates_if": "TE(43122) overall is positive and comparable to TE(43022) overall (~+0.689); DiD overall near zero.",
                "treatment_specific_loss_if": "DiD overall clearly negative, concentrated in specific columns (watch EWoK).",
            },
        }
        result["reinvest_seed43122_overall"] = overall9(REINVEST_43122_FULL)

    # Key numbers for narrative.
    result["headline"] = {
        "clean_seed_spread_overall": overall9(CLEAN_43122) - overall9(CLEAN_43022),
        "reinvest_TE_seed43022_overall": te_43022_overall,
        "reinvest_seed43022_overall": overall9(REINVEST_43022),
        "clean_seed43022_overall": overall9(CLEAN_43022),
        "clean_seed43122_overall": overall9(CLEAN_43122),
        "reasoning": (
            "The clean base recipe already loses %.3f Overall from seed43022 to seed43122. "
            "The reinvest treatment adds +%.3f Overall at seed43022. If the same +~0.69 treatment effect "
            "holds at seed43122, reinvest_seed43122 would land near clean_seed43122 + TE = %.3f + %.3f = %.3f Overall, "
            "which would be below 41.8 purely because of inherited seed variance, NOT because the learning principle failed."
        ) % (
            overall9(CLEAN_43122) - overall9(CLEAN_43022),
            te_43022_overall,
            overall9(CLEAN_43122), te_43022_overall,
            overall9(CLEAN_43122) + te_43022_overall,
        ),
    }

    out_json = OUT_DIR / "treatment_effect_2x2.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Markdown note.
    lines = []
    lines.append("# research — 2x2 treatment (reinvest vs clean) x seed (43022/43122) DiD\n\n")
    lines.append("absolute score = endpoint robustness; within-seed treatment effect = whether the learning principle replicates despite inherited seed variance.\n\n")
    lines.append("## Known full-surface Overall (nine-column mean)\n")
    lines.append(f"- clean_seed43022: {overall9(CLEAN_43022):.4f}\n")
    lines.append(f"- clean_seed43122: {overall9(CLEAN_43122):.4f}\n")
    lines.append(f"- reinvest_seed43022: {overall9(REINVEST_43022):.4f}\n")
    lines.append(f"- reinvest_seed43122: PENDING (A01 full official vector)\n\n")
    lines.append("## Inherited seed spread (clean base recipe)\n")
    lines.append(f"- Overall: {overall9(CLEAN_43122) - overall9(CLEAN_43022):+.4f}\n")
    for c in COLS9:
        lines.append(f"  - {c}: {clean_seed_gap[c]:+.3f}\n")
    lines.append("\n## Treatment effect at seed43022 (reinvest - clean, full official)\n")
    lines.append(f"- Overall: {te_43022_overall:+.4f}\n")
    for c in COLS9:
        lines.append(f"  - {c}: {te_43022[c]:+.3f}\n")
    lines.append("\n## Preliminary DiD (7-col fast reinvest vs full clean)\n")
    lines.append(f"- DiD mean (7-col): {statistics.mean(did_prelim.values()):+.4f}\n")
    for c in COLS7_NOAOA_NOSG:
        lines.append(f"  - {c}: TE43022={te_43022_fastclean[c]:+.3f}, TE43122={te_43122_prelim[c]:+.3f}, DiD={did_prelim[c]:+.3f}\n")
    lines.append("\n## Headline reasoning\n")
    lines.append(result["headline"]["reasoning"] + "\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "treatment_effect_2x2.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "clean_seed_spread_overall": overall9(CLEAN_43122) - overall9(CLEAN_43022),
        "reinvest_TE_seed43022_overall": te_43022_overall,
        "projected_reinvest_43122_if_TE_replicates": overall9(CLEAN_43122) + te_43022_overall,
        "did_mean_7col_prelim": statistics.mean(did_prelim.values()),
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
