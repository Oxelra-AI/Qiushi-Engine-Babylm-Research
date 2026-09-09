#!/usr/bin/env python3
"""research: apply a delivered compact_view_reinvest seed43122 official vector to the 2x2 analysis.

Input: pristine seed43122 staged official-coordinate summary, expected at:
  experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json

Output: A full-surface treatment x seed comparison:
  clean-Qwen seed43022, clean-Qwen seed43122, reinvest seed43022, reinvest seed43122

Scientific purpose:
  The absolute seed43122 score measures endpoint robustness.
  The matched within-seed effect reinvest_seed43122 - clean_seed43122 measures whether
  compact views plus reinvested diversity replicate despite inherited clean-Qwen seed spread.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = STUDY / "data" / "seed43122_2x2_application"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLS9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA"]
HUMAN_COLS = ["Reading", "AoA"]
VISIBLE_LEADER = 41.8

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
REINVEST_43022 = {
    "BLiMP": 66.87232315173485, "Supplement": 63.27576417952158,
    "EWoK": 53.536575594886855, "Entity": 27.745741097952372,
    "COMPS": 51.968828052457084, "SuperGLUE": 71.03604952825312,
    "GlobalPIQA": 35.62135922330097, "Reading": 8.241572282566393, "AoA": 0.0,
}


def mean_cols(v: dict[str, float], cols: list[str]) -> float:
    return statistics.mean(float(v[c]) for c in cols)


def extract_scores(summary: dict[str, Any]) -> dict[str, float]:
    """Handle research/research schemas and likely variants."""
    candidates = [
        summary.get("score_summary", {}).get("official_overall", {}).get("scores"),
        summary.get("score_summary", {}).get("scores"),
        summary.get("official_overall", {}).get("scores"),
        summary.get("scores"),
    ]
    for cand in candidates:
        if isinstance(cand, dict) and all(k in cand for k in COLS9):
            return {k: float(cand[k]) for k in COLS9}
    # Some records flatten the official row directly.
    if all(k in summary for k in COLS9):
        return {k: float(summary[k]) for k in COLS9}
    raise KeyError("Could not locate all nine official columns in summary")


def extract_validation(summary: dict[str, Any]) -> dict[str, Any]:
    coll = summary.get("collated_summary", {}) if isinstance(summary.get("collated_summary"), dict) else {}
    score = summary.get("score_summary", {}) if isinstance(summary.get("score_summary"), dict) else {}
    official = score.get("official_overall", score) if isinstance(score, dict) else {}
    return {
        "status": summary.get("status"),
        "collate_returncode": summary.get("collate_run", {}).get("returncode") if isinstance(summary.get("collate_run"), dict) else None,
        "null_keys": coll.get("null_keys"),
        "ewok_total": sum(coll.get("ewok_lengths", {}).values()) if isinstance(coll.get("ewok_lengths"), dict) else None,
        "aoa_row_count_values": coll.get("aoa_surprisal_row_count_values"),
        "reported_overall": official.get("Overall") if isinstance(official, dict) else None,
        "margin_over_visible_leader_41p8": official.get("margin_over_visible_leader_41p8") or official.get("margin_over_visible_leader") if isinstance(official, dict) else None,
    }


def diff(a: dict[str, float], b: dict[str, float], cols: list[str]) -> dict[str, float]:
    return {c: float(a[c]) - float(b[c]) for c in cols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed43122-summary", default="experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json")
    ap.add_argument("--allow-missing", action="store_true", help="write a waiting record if the summary is not present")
    args = ap.parse_args()

    seed43122_path = ROOT / args.seed43122_summary
    base: dict[str, Any] = {
        "status": "SEED43122_2x2_APPLICATION",
        "seed43122_summary_path": str(seed43122_path.relative_to(ROOT) if seed43122_path.is_absolute() and ROOT in seed43122_path.parents else args.seed43122_summary),
        "purpose": "Apply delivered seed43122 official vector to matched within-seed treatment-effect analysis.",
        "official_nine_columns": COLS9,
        "known_cells": {
            "clean_seed43022": CLEAN_43022,
            "clean_seed43122": CLEAN_43122,
            "reinvest_seed43022": REINVEST_43022,
        },
        "known_overalls": {
            "clean_seed43022": mean_cols(CLEAN_43022, COLS9),
            "clean_seed43122": mean_cols(CLEAN_43122, COLS9),
            "reinvest_seed43022": mean_cols(REINVEST_43022, COLS9),
        },
    }

    if not seed43122_path.exists():
        base["status"] = "SEED43122_2x2_APPLICATION_AWAITING_VECTOR"
        base["missing"] = str(seed43122_path)
        base["why_missing_matters"] = "Official seed43122 vector is required for the full-surface within-seed treatment effect TE43122 = reinvest43122 - clean43122."
        base["reference_calculations"] = {
            "clean_seed_spread_43122_minus_43022_overall": mean_cols(CLEAN_43122, COLS9) - mean_cols(CLEAN_43022, COLS9),
            "te_seed43022_full_overall": mean_cols(REINVEST_43022, COLS9) - mean_cols(CLEAN_43022, COLS9),
            "projected_reinvest43122_if_te43022_replicates": mean_cols(CLEAN_43122, COLS9) + (mean_cols(REINVEST_43022, COLS9) - mean_cols(CLEAN_43022, COLS9)),
            "te43122_required_for_absolute_41p8": VISIBLE_LEADER - mean_cols(CLEAN_43122, COLS9),
        }
        out_json = OUT_DIR / "seed43122_2x2_application_awaiting.json"
        out_json.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": base["status"], "out_json": str(out_json)}, indent=2))
        if not args.allow_missing:
            raise SystemExit(2)
        return

    summary = json.loads(seed43122_path.read_text(encoding="utf-8"))
    r431 = extract_scores(summary)
    validation = extract_validation(summary)
    te430 = diff(REINVEST_43022, CLEAN_43022, COLS9)
    te431 = diff(r431, CLEAN_43122, COLS9)
    did = {c: te431[c] - te430[c] for c in COLS9}
    clean_seed_gap = diff(CLEAN_43122, CLEAN_43022, COLS9)
    reinvest_seed_gap = diff(r431, REINVEST_43022, COLS9)
    overall_clean_gap = mean_cols(CLEAN_43122, COLS9) - mean_cols(CLEAN_43022, COLS9)
    overall_r_gap = mean_cols(r431, COLS9) - mean_cols(REINVEST_43022, COLS9)
    overall_te430 = mean_cols(REINVEST_43022, COLS9) - mean_cols(CLEAN_43022, COLS9)
    overall_te431 = mean_cols(r431, COLS9) - mean_cols(CLEAN_43122, COLS9)
    result = base | {
        "reinvest_seed43122": r431,
        "validation": validation,
        "overalls": {
            **base["known_overalls"],
            "reinvest_seed43122": mean_cols(r431, COLS9),
        },
        "absolute_endpoint": {
            "reinvest_seed43022_margin_over_41p8": mean_cols(REINVEST_43022, COLS9) - VISIBLE_LEADER,
            "reinvest_seed43122_margin_over_41p8": mean_cols(r431, COLS9) - VISIBLE_LEADER,
        },
        "within_seed_treatment_effects": {
            "seed43022_reinvest_minus_clean": {"per_column": te430, "overall": overall_te430, "nlp_average": mean_cols(REINVEST_43022, NLP_COLS) - mean_cols(CLEAN_43022, NLP_COLS), "human_average": mean_cols(REINVEST_43022, HUMAN_COLS) - mean_cols(CLEAN_43022, HUMAN_COLS)},
            "seed43122_reinvest_minus_clean": {"per_column": te431, "overall": overall_te431, "nlp_average": mean_cols(r431, NLP_COLS) - mean_cols(CLEAN_43122, NLP_COLS), "human_average": mean_cols(r431, HUMAN_COLS) - mean_cols(CLEAN_43122, HUMAN_COLS)},
        },
        "seed_spread": {
            "clean_43122_minus_43022": {"per_column": clean_seed_gap, "overall": overall_clean_gap},
            "reinvest_43122_minus_43022": {"per_column": reinvest_seed_gap, "overall": overall_r_gap},
        },
        "difference_in_differences": {
            "per_column": did,
            "overall": overall_te431 - overall_te430,
            "nlp_average": (mean_cols(r431, NLP_COLS) - mean_cols(CLEAN_43122, NLP_COLS)) - (mean_cols(REINVEST_43022, NLP_COLS) - mean_cols(CLEAN_43022, NLP_COLS)),
            "human_average": (mean_cols(r431, HUMAN_COLS) - mean_cols(CLEAN_43122, HUMAN_COLS)) - (mean_cols(REINVEST_43022, HUMAN_COLS) - mean_cols(CLEAN_43022, HUMAN_COLS)),
        },
        "interpretation_numbers": {
            "positive_TE43122_means_reinvest_beats_matched_clean_seed43122": overall_te431,
            "TE43122_minus_TE43022_near_zero_means_replication_of_treatment_effect": overall_te431 - overall_te430,
            "absolute_41p8_requires_TE43122": VISIBLE_LEADER - mean_cols(CLEAN_43122, COLS9),
            "extra_TE43122_above_TE43022_needed_to_clear_41p8": (VISIBLE_LEADER - mean_cols(CLEAN_43122, COLS9)) - overall_te430,
        },
    }
    out_json = OUT_DIR / "seed43122_2x2_application.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # concise markdown summary
    lines = []
    lines.append("# research — seed43122 official vector applied to 2x2 treatment analysis\n\n")
    lines.append(f"Seed43122 staged summary: `{args.seed43122_summary}`\n\n")
    lines.append("## Overall\n")
    for name, val in result["overalls"].items():
        lines.append(f"- {name}: {val:.4f}\n")
    lines.append("\n## Within-seed treatment effect (reinvest - clean)\n")
    lines.append(f"- seed43022: {overall_te430:+.4f}\n")
    lines.append(f"- seed43122: {overall_te431:+.4f}\n")
    lines.append(f"- DiD: {overall_te431 - overall_te430:+.4f}\n")
    lines.append("\n## Component DiD\n")
    for c in COLS9:
        lines.append(f"- {c}: TE43022={te430[c]:+.3f}, TE43122={te431[c]:+.3f}, DiD={did[c]:+.3f}\n")
    lines.append("\n## Interpretation numbers\n")
    lines.append(f"- clean seed spread 43122-43022: {overall_clean_gap:+.4f}\n")
    lines.append(f"- reinvest seed spread 43122-43022: {overall_r_gap:+.4f}\n")
    lines.append(f"- absolute seed43122 margin over 41.8: {mean_cols(r431, COLS9)-VISIBLE_LEADER:+.4f}\n")
    lines.append(f"- absolute 41.8 would require TE43122={VISIBLE_LEADER - mean_cols(CLEAN_43122, COLS9):+.4f}, i.e. {((VISIBLE_LEADER - mean_cols(CLEAN_43122, COLS9)) - overall_te430):+.4f} above the seed43022 treatment effect.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    (OUT_DIR / "seed43122_2x2_application.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "reinvest_seed43122_overall": mean_cols(r431, COLS9),
        "te43122_overall": overall_te431,
        "did_overall": overall_te431 - overall_te430,
        "margin_over_41p8": mean_cols(r431, COLS9)-VISIBLE_LEADER,
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
