#!/usr/bin/env python3
"""research: interpret mature legal-tokenizer treatment patterns.

This CPU-only helper reads the cheap-column trajectory comparison and turns the
70M/80M reinvest-minus-clean deltas into a compact mechanism signature.  It does
not launch training and it does not make an endpoint choice.  Its role is to keep
future intervention choice aligned with the measured pattern:

- broad positive reinvestment effect: compact-view reinvestment survives the legal
  tokenizer coordinate; preserve the data mechanism and investigate legal
  representation/optimization rather than masking/objective changes.
- selective relation-sensitive weakness: relation-weighted masking is a plausible
  single-variable learning-signal test.
- broad disappearance: explicit source-view consistency becomes the more matched
  intervention family.

The helper waits for complete 70M and 80M comparison rows.  With missing rows it
writes a not-ready summary so the same command can be rerun unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_COMPARISON = STUDY / "data/legal_treatment_trajectory/legal_treatment_trajectory_comparison.json"
DEFAULT_OUT_DIR = STUDY / "data/legal_treatment_pattern"
MATURE_EXPOSURES = (70, 80)
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RELATION_COLUMNS = ["Supplement", "EWoK", "Entity"]
BROAD_COLUMNS = ["BLiMP", "COMPS", "GlobalPIQA", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(xs: list[float | None]) -> list[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def mean_or_none(xs: list[float | None]) -> float | None:
    ys = finite(xs)
    return sum(ys) / len(ys) if len(ys) == len(xs) and ys else None


def sample_sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def exposure_signature(row: dict[str, Any]) -> dict[str, Any]:
    deltas = {col: row.get(f"delta_{col}") for col in COLUMNS}
    relation_vals = finite([deltas[c] for c in RELATION_COLUMNS])
    broad_vals = finite([deltas[c] for c in BROAD_COLUMNS])
    all_vals = finite([deltas[c] for c in COLUMNS])
    complete = bool(row.get("complete_pair")) and len(all_vals) == len(COLUMNS)
    if complete:
        relation_mean = sum(relation_vals) / len(relation_vals)
        broad_mean = sum(broad_vals) / len(broad_vals)
        mean7 = sum(all_vals) / len(all_vals)
        pos_cols = sum(1 for v in all_vals if v > 0)
        neg_cols = sum(1 for v in all_vals if v < 0)
        relational_gap = relation_mean - broad_mean
    else:
        relation_mean = broad_mean = mean7 = relational_gap = None
        pos_cols = neg_cols = None
    return {
        "exposure_m": row.get("exposure_m"),
        "complete_pair": complete,
        "deltas": deltas,
        "mean7_delta": mean7,
        "positive_columns": pos_cols,
        "negative_columns": neg_cols,
        "relation_columns": RELATION_COLUMNS,
        "relation_mean_delta": relation_mean,
        "broad_columns": BROAD_COLUMNS,
        "broad_mean_delta": broad_mean,
        "relation_minus_broad_delta": relational_gap,
    }


def combined_pattern(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [s for s in signatures if s.get("complete_pair")]
    if len(complete) != len(MATURE_EXPOSURES):
        return {
            "ready": False,
            "reason": "70M and 80M complete reinvest-clean pairs are both required before interpreting the mature legal-tokenizer treatment pattern.",
            "missing_exposures": [m for m in MATURE_EXPOSURES if not any(s.get("complete_pair") and s.get("exposure_m") == m for s in signatures)],
        }
    mean7_vals = [float(s["mean7_delta"]) for s in complete]
    relation_vals = [float(s["relation_mean_delta"]) for s in complete]
    broad_vals = [float(s["broad_mean_delta"]) for s in complete]
    rel_gap_vals = [float(s["relation_minus_broad_delta"]) for s in complete]
    pos_cols_vals = [int(s["positive_columns"]) for s in complete]
    per_col_means = {}
    per_col_by_exp = {}
    for col in COLUMNS:
        vals = [float(s["deltas"][col]) for s in complete]
        per_col_by_exp[col] = vals
        per_col_means[col] = sum(vals) / len(vals)

    mean7_avg = sum(mean7_vals) / len(mean7_vals)
    relation_avg = sum(relation_vals) / len(relation_vals)
    broad_avg = sum(broad_vals) / len(broad_vals)
    rel_gap_avg = sum(rel_gap_vals) / len(rel_gap_vals)
    min_pos_cols = min(pos_cols_vals)
    max_abs_mean7_change = max(abs(mean7_vals[i] - mean7_vals[i - 1]) for i in range(1, len(mean7_vals))) if len(mean7_vals) > 1 else 0.0

    # Non-binding scientific pattern labels.  Thresholds are intentionally small
    # relative to BabyLM column noise; the output records all measured numbers so
    # interpretation can use the measured values rather than the label alone.
    if mean7_avg >= 0.35 and min_pos_cols >= 5 and broad_avg >= 0.20:
        pattern = "broad_positive_reinvestment_survival"
        matched_intervention_family = "preserve compact-view reinvestment; investigate legal representation or optimization, coordinated with A01's representation route"
    elif mean7_avg > -0.10 and broad_avg >= 0.10 and relation_avg < broad_avg - 0.40:
        pattern = "selective_relation_sensitive_weakness"
        matched_intervention_family = "single-variable relation-weighted masking screen, preferably relation_only_v1 after full static-prior budget verification"
    elif mean7_avg <= 0.10 and min_pos_cols <= 3:
        pattern = "broad_reinvestment_disappearance"
        matched_intervention_family = "explicit source-view consistency construction and matched trajectory, kept separate from static-prior masking"
    else:
        pattern = "mixed_mature_pattern"
        matched_intervention_family = "inspect the per-column evidence and the corresponding representation comparison before choosing a new training experiment"

    return {
        "ready": True,
        "mature_exposures": list(MATURE_EXPOSURES),
        "mean7_delta_average": mean7_avg,
        "mean7_delta_by_exposure": dict(zip([str(m) for m in MATURE_EXPOSURES], mean7_vals)),
        "mean7_delta_sd": sample_sd(mean7_vals),
        "relation_mean_delta_average": relation_avg,
        "broad_mean_delta_average": broad_avg,
        "relation_minus_broad_delta_average": rel_gap_avg,
        "positive_columns_by_exposure": dict(zip([str(m) for m in MATURE_EXPOSURES], pos_cols_vals)),
        "per_column_mean_deltas": per_col_means,
        "per_column_deltas_by_exposure_order": per_col_by_exp,
        "max_abs_mean7_change_between_mature_exposures": max_abs_mean7_change,
        "pattern_label": pattern,
        "matched_intervention_family": matched_intervention_family,
        "interpretation_limits": [
            "This is cheap-column evidence only: SuperGLUE and AoA are absent.",
            "Clean-Qwen uses the reinvest-trained legal tokenizer as a scientific control and is not an independently valid Strict-Small endpoint.",
            "The label is a compact summary of the measured deltas, not a substitute for reading the per-column numbers and training/evaluation logs.",
        ],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-json", default=str(DEFAULT_COMPARISON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path = pathlib.Path(args.comparison_json)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        rows = comparison.get("rows", [])
    else:
        comparison = None
        rows = []
    mature_rows = [r for r in rows if int(r.get("exposure_m", -1)) in MATURE_EXPOSURES]
    signatures = [exposure_signature(r) for r in mature_rows]
    pattern = combined_pattern(signatures)
    summary = {
        "status": "LEGAL_TREATMENT_PATTERN",
        "created_utc": now_utc(),
        "purpose": "Quantify the mature 70M/80M legal-tokenizer treatment pattern so the next intervention matches the observed mechanism rather than the easiest available implementation.",
        "inputs": {
            "comparison_json": str(comparison_path),
            "comparison_exists": comparison_path.exists(),
            "mature_exposures": list(MATURE_EXPOSURES),
            "columns": COLUMNS,
            "relation_columns": RELATION_COLUMNS,
            "broad_columns": BROAD_COLUMNS,
        },
        "comparison_missing": comparison.get("missing") if isinstance(comparison, dict) else None,
        "mature_signatures": signatures,
        "pattern": pattern,
    }
    out_json = out_dir / "legal_treatment_pattern.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research legal-tokenizer treatment pattern",
        "",
        summary["purpose"],
        "",
        f"Comparison file: `{comparison_path}` exists={comparison_path.exists()}",
        f"Comparison missing entries: `{summary['comparison_missing']}`",
        "",
        "## Mature exposure signatures",
        "",
        "| exposure | complete | mean7 Δ | relation Δ | broad Δ | relation-broad Δ | positive cols |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for sig in signatures:
        def fmt(v: Any) -> str:
            return "NA" if v is None else f"{float(v):.4f}"
        lines.append(
            f"| {sig.get('exposure_m')} | {sig.get('complete_pair')} | {fmt(sig.get('mean7_delta'))} | {fmt(sig.get('relation_mean_delta'))} | {fmt(sig.get('broad_mean_delta'))} | {fmt(sig.get('relation_minus_broad_delta'))} | {sig.get('positive_columns')} |"
        )
    if not signatures:
        lines.append("| NA | False | NA | NA | NA | NA | NA |")
    lines += ["", "## Pattern summary", ""]
    for k, v in pattern.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md = out_dir / "legal_treatment_pattern.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "ready": pattern.get("ready"),
        "pattern_label": pattern.get("pattern_label"),
        "missing_exposures": pattern.get("missing_exposures"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
