#!/usr/bin/env python3
"""research: stronger causal GPT compact-vs-repeat selected-trajectory comparator.

Compares the six selected official-compatible cheap7 checkpoints by endpoint name, not
raw charged-word values, and summarizes breadth/stability so compact-view transfer is
not inferred from one volatile column or one checkpoint. This script consumes only
completed selected-evaluation JSON files; it does not evaluate models.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from statistics import mean, pstdev
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CHEAP6_NO_GLOBAL = [c for c in CHEAP_COLUMNS if c != "GlobalPIQA"]
CHEAP5_NO_GLOBAL_READING = [c for c in CHEAP_COLUMNS if c not in {"GlobalPIQA", "Reading"}]
SYNTAX_SURFACE = ["BLiMP", "Supplement", "COMPS"]
RELATION_STATE = ["EWoK", "Entity"]
VOLATILE_SMALL = ["GlobalPIQA", "Reading"]
EFFECTIVE_TOKEN_REL_DELTA = 0.002216046994266418  # research/172 compact active positions per epoch vs repeat.


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            data = data["rows"]
        elif isinstance(data.get("trajectory"), list):
            data = data["trajectory"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like selected trajectory in {path}")
    return [r for r in data if r.get("cheap7") is not None and not r.get("error")]


def avg_cols(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [row.get(c) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def fmt(v: Any, nd: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{float(v):.{nd}f}"
    return str(v)


def positive_share(row: dict[str, Any]) -> float | None:
    deltas = [row.get(f"delta_{c}") for c in CHEAP_COLUMNS if row.get(f"delta_{c}") is not None]
    pos = [d for d in deltas if d > 0]
    if not pos:
        return 0.0
    return max(pos) / sum(pos) if sum(pos) != 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", type=pathlib.Path, required=True)
    ap.add_argument("--repeat", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--accounting-json", type=pathlib.Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    compact_rows = {str(r.get("endpoint")): r for r in read_rows(args.compact)}
    repeat_rows = {str(r.get("endpoint")): r for r in read_rows(args.repeat)}
    common_eps = sorted(set(compact_rows) & set(repeat_rows), key=lambda e: int(e.split("_")[1][:-1]) if e.startswith("chck_") and e.endswith("M") else 10**12)

    contrasts: list[dict[str, Any]] = []
    for ep in common_eps:
        c = compact_rows[ep]
        r = repeat_rows[ep]
        row: dict[str, Any] = {
            "endpoint": ep,
            "nominal_words": int(ep.split("_")[1][:-1]) * 1_000_000 if ep.startswith("chck_") and ep.endswith("M") else c.get("words"),
            "compact_cheap7": float(c["cheap7"]),
            "repeat_cheap7": float(r["cheap7"]),
            "delta_cheap7": float(c["cheap7"]) - float(r["cheap7"]),
        }
        for group_name, cols in [
            ("cheap6_no_global", CHEAP6_NO_GLOBAL),
            ("cheap5_no_global_reading", CHEAP5_NO_GLOBAL_READING),
            ("syntax_surface", SYNTAX_SURFACE),
            ("relation_state", RELATION_STATE),
            ("volatile_small", VOLATILE_SMALL),
        ]:
            ca = avg_cols(c, cols)
            ra = avg_cols(r, cols)
            row[f"compact_{group_name}"] = ca
            row[f"repeat_{group_name}"] = ra
            row[f"delta_{group_name}"] = ca - ra if ca is not None and ra is not None else None
        n_pos = 0
        n_neg = 0
        for col in CHEAP_COLUMNS:
            if c.get(col) is not None and r.get(col) is not None:
                d = float(c[col]) - float(r[col])
                row[f"delta_{col}"] = d
                if d > 0:
                    n_pos += 1
                elif d < 0:
                    n_neg += 1
        row["n_positive_columns"] = n_pos
        row["n_negative_columns"] = n_neg
        row["largest_positive_column_share"] = positive_share(row)
        row["volatile_contribution_fraction_of_delta"] = (
            row.get("delta_volatile_small") / row["delta_cheap7"] if row["delta_cheap7"] not in (0, None) and row.get("delta_volatile_small") is not None else None
        )
        contrasts.append(row)

    accounting: dict[str, Any] | None = None
    if args.accounting_json and args.accounting_json.exists():
        accounting = json.loads(args.accounting_json.read_text(encoding="utf-8"))

    deltas = [r["delta_cheap7"] for r in contrasts]
    pos_points = [r for r in contrasts if r["delta_cheap7"] > 0]
    broad_points = [r for r in contrasts if r["delta_cheap7"] > 0 and r["n_positive_columns"] >= 4]
    robust_points = [r for r in contrasts if r["delta_cheap6_no_global"] is not None and r["delta_cheap6_no_global"] > 0 and r["n_positive_columns"] >= 4]
    best = max(contrasts, key=lambda r: r["delta_cheap7"]) if contrasts else None
    worst = min(contrasts, key=lambda r: r["delta_cheap7"]) if contrasts else None

    summary = {
        "status": "CAUSAL_COMPACT_REPEAT_SELECTED_TRAJECTORY_CONTRAST" if contrasts else "NO_COMMON_ENDPOINTS",
        "compact_path": str(args.compact),
        "repeat_path": str(args.repeat),
        "accounting_json": str(args.accounting_json) if args.accounting_json else None,
        "n_common_endpoints": len(contrasts),
        "common_endpoints": common_eps,
        "effective_token_relative_delta_compact_minus_repeat": EFFECTIVE_TOKEN_REL_DELTA,
        "mean_delta_cheap7": mean(deltas) if deltas else None,
        "std_delta_cheap7": pstdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None,
        "n_positive_cheap7_endpoints": len(pos_points),
        "n_broad_positive_endpoints_pos4plus": len(broad_points),
        "n_robust_no_global_positive_endpoints": len(robust_points),
        "mean_positive_columns": mean([r["n_positive_columns"] for r in contrasts]) if contrasts else None,
        "mean_delta_cheap6_no_global": mean([r["delta_cheap6_no_global"] for r in contrasts if r.get("delta_cheap6_no_global") is not None]) if contrasts else None,
        "mean_delta_cheap5_no_global_reading": mean([r["delta_cheap5_no_global_reading"] for r in contrasts if r.get("delta_cheap5_no_global_reading") is not None]) if contrasts else None,
        "mean_delta_relation_state": mean([r["delta_relation_state"] for r in contrasts if r.get("delta_relation_state") is not None]) if contrasts else None,
        "mean_delta_syntax_surface": mean([r["delta_syntax_surface"] for r in contrasts if r.get("delta_syntax_surface") is not None]) if contrasts else None,
        "mean_delta_volatile_small": mean([r["delta_volatile_small"] for r in contrasts if r.get("delta_volatile_small") is not None]) if contrasts else None,
        "best_delta_endpoint": best,
        "worst_delta_endpoint": worst,
        "contrasts": contrasts,
        "interpretation_rules": [
            "Sustained compact advantage across several endpoints and at least four of seven columns is architecture-transfer evidence for compact semantic same-window views.",
            "A positive cheap7 delta that vanishes without GlobalPIQA/Reading or is dominated by one column is endpoint redistribution, not a transferable learning principle.",
            "The compact arm has only a +0.2216% active-token exposure advantage and higher training loss; official score deltas must be interpreted against this small implementation asymmetry, not against loss."
        ],
    }
    if accounting is not None:
        summary["training_accounting_compact_minus_repeat"] = accounting.get("comparisons", {}).get("compact_minus_repeat")

    out_json = args.out_dir / "causal_compact_repeat_selected_contrast.json"
    out_md = args.out_dir / "causal_compact_repeat_selected_contrast.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = ["# research causal GPT compact-vs-repeat selected contrast\n\n"]
    md.append(f"Common endpoints: {len(contrasts)}; compact active-token exposure asymmetry: +{EFFECTIVE_TOKEN_REL_DELTA*100:.4f}% relative to repeat.\n\n")
    if contrasts:
        md.append(f"Mean Δcheap7: {summary['mean_delta_cheap7']:+.6f}; positive cheap7 endpoints: {summary['n_positive_cheap7_endpoints']}/{len(contrasts)}; broad positive endpoints (Δcheap7>0 and ≥4 positive columns): {summary['n_broad_positive_endpoints_pos4plus']}/{len(contrasts)}.\n\n")
        md.append(f"Mean Δcheap6(no GlobalPIQA): {summary['mean_delta_cheap6_no_global']:+.6f}; mean Δcheap5(no GlobalPIQA/Reading): {summary['mean_delta_cheap5_no_global_reading']:+.6f}; mean Δrelation_state(EWoK+Entity): {summary['mean_delta_relation_state']:+.6f}.\n\n")
        md.append(f"Best endpoint: {best['endpoint']} Δcheap7={best['delta_cheap7']:+.6f}, +cols={best['n_positive_columns']}/7; worst endpoint: {worst['endpoint']} Δcheap7={worst['delta_cheap7']:+.6f}.\n\n")
        md.append("| endpoint | compact | repeat | Δcheap7 | Δcheap6 noG | Δcheap5 noG/R | Δrel/state | Δsyntax | Δvolatile | +cols | max +col share |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in contrasts:
            md.append(
                f"| {row['endpoint']} | {row['compact_cheap7']:.6f} | {row['repeat_cheap7']:.6f} | {row['delta_cheap7']:+.6f} | "
                f"{fmt(row.get('delta_cheap6_no_global'))} | {fmt(row.get('delta_cheap5_no_global_reading'))} | {fmt(row.get('delta_relation_state'))} | {fmt(row.get('delta_syntax_surface'))} | {fmt(row.get('delta_volatile_small'))} | "
                f"{row['n_positive_columns']} | {fmt(row.get('largest_positive_column_share'), 3)} |\n"
            )
    md.append("\nInterpretation: this comparison can support the compact-view principle only if compact wins across multiple checkpoints and families. A spike in GlobalPIQA/Reading or one grammaticality family remains a redistribution endpoint.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
