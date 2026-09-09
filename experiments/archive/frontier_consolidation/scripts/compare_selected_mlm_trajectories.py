#!/usr/bin/env python3
"""research: compare selected DeBERTa MLM checkpoint trajectories.

Consumes selected_trajectory.json files produced by selected_mlm_checkpoint_eval.py.
The purpose is scientific interpretation of the dense 70--100M grid: peak timing,
near-peak width, family movement, sensitivity to volatile columns, and pairwise
contrasts against the protected scale1.75 seed43022 reference.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from statistics import mean, pstdev
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
NO_GPIQA_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
REL_STATE_COLUMNS = ["EWoK", "Entity"]
SYNTAX_COMP_COLUMNS = ["BLiMP", "Supplement", "COMPS"]
VOLATILE_COLUMNS = ["GlobalPIQA", "Reading"]
WINDOWS = {
    "70_76M": (70_000_000, 76_000_000),
    "78_86M": (78_000_000, 86_000_000),
    "90_100M": (90_000_000, 100_000_000),
    "70_100M": (70_000_000, 100_000_000),
}


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def mean_cols(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [fnum(row.get(c)) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(mean(vals))


def words_m(row: dict[str, Any]) -> float:
    return float(row["words"]) / 1_000_000.0


def load_trajectory(path: pathlib.Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError(f"Expected list trajectory in {path}")
    rows: list[dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict) or fnum(raw.get("cheap7")) is None:
            continue
        row = dict(raw)
        row["cheap7"] = float(row["cheap7"])
        row["cheap6_no_GlobalPIQA"] = mean_cols(row, NO_GPIQA_COLUMNS)
        row["cheap5_no_GlobalPIQA_Reading"] = mean_cols(row, CORE_COLUMNS)
        row["relation_state_mean"] = mean_cols(row, REL_STATE_COLUMNS)
        row["syntax_comp_mean"] = mean_cols(row, SYNTAX_COMP_COLUMNS)
        row["volatile_mean"] = mean_cols(row, VOLATILE_COLUMNS)
        rows.append(row)
    return sorted(rows, key=lambda r: int(r["words"]))


def summarize(rows: list[dict[str, Any]], width_drop: float) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    best = max(rows, key=lambda r: r["cheap7"])
    near = [r for r in rows if r["cheap7"] >= best["cheap7"] - width_drop]
    final = max(rows, key=lambda r: int(r["words"]))
    windows: dict[str, Any] = {}
    for name, (lo, hi) in WINDOWS.items():
        inside = [r for r in rows if lo <= int(r["words"]) <= hi]
        win: dict[str, Any] = {"n": len(inside)}
        for key in ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean", "syntax_comp_mean", "volatile_mean"]:
            vals = [fnum(r.get(key)) for r in inside]
            vals = [v for v in vals if v is not None]
            win[f"mean_{key}"] = float(mean(vals)) if vals else None
            win[f"std_{key}"] = float(pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None
        windows[name] = win
    column_maxima: dict[str, Any] = {}
    peak_words = []
    for col in CHEAP_COLUMNS:
        valid = [r for r in rows if fnum(r.get(col)) is not None]
        if not valid:
            continue
        cbest = max(valid, key=lambda r: float(r[col]))
        column_maxima[col] = {"endpoint": cbest.get("endpoint"), "words": int(cbest["words"]), "score": float(cbest[col])}
        peak_words.append(int(cbest["words"]))
    return {
        "status": "ok",
        "n_rows": len(rows),
        "first_words": int(rows[0]["words"]),
        "last_words": int(rows[-1]["words"]),
        "best_endpoint": best.get("endpoint"),
        "best_words": int(best["words"]),
        "best_cheap7": best["cheap7"],
        "best_cheap6_no_GlobalPIQA": best.get("cheap6_no_GlobalPIQA"),
        "best_relation_state_mean": best.get("relation_state_mean"),
        "final_endpoint": final.get("endpoint"),
        "final_cheap7": final.get("cheap7"),
        "final_minus_best_cheap7": float(final["cheap7"] - best["cheap7"]),
        "near_best_drop": width_drop,
        "near_best_endpoints": [r.get("endpoint") for r in near],
        "near_best_span_m": (max(int(r["words"]) for r in near) - min(int(r["words"]) for r in near)) / 1_000_000.0 if near else None,
        "mean_cheap7": float(mean(r["cheap7"] for r in rows)),
        "std_cheap7": float(pstdev([r["cheap7"] for r in rows])) if len(rows) > 1 else 0.0,
        "windows": windows,
        "column_maxima": column_maxima,
        "column_peak_spread_m": (max(peak_words) - min(peak_words)) / 1_000_000.0 if peak_words else None,
        "rows": rows,
    }


def delta_row(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    out = {"words": int(row["words"]), "endpoint": row.get("endpoint"), "reference_endpoint": ref.get("endpoint")}
    for key in ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean", "syntax_comp_mean", "volatile_mean"] + CHEAP_COLUMNS:
        a = fnum(row.get(key)); b = fnum(ref.get(key))
        if a is not None and b is not None:
            out[f"delta_{key}"] = float(a - b)
    deltas = [out.get(f"delta_{c}") for c in CHEAP_COLUMNS if out.get(f"delta_{c}") is not None]
    out["n_positive_columns"] = sum(1 for d in deltas if float(d) > 0)
    out["n_negative_columns"] = sum(1 for d in deltas if float(d) < 0)
    pos_sum = sum(max(float(d), 0.0) for d in deltas)
    g = out.get("delta_GlobalPIQA")
    out["positive_gain_share_GlobalPIQA"] = (max(float(g), 0.0) / pos_sum) if g is not None and pos_sum > 0 else None
    return out


def summarize_comparison(rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref_by_words = {int(r["words"]): r for r in ref_rows}
    paired = [delta_row(r, ref_by_words[int(r["words"])]) for r in rows if int(r["words"]) in ref_by_words]
    if not paired:
        return {"status": "no_common_words"}
    best_delta = max(paired, key=lambda r: fnum(r.get("delta_cheap7")) or -999.0)
    best_no_gpiqa = max(paired, key=lambda r: fnum(r.get("delta_cheap6_no_GlobalPIQA")) or -999.0)
    def mean_key(key: str, subset: list[dict[str, Any]]) -> float | None:
        vals = [fnum(r.get(key)) for r in subset]
        vals = [v for v in vals if v is not None]
        return float(mean(vals)) if vals else None
    windows: dict[str, Any] = {}
    for name, (lo, hi) in WINDOWS.items():
        inside = [r for r in paired if lo <= int(r["words"]) <= hi]
        windows[name] = {
            "n": len(inside),
            "mean_delta_cheap7": mean_key("delta_cheap7", inside),
            "mean_delta_cheap6_no_GlobalPIQA": mean_key("delta_cheap6_no_GlobalPIQA", inside),
            "mean_delta_relation_state_mean": mean_key("delta_relation_state_mean", inside),
            "mean_positive_columns": mean_key("n_positive_columns", inside),
        }
    positive_rows = [r for r in paired if (fnum(r.get("delta_cheap7")) or 0.0) > 0]
    broad_rows = [r for r in paired if (r.get("n_positive_columns") or 0) >= 4 and (fnum(r.get("delta_cheap6_no_GlobalPIQA")) or -999) > 0]
    return {
        "status": "ok",
        "n_common": len(paired),
        "mean_delta_cheap7": mean_key("delta_cheap7", paired),
        "mean_delta_cheap6_no_GlobalPIQA": mean_key("delta_cheap6_no_GlobalPIQA", paired),
        "mean_delta_relation_state_mean": mean_key("delta_relation_state_mean", paired),
        "n_positive_cheap7_rows": len(positive_rows),
        "n_rows_with_4plus_positive_columns_and_positive_noGPIQA": len(broad_rows),
        "best_delta_cheap7_row": best_delta,
        "best_delta_noGPIQA_row": best_no_gpiqa,
        "windows": windows,
        "deltas": paired,
    }


def parse_labeled_path(text: str) -> tuple[str, pathlib.Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, path = text.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Empty label")
    return label, pathlib.Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", type=parse_labeled_path, required=True, help="LABEL=selected_trajectory.json")
    ap.add_argument("--reference-label", required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--width-drop", type=float, default=0.2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for label, path in args.trajectory:
        trajectories[label] = load_trajectory(path)
    if args.reference_label not in trajectories:
        raise ValueError(f"Reference label {args.reference_label!r} not among {list(trajectories)}")

    summaries = {label: summarize(rows, args.width_drop) for label, rows in trajectories.items()}
    ref_rows = trajectories[args.reference_label]
    comparisons = {label: summarize_comparison(rows, ref_rows) for label, rows in trajectories.items() if label != args.reference_label}
    result = {
        "status": "SELECTED_MLM_TRAJECTORY_COMPARISON",
        "width_drop": args.width_drop,
        "reference_label": args.reference_label,
        "labels": list(trajectories),
        "derived_columns": {
            "cheap6_no_GlobalPIQA": NO_GPIQA_COLUMNS,
            "cheap5_no_GlobalPIQA_Reading": CORE_COLUMNS,
            "relation_state_mean": REL_STATE_COLUMNS,
            "syntax_comp_mean": SYNTAX_COMP_COLUMNS,
            "volatile_mean": VOLATILE_COLUMNS,
        },
        "summaries": summaries,
        "comparisons_to_reference": comparisons,
        "interpretation_note": "Use the common late grid to compare timing and family movement. Scale1.25 is a clean adapter-scale contrast against the reference; scale1.75 seed43122 mixes initialization and mask-stream changes. A useful amplitude result should broaden or delay high cheap7 without relying on GlobalPIQA alone, and should not hide EWoK/Entity movement behind syntax-only gains.",
    }
    out_json = args.out_dir / "selected_mlm_trajectory_comparison.json"
    out_md = args.out_dir / "selected_mlm_trajectory_comparison.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = ["# research selected MLM trajectory comparison\n\n"]
    md.append(f"Reference: `{args.reference_label}`. Near-best band uses cheap7 within {args.width_drop:.3f} of each trajectory peak.\n\n")
    md.append("## Trajectory summaries\n\n")
    md.append("| label | n | best | best cheap7 | final-best | near-best endpoints | cheap6(no GPIQA) at best | EWoK/Entity at best |\n")
    md.append("|---|---:|---|---:|---:|---|---:|---:|\n")
    for label, s in summaries.items():
        if s.get("status") != "ok":
            md.append(f"| {label} | 0 | — | — | — | — | — | — |\n")
            continue
        md.append(
            f"| {label} | {s['n_rows']} | {s['best_endpoint']} | {s['best_cheap7']:.6f} | "
            f"{s['final_minus_best_cheap7']:+.6f} | {', '.join(s['near_best_endpoints'])} | "
            f"{(s.get('best_cheap6_no_GlobalPIQA') if s.get('best_cheap6_no_GlobalPIQA') is not None else float('nan')):.6f} | "
            f"{(s.get('best_relation_state_mean') if s.get('best_relation_state_mean') is not None else float('nan')):.6f} |\n"
        )
    if comparisons:
        md.append("\n## Contrasts against reference\n\n")
        md.append("| label | common | mean Δcheap7 | mean Δcheap6(no GPIQA) | mean ΔEWoK/Entity | positive Δcheap7 rows | broad rows | best Δ endpoint | best Δcheap7 | GPIQA gain share at best |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|\n")
        for label, c in comparisons.items():
            if c.get("status") != "ok":
                md.append(f"| {label} | 0 | — | — | — | — | — | — | — | — |\n")
                continue
            b = c["best_delta_cheap7_row"]
            gshare = b.get("positive_gain_share_GlobalPIQA")
            md.append(
                f"| {label} | {c['n_common']} | {c['mean_delta_cheap7']:+.6f} | "
                f"{c['mean_delta_cheap6_no_GlobalPIQA']:+.6f} | {c['mean_delta_relation_state_mean']:+.6f} | "
                f"{c['n_positive_cheap7_rows']} | {c['n_rows_with_4plus_positive_columns_and_positive_noGPIQA']} | "
                f"{b.get('endpoint')} | {b.get('delta_cheap7'):+.6f} | "
                f"{(gshare if gshare is not None else float('nan')):.3f} |\n"
            )
    md.append("\nInterpretation note: the file records measurements only. The scale1.25 run should be read against the seed/mask-matched reference; the scale1.75 seed43122 run cannot separate model initialization from mask-stream randomness.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
