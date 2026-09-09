#!/usr/bin/env python3
"""research: interpret the pending semantic-view no-AoA component trajectories.

This script is meant to be run after
`semantic_view_packet_local_delta_summary.json` exists. It reads the matched
trajectory and writes a component-level route interpretation using the current
research/research research logic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

TARGET_COLS = ["EWoK", "Entity", "COMPS", "GlobalPIQA"]
PROTECTED_COLS = ["Supplement", "Reading", "SuperGLUE"]
DISPLAY_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "equal7_full_eval"]


def as_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def ck_num(name: str) -> int:
    try:
        return int(name.split("_")[1].rstrip("M"))
    except Exception:
        return 10**9


def classify(rows):
    usable = []
    for ck, row in rows.items():
        d = row.get("delta_treatment_minus_control", {})
        equal7 = as_float(d.get("equal7_full_eval"))
        if equal7 is None:
            continue
        ksum = sum(as_float(d.get(c)) or 0.0 for c in TARGET_COLS)
        protected_known = [as_float(d.get(c)) for c in PROTECTED_COLS if as_float(d.get(c)) is not None]
        protect_sum = sum(protected_known)
        protect_min = min(protected_known) if protected_known else None
        usable.append({
            "checkpoint": ck,
            "checkpoint_mwords": ck_num(ck),
            "delta_equal7": equal7,
            "delta_ksum": ksum,
            "delta_protected_sum": protect_sum,
            "delta_protected_min": protect_min,
            "deltas": {c: as_float(d.get(c)) for c in DISPLAY_COLS + PROTECTED_COLS},
        })
    usable.sort(key=lambda r: r["checkpoint_mwords"])
    if not usable:
        return {"status": "no_usable_rows", "usable_rows": []}

    best_equal7 = max(usable, key=lambda r: r["delta_equal7"])
    best_ksum = max(usable, key=lambda r: r["delta_ksum"])
    best_balanced = max(usable, key=lambda r: (r["delta_ksum"] + 0.5 * r["delta_equal7"] + 0.25 * r["delta_protected_sum"]))
    late = [r for r in usable if r["checkpoint_mwords"] >= 70]
    last = usable[-1]
    positives = [r for r in usable if r["delta_equal7"] >= 0.25 and r["delta_ksum"] > 0]
    target_positive = [r for r in usable if r["delta_ksum"] >= 1.0]
    protected_damage = [r for r in usable if r["delta_protected_min"] is not None and r["delta_protected_min"] <= -1.0]

    if best_balanced["delta_ksum"] >= 2.0 and best_balanced["delta_equal7"] >= 0.25 and not (best_balanced["delta_protected_min"] is not None and best_balanced["delta_protected_min"] <= -1.5):
        route = "semantic_views_are_supported_enough_to_pair_with_fineweb_three_arm_design"
        next_work = "Use the positive same-source view signal to build a scaled FineWeb B-vs-C contrast from identical selected sources plus accepted relation-preserving rewrites; include protected official-slot A unless a truly matching A already exists."
    elif best_ksum["delta_ksum"] >= 1.0 and (best_equal7["delta_equal7"] < 0.25 or protected_damage):
        route = "semantic_views_move_target_columns_but_trade_against_protected_strengths"
        next_work = "Treat generated views as an interference-prone signal. Compare static insertion with late protected replay before combining it with a large source-breadth arm."
    elif best_equal7["delta_equal7"] >= 0.25 and best_ksum["delta_ksum"] <= 0.5:
        route = "semantic_views_are_linguistic_or_surface_help_not_main_gap_solution"
        next_work = "Do not train the semantic-view hybrid as the main route. Use broad FineWeb source-breadth A-vs-B at meaningful scale; keep rewrite as a secondary factor."
    else:
        route = "same_source_semantic_views_weak_for_sota_gap"
        next_work = "Prioritize a scaled source-breadth contrast: protected official-slot A versus cleaned FineWeb source-repetition B. In parallel, improve relation-preserving compression prompts before any C arm."

    return {
        "status": "interpreted",
        "usable_rows": usable,
        "best_equal7": best_equal7,
        "best_ksum": best_ksum,
        "best_balanced": best_balanced,
        "last_checkpoint": last,
        "late_mean": {
            "n": len(late),
            "delta_equal7": mean([r["delta_equal7"] for r in late]) if late else None,
            "delta_ksum": mean([r["delta_ksum"] for r in late]) if late else None,
            "delta_protected_sum": mean([r["delta_protected_sum"] for r in late]) if late else None,
        },
        "positive_rows_equal7_and_ksum": positives,
        "target_positive_rows": target_positive,
        "protected_damage_rows": protected_damage,
        "route_reading": route,
        "next_work": next_work,
        "interpretation_scale": {
            "sota_gap_overall": 0.4546,
            "summed_column_points_needed_if_other_columns_fixed": 4.10,
            "safer_one_run_summed_target": 4.5,
            "ksum_columns": TARGET_COLS,
            "protected_columns": PROTECTED_COLS,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json")
    ap.add_argument("--out-dir", default="experiments/archive/representation_and_objectives/data/semantic_view_interpretation")
    args = ap.parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"pending semantic-view summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("matched_rows", {})
    result = {
        "status": "SEMANTIC_VIEW_NOAOA_INTERPRETED",
        "summary_path": str(summary_path),
        "source_status": payload.get("status"),
        "treatment_target": payload.get("treatment_target"),
        "control_target": payload.get("control_target"),
        "interpretation": classify(rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "semantic_view_noaoa_route_interpretation.json"
    out_note = out_dir / "semantic_view_noaoa_route_interpretation.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    interp = result["interpretation"]
    lines = ["# Semantic-view no-AoA route interpretation\n\n"]
    if interp.get("status") != "interpreted":
        lines.append("No usable matched rows were found.\n")
    else:
        for label in ["best_equal7", "best_ksum", "best_balanced", "last_checkpoint"]:
            r = interp[label]
            lines.append(f"- {label}: {r['checkpoint']} equal7 Δ={r['delta_equal7']:.4f}, Ksum Δ={r['delta_ksum']:.4f}, protected-sum Δ={r['delta_protected_sum']:.4f}\n")
        lm = interp["late_mean"]
        lines.append(f"- late mean over {lm['n']} rows: equal7 Δ={lm['delta_equal7']}, Ksum Δ={lm['delta_ksum']}, protected-sum Δ={lm['delta_protected_sum']}\n")
        lines.append(f"\nRoute reading: `{interp['route_reading']}`\n\n")
        lines.append(f"Next work: {interp['next_work']}\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    out_note.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_note": str(out_note), "route_reading": interp.get("route_reading")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
