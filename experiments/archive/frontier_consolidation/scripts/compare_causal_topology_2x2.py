#!/usr/bin/env python3
"""research: compare a future causal topology 2x2 screen.

Consumes four completed selected-causal-trajectory JSON files:
  compact_oneway, repeat_oneway, compact_reciprocal, repeat_reciprocal

Reports semantic effects, topology effects, and the interaction:
  interaction = (compact_reciprocal - repeat_reciprocal)
                - (compact_oneway - repeat_oneway)
              = (compact_reciprocal - compact_oneway)
                - (repeat_reciprocal - repeat_oneway)

This is intentionally stricter than comparing reciprocal compact to reciprocal
repeat alone. The route is scientifically live only if the interaction is broad
across official families and survives removal of GlobalPIQA/Reading volatility.
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
GROUPS = {
    "cheap7": CHEAP_COLUMNS,
    "cheap6_no_global": CHEAP6_NO_GLOBAL,
    "cheap5_no_global_reading": CHEAP5_NO_GLOBAL_READING,
    "syntax_surface": SYNTAX_SURFACE,
    "relation_state": RELATION_STATE,
    "volatile_small": VOLATILE_SMALL,
}


def read_rows(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ["rows", "trajectory", "contrasts"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Expected list trajectory in {path}")
    rows: dict[str, dict[str, Any]] = {}
    for r in data:
        if r.get("cheap7") is None or r.get("error"):
            continue
        ep = str(r.get("endpoint"))
        rows[ep] = r
    return rows


def endpoint_key(ep: str) -> int:
    if ep.startswith("chck_") and ep.endswith("M"):
        return int(ep.split("_")[1][:-1])
    return 10**9


def avg(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [row.get(c) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def get_metric(row: dict[str, Any], metric: str) -> float | None:
    if metric == "cheap7":
        return float(row["cheap7"])
    return avg(row, GROUPS[metric])


def diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def fmt(v: Any, nd: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, int)):
        return f"{float(v):.{nd}f}"
    return str(v)


def summarize_values(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "positive": 0}
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
        "min": float(min(xs)),
        "max": float(max(xs)),
        "positive": sum(1 for x in xs if x > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-oneway", type=pathlib.Path, required=True)
    ap.add_argument("--repeat-oneway", type=pathlib.Path, required=True)
    ap.add_argument("--compact-reciprocal", type=pathlib.Path, required=True)
    ap.add_argument("--repeat-reciprocal", type=pathlib.Path, required=True)
    ap.add_argument("--scaffold-manifest", type=pathlib.Path, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    CO = read_rows(args.compact_oneway)
    RO = read_rows(args.repeat_oneway)
    CR = read_rows(args.compact_reciprocal)
    RR = read_rows(args.repeat_reciprocal)
    common_eps = sorted(set(CO) & set(RO) & set(CR) & set(RR), key=endpoint_key)

    rows: list[dict[str, Any]] = []
    for ep in common_eps:
        row: dict[str, Any] = {"endpoint": ep, "nominal_words": endpoint_key(ep) * 1_000_000 if endpoint_key(ep) < 10**9 else None}
        for metric in GROUPS:
            co = get_metric(CO[ep], metric)
            ro = get_metric(RO[ep], metric)
            cr = get_metric(CR[ep], metric)
            rr = get_metric(RR[ep], metric)
            sem_one = diff(co, ro)
            sem_rec = diff(cr, rr)
            top_compact = diff(cr, co)
            top_repeat = diff(rr, ro)
            interaction = diff(sem_rec, sem_one)
            row[f"compact_oneway_{metric}"] = co
            row[f"repeat_oneway_{metric}"] = ro
            row[f"compact_reciprocal_{metric}"] = cr
            row[f"repeat_reciprocal_{metric}"] = rr
            row[f"semantic_effect_oneway_{metric}"] = sem_one
            row[f"semantic_effect_reciprocal_{metric}"] = sem_rec
            row[f"topology_effect_compact_{metric}"] = top_compact
            row[f"topology_effect_repeat_{metric}"] = top_repeat
            row[f"interaction_{metric}"] = interaction
        pos_col_interactions = 0
        neg_col_interactions = 0
        col_interactions: dict[str, float] = {}
        for c in CHEAP_COLUMNS:
            co = float(CO[ep][c]); ro = float(RO[ep][c]); cr = float(CR[ep][c]); rr = float(RR[ep][c])
            val = (cr - rr) - (co - ro)
            row[f"interaction_{c}"] = val
            col_interactions[c] = val
            if val > 0:
                pos_col_interactions += 1
            elif val < 0:
                neg_col_interactions += 1
        row["positive_column_interactions"] = pos_col_interactions
        row["negative_column_interactions"] = neg_col_interactions
        positive_sum = sum(v for v in col_interactions.values() if v > 0)
        row["largest_positive_column_interaction_share"] = (max([v for v in col_interactions.values() if v > 0], default=0.0) / positive_sum) if positive_sum > 0 else 0.0
        row["volatile_interaction_fraction_of_cheap7"] = (row["interaction_volatile_small"] / row["interaction_cheap7"]) if row.get("interaction_cheap7") not in (None, 0) and row.get("interaction_volatile_small") is not None else None
        rows.append(row)

    metric_summaries = {}
    for metric in GROUPS:
        vals = [r[f"interaction_{metric}"] for r in rows if r.get(f"interaction_{metric}") is not None]
        metric_summaries[metric] = summarize_values([float(v) for v in vals])

    broad_positive = [r for r in rows if r.get("interaction_cheap7", 0) > 0 and r.get("interaction_cheap6_no_global", 0) > 0 and r.get("positive_column_interactions", 0) >= 4]
    robust_positive = [r for r in broad_positive if r.get("interaction_cheap5_no_global_reading", 0) > 0]
    best = max(rows, key=lambda r: r.get("interaction_cheap7", -1e9)) if rows else None
    worst = min(rows, key=lambda r: r.get("interaction_cheap7", 1e9)) if rows else None

    scaffold = None
    if args.scaffold_manifest and args.scaffold_manifest.exists():
        scaffold = json.loads(args.scaffold_manifest.read_text(encoding="utf-8"))

    summary = {
        "status": "CAUSAL_TOPOLOGY_2X2_INTERACTION_COMPARISON" if rows else "NO_COMMON_ENDPOINTS",
        "inputs": {
            "compact_oneway": str(args.compact_oneway),
            "repeat_oneway": str(args.repeat_oneway),
            "compact_reciprocal": str(args.compact_reciprocal),
            "repeat_reciprocal": str(args.repeat_reciprocal),
            "scaffold_manifest": str(args.scaffold_manifest) if args.scaffold_manifest else None,
        },
        "common_endpoints": common_eps,
        "n_common_endpoints": len(rows),
        "metric_interaction_summaries": metric_summaries,
        "n_broad_positive_endpoints": len(broad_positive),
        "n_robust_positive_endpoints_no_global_no_reading": len(robust_positive),
        "best_interaction_endpoint": best,
        "worst_interaction_endpoint": worst,
        "rows": rows,
        "scaffold_token_comparisons": (scaffold or {}).get("token_comparisons"),
        "scientific_readout": {
            "live_semantic_by_topology_pattern": "interaction_cheap7 > 0, interaction_cheap6_no_global > 0, interaction_cheap5_no_global_reading > 0, and >=4 positive column interactions at multiple endpoints",
            "stop_pattern": "interaction vanishes without GlobalPIQA/Reading, is dominated by one column, or repeat topology gains as much as compact topology",
            "reason": "The route is reciprocal semantic learning only if reciprocal topology specifically increases the compact-vs-repeat advantage, not if recurrence or exact repetition improves both arms equally."
        }
    }

    out_json = args.out_dir / "causal_topology_2x2_interaction_comparison.json"
    out_md = args.out_dir / "causal_topology_2x2_interaction_comparison.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# research causal topology 2×2 interaction comparison\n\n")
    md.append("This file compares completed future trajectories; it does not evaluate or train models.\n\n")
    md.append(f"Common endpoints: {len(rows)}. Broad positive endpoints: {len(broad_positive)}; robust without GlobalPIQA/Reading: {len(robust_positive)}.\n\n")
    if rows:
        md.append("## Interaction summaries\n\n")
        for metric, s in metric_summaries.items():
            md.append(f"- {metric}: mean {fmt(s['mean'])}, std {fmt(s['std'])}, positive {s['positive']}/{s['n']}, min {fmt(s['min'])}, max {fmt(s['max'])}.\n")
        md.append("\n")
        md.append(f"Best endpoint by cheap7 interaction: {best['endpoint']} ({fmt(best['interaction_cheap7'])}); worst: {worst['endpoint']} ({fmt(worst['interaction_cheap7'])}).\n\n")
        md.append("| endpoint | sem one cheap7 | sem rec cheap7 | topo compact | topo repeat | interaction cheap7 | int cheap6 noG | int cheap5 noG/R | int rel/state | int syntax | int volatile | +int cols | max +share |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            md.append(
                f"| {r['endpoint']} | {fmt(r['semantic_effect_oneway_cheap7'])} | {fmt(r['semantic_effect_reciprocal_cheap7'])} | "
                f"{fmt(r['topology_effect_compact_cheap7'])} | {fmt(r['topology_effect_repeat_cheap7'])} | {fmt(r['interaction_cheap7'])} | "
                f"{fmt(r['interaction_cheap6_no_global'])} | {fmt(r['interaction_cheap5_no_global_reading'])} | {fmt(r['interaction_relation_state'])} | {fmt(r['interaction_syntax_surface'])} | {fmt(r['interaction_volatile_small'])} | "
                f"{r['positive_column_interactions']} | {fmt(r['largest_positive_column_interaction_share'], 3)} |\n"
            )
    md.append("\n## Scientific reading\n\n")
    md.append("The topology route remains live only if reciprocal topology specifically increases the compact semantic advantage over repeat, with breadth across official families and persistence after removing GlobalPIQA/Reading. Otherwise the result is recurrence/copy/volatile-column redistribution.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "n_common": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
