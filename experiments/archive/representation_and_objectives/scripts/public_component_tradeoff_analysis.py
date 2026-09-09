#!/usr/bin/env python3
"""Analyze strict-small leaderboard component tradeoffs against COMPACT_EXPERIENCE local best.

This is a CPU-only evidence script for route choice while the managed semantic-view
evaluator is running. It does not train or evaluate models.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_ROOT = ROOT / "data" / "public_component_tradeoffs"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/public_component_tradeoffs.md')
STRICT_SMALL = ROOT / "data" / "babylm2026_live_surface" / "strict_small_top30.json"

OURS = {
    "Model_plain": "qwen_clean_aligned_COMPACT_EXPERIENCE",
    "Overall Average": 41.34429066479573,
    "BLiMP": 66.84,
    "BLiMP Supplement": 62.84,
    "EWoK": 50.19,
    "Entity Tracking": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "(Super)GLUE": 70.30861598316157,
    "Reading": 7.76,
    "AoA": 0.0,
}
KEYS = [
    ("BLiMP", "BLiMP"),
    ("Supplement", "BLiMP Supplement"),
    ("EWoK", "EWoK"),
    ("Entity", "Entity Tracking"),
    ("COMPS", "COMPS"),
    ("GlobalPIQA", "GlobalPIQA"),
    ("SuperGLUE", "(Super)GLUE"),
    ("Reading", "Reading"),
    ("AoA", "AoA"),
]
CLUSTER = ["EWoK", "Entity Tracking", "COMPS", "GlobalPIQA"]


def get(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def overall(row: dict[str, Any]) -> float:
    return mean([float(row[col]) for _, col in KEYS])


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def main() -> None:
    rows = json.loads(STRICT_SMALL.read_text())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    leader = rows[0]
    gap_rows = []
    for short, col in KEYS:
        ov = float(OURS[col])
        lv = float(leader[col])
        gap_rows.append({
            "column": short,
            "ours": ov,
            "strict_small_leader": lv,
            "leader_minus_ours": lv - ov,
            "overall_contribution": (lv - ov) / 9.0,
        })

    top_rows = []
    for rank, row in enumerate(rows, 1):
        rec = {
            "rank": rank,
            "model": row.get("Model_plain"),
            "repo": row.get("HF Repo"),
            "overall": get(row, "Overall Average"),
            "knowledge_cluster_mean": mean([float(row[c]) for c in CLUSTER]),
        }
        for short, col in KEYS:
            rec[short] = get(row, col)
        top_rows.append(rec)

    component_ranges = {}
    for short, col in KEYS:
        xs = [get(r, col) for r in rows if get(r, col) is not None]
        xs_f = [float(x) for x in xs if x is not None]
        arg = max(rows, key=lambda r: get(r, col) if get(r, col) is not None else -1e9)
        component_ranges[short] = {
            "min": min(xs_f),
            "median": statistics.median(xs_f),
            "max": max(xs_f),
            "max_model": arg.get("Model_plain"),
            "ours": float(OURS[col]),
        }

    # Tradeoffs: correlation among top30 between components and overall/each other.
    corrs = {}
    for short, col in KEYS[:-1]:
        xs, ys = [], []
        for r in rows:
            x, y = get(r, col), get(r, "Overall Average")
            if x is not None and y is not None:
                xs.append(x); ys.append(y)
        corrs[f"{short}_vs_overall"] = corr(xs, ys)
    # Main worry: high knowledge cluster may trade off against Supplement/Reading.
    xs_cluster = []
    ys_supp = []
    ys_read = []
    ys_overall = []
    for r in rows:
        vals = [get(r, c) for c in CLUSTER]
        if all(v is not None for v in vals):
            xs_cluster.append(mean([float(v) for v in vals]))
            ys_supp.append(float(get(r, "BLiMP Supplement")))
            ys_read.append(float(get(r, "Reading")))
            ys_overall.append(float(get(r, "Overall Average")))
    corrs["knowledge_cluster_vs_supplement"] = corr(xs_cluster, ys_supp)
    corrs["knowledge_cluster_vs_reading"] = corr(xs_cluster, ys_read)
    corrs["knowledge_cluster_vs_overall"] = corr(xs_cluster, ys_overall)

    preserve_models = []
    ours_cluster = mean([float(OURS[c]) for c in CLUSTER])
    for r in rows:
        supp = get(r, "BLiMP Supplement")
        reading = get(r, "Reading")
        if supp is not None and reading is not None and supp >= 60 and reading >= 7:
            kc = mean([float(r[c]) for c in CLUSTER])
            preserve_models.append({
                "model": r.get("Model_plain"),
                "overall": get(r, "Overall Average"),
                "knowledge_cluster_mean": kc,
                "cluster_delta_vs_ours": kc - ours_cluster,
                "BLiMP": get(r, "BLiMP"),
                "Supplement": supp,
                "Reading": reading,
                "EWoK": get(r, "EWoK"),
                "Entity": get(r, "Entity Tracking"),
                "COMPS": get(r, "COMPS"),
                "GlobalPIQA": get(r, "GlobalPIQA"),
            })

    high_ewok_models = []
    for r in rows:
        if get(r, "EWoK") is not None and get(r, "EWoK") >= 55:
            high_ewok_models.append({
                "model": r.get("Model_plain"),
                "overall": get(r, "Overall Average"),
                "EWoK": get(r, "EWoK"),
                "Supplement": get(r, "BLiMP Supplement"),
                "Entity": get(r, "Entity Tracking"),
                "GlobalPIQA": get(r, "GlobalPIQA"),
                "Reading": get(r, "Reading"),
            })

    synth = dict(OURS)
    component_maxes = {}
    for short, col in KEYS:
        arg = max(rows, key=lambda r: get(r, col) if get(r, col) is not None else -1e9)
        mx = float(arg[col])
        component_maxes[short] = {"max": mx, "max_model": arg.get("Model_plain"), "ours": float(OURS[col])}
        if mx > float(synth[col]):
            synth[col] = mx
    synthetic_componentwise_upper = overall(synth)

    result = {
        "status": "PUBLIC_COMPONENT_TRADEOFFS_ANALYZED",
        "strict_small_rows": len(rows),
        "strict_small_leader": {"model": leader.get("Model_plain"), "repo": leader.get("HF Repo"), "overall": get(leader, "Overall Average")},
        "ours": OURS,
        "gap_leader_minus_ours": gap_rows,
        "top15": top_rows[:15],
        "component_ranges_top30": component_ranges,
        "correlations_top30": corrs,
        "models_preserving_supp_ge60_reading_ge7": preserve_models,
        "models_ewok_ge55": high_ewok_models,
        "component_maxes_top30": component_maxes,
        "synthetic_componentwise_upper_preserving_ours_when_better": synthetic_componentwise_upper,
        "interpretation": "The deficit is concentrated in EWoK/Entity/COMPS/GlobalPIQA; ours already exceeds leader on Supplement, Reading, and SuperGLUE. The next training bet should target factual/world-knowledge breadth while preserving the developmental/conversational strengths.",
    }
    (OUT_ROOT / "public_component_tradeoffs.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    lines = []
    lines.append("# research public strict-small component tradeoffs\n")
    lines.append(f"Analyzed `{STRICT_SMALL}` against inherited COMPACT_EXPERIENCE `qwen_clean_aligned` (Overall {OURS['Overall Average']:.4f}).\n")
    lines.append(f"Strict-small leader: {leader.get('Model_plain')} at {get(leader, 'Overall Average'):.2f}.\n")
    lines.append("\n## Leader minus ours\n")
    lines.append("| column | ours | leader | delta | overall contribution |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for g in gap_rows:
        lines.append(f"| {g['column']} | {g['ours']:.3f} | {g['strict_small_leader']:.3f} | {g['leader_minus_ours']:+.3f} | {g['overall_contribution']:+.3f} |\n")
    lines.append("\n## Top30 empirical tradeoff clues\n")
    lines.append(f"Knowledge-cluster mean (EWoK/Entity/COMPS/GlobalPIQA) correlation with Supplement among top30: {corrs['knowledge_cluster_vs_supplement']:.3f}.\n")
    lines.append(f"Knowledge-cluster mean correlation with Reading among top30: {corrs['knowledge_cluster_vs_reading']:.3f}.\n")
    lines.append(f"Knowledge-cluster mean correlation with Overall among top30: {corrs['knowledge_cluster_vs_overall']:.3f}.\n")
    lines.append("\nModels in top30 with Supplement>=60 and Reading>=7:\n")
    if preserve_models:
        for m in preserve_models:
            lines.append(f"- {m['model']}: Overall {m['overall']:.2f}, cluster {m['knowledge_cluster_mean']:.2f}, cluster delta vs ours {m['cluster_delta_vs_ours']:+.2f}, Supp {m['Supplement']:.2f}, Reading {m['Reading']:.2f}.\n")
    else:
        lines.append("- None. This suggests a real tradeoff in public models; our route must preserve the COMPACT_EXPERIENCE strengths while adding factual breadth modestly.\n")
    lines.append("\nHigh-EWoK public models (EWoK>=55):\n")
    for m in high_ewok_models:
        lines.append(f"- {m['model']}: Overall {m['overall']:.2f}, EWoK {m['EWoK']:.2f}, Supp {m['Supplement']:.2f}, Entity {m['Entity']:.2f}, GPIQA {m['GlobalPIQA']:.2f}, Reading {m['Reading']:.2f}.\n")
    lines.append("\n## Research consequence\n")
    lines.append("The public tradeoff picture reinforces the component gap analysis: the SOTA path is not to copy the leader wholesale, because we already have rare Supplement/Reading strength. The next H100 allocation should seek a narrow factual-breadth intervention on top of the protected clean-Qwen/developmental mixture, then evaluate whether EWoK/Entity/COMPS/GlobalPIQA move without erasing Supplement/Reading. Same-source semantic-view evidence remains scientifically informative but low-ceiling for the actual 41.8 gap.\n")
    lines.append(f"\nFull JSON: `{OUT_ROOT / 'public_component_tradeoffs.json'}`\n")
    NOTE.write_text("".join(lines))

    print(json.dumps({"status": result["status"], "out": str(OUT_ROOT / "public_component_tradeoffs.json"), "note": str(NOTE), "leader": result["strict_small_leader"], "knowledge_cluster_vs_supplement_corr": corrs["knowledge_cluster_vs_supplement"], "knowledge_cluster_vs_reading_corr": corrs["knowledge_cluster_vs_reading"], "preserve_models_count": len(preserve_models)}, indent=2))

if __name__ == "__main__":
    main()
