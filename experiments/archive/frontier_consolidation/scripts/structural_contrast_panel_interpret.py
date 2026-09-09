#!/usr/bin/env python3
"""research: interpret the frozen structural contrast-margin v1 panel.

This script reads the completed research scale1.75 late-window panel and summarizes
whether the already-frozen composite or family readouts track the known official
cheap7 late trajectory.  It does not rescore models and does not alter the probe.
"""
from __future__ import annotations
import json, math, statistics, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("experiments/archive/frontier_consolidation")
PANEL = ROOT / "data/structural_contrast_scores/scale1p75_late_panel.json"
QUALITY = ROOT / "data/structural_probe_quality/structural_probe_quality_report.json"
OUT_DIR = ROOT / "data/structural_contrast_panel_interpretation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRUCT_WEIGHTS = {
    "entity_state": 0.30,
    "temporal_order": 0.25,
    "directed_relation": 0.20,
    "belief_report": 0.15,
    "polarity_relation": 0.10,
}
SURFACE_WEIGHT = -0.25


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(vx*vy)


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0]*len(xs)
    i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and xs[order[j+1]] == xs[order[i]]:
            j += 1
        avg = (i + j)/2 + 1
        for k in range(i, j+1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    return pearson(rankdata([p[0] for p in pairs]), rankdata([p[1] for p in pairs]))


def best_ckpt(ckpts, vals):
    idx = max(range(len(vals)), key=lambda i: vals[i])
    return ckpts[idx], vals[idx]


def rank_of(ckpts, vals, target="chck_82M"):
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    for pos, idx in enumerate(order, 1):
        if ckpts[idx] == target:
            return pos
    return None


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main():
    t0 = time.time()
    panel = json.load(open(PANEL, encoding="utf-8"))
    quality = json.load(open(QUALITY, encoding="utf-8")) if QUALITY.exists() else None
    records = sorted(panel["per_checkpoint"], key=lambda r: r["checkpoint_M"])
    ckpts = [r["checkpoint"] for r in records]
    cheap7 = [r["cheap7"] for r in records]
    composite = [r["aggregation"]["composite_score"] for r in records]

    series = {"frozen_composite": composite}
    structural_only = []
    surface_terms = []
    for r in records:
        fam = r["aggregation"]["family_results"]
        s = sum(STRUCT_WEIGHTS[k] * fam[k]["trimmed_mean_margin"] for k in STRUCT_WEIGHTS)
        structural_only.append(s)
        surface_terms.append(SURFACE_WEIGHT * fam["surface_control"]["trimmed_mean_margin"])
        for family, fres in fam.items():
            series.setdefault(f"family_trimmed/{family}", []).append(fres["trimmed_mean_margin"])
            series.setdefault(f"family_mean/{family}", []).append(fres["mean_margin"])
            series.setdefault(f"family_hard_accuracy/{family}", []).append(fres["hard_accuracy"])
            series.setdefault(f"family_q25/{family}", []).append(fres["q25_margin"])
        for dk, dv in r["aggregation"].get("entity_state_by_depth", {}).items():
            series.setdefault(f"entity_depth_mean/{dk}", []).append(dv["mean_margin"])
            series.setdefault(f"entity_depth_hard_accuracy/{dk}", []).append(dv["hard_accuracy"])
    series["structural_weighted_without_surface"] = structural_only
    series["surface_subtraction_term"] = surface_terms

    table = []
    for name, vals in sorted(series.items()):
        if len(vals) != len(ckpts):
            continue
        best_name, best_val = best_ckpt(ckpts, vals)
        entry = {
            "readout": name,
            "values": {c: round(v, 6) for c, v in zip(ckpts, vals)},
            "best_checkpoint": best_name,
            "best_value": round(best_val, 6),
            "rank_of_chck_82M": rank_of(ckpts, vals),
            "pearson_vs_cheap7": round(pearson(vals, cheap7), 6) if pearson(vals, cheap7) is not None else None,
            "spearman_vs_cheap7": round(spearman(vals, cheap7), 6) if spearman(vals, cheap7) is not None else None,
            "delta_80M_to_82M": round(vals[ckpts.index("chck_82M")] - vals[ckpts.index("chck_80M")], 6) if "chck_80M" in ckpts and "chck_82M" in ckpts else None,
            "delta_82M_to_100M": round(vals[ckpts.index("chck_100M")] - vals[ckpts.index("chck_82M")], 6) if "chck_100M" in ckpts and "chck_82M" in ckpts else None,
        }
        table.append(entry)

    adjacent = []
    for i in range(1, len(ckpts)):
        ch_delta = cheap7[i] - cheap7[i-1]
        co_delta = composite[i] - composite[i-1]
        st_delta = structural_only[i] - structural_only[i-1]
        adjacent.append({
            "from": ckpts[i-1], "to": ckpts[i],
            "cheap7_delta": round(ch_delta, 6),
            "composite_delta": round(co_delta, 6),
            "structural_without_surface_delta": round(st_delta, 6),
            "composite_sign_matches_cheap7": sign(ch_delta) == sign(co_delta),
            "structural_sign_matches_cheap7": sign(ch_delta) == sign(st_delta),
        })

    # A compact statement of the actual frozen-v1 result.
    frozen = next(x for x in table if x["readout"] == "frozen_composite")
    structural = next(x for x in table if x["readout"] == "structural_weighted_without_surface")
    interesting_positive = [x for x in table if (x["pearson_vs_cheap7"] or 0) > 0.3 or (x["spearman_vs_cheap7"] or 0) > 0.3]

    result = {
        "status": "STRUCTURAL_CONTRAST_V1_PANEL_INTERPRETED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panel_path": str(PANEL),
        "quality_report_path": str(QUALITY) if QUALITY.exists() else None,
        "checkpoints": ckpts,
        "cheap7_series": {c: round(v, 6) for c, v in zip(ckpts, cheap7)},
        "frozen_composite_summary": frozen,
        "structural_without_surface_summary": structural,
        "all_readout_summaries": table,
        "adjacent_movement": adjacent,
        "positive_but_not_peak_readouts": interesting_positive,
        "static_quality_summary": {
            "n_pairs": quality.get("n_pairs") if quality else None,
            "global_flag_counts": quality.get("global_flag_counts") if quality else None,
            "zero_focus_by_family": {k: v.get("zero_focus_pairs") for k, v in quality.get("family_summary", {}).items()} if quality else None,
            "focus_imbalance_ge3_by_family": {k: v.get("focus_imbalance_ge3") for k, v in quality.get("family_summary", {}).items()} if quality else None,
        },
        "scientific_reading": {
            "frozen_v1_result": "The frozen composite does not track the known scale1.75 late official cheap7 peak: Pearson is negative and its maximum is chck_77M, not chck_82M.",
            "family_result": "Some single-family readouts have modest positive association with cheap7, especially directed_relation, polarity_relation, and entity_state depth3, but none produces a reliable 82M peak and their best checkpoints differ.",
            "quality_context": "The scorer sees every pair, but the v1 pair set contains construction artifacts: temporal perturbations often start lowercase and carry double punctuation, entity_state uses synthetic templates with some awkward put-to wording, and surface substitutions produce very large margins that dominate the frozen surface subtraction.",
            "next_research_direction": "Do not spend official evaluation on a cross-trajectory v1 test or retune v1 after seeing these scores. Use the failure to refine the mechanism search or build a cleaner structural source only from a newly frozen construction before model scoring."
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }

    out_json = OUT_DIR / "structural_contrast_v1_panel_interpretation.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# research interpretation of frozen structural contrast-margin v1")
    md.append("")
    md.append(f"Panel: `{PANEL}`")
    md.append(f"Quality report: `{QUALITY}`")
    md.append("")
    md.append("## Main result")
    md.append("")
    md.append(f"- Frozen composite Pearson vs known scale1.75 cheap7: **{frozen['pearson_vs_cheap7']}**; Spearman: **{frozen['spearman_vs_cheap7']}**.")
    md.append(f"- Frozen composite maximum: **{frozen['best_checkpoint']}** with value {frozen['best_value']}; rank of `chck_82M`: **{frozen['rank_of_chck_82M']}**.")
    md.append(f"- Structural-only weighted score without the surface subtraction maximum: **{structural['best_checkpoint']}**; Pearson {structural['pearson_vs_cheap7']}; rank of `chck_82M`: **{structural['rank_of_chck_82M']}**.")
    md.append(f"- Composite movement from 80M to 82M: {frozen['delta_80M_to_82M']}; from 82M to 100M: {frozen['delta_82M_to_100M']}.")
    md.append("")
    md.append("## Readouts with positive association but no 82M peak")
    md.append("")
    md.append("| readout | best ckpt | rank of 82M | Pearson | Spearman | 80→82 | 82→100 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for x in interesting_positive:
        md.append(f"| {x['readout']} | {x['best_checkpoint']} | {x['rank_of_chck_82M']} | {x['pearson_vs_cheap7']} | {x['spearman_vs_cheap7']} | {x['delta_80M_to_82M']} | {x['delta_82M_to_100M']} |")
    md.append("")
    md.append("## Static construction context")
    md.append("")
    flags = result["static_quality_summary"]["global_flag_counts"] or {}
    for k, v in list(flags.items())[:12]:
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Scientific reading")
    md.append("")
    for k, v in result["scientific_reading"].items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append(f"JSON: `{out_json}`")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/structural_contrast_panel_interpretation/structural_contrast_v1_panel_interpretation.md')).write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "frozen_composite_best": frozen["best_checkpoint"],
        "frozen_composite_pearson": frozen["pearson_vs_cheap7"],
        "structural_without_surface_best": structural["best_checkpoint"],
        "n_positive_readouts": len(interesting_positive),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
