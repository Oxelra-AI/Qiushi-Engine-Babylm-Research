#!/usr/bin/env python3
"""Step243b: baseline/subtask-conditioned transfer test for DeBERTa compact components.

independent review noted that research demotion is justified but a final no-training refinement remains:
research decomposed DeBERTa compact into A=adjbreak-repeat and B=view-adjbreak. Test whether
A or B predicts RoBERTa compact-repeat or RoBERTa factorial movement after controlling for
subtask/template and baseline correctness. This script uses frozen prediction artifacts only.
"""
from __future__ import annotations

import collections
import csv
import importlib.util
import json
import math
import pathlib
import random
import sys
from typing import Any, Dict, Iterable, List, Tuple

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data" / "component_transfer_conditioned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_SCRIPT = ROOT / "scripts" / "frozen_compact_transfer_predictor.py"
spec = importlib.util.spec_from_file_location("step243pred", PRED_SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["step243pred"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

STABLE_NONBLIMP = {"Supplement", "EWoK", "COMPS"}
STABLE_WITH_BLIMP = {"BLiMP", "Supplement", "EWoK", "COMPS"}


def load_joined() -> List[dict]:
    view = mod.score_arm(mod.VIEW_ROOT)
    rep = mod.score_arm(mod.REPEAT_ROOT)
    adj = mod.score_arm(mod.ADJBREAK_ROOT)
    rob_cr = mod.load_roberta_compact_repeat()
    rob_fact = {}
    with open(mod.ROBERTA_FACT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rob_fact[r["item_id"]] = {
                "column": r["column"],
                "subtask": r["subtask"],
                "hs_correct": int(r["hs_correct"]),
                "ls_correct": int(r["ls_correct"]),
                "hd_correct": int(r["hd_correct"]),
                "ld_correct": int(r["ld_correct"]),
                "same_anchor_effect": int(r["same_anchor_effect"]),
                "deranged_effect": int(r["deranged_effect"]),
                "interaction": int(r["interaction"]),
            }
    common = sorted(set(view) & set(rep) & set(adj) & set(rob_cr) & set(rob_fact))
    rows = []
    for item_id in common:
        col = view[item_id]["column"]
        sub = view[item_id]["subtask"]
        rc = rob_cr[item_id]
        rf = rob_fact[item_id]
        rows.append({
            "item_id": item_id,
            "column": col,
            "subtask": sub,
            "d_view": int(view[item_id]["correct"]),
            "d_repeat": int(rep[item_id]["correct"]),
            "d_adjbreak": int(adj[item_id]["correct"]),
            "A_adjbreak_minus_repeat": int(adj[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "B_view_minus_adjbreak": int(view[item_id]["correct"]) - int(adj[item_id]["correct"]),
            "T_view_minus_repeat": int(view[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "r_repeat": int(rc["r_left_correct"]),
            "r_compact": int(rc["r_right_correct"]),
            "R_compact_minus_repeat": int(rc["r_compact_minus_repeat"]),
            "f_hs": rf["hs_correct"],
            "f_ls": rf["ls_correct"],
            "f_hd": rf["hd_correct"],
            "f_ld": rf["ld_correct"],
            "F_same_anchor": rf["same_anchor_effect"],
            "F_deranged": rf["deranged_effect"],
            "F_interaction": rf["interaction"],
        })
    return rows


def mean(xs: List[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def residual_pairs(rows: List[dict], xkey: str, ykey: str, group_keys: Tuple[str, ...]) -> Tuple[List[float], List[float]]:
    buckets: Dict[Tuple[Any, ...], List[dict]] = collections.defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in group_keys)].append(r)
    xs, ys = [], []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        mx = mean([float(r[xkey]) for r in bucket])
        my = mean([float(r[ykey]) for r in bucket])
        if mx is None or my is None:
            continue
        for r in bucket:
            xs.append(float(r[xkey]) - mx)
            ys.append(float(r[ykey]) - my)
    return xs, ys


def slope_corr(xs: List[float], ys: List[float]) -> dict:
    n = len(xs)
    if n < 2:
        return {"n_residual_items": n, "slope_pp_per_unit_x": None, "corr": None}
    sx2 = sum(x*x for x in xs)
    sy2 = sum(y*y for y in ys)
    sxy = sum(x*y for x, y in zip(xs, ys))
    slope = sxy / sx2 if sx2 > 0 else None
    corr = sxy / math.sqrt(sx2 * sy2) if sx2 > 0 and sy2 > 0 else None
    return {"n_residual_items": n, "slope_pp_per_unit_x": None if slope is None else 100.0 * slope, "corr": corr}


def raw_by_x(rows: List[dict], xkey: str, ykey: str) -> dict:
    groups: Dict[int, List[int]] = collections.defaultdict(list)
    for r in rows:
        groups[int(r[xkey])].append(int(r[ykey]))
    out = {}
    for k in sorted(groups):
        vals = groups[k]
        out[str(k)] = {"n": len(vals), "mean_pp": 100.0 * mean(vals), "sum": sum(vals)}
    if 1 in groups and -1 in groups:
        out["plus1_minus_minus1_mean_pp"] = 100.0 * (mean(groups[1]) - mean(groups[-1]))
    return out


def cluster_bootstrap(rows: List[dict], xkey: str, ykey: str, group_keys: Tuple[str, ...], cluster_key: str = "subtask", iters: int = 500, seed: int = 243) -> dict:
    clusters: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        clusters[str(r[cluster_key])].append(r)
    names = sorted(clusters)
    if len(names) < 2:
        return {"iters": 0, "cluster_count": len(names), "p_slope_gt0": None, "ci95_slope_pp": None}
    rng = random.Random(seed)
    slopes = []
    for _ in range(iters):
        sample_rows = []
        for name in (rng.choice(names) for _ in names):
            sample_rows.extend(clusters[name])
        xs, ys = residual_pairs(sample_rows, xkey, ykey, group_keys)
        rec = slope_corr(xs, ys)
        s = rec["slope_pp_per_unit_x"]
        if s is not None:
            slopes.append(s)
    if not slopes:
        return {"iters": iters, "cluster_count": len(names), "p_slope_gt0": None, "ci95_slope_pp": None}
    slopes.sort()
    lo = slopes[int(0.025 * (len(slopes) - 1))]
    hi = slopes[int(0.975 * (len(slopes) - 1))]
    return {"iters": len(slopes), "cluster_count": len(names), "p_slope_gt0": sum(s > 0 for s in slopes) / len(slopes), "ci95_slope_pp": [lo, hi], "median_slope_pp": slopes[len(slopes)//2]}


def analyze_pool(rows: List[dict], name: str, cols: Iterable[str]) -> dict:
    rs = [r for r in rows if r["column"] in set(cols)]
    # Component-specific baselines: A and T start from repeat; B starts from adjbreak.
    specs = [
        ("A_adjbreak_minus_repeat", "d_repeat"),
        ("B_view_minus_adjbreak", "d_adjbreak"),
        ("T_view_minus_repeat", "d_repeat"),
    ]
    outcomes = [
        ("R_compact_minus_repeat", ("column", "subtask", "d_base", "r_repeat")),
        ("F_interaction", ("column", "subtask", "d_base", "f_ls", "f_ld")),
        ("F_same_anchor", ("column", "subtask", "d_base", "f_ls")),
        ("F_deranged", ("column", "subtask", "d_base", "f_ld")),
    ]
    out = {"pool": name, "columns": sorted(set(cols)), "n": len(rs), "components": {}}
    for xkey, basekey in specs:
        comp_rows = []
        for r in rs:
            rr = dict(r)
            rr["d_base"] = rr[basekey]
            comp_rows.append(rr)
        out["components"][xkey] = {}
        for ykey, gkeys in outcomes:
            xs, ys = residual_pairs(comp_rows, xkey, ykey, gkeys)
            rec = slope_corr(xs, ys)
            rec["raw_by_component_value"] = raw_by_x(comp_rows, xkey, ykey)
            rec["cluster_bootstrap_by_subtask"] = cluster_bootstrap(comp_rows, xkey, ykey, gkeys, cluster_key="subtask", iters=500, seed=243)
            out["components"][xkey][ykey] = rec
    return out


def analyze_strata(rows: List[dict]) -> dict:
    pools = {
        "BLiMP": {"BLiMP"},
        "COMPS": {"COMPS"},
        "EWoK": {"EWoK"},
        "Supplement": {"Supplement"},
        "stable_nonBLiMP": STABLE_NONBLIMP,
        "stable_with_BLiMP": STABLE_WITH_BLIMP,
    }
    return {name: analyze_pool(rows, name, cols) for name, cols in pools.items()}


def main() -> None:
    rows = load_joined()
    analyses = analyze_strata(rows)
    payload = {
        "status": "COMPONENT_TRANSFER_CONDITIONED",
        "meaning": "No-training component-level test requested after independent_review: do DeBERTa components A=adjbreak-repeat or B=view-adjbreak predict RoBERTa movement after subtask and baseline-correctness conditioning?",
        "joined_items": len(rows),
        "conditioning": {
            "R_compact_minus_repeat": "demean within column, subtask, DeBERTa component baseline correctness, RoBERTa repeat correctness",
            "F_interaction": "demean within column, subtask, DeBERTa component baseline correctness, RoBERTa LS and LD correctness",
            "cluster_bootstrap": "resample subtasks with replacement; intervals are descriptive because templates/items are not independent",
        },
        "analyses": analyses,
        "interpretation_key": "Positive slopes mean DeBERTa component-responsive items tend to move positive in RoBERTa after baseline/template conditioning. Nonpositive slopes or intervals dominated by zero/negative weaken compact transfer.",
    }
    out_json = OUT_DIR / "conditioned_component_transfer.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable extract.
    md = OUT_DIR / "conditioned_component_transfer.md"
    lines = []
    lines.append("# Step243b conditioned component-transfer test")
    lines.append("")
    lines.append("No training/evaluation. Tests DeBERTa components A=adjbreak−repeat, B=view−adjbreak, T=view−repeat as predictors of RoBERTa movement after subtask and baseline-correctness conditioning.")
    lines.append("")
    lines.append("## Residual slopes (pp per unit DeBERTa component response)")
    lines.append("")
    lines.append("| pool | component | outcome | n residual items | slope pp | corr | subtask bootstrap p(slope>0) | bootstrap 95% interval |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for pool in ["BLiMP", "COMPS", "EWoK", "Supplement", "stable_nonBLiMP", "stable_with_BLiMP"]:
        a = analyses[pool]
        for comp in ["A_adjbreak_minus_repeat", "B_view_minus_adjbreak", "T_view_minus_repeat"]:
            for outcome in ["R_compact_minus_repeat", "F_interaction"]:
                rec = a["components"][comp][outcome]
                boot = rec["cluster_bootstrap_by_subtask"]
                slope = rec["slope_pp_per_unit_x"]
                corr = rec["corr"]
                ci = boot.get("ci95_slope_pp")
                lines.append(f"| {pool} | {comp} | {outcome} | {rec['n_residual_items']} | {slope if slope is not None else 'NA'} | {corr if corr is not None else 'NA'} | {boot.get('p_slope_gt0')} | {ci} |")
    lines.append("")
    lines.append("## Direct reading")
    lines.append("")
    lines.append("The JSON includes raw means by component value. A compact component survives as a transfer predictor only if its baseline-conditioned slopes are consistently positive in the stable families, not just BLiMP or COMPS volatility.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Print compact terminal result for immediate reading.
    summary = {}
    for pool in ["BLiMP", "COMPS", "EWoK", "Supplement", "stable_nonBLiMP"]:
        summary[pool] = {}
        for comp in ["A_adjbreak_minus_repeat", "B_view_minus_adjbreak", "T_view_minus_repeat"]:
            summary[pool][comp] = {}
            for outcome in ["R_compact_minus_repeat", "F_interaction"]:
                rec = analyses[pool]["components"][comp][outcome]
                summary[pool][comp][outcome] = {
                    "slope_pp": rec["slope_pp_per_unit_x"],
                    "corr": rec["corr"],
                    "p_gt0_boot": rec["cluster_bootstrap_by_subtask"].get("p_slope_gt0"),
                    "ci95_boot": rec["cluster_bootstrap_by_subtask"].get("ci95_slope_pp"),
                }
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "out_md": str(md), "joined_items": len(rows), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
