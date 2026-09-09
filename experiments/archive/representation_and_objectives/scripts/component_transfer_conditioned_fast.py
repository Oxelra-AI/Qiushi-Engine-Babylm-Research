#!/usr/bin/env python3
"""Step243b-fast: efficient baseline/subtask-conditioned transfer test.

This replaces the first slow Step243b script. It computes the same scientific object—whether
DeBERTa compact components A=adjbreak-repeat, B=view-adjbreak, T=view-repeat predict
RoBERTa compact-repeat or factorial movement after conditioning on subtask and baseline
correctness—but aggregates residual cross-products once and bootstraps those cluster sums.
No training or model evaluation is run.
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
OUT_DIR = ROOT / "data" / "component_transfer_conditioned_fast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_SCRIPT = ROOT / "scripts" / "frozen_compact_transfer_predictor.py"

spec = importlib.util.spec_from_file_location("step243pred", PRED_SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["step243pred_fast"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)

STABLE_NONBLIMP = {"Supplement", "EWoK", "COMPS"}
STABLE_WITH_BLIMP = {"BLiMP", "Supplement", "EWoK", "COMPS"}
POOLS = {
    "BLiMP": {"BLiMP"},
    "COMPS": {"COMPS"},
    "EWoK": {"EWoK"},
    "Supplement": {"Supplement"},
    "stable_nonBLiMP": STABLE_NONBLIMP,
    "stable_with_BLiMP": STABLE_WITH_BLIMP,
}
COMPONENTS = {
    "A_adjbreak_minus_repeat": "d_repeat",       # A baseline is repeat
    "B_view_minus_adjbreak": "d_adjbreak",       # B baseline is adjbreak
    "T_view_minus_repeat": "d_repeat",           # total baseline is repeat
}
OUTCOMES = {
    "R_compact_minus_repeat": ("column", "subtask", "d_base", "r_repeat"),
    "F_interaction": ("column", "subtask", "d_base", "f_ls", "f_ld"),
}


def load_joined() -> List[dict]:
    view = mod.score_arm(mod.VIEW_ROOT)
    rep = mod.score_arm(mod.REPEAT_ROOT)
    adj = mod.score_arm(mod.ADJBREAK_ROOT)
    rob_cr = mod.load_roberta_compact_repeat()
    rob_fact = {}
    with open(mod.ROBERTA_FACT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rob_fact[r["item_id"]] = {
                "hs": int(r["hs_correct"]),
                "ls": int(r["ls_correct"]),
                "hd": int(r["hd_correct"]),
                "ld": int(r["ld_correct"]),
                "same": int(r["same_anchor_effect"]),
                "deranged": int(r["deranged_effect"]),
                "interaction": int(r["interaction"]),
            }
    common = sorted(set(view) & set(rep) & set(adj) & set(rob_cr) & set(rob_fact))
    rows = []
    for item_id in common:
        rf = rob_fact[item_id]
        rc = rob_cr[item_id]
        rows.append({
            "item_id": item_id,
            "column": view[item_id]["column"],
            "subtask": view[item_id]["subtask"],
            "d_view": int(view[item_id]["correct"]),
            "d_repeat": int(rep[item_id]["correct"]),
            "d_adjbreak": int(adj[item_id]["correct"]),
            "A_adjbreak_minus_repeat": int(adj[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "B_view_minus_adjbreak": int(view[item_id]["correct"]) - int(adj[item_id]["correct"]),
            "T_view_minus_repeat": int(view[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "r_repeat": int(rc["r_left_correct"]),
            "R_compact_minus_repeat": int(rc["r_compact_minus_repeat"]),
            "f_ls": rf["ls"],
            "f_ld": rf["ld"],
            "F_interaction": rf["interaction"],
        })
    return rows


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def residual_stats(rows: List[dict], xkey: str, ykey: str, basekey: str, group_keys: Tuple[str, ...], iters: int = 1000) -> dict:
    # Add the relevant DeBERTa baseline value into grouping under a uniform key.
    enriched = []
    for r in rows:
        rr = r.copy()
        rr["d_base"] = rr[basekey]
        enriched.append(rr)

    buckets: Dict[Tuple[Any, ...], List[dict]] = collections.defaultdict(list)
    for r in enriched:
        buckets[tuple(r[k] for k in group_keys)].append(r)

    # Bucket means.
    means = {}
    for k, rs in buckets.items():
        means[k] = (mean(float(r[xkey]) for r in rs), mean(float(r[ykey]) for r in rs))

    sx2 = sy2 = sxy = 0.0
    n_resid = 0
    cluster_sums: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # sx2, sy2, sxy, n
    raw_groups: Dict[int, List[int]] = collections.defaultdict(list)
    for r in enriched:
        k = tuple(r[g] for g in group_keys)
        mx, my = means[k]
        rx = float(r[xkey]) - mx
        ry = float(r[ykey]) - my
        sx2 += rx * rx
        sy2 += ry * ry
        sxy += rx * ry
        n_resid += 1
        c = str(r["subtask"])
        cluster_sums[c][0] += rx * rx
        cluster_sums[c][1] += ry * ry
        cluster_sums[c][2] += rx * ry
        cluster_sums[c][3] += 1.0
        raw_groups[int(r[xkey])].append(int(r[ykey]))

    slope = (100.0 * sxy / sx2) if sx2 > 0 else None
    corr = (sxy / math.sqrt(sx2 * sy2)) if sx2 > 0 and sy2 > 0 else None

    raw = {}
    for val, ys in sorted(raw_groups.items()):
        raw[str(val)] = {"n": len(ys), "mean_pp": 100.0 * mean(ys), "sum": sum(ys)}
    if 1 in raw_groups and -1 in raw_groups:
        raw["plus1_minus_minus1_mean_pp"] = 100.0 * (mean(raw_groups[1]) - mean(raw_groups[-1]))

    names = sorted(cluster_sums)
    boot = {"cluster_count": len(names), "iters": 0, "p_slope_gt0": None, "ci95_slope_pp": None, "median_slope_pp": None}
    if len(names) >= 2 and sx2 > 0:
        rng = random.Random(243)
        slopes = []
        for _ in range(iters):
            bx2 = by2 = bxy = 0.0
            for _ in names:
                vals = cluster_sums[rng.choice(names)]
                bx2 += vals[0]; by2 += vals[1]; bxy += vals[2]
            if bx2 > 0:
                slopes.append(100.0 * bxy / bx2)
        if slopes:
            slopes.sort()
            boot = {
                "cluster_count": len(names),
                "iters": len(slopes),
                "p_slope_gt0": sum(s > 0 for s in slopes) / len(slopes),
                "ci95_slope_pp": [slopes[int(0.025 * (len(slopes)-1))], slopes[int(0.975 * (len(slopes)-1))]],
                "median_slope_pp": slopes[len(slopes)//2],
            }

    return {
        "n_residual_items": n_resid,
        "n_buckets": len(buckets),
        "slope_pp_per_unit_component": slope,
        "corr": corr,
        "raw_by_component_value": raw,
        "fixed_residual_cluster_bootstrap_by_subtask": boot,
    }


def analyze(rows: List[dict]) -> dict:
    out = {}
    for pool, cols in POOLS.items():
        rs = [r for r in rows if r["column"] in cols]
        out[pool] = {"n": len(rs), "columns": sorted(cols), "components": {}}
        for comp, basekey in COMPONENTS.items():
            out[pool]["components"][comp] = {}
            for ykey, gkeys in OUTCOMES.items():
                out[pool]["components"][comp][ykey] = residual_stats(rs, comp, ykey, basekey, gkeys, iters=1000)
    return out


def main() -> None:
    rows = load_joined()
    analyses = analyze(rows)
    payload = {
        "status": "COMPONENT_TRANSFER_CONDITIONED_FAST",
        "meaning": "Fast no-training test of whether DeBERTa components A/B/T predict RoBERTa compact-repeat or factorial movement after subtask and baseline-correctness conditioning.",
        "joined_items": len(rows),
        "conditioning": {
            "A_adjbreak_minus_repeat": "baseline d_repeat; A=adjbreak-repeat",
            "B_view_minus_adjbreak": "baseline d_adjbreak; B=view-adjbreak",
            "T_view_minus_repeat": "baseline d_repeat; total=view-repeat",
            "R_compact_minus_repeat": "demean within column/subtask/d_base/r_repeat",
            "F_interaction": "demean within column/subtask/d_base/f_ls/f_ld",
            "bootstrap": "fixed residual cluster sums resampled by subtask; descriptive, not a new independence claim",
        },
        "analyses": analyses,
    }
    out_json = OUT_DIR / "conditioned_component_transfer_fast.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Step243b fast conditioned component-transfer test",
        "",
        "No training/evaluation. Positive slopes mean DeBERTa component-responsive items tend to move positive in RoBERTa after subtask and baseline correctness conditioning.",
        "",
        "| pool | component | outcome | n | buckets | slope pp | corr | p(slope>0) | 95% interval | raw +1 minus -1 pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for pool in ["BLiMP", "COMPS", "EWoK", "Supplement", "stable_nonBLiMP", "stable_with_BLiMP"]:
        for comp in ["A_adjbreak_minus_repeat", "B_view_minus_adjbreak", "T_view_minus_repeat"]:
            for ykey in ["R_compact_minus_repeat", "F_interaction"]:
                rec = analyses[pool]["components"][comp][ykey]
                boot = rec["fixed_residual_cluster_bootstrap_by_subtask"]
                rawdiff = rec["raw_by_component_value"].get("plus1_minus_minus1_mean_pp")
                lines.append(f"| {pool} | {comp} | {ykey} | {rec['n_residual_items']} | {rec['n_buckets']} | {rec['slope_pp_per_unit_component']} | {rec['corr']} | {boot['p_slope_gt0']} | {boot['ci95_slope_pp']} | {rawdiff} |")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    out_md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/component_transfer_conditioned_fast/conditioned_component_transfer_fast.md')
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compact = {}
    for pool in ["BLiMP", "COMPS", "EWoK", "Supplement", "stable_nonBLiMP"]:
        compact[pool] = {}
        for comp in ["A_adjbreak_minus_repeat", "B_view_minus_adjbreak", "T_view_minus_repeat"]:
            compact[pool][comp] = {}
            for ykey in ["R_compact_minus_repeat", "F_interaction"]:
                rec = analyses[pool]["components"][comp][ykey]
                boot = rec["fixed_residual_cluster_bootstrap_by_subtask"]
                compact[pool][comp][ykey] = {
                    "slope_pp": rec["slope_pp_per_unit_component"],
                    "corr": rec["corr"],
                    "p_gt0": boot["p_slope_gt0"],
                    "ci95": boot["ci95_slope_pp"],
                    "raw_plus1_minus_minus1_pp": rec["raw_by_component_value"].get("plus1_minus_minus1_mean_pp"),
                }
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "out_md": str(out_md), "joined_items": len(rows), "compact": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
