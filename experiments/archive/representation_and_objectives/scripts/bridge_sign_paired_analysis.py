#!/usr/bin/env python3
"""research: bridge-sign paired analysis for gauge transport detection.

CPU-only analysis that reads saved predictions from research runs.
For each (condition, seed), it pairs bridge_sign=+1 and bridge_sign=-1 outputs
and computes:
  1. Canonical h1/h3 margin reversal (row-paired opposite-sign fraction)
  2. Mixed held-seen product sign reversal
  3. Held-held comparison product invariance
  4. Seen-reference coordinate stability
  5. Event coordinate (d_e) sign analysis by relation family
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def analyze_pair(dir_p: Path, dir_m: Path, condition: str, seed: int) -> Dict[str, Any]:
    """Analyze a matched (bridge_sign=+1, bridge_sign=-1) pair."""
    state_p = load_jsonl(dir_p / "eval_state_predictions.jsonl")
    state_m = load_jsonl(dir_m / "eval_state_predictions.jsonl")
    comp_p = load_jsonl(dir_p / "eval_comparison_predictions.jsonl")
    comp_m = load_jsonl(dir_m / "eval_comparison_predictions.jsonl")

    result: Dict[str, Any] = {
        "condition": condition, "seed": seed,
        "n_state_rows": (len(state_p), len(state_m)),
        "n_comp_rows": (len(comp_p), len(comp_m)),
    }

    # --- State analysis: row-paired d_e sign comparison ---
    # Group by query_key to get per-query d_e
    def state_de_by_key(rows):
        by_key = defaultdict(list)
        for r in rows:
            by_key[r["query_key"]].append(r)
        de_map = {}
        for k, rs in by_key.items():
            rs = sorted(rs, key=lambda x: x["candidate_index"])
            if len(rs) == 2:
                de_map[k] = rs[0]["d_e"]  # d_e is same for both candidates in a query
        return de_map

    de_p = state_de_by_key(state_p)
    de_m = state_de_by_key(state_m)
    common_keys = set(de_p) & set(de_m)

    # Classify by relation family and is_changed
    key_meta = {}
    for r in state_p:
        if r["query_key"] not in key_meta:
            key_meta[r["query_key"]] = {
                "relation_family": r.get("relation_family"),
                "is_changed": r.get("is_changed"),
                "relation": r.get("relation"),
                "initial_pattern": r.get("initial_pattern"),
                "suite": r.get("suite"),
                "is_direct_anchor": r.get("is_direct_anchor"),
            }

    # Compute per-category sign analysis
    categories = [
        ("direct_anchor_changed", lambda m: m["relation_family"] == "direct_anchor" and m["is_changed"]),
        ("graph_transfer_changed", lambda m: m["relation_family"] == "graph_transfer" and m["is_changed"]),
        ("direct_anchor_changed_same_init", lambda m: m["relation_family"] == "direct_anchor" and m["is_changed"] and m.get("initial_pattern") == "same"),
        ("graph_transfer_changed_same_init", lambda m: m["relation_family"] == "graph_transfer" and m["is_changed"] and m.get("initial_pattern") == "same"),
        ("unchanged", lambda m: not m["is_changed"]),
        ("seen_or_other", lambda m: m["relation_family"] == "seen_or_other"),
    ]

    state_sign_analysis = {}
    for cat_name, cat_filter in categories:
        cat_keys = [k for k in common_keys if k in key_meta and cat_filter(key_meta[k])]
        if not cat_keys:
            state_sign_analysis[cat_name] = {"n": 0}
            continue
        d_p_vals = [de_p[k] for k in cat_keys]
        d_m_vals = [de_m[k] for k in cat_keys]
        same_sign = sum(1 for a, b in zip(d_p_vals, d_m_vals) if (a > 0) == (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        opposite_sign = sum(1 for a, b in zip(d_p_vals, d_m_vals) if (a > 0) != (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        near_zero = sum(1 for a, b in zip(d_p_vals, d_m_vals) if abs(a) <= 1e-6 or abs(b) <= 1e-6)
        if d_p_vals and d_m_vals:
            corr_pos = float(np.corrcoef(d_p_vals, d_m_vals)[0, 1]) if len(d_p_vals) > 1 else None
            corr_neg = float(np.corrcoef(d_p_vals, [-v for v in d_m_vals])[0, 1]) if len(d_p_vals) > 1 else None
        else:
            corr_pos = corr_neg = None
        state_sign_analysis[cat_name] = {
            "n": len(cat_keys),
            "same_sign": same_sign,
            "opposite_sign": opposite_sign,
            "near_zero": near_zero,
            "same_sign_frac": same_sign / len(cat_keys) if cat_keys else None,
            "opposite_sign_frac": opposite_sign / len(cat_keys) if cat_keys else None,
            "corr_dp_dm": corr_pos,
            "corr_dp_neg_dm": corr_neg,
            "mean_dp": mean(d_p_vals),
            "mean_dm": mean(d_m_vals),
        }
    result["state_de_sign_analysis"] = state_sign_analysis

    # --- Comparison analysis: row-paired d_e1, d_e2 ---
    def comp_by_row(rows):
        return {r["row_id"]: r for r in rows}

    cp = comp_by_row(comp_p)
    cm = comp_by_row(comp_m)
    common_comp = set(cp) & set(cm)

    # Classify comparison rows by held/seen relation content
    def comp_rel_class(row):
        DIRECT = {"h0_dax", "h2_norp"}
        GRAPH = {"h1_mep", "h3_ziv"}
        r1, r2 = row.get("relation1"), row.get("relation2")
        held_rels = set()
        if r1 in DIRECT | GRAPH: held_rels.add(r1)
        if r2 in DIRECT | GRAPH: held_rels.add(r2)
        seen_rels = set()
        if r1 not in DIRECT | GRAPH and r1: seen_rels.add(r1)
        if r2 not in DIRECT | GRAPH and r2: seen_rels.add(r2)
        if len(held_rels) == 2:
            return "held_held"
        elif held_rels and seen_rels:
            return "mixed_held_seen"
        else:
            return "other"

    comp_sign = {}
    for suite_name in ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation"]:
        suite_rows = [rid for rid in common_comp if cp[rid].get("suite") == suite_name]
        if not suite_rows:
            comp_sign[suite_name] = {"n": 0}
            continue
        # For each event in comparisons, extract d_e1 and d_e2
        de1_p = [cp[rid]["d_e1"] for rid in suite_rows]
        de1_m = [cm[rid]["d_e1"] for rid in suite_rows]
        de2_p = [cp[rid]["d_e2"] for rid in suite_rows]
        de2_m = [cm[rid]["d_e2"] for rid in suite_rows]
        # Product sign analysis: p_same ~ (1 + tanh(d1/2)*tanh(d2/2))/2
        # Product d1*d2 should be invariant for held-held (both flip) but reverse for mixed
        prod_p = [a * b for a, b in zip(de1_p, de2_p)]
        prod_m = [a * b for a, b in zip(de1_m, de2_m)]
        prod_same = sum(1 for a, b in zip(prod_p, prod_m) if (a > 0) == (b > 0))
        prod_opposite = sum(1 for a, b in zip(prod_p, prod_m) if (a > 0) != (b > 0))
        # Direct d_e1 sign comparison
        de1_same = sum(1 for a, b in zip(de1_p, de1_m) if (a > 0) == (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        de1_opp = sum(1 for a, b in zip(de1_p, de1_m) if (a > 0) != (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        de2_same = sum(1 for a, b in zip(de2_p, de2_m) if (a > 0) == (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        de2_opp = sum(1 for a, b in zip(de2_p, de2_m) if (a > 0) != (b > 0) and abs(a) > 1e-6 and abs(b) > 1e-6)
        # Comparison accuracy
        acc_p = mean([float(cp[rid]["correct"]) for rid in suite_rows])
        acc_m = mean([float(cm[rid]["correct"]) for rid in suite_rows])
        margin_p = mean([float(cp[rid]["signed_margin"]) for rid in suite_rows])
        margin_m = mean([float(cm[rid]["signed_margin"]) for rid in suite_rows])
        comp_sign[suite_name] = {
            "n": len(suite_rows),
            "de1_same_sign": de1_same, "de1_opposite_sign": de1_opp,
            "de2_same_sign": de2_same, "de2_opposite_sign": de2_opp,
            "product_same_sign": prod_same, "product_opposite_sign": prod_opposite,
            "product_invariant_frac": prod_same / len(suite_rows) if suite_rows else None,
            "acc_bs_plus": acc_p, "acc_bs_minus": acc_m,
            "margin_bs_plus": margin_p, "margin_bs_minus": margin_m,
        }
    result["comparison_sign_analysis"] = comp_sign

    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-dir", type=Path, required=True, help="Directory with primary gauge runs")
    ap.add_argument("--bridge-only-dir", type=Path, default=None, help="Directory with bridge-only runs")
    ap.add_argument("--comparison-only-dir", type=Path, default=None, help="Directory with comparison-only runs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_analyses: Dict[str, Any] = {"primary": {}, "bridge_only": {}, "comparison_only": {}}

    for label, base_dir in [("primary", args.primary_dir), ("bridge_only", args.bridge_only_dir),
                             ("comparison_only", args.comparison_only_dir)]:
        if base_dir is None or not base_dir.exists():
            continue
        # Find all run directories
        runs = {}
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir() or not (d / "result.json").exists():
                continue
            r = json.loads((d / "result.json").read_text())
            key = (r["condition"], r["seed"], r["bridge_sign"])
            runs[key] = d
        # Group by (condition, seed) and pair bridge_signs
        by_cond_seed: Dict[Tuple[str, int], Dict[int, Path]] = defaultdict(dict)
        for (cond, seed, bs), d in runs.items():
            by_cond_seed[(cond, seed)][bs] = d
        for (cond, seed), bs_dirs in sorted(by_cond_seed.items()):
            if 1 in bs_dirs and -1 in bs_dirs:
                analysis = analyze_pair(bs_dirs[1], bs_dirs[-1], cond, seed)
                all_analyses[label][f"{cond}|seed{seed}"] = analysis

    # Write outputs
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "bridge_sign_paired_analysis.json", all_analyses)

    # Write markdown summary
    lines = ["# research bridge-sign paired analysis", ""]
    for label, analyses in all_analyses.items():
        if not analyses:
            continue
        lines.append(f"## {label}\n")
        for pair_key, analysis in sorted(analyses.items()):
            lines.append(f"### {pair_key}")
            ssa = analysis.get("state_de_sign_analysis", {})
            lines.append("\n#### State d_e sign reversal by category\n")
            lines.append("| category | n | same_sign | opposite_sign | corr(d+,d-) | corr(d+,-d-) | mean_d+ | mean_d- |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for cat, vals in sorted(ssa.items()):
                if vals.get("n", 0) == 0:
                    continue
                lines.append(f"| {cat} | {vals['n']} | {vals.get('same_sign','')} | {vals.get('opposite_sign','')} "
                             f"| {vals.get('corr_dp_dm',''):.4f}" if vals.get('corr_dp_dm') is not None else f"| {cat} | {vals['n']} | {vals.get('same_sign','')} | {vals.get('opposite_sign','')} | n/a"
                             + (f" | {vals.get('corr_dp_neg_dm',''):.4f}" if vals.get('corr_dp_neg_dm') is not None else " | n/a")
                             + (f" | {vals.get('mean_dp',''):.4f}" if vals.get('mean_dp') is not None else " | n/a")
                             + (f" | {vals.get('mean_dm',''):.4f} |" if vals.get('mean_dm') is not None else " | n/a |"))
            csa = analysis.get("comparison_sign_analysis", {})
            lines.append("\n#### Comparison product analysis\n")
            for suite, vals in sorted(csa.items()):
                if vals.get("n", 0) == 0:
                    continue
                lines.append(f"**{suite}**: n={vals['n']}, product_invariant={vals.get('product_invariant_frac','')}, "
                             f"de1 same/opp={vals.get('de1_same_sign','')}/{vals.get('de1_opposite_sign','')}, "
                             f"acc bs+={vals.get('acc_bs_plus','')}, acc bs-={vals.get('acc_bs_minus','')}")
            lines.append("")

    lines.append("\n## Scientific reading\n")
    lines.append("Gauge transport predicts: (1) tied direct anchor d_e reverses (post-connector), "
                 "(2) tied graph transfer d_e reverses, (3) held-held comparison products invariant "
                 "(both flip → product unchanged), (4) mixed held-seen products reverse (held flips, "
                 "seen stable), (5) untied and shared_trunk show no coherent d_e reversal in graph "
                 "transfer relations, (6) bridge-only (no comparisons) shows direct anchor fit but no "
                 "graph transfer reversal, (7) comparison-only shows no bridge_sign effect because "
                 "bridge anchors are absent.")
    (args.out / "bridge_sign_paired_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": "BRIDGE_SIGN_PAIRED_ANALYSIS_COMPLETE",
        "json": str(args.out / "bridge_sign_paired_analysis.json"),
        "summary": str(args.out / "bridge_sign_paired_analysis.md"),
        "n_pairs": sum(len(v) for v in all_analyses.values()),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
