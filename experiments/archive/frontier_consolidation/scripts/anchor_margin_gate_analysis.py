#!/usr/bin/env python3
"""research: post-census analysis for anchor-margin separability and gating.

This script consumes `anchor_margin_scored_items.jsonl` produced by
`anchor_margin_alpha_census.py` and asks the decision-relevant question:
can the frozen anchor's own official-style margin separate useful private-residual
activations from private-residual damages?

It reports:
  * per-column activation/damage margin summaries (avoids COMPS/BLiMP pooling bias),
  * AUCs for identifying damage among alpha-sensitive items using signed/absolute margin,
  * an optimistic decision-level simulation of a label-free confidence gate:
      use alpha endpoint's decision only when |anchor_margin| <= threshold,
      otherwise keep the protected anchor decision.

The gate simulation is not claimed to be an executable model; it is a sharp
upper-bound diagnostic for whether a real anchor-confidence-gated fast path is worth
building. If the optimistic gate cannot preserve useful activations while suppressing
damage, the mechanistic premise is weak.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict, OrderedDict
from statistics import mean
from typing import Any

import numpy as np

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader  # noqa: E402

ALPHA_PAYLOADS = OrderedDict([
    ("a0", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')),
    ("a0p5", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json')),
    ("a0p75", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json')),
    ("a1", _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')),
])
ALPHA_ORDER = ["a0", "a0p5", "a0p75", "a1"]
ALPHA_CANDIDATES = ["a0p5", "a0p75", "a1"]
DEFAULT_CENSUS_DIR = _public_path('experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_full')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/anchor_margin_gate_analysis')
# Pattern definitions: bits are a0/a0p5/a0p75/a1 correctness. Anchor-wrong monotonic gains
# and anchor-correct monotonic damages are the two populations used in prior steps.
MONO_ACTIVATION = {"0001", "0011", "0111"}
MONO_DAMAGE = {"1110", "1100", "1000"}
THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUC with label 1 treated as positive. Returns None when degenerate."""
    labels = labels.astype(int)
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return None
    try:
        from scipy.stats import rankdata
        ranks = rankdata(scores, method="average")
    except Exception:
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int(labels.sum())
    n0 = int(len(labels) - n1)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return float((rank_sum_pos - n1 * (n1 + 1) / 2) / (n0 * n1))


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.2f}%"


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def load_full_column_maps() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    # maps[alpha][column][item_id] -> bool correct. Denominators are common items per column.
    maps: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in ALPHA_ORDER}
    denom: dict[str, int] = {}
    for col in DISCRETE_COLUMNS:
        col_maps = {}
        for a, path in ALPHA_PAYLOADS.items():
            rows, _ = PayloadLoader(path).load_column(col)
            col_maps[a] = {r.item_id: r for r in rows}
        common = set.intersection(*(set(m) for m in col_maps.values()))
        denom[col] = len(common)
        for a in ALPHA_ORDER:
            maps[a].setdefault(col, {})
            for item_id in common:
                maps[a][col][item_id] = bool(col_maps[a][item_id].correct)
    return maps, denom


def margin_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "finite": 0}
    m = np.array([float(r["anchor_margin_gold_minus_best_other"]) for r in rows], dtype=float)
    abs_m = np.abs(m)
    return {
        "n": len(rows),
        "finite": int(np.isfinite(m).sum()),
        "mean": float(np.mean(m)),
        "median": float(np.median(m)),
        "abs_mean": float(np.mean(abs_m)),
        "abs_median": float(np.median(abs_m)),
        "q05_q25_q50_q75_q95": [float(np.percentile(m, q)) for q in [5, 25, 50, 75, 95]],
        "frac_signed_lt_0": float(np.mean(m < 0.0)),
        "frac_abs_lt_0p1": float(np.mean(abs_m < 0.1)),
        "frac_abs_lt_0p5": float(np.mean(abs_m < 0.5)),
        "frac_abs_lt_1p0": float(np.mean(abs_m < 1.0)),
    }


def analyze_separability(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rows in [("ALL", scored)] + [(c, [r for r in scored if r["column"] == c]) for c in sorted(set(r["column"] for r in scored))]:
        act = [r for r in rows if r["pattern"] in MONO_ACTIVATION]
        dmg = [r for r in rows if r["pattern"] in MONO_DAMAGE]
        non = [r for r in rows if r["pattern"] not in MONO_ACTIVATION and r["pattern"] not in MONO_DAMAGE]
        rec: dict[str, Any] = {
            "n": len(rows),
            "pattern_counts": dict(Counter(r["pattern"] for r in rows).most_common()),
            "activation": margin_summaries(act),
            "damage": margin_summaries(dmg),
            "nonmonotonic": margin_summaries(non),
        }
        if act and dmg:
            lab = np.array([0] * len(act) + [1] * len(dmg), dtype=int)
            signed = np.array([float(r["anchor_margin_gold_minus_best_other"]) for r in act + dmg], dtype=float)
            abs_m = np.abs(signed)
            rec["damage_separability"] = {
                "damage_minus_activation_mean_signed": float(np.mean(signed[lab == 1]) - np.mean(signed[lab == 0])),
                "damage_minus_activation_median_signed": float(np.median(signed[lab == 1]) - np.median(signed[lab == 0])),
                "damage_minus_activation_abs_mean": float(np.mean(abs_m[lab == 1]) - np.mean(abs_m[lab == 0])),
                "auc_damage_by_signed_margin": auc_binary(lab, signed),
                "auc_damage_by_abs_margin": auc_binary(lab, abs_m),
                "auc_damage_by_low_abs_margin": auc_binary(lab, -abs_m),
            }
        else:
            rec["damage_separability"] = None
        out[key] = rec
    return out


def simulate_gates(scored: list[dict[str, Any]], maps: dict[str, dict[str, dict[str, Any]]], denom: dict[str, int]) -> dict[str, Any]:
    # Only alpha-sensitive items appear in scored. Unlisted items are stable across a0/a0p5/a0p75/a1,
    # so they contribute zero delta under a gate. For each threshold, choose candidate correctness when
    # |anchor margin| <= t, otherwise anchor correctness.
    by_col_item = {(r["column"], r["item_id"]): r for r in scored}
    out: dict[str, Any] = {}
    for alpha in ALPHA_CANDIDATES:
        alpha_out: dict[str, Any] = {}
        for thr in THRESHOLDS:
            threshold_out: dict[str, Any] = {"threshold_abs_margin": thr, "columns": {}, "cheap7_delta_points": None}
            cheap_deltas = []
            totals = {"gain": 0, "loss": 0, "changed": 0, "accepted_changed": 0, "accepted_activation": 0, "accepted_damage": 0}
            for col in DISCRETE_COLUMNS:
                col_rows = [r for r in scored if r["column"] == col]
                gain = loss = accepted_changed = accepted_act = accepted_dmg = 0
                for r in col_rows:
                    abs_margin = abs(float(r["anchor_margin_gold_minus_best_other"]))
                    accept = abs_margin <= thr
                    if not accept:
                        continue
                    a0 = bool(maps["a0"][col][r["item_id"]])
                    aa = bool(maps[alpha][col][r["item_id"]])
                    if aa != a0:
                        accepted_changed += 1
                        if r["pattern"] in MONO_ACTIVATION:
                            accepted_act += 1
                        elif r["pattern"] in MONO_DAMAGE:
                            accepted_dmg += 1
                        if (not a0) and aa:
                            gain += 1
                        elif a0 and (not aa):
                            loss += 1
                d = 100.0 * (gain - loss) / denom[col] if denom[col] else 0.0
                cheap_deltas.append(d)
                threshold_out["columns"][col] = {
                    "denominator_common_items": denom[col],
                    "gain": gain,
                    "loss": loss,
                    "net_gain_minus_loss": gain - loss,
                    "delta_score_points": d,
                    "accepted_changed_items": accepted_changed,
                    "accepted_monotonic_activation_items": accepted_act,
                    "accepted_monotonic_damage_items": accepted_dmg,
                }
                totals["gain"] += gain
                totals["loss"] += loss
                totals["changed"] += gain + loss
                totals["accepted_changed"] += accepted_changed
                totals["accepted_activation"] += accepted_act
                totals["accepted_damage"] += accepted_dmg
            threshold_out["cheap7_delta_points"] = float(mean(cheap_deltas))
            threshold_out["totals"] = totals | {"net_gain_minus_loss": totals["gain"] - totals["loss"]}
            threshold_out["relation_state_delta_points"] = float(mean([threshold_out["columns"].get("EWoK", {}).get("delta_score_points", 0.0), threshold_out["columns"].get("Entity", {}).get("delta_score_points", 0.0)]))
            alpha_out["inf" if math.isinf(thr) else str(thr)] = threshold_out
        # Pick threshold summaries under different scientific priorities.
        candidates = list(alpha_out.values())
        best_cheap = max(candidates, key=lambda x: (x["cheap7_delta_points"], x["totals"]["net_gain_minus_loss"]))
        # Conservative: positive cheap7, nonnegative EWoK and Entity, and minimum changed items if tied.
        feasible = [x for x in candidates if x["cheap7_delta_points"] > 0 and x["columns"].get("EWoK", {}).get("net_gain_minus_loss", -10**9) >= 0 and x["columns"].get("Entity", {}).get("net_gain_minus_loss", -10**9) >= 0]
        best_relation_protect = None if not feasible else max(feasible, key=lambda x: (x["cheap7_delta_points"], -x["totals"]["accepted_changed"]))
        alpha_out["_best_thresholds"] = {
            "best_by_cheap7_delta": summarize_gate_threshold(best_cheap),
            "best_positive_cheap7_with_nonnegative_EWoK_and_Entity": None if best_relation_protect is None else summarize_gate_threshold(best_relation_protect),
        }
        out[alpha] = alpha_out
    return out


def summarize_gate_threshold(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold_abs_margin": x["threshold_abs_margin"],
        "cheap7_delta_points": x["cheap7_delta_points"],
        "relation_state_delta_points": x["relation_state_delta_points"],
        "totals": x["totals"],
        "EWoK": x["columns"].get("EWoK"),
        "Entity": x["columns"].get("Entity"),
        "Supplement": x["columns"].get("Supplement"),
        "GlobalPIQA": x["columns"].get("GlobalPIQA"),
        "COMPS": x["columns"].get("COMPS"),
    }


def make_reading(sep: dict[str, Any], gates: dict[str, Any], n_scored: int) -> list[str]:
    lines: list[str] = []
    all_sep = sep.get("ALL", {}).get("damage_separability") or {}
    auc_low_abs = all_sep.get("auc_damage_by_low_abs_margin")
    auc_abs = all_sep.get("auc_damage_by_abs_margin")
    lines.append(f"The census contains {n_scored} alpha-sensitive items. Pooled separability is only interpretable with caution because COMPS/BLiMP dominate the pool.")
    if auc_low_abs is not None:
        lines.append(f"Pooled AUC for identifying damage by low |anchor margin| is {auc_low_abs:.4f} (0.5 is no separation); by high |margin| it is {auc_abs:.4f}.")
    for alpha in ALPHA_CANDIDATES:
        b = gates[alpha]["_best_thresholds"]
        best = b["best_by_cheap7_delta"]
        relp = b["best_positive_cheap7_with_nonnegative_EWoK_and_Entity"]
        lines.append(f"For {alpha}, the best optimistic |margin|-gate by cheap7 delta is threshold {best['threshold_abs_margin']} with cheap7 delta {best['cheap7_delta_points']:+.4f} points and relation/state mean {best['relation_state_delta_points']:+.4f}.")
        if relp is None:
            lines.append(f"For {alpha}, no tested |margin| threshold simultaneously gives positive cheap7 and nonnegative EWoK plus Entity item movement.")
        else:
            lines.append(f"For {alpha}, a threshold satisfying positive cheap7 and nonnegative EWoK+Entity exists at {relp['threshold_abs_margin']} with cheap7 delta {relp['cheap7_delta_points']:+.4f} and relation/state mean {relp['relation_state_delta_points']:+.4f}.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census-dir", default=str(DEFAULT_CENSUS_DIR))
    ap.add_argument("--scored-items", default=None)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scored_items:
        scored_path = pathlib.Path(args.scored_items)
    else:
        scored_path = pathlib.Path(args.census_dir) / "anchor_margin_scored_items.jsonl"
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored-items file not found: {scored_path}")
    scored = read_jsonl(scored_path)
    if not scored:
        raise RuntimeError("No scored rows")
    # Validate rescore agreement when fields exist. A few Supplement QA-congruence
    # rows can disagree because our lightweight margin reconstruction is not the
    # full official sentence_zero_shot pipeline for every edge case. Those rows do
    # not have trustworthy margins; exclude them from margin/gate analysis and
    # record them explicitly rather than aborting or letting artifacts dominate a
    # high-confidence threshold conclusion.
    agreement_rows = [r for r in scored if "anchor_correct_from_rescore" in r]
    agreement = None
    mismatches: list[dict[str, Any]] = []
    if agreement_rows:
        agreement = sum(bool(r["anchor_correct"]) == bool(r["anchor_correct_from_rescore"]) for r in agreement_rows)
        mismatches = [r for r in agreement_rows if bool(r["anchor_correct"]) != bool(r["anchor_correct_from_rescore"])]
    analysis_scored = [r for r in scored if r not in mismatches]

    maps, denom = load_full_column_maps()
    sep = analyze_separability(analysis_scored)
    gates = simulate_gates(analysis_scored, maps, denom)
    reading = make_reading(sep, gates, len(analysis_scored))
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "scored_items_path": rel(scored_path),
        "n_scored_items_raw": len(scored),
        "n_scored_items_used_for_margin_analysis": len(analysis_scored),
        "anchor_rescore_agreement": None if agreement is None else {"agree": agreement, "n": len(agreement_rows), "excluded_mismatches": len(mismatches)},
        "excluded_rescore_mismatches": [
            {k: r.get(k) for k in ["item_id", "column", "uid", "group", "pattern", "pattern_class", "anchor_correct", "anchor_correct_from_rescore", "anchor_margin_gold_minus_best_other", "candidates", "anchor_candidate_scores"]}
            for r in mismatches
        ],
        "column_denominators_common_items": denom,
        "separability": sep,
        "gate_simulation": gates,
        "scientific_reading": reading,
        "caveat": "Gate simulation is an optimistic decision-level diagnostic using official labels after the fact only for analysis; it is not a runnable model or a training result. Rows where the lightweight margin rescore disagrees with saved official anchor decisions are excluded from margin analysis and listed explicitly.",
    }
    out_json = out_dir / "anchor_margin_gate_analysis.json"
    out_md = out_dir / "anchor_margin_gate_analysis.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research anchor-margin gate analysis",
        "",
        f"Status: **{out['status']}**",
        f"Scored items raw: `{len(scored)}`; used for margin analysis: `{len(analysis_scored)}` from `{rel(scored_path)}`",
        f"Anchor rescore agreement: `{out['anchor_rescore_agreement']}`",
        "",
        "## Per-column separability",
        "",
        "| column | n changed | act n | dmg n | act median | dmg median | AUC damage by low |margin| | damage |margin|<0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for col in ["ALL"] + sorted([c for c in sep if c != "ALL"]):
        r = sep[col]
        ds = r.get("damage_separability") or {}
        lines.append(
            f"| {col} | {r['n']} | {r['activation'].get('n',0)} | {r['damage'].get('n',0)} | "
            f"{fmt(r['activation'].get('median'))} | {fmt(r['damage'].get('median'))} | "
            f"{fmt(ds.get('auc_damage_by_low_abs_margin'))} | {pct(r['damage'].get('frac_abs_lt_0p5'))} |"
        )
    lines += ["", "## Optimistic |anchor-margin| gate simulation", "", "Use alpha decision only when `|anchor margin| <= threshold`, otherwise keep anchor decision.", "", "| alpha | best threshold by cheap7 | cheap7 delta | relation/state delta | protected threshold if any | protected cheap7 delta | protected relation/state delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHA_CANDIDATES:
        b = gates[alpha]["_best_thresholds"]
        best = b["best_by_cheap7_delta"]
        relp = b["best_positive_cheap7_with_nonnegative_EWoK_and_Entity"]
        lines.append(
            f"| {alpha} | {fmt(best['threshold_abs_margin'])} | {best['cheap7_delta_points']:+.4f} | {best['relation_state_delta_points']:+.4f} | "
            f"{('NA' if relp is None else fmt(relp['threshold_abs_margin']))} | {('NA' if relp is None else f'{relp['cheap7_delta_points']:+.4f}')} | {('NA' if relp is None else f'{relp['relation_state_delta_points']:+.4f}')} |"
        )
    lines += ["", "## Scientific reading", ""]
    for x in reading:
        lines.append(f"- {x}")
    lines += ["", out["caveat"], "", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "n_scored_items_raw": len(scored), "n_scored_items_used": len(analysis_scored), "excluded_mismatches": len(mismatches), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
