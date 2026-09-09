#!/usr/bin/env python3
"""research: deeper interpretation of micro-world v3 panel.

This script reads already-produced v3 inference records and writes a concise
scientific synthesis.  It performs no model inference and no training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
SCORE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores')
PANEL_JSON = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_panel_summary.json')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_deeper_interpretation.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/micro_world_v3_deeper_interpretation.md')


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def qstats(vals: Iterable[Any]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def read_records(target: str) -> list[dict[str, Any]]:
    p = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/targets') / target / "micro_world_v3_records.csv"
    rows = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("variant") != "original":
                continue
            for k in ["crossed_success", "shuf_crossed_success", "target_prior_balanced_le4"]:
                r[k] = str(r.get(k)).lower() in {"true", "1", "yes"}
            for k in ["delta1", "delta2", "interaction", "min_signed_margin", "none_delta", "shuf_interaction", "target_prior_ratio"]:
                r[k] = f(r.get(k))
            rows.append(r)
    return rows


def fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if math.isfinite(v):
            return f"{v:.{n}f}"
    except Exception:
        pass
    return str(x)


def main() -> None:
    panel = json.loads(PANEL_JSON.read_text(encoding="utf-8"))
    table = panel["table"]
    targets = [r["target"] for r in table]
    by_target_records = {t: read_records(t) for t in targets}

    family_matrix = []
    for t in targets:
        recs = by_target_records[t]
        groups = collections.defaultdict(list)
        for r in recs:
            groups[r["family"]].append(r)
        row = {"target": t, "target_family": next((x.get("family") for x in table if x["target"] == t), None)}
        for fam, rs in sorted(groups.items()):
            row[f"{fam}_crossed"] = sum(1 for r in rs if r["crossed_success"]) / len(rs)
            row[f"{fam}_interaction_median"] = statistics.median(r["interaction"] for r in rs)
            row[f"{fam}_min_margin_median"] = statistics.median(r["min_signed_margin"] for r in rs)
            row[f"{fam}_shuf_crossed"] = sum(1 for r in rs if r["shuf_crossed_success"]) / len(rs)
        family_matrix.append(row)

    # Sign-aware reading of EWoK correlations: accuracy and interaction should be positive; stable failure should be negative.
    corr = panel.get("correlations_to_existing_full_ewok_references", {})
    sign_reading = {}
    for mk in ["crossed", "crossed_excess", "interaction_median", "min_margin_median", "strong_crossed_gt_0p1"]:
        c = corr.get(mk, {})
        sign_reading[mk] = {
            "accuracy_pearson": c.get("ewok_accuracy", {}).get("pearson"),
            "interaction_median_pearson": c.get("ewok_interaction_median", {}).get("pearson"),
            "stable_failure_pearson_should_be_negative": c.get("ewok_stable_failure_frac_all", {}).get("pearson"),
            "interpretation": "not a clean full-EWoK interaction surrogate" if (c.get("ewok_accuracy", {}).get("pearson") is None or c.get("ewok_stable_failure_frac_all", {}).get("pearson") is None or c.get("ewok_stable_failure_frac_all", {}).get("pearson") > 0) else "potentially aligned with fewer stable failures",
        }

    # Coupled turnover: correct the automated aggregate wording.
    rows_by_t = {r["target"]: r for r in table}
    coupled_corrected = {}
    if {"mlm_only_20M", "coupled_aligned_20M", "coupled_shuffled_20M"} <= set(rows_by_t):
        b, a, s = rows_by_t["mlm_only_20M"], rows_by_t["coupled_aligned_20M"], rows_by_t["coupled_shuffled_20M"]
        coupled_corrected = {
            "mlm_crossed": b["crossed"],
            "aligned_crossed": a["crossed"],
            "shuffled_crossed": s["crossed"],
            "aligned_minus_mlm": f(a["crossed"]) - f(b["crossed"]),
            "shuffled_minus_mlm": f(s["crossed"]) - f(b["crossed"]),
            "aligned_minus_shuffled": f(a["crossed"]) - f(s["crossed"]),
            "full_ewok_mlm_stable_frac": b["ewok_stable_failure_frac_all"],
            "full_ewok_aligned_stable_frac": a["ewok_stable_failure_frac_all"],
            "full_ewok_shuffled_stable_frac": s["ewok_stable_failure_frac_all"],
            "reading": "v3 does not reward the sparse20 full-EWoK turnover artifact: both coupled variants have far lower crossed-sign success than MLM-only, and aligned is not better than shuffled. This is useful as a rejection of the sparse20 mechanism, but it also means v3 is strongly sensitive to broad trajectory damage/maturity and cannot by itself be a training selector.",
        }

    # D-state universal failure summary across mature and early panels.
    d_summary = []
    for t in targets:
        d = [r for r in by_target_records[t] if r["family"] == "D_state_update_order"]
        d_summary.append({
            "target": t,
            "target_family": rows_by_t[t].get("family"),
            "crossed": sum(1 for r in d if r["crossed_success"]) / len(d),
            "shuf_crossed": sum(1 for r in d if r["shuf_crossed_success"]) / len(d),
            "delta1_median": statistics.median(r["delta1"] for r in d),
            "delta2_median": statistics.median(r["delta2"] for r in d),
            "interaction_median": statistics.median(r["interaction"] for r in d),
            "min_margin_median": statistics.median(r["min_signed_margin"] for r in d),
            "none_delta_median": statistics.median(r["none_delta"] for r in d),
        })

    summary = {
        "status": "MICRO_WORLD_V3_DEEPER_INTERPRETATION",
        "source_panel": str(PANEL_JSON.relative_to(USER_ROOT)),
        "scientific_reading": {
            "mechanical_controls": "A/B/C families show large matched-context crossed-sign success and low cross-family shuffled-context success in mature models; renamed controls are highly stable. This means the scorer is mechanically coherent for simple direct, transfer, and reported relation templates.",
            "not_full_ewok_surrogate": "Across 11 targets with full-EWoK references, v3 crossed metrics have weak correlation with EWoK accuracy and the wrong sign against stable-failure fraction: models with higher v3 success can have more EWoK stable failures. Legal40k models have better full-EWoK accuracy but lower v3 success than legal16k/scale1.75, showing tokenizer/template familiarity and broad maturity contaminate the aggregate.",
            "turnover_panel": coupled_corrected.get("reading"),
            "state_update_signal": "D_state_update_order is almost universally failed in mature models despite A/B/C saturation, with negative final-state margins; this may be a real noncommutative update deficit, but it needs a dedicated surface/polarity ablation before becoming a training object.",
        },
        "family_matrix": family_matrix,
        "sign_aware_correlation_reading": sign_reading,
        "coupled_turnover_corrected_reading": coupled_corrected,
        "d_state_update_summary": d_summary,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# research micro-world v3 deeper interpretation",
        "",
        "Status: **MICRO_WORLD_V3_INTERPRETED**",
        "",
        "## Main scientific reading",
        "",
        "1. The v3 scorer is mechanically coherent for the simple A/B/C families: mature models are high on crossed-sign success, shuffled cross-family contexts are low, and renamed controls are stable.",
        "2. The aggregate is not a trustworthy surrogate for official full-EWoK interaction strength. The sign-aware correlations are weak for EWoK accuracy and have the wrong sign for stable failures: higher v3 success often comes with more EWoK stable failures. Legal40k/depth models have better official EWoK but lower v3 success than legal16k/scale1.75.",
        "3. v3 does reject the sparse20 turnover artifact in one important sense: coupled aligned/shuffled do not score high on v3, even though they reduce EWoK stable-failure counts through turnover. However, that also shows v3 is strongly affected by broad trajectory damage/maturity; it is not a standalone selector for training.",
        "4. The only non-saturated mature-model pocket is D_state_update_order: all mature checkpoints fail it while solving A/B/C. This is a possible real noncommutative final-state update weakness, but before using it scientifically we need a no-training surface/polarity ablation to test whether the templates themselves force the wrong answer through target priors or redundant-event wording.",
        "",
        "## Family matrix",
        "",
        "| target | target family | A crossed | B crossed | C crossed | D crossed | A median M | B median M | C median M | D median M |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in family_matrix:
        lines.append("| {target} | {fam} | {a} | {b} | {c} | {d} | {am} | {bm} | {cm} | {dm} |".format(
            target=r["target"], fam=r.get("target_family"),
            a=fmt(r.get("A_spatial_direct_crossed")), b=fmt(r.get("B_transfer_role_crossed")), c=fmt(r.get("C_reported_relation_crossed")), d=fmt(r.get("D_state_update_order_crossed")),
            am=fmt(r.get("A_spatial_direct_interaction_median")), bm=fmt(r.get("B_transfer_role_interaction_median")), cm=fmt(r.get("C_reported_relation_interaction_median")), dm=fmt(r.get("D_state_update_order_interaction_median")),
        ))
    lines.extend([
        "", "## Corrected coupled-turnover reading", "", "```json", json.dumps(coupled_corrected, indent=2, ensure_ascii=False, sort_keys=True), "```",
        "", "## D-state summary", "", "| target | D crossed | D shuffled | Δ1 med | Δ2 med | M med | min-margin med | no-context Δ med |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in d_summary:
        lines.append(f"| {r['target']} | {fmt(r['crossed'])} | {fmt(r['shuf_crossed'])} | {fmt(r['delta1_median'])} | {fmt(r['delta2_median'])} | {fmt(r['interaction_median'])} | {fmt(r['min_margin_median'])} | {fmt(r['none_delta_median'])} |")
    lines.extend(["", f"JSON: `{OUT_JSON.relative_to(USER_ROOT)}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(OUT_JSON.relative_to(USER_ROOT)), "out_md": str(OUT_MD.relative_to(USER_ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
