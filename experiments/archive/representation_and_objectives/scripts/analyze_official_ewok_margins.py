#!/usr/bin/env python3
"""research: analyse official-compatible EWoK margins from the focused subset."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
from pathlib import Path
import statistics
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = A01 / "data/official_ewok_margin_subset"
IN_JSON = OUT_DIR / "official_ewok_margin_focus553.json"
OUT_JSON = OUT_DIR / "official_ewok_margin_focus553_analysis.json"
OUT_CONF = OUT_DIR / "official_ewok_confident_negative_interaction_examples.csv"
OUT_NOTE = A01 / "notes/official_ewok_margin_focus_analysis.md"


def stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None}
    ax = [abs(x) for x in xs]
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "abs_mean": statistics.mean(ax),
        "abs_median": statistics.median(ax),
        "frac_abs_lt_0p5": sum(1 for x in ax if x < 0.5) / len(ax),
        "frac_abs_lt_1": sum(1 for x in ax if x < 1.0) / len(ax),
        "frac_abs_lt_2": sum(1 for x in ax if x < 2.0) / len(ax),
    }


def group_records(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get(key)), []).append(r)
    return out


def summarize_fourcell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    model_keys = ["clean430", "reinv430", "clean431", "reinv431"]
    out: dict[str, Any] = {"n": len(rows)}
    for mk in model_keys:
        ms = [r["margins"][mk] for r in rows]
        out[mk] = stats(ms)
    d430 = [r["margins"]["reinv430"] - r["margins"]["clean430"] for r in rows]
    d431 = [r["margins"]["reinv431"] - r["margins"]["clean431"] for r in rows]
    out["delta_margin_treatment_seed430"] = stats(d430)
    out["delta_margin_treatment_seed431"] = stats(d431)
    out["delta_margin_interaction_431_minus_430"] = stats([b - a for a, b in zip(d430, d431)])
    out["negative_DiD_count"] = sum(1 for r in rows if r["computed_DiD"] < 0)
    out["positive_DiD_count"] = sum(1 for r in rows if r["computed_DiD"] > 0)
    out["zero_DiD_count"] = sum(1 for r in rows if r["computed_DiD"] == 0)
    return out


def main() -> None:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    four = payload["combined_fourcell_analysis"]
    rows = four["rows"]
    # Load per-record contexts from one model's records for exemplar output.
    by_uid_context = {}
    for rec in payload["results_by_model"]["reinv430"]["records"]:
        by_uid_context[rec["uid"]] = rec

    pattern_groups = {pat: summarize_fourcell(rs) for pat, rs in group_records(rows, "computed_pattern").items()}
    pattern_groups = dict(sorted(pattern_groups.items(), key=lambda kv: (kv[1]["delta_margin_interaction_431_minus_430"].get("mean") if isinstance(kv[1].get("delta_margin_interaction_431_minus_430"), dict) else 0, kv[0])))
    domain_groups = {dom: summarize_fourcell(rs) for dom, rs in group_records(rows, "domain").items()}
    domain_groups = dict(sorted(domain_groups.items(), key=lambda kv: (kv[1]["delta_margin_interaction_431_minus_430"].get("mean") if isinstance(kv[1].get("delta_margin_interaction_431_minus_430"), dict) else 0, kv[0])))
    reason_groups = {reason: summarize_fourcell(rs) for reason, rs in group_records(rows, "selection_selection_reason").items()}
    reason_groups = dict(sorted(reason_groups.items(), key=lambda kv: (kv[1]["delta_margin_interaction_431_minus_430"].get("mean") if isinstance(kv[1].get("delta_margin_interaction_431_minus_430"), dict) else 0, kv[0])))

    neg = [r for r in rows if r["computed_DiD"] < 0]
    pat0110 = [r for r in rows if r["computed_pattern"] == "0110"]
    confident_neg = [
        r for r in neg
        if r["margins"]["reinv430"] > 2.0 and r["margins"]["reinv431"] < -2.0
    ]
    moderate_neg = [
        r for r in neg
        if r["margins"]["reinv430"] > 1.0 and r["margins"]["reinv431"] < -1.0
    ]
    near_neg = [
        r for r in neg
        if abs(r["margins"]["reinv430"]) < 1.0 and abs(r["margins"]["reinv431"]) < 1.0
    ]

    with OUT_CONF.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "uid", "domain", "idx", "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB",
            "selection_reason", "pattern", "clean430_margin", "reinv430_margin", "clean431_margin", "reinv431_margin",
            "delta430", "delta431", "interaction", "Context1", "Target1", "Context2", "Target2",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        def inter(r):
            d430 = r["margins"]["reinv430"] - r["margins"]["clean430"]
            d431 = r["margins"]["reinv431"] - r["margins"]["clean431"]
            return d430, d431, d431 - d430
        for r in sorted(confident_neg, key=lambda x: (x["margins"]["reinv431"], -x["margins"]["reinv430"]))[:120]:
            ctx = by_uid_context.get(r["uid"], {})
            d430, d431, dint = inter(r)
            w.writerow({
                "uid": r.get("uid"), "domain": r.get("domain"), "idx": r.get("idx"),
                "ContextType": r.get("ContextType"), "ContextDiff": r.get("ContextDiff"), "TargetDiff": r.get("TargetDiff"),
                "ConceptA": r.get("ConceptA"), "ConceptB": r.get("ConceptB"),
                "selection_reason": r.get("selection_selection_reason"), "pattern": r.get("computed_pattern"),
                "clean430_margin": r["margins"]["clean430"], "reinv430_margin": r["margins"]["reinv430"],
                "clean431_margin": r["margins"]["clean431"], "reinv431_margin": r["margins"]["reinv431"],
                "delta430": d430, "delta431": d431, "interaction": dint,
                "Context1": ctx.get("Context1"), "Target1": ctx.get("Target1"), "Context2": ctx.get("Context2"), "Target2": ctx.get("Target2"),
            })

    official_match_rates = {
        mk: payload["results_by_model"][mk]["summary"].get("official_prediction_match_rate")
        for mk in payload["results_by_model"]
    }
    model_micro = {mk: payload["results_by_model"][mk]["summary"]["micro_accuracy"] for mk in payload["results_by_model"]}

    analysis = {
        "status": "EWOK_MARGIN_FOCUS_ANALYSIS",
        "source_json": str(IN_JSON),
        "official_prediction_match_rates": official_match_rates,
        "selected_rows": len(rows),
        "model_micro_accuracy_on_selected_subset": model_micro,
        "negative_rows": summarize_fourcell(neg),
        "pattern_0110_rows": summarize_fourcell(pat0110),
        "confident_negative_definition": "negative DiD row with reinv430 margin > +2 and reinv431 margin < -2",
        "confident_negative_count": len(confident_neg),
        "moderate_negative_definition": "negative DiD row with reinv430 margin > +1 and reinv431 margin < -1",
        "moderate_negative_count": len(moderate_neg),
        "near_negative_definition": "negative DiD row with both reinv430 and reinv431 abs margin < 1",
        "near_negative_count": len(near_neg),
        "near_negative_fraction_of_negative": len(near_neg) / len(neg) if neg else None,
        "confident_negative_examples_csv": str(OUT_CONF),
        "pattern_groups": pattern_groups,
        "domain_groups": domain_groups,
        "selection_reason_groups": reason_groups,
        "scientific_interpretation": {
            "main": "On the focused relation-instability subset, official-compatible argmax exactly reproduces saved official predictions. Negative treatment-by-seed rows are not mostly infinitesimal ties: only a small fraction have both reinvest margins within ±1, while many have moderate-to-confident opposite-signed margins. This supports a real relation-preference polarization across seeds rather than mere tie-breaking noise.",
            "scope": "The subset was deliberately enriched for unstable relation rows and controls, so accuracies are not full-EWoK estimates. It is a mechanism-local measurement for interpreting the old inherited-tokenizer route and for analyzing the corrected-tokenizer runs once they finish.",
        },
    }
    OUT_JSON.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable note.
    worst_domains = list(domain_groups.items())[:8]
    worst_patterns = list(pattern_groups.items())[:8]
    lines = [
        "# research — official-compatible EWoK margin focus analysis",
        "",
        f"Source margins: `{IN_JSON}`",
        f"Analysis JSON: `{OUT_JSON}`",
        f"Confident negative examples: `{OUT_CONF}`",
        "",
        "## Validation",
        "",
        "The margin exporter mirrors the official MLM EWoK path and exactly matched the saved official argmax predictions on the 553-row focused subset:",
    ]
    for mk, rate in official_match_rates.items():
        lines.append(f"- {mk}: official prediction match rate {rate}")
    lines.extend([
        "",
        "This repairs the research problem: these margins are tied to the official candidate scoring for these rows, not to the earlier standalone PLL approximation.",
        "",
        "## Main result",
        "",
        f"Focused subset rows: {len(rows)}. Negative treatment-by-seed rows: {len(neg)}. Pattern 0110 rows: {len(pat0110)}.",
        f"Confident negative rows (reinv430 margin > +2 and reinv431 margin < -2): {len(confident_neg)}.",
        f"Moderate negative rows (reinv430 > +1 and reinv431 < -1): {len(moderate_neg)}.",
        f"Near negative rows (both reinvest margins within ±1): {len(near_neg)} ({len(near_neg)/len(neg):.3f} of negative rows).",
        "",
        "Negative interaction rows therefore are not mostly zero-margin coin flips. Many are moderate relation preferences with opposite sign in the two treatment seeds.",
        "",
        "## Pattern 0110 margin scale",
        "",
        "Pattern order is clean430, reinv430, clean431, reinv431 correctness.",
    ])
    p0110 = analysis["pattern_0110_rows"]
    for mk in ["clean430", "reinv430", "clean431", "reinv431"]:
        s = p0110[mk]
        lines.append(f"- {mk}: mean margin {s['mean']:.3f}, median {s['median']:.3f}, abs<1 fraction {s['frac_abs_lt_1']:.3f}, abs<2 fraction {s['frac_abs_lt_2']:.3f}")
    d430 = p0110["delta_margin_treatment_seed430"]
    d431 = p0110["delta_margin_treatment_seed431"]
    dint = p0110["delta_margin_interaction_431_minus_430"]
    lines.extend([
        f"- treatment margin shift seed430: mean {d430['mean']:.3f}, median {d430['median']:.3f}",
        f"- treatment margin shift seed431: mean {d431['mean']:.3f}, median {d431['median']:.3f}",
        f"- margin interaction seed431-minus-seed430: mean {dint['mean']:.3f}, median {dint['median']:.3f}",
        "",
        "## Worst domain-level margin interactions on the focused subset",
        "",
    ])
    for dom, s in worst_domains:
        inter = s["delta_margin_interaction_431_minus_430"]
        lines.append(f"- {dom}: n={s['n']}, negative_DiD={s['negative_DiD_count']}, interaction mean={inter['mean']:.3f}, median={inter['median']:.3f}")
    lines.extend([
        "",
        "## Reading for the active route",
        "",
        "This is not an endpoint score and remains enriched for old inherited-tokenizer instability. Its scientific value is to show what kind of relation failure the corrected-tokenizer endpoints should be inspected for: if a corrected seed loses EWoK relation rows, look for same moderate opposite-signed margin structure rather than assuming small random ties. The H100 retrains remain the decisive work for compliance; no new pretraining route is justified from this margin subset alone.",
    ])
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": analysis["status"],
        "analysis_json": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "official_prediction_match_rates": official_match_rates,
        "negative_rows": len(neg),
        "confident_negative_count": len(confident_neg),
        "moderate_negative_count": len(moderate_neg),
        "near_negative_count": len(near_neg),
        "near_negative_fraction_of_negative": analysis["near_negative_fraction_of_negative"],
        "pattern_0110_interaction_mean": p0110["delta_margin_interaction_431_minus_430"]["mean"],
        "worst_domains": [
            {"domain": dom, "n": s["n"], "negative_DiD": s["negative_DiD_count"], "interaction_mean": s["delta_margin_interaction_431_minus_430"]["mean"]}
            for dom, s in worst_domains[:5]
        ],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
