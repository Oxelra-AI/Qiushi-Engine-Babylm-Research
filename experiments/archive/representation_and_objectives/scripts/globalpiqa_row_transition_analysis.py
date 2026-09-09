#!/usr/bin/env python3
"""research: row-level GlobalPIQA transition analysis for scale1.75.

Uses already-produced row-level GlobalPIQA readouts. It tests whether the
scale1.75 80M/100M behavior looks like coherent relation-memory forgetting or
mostly scattered ranking drift. This is an analysis-only artifact; it does not
train and it must not be used as a benchmark-shaped training set.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from collections import Counter, defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
OUT = ROOT / "experiments/archive/representation_and_objectives/data/globalpiqa_row_transition_analysis"
NOTE = ROOT / "research/notes/representation_and_objectives/scale1p75_globalpiqa_transitions.md"
RAW154 = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/raw_rows"
RAW155 = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/raw_rows"
SUMMARY155 = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_globalpiqa_hard_surface/scale1p75_100m_globalpiqa_summary.json"
SUMMARY154 = ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_inference_ablation/scale1p75_inference_ablation_summary.json"
HARD52_SOURCE = ROOT / "experiments/archive/representation_and_objectives/data/globalpiqa_margin_synthesis"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def load_rows(path: pathlib.Path) -> dict[str, dict[str, dict[str, Any]]]:
    d = read_json(path)
    return {mode: {r["example_id"]: r for r in rows} for mode, rows in d.items()}


def margin(row: dict[str, Any]) -> float:
    return float(row.get("top_minus_correct", 0.0))


def rank(row: dict[str, Any]) -> int:
    return int(row.get("correct_rank"))


def is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def summarise_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = list(rows.values())
    margins = [margin(r) for r in vals]
    ranks = Counter(rank(r) for r in vals)
    return {
        "n": len(vals),
        "accuracy": 100.0 * sum(is_correct(r) for r in vals) / max(1, len(vals)),
        "correct_rank_counts": dict(sorted(ranks.items())),
        "mean_top_minus_correct": statistics.fmean(margins) if margins else None,
        "median_top_minus_correct": statistics.median(margins) if margins else None,
        "small_wrong_margin_le_0p25": sum((not is_correct(r)) and margin(r) <= 0.25 for r in vals),
        "small_wrong_margin_le_0p50": sum((not is_correct(r)) and margin(r) <= 0.50 for r in vals),
    }


def compare_rows(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], name_a: str, name_b: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = sorted(set(a) & set(b))
    trans = Counter()
    rank_delta = Counter()
    rows = []
    delta_margins = []
    abs_delta_margins = []
    for eid in ids:
        ra, rb = a[eid], b[eid]
        ca, cb = is_correct(ra), is_correct(rb)
        key = ("C" if ca else "W") + "->" + ("C" if cb else "W")
        trans[key] += 1
        rd = rank(rb) - rank(ra)
        rank_delta[rd] += 1
        dm = margin(rb) - margin(ra)
        delta_margins.append(dm)
        abs_delta_margins.append(abs(dm))
        rows.append({
            "example_id": eid,
            "mode": ra.get("mode"),
            "label": ra.get("label"),
            f"choice_{name_a}": ra.get("choice"),
            f"choice_{name_b}": rb.get("choice"),
            f"correct_{name_a}": ca,
            f"correct_{name_b}": cb,
            f"rank_{name_a}": rank(ra),
            f"rank_{name_b}": rank(rb),
            "rank_delta_b_minus_a": rd,
            f"margin_{name_a}": margin(ra),
            f"margin_{name_b}": margin(rb),
            "margin_delta_b_minus_a": dm,
            "same_choice": ra.get("choice") == rb.get("choice"),
            "prompt": ra.get("prompt"),
            "completions": " ||| ".join(ra.get("completions", [])),
        })
    # Coherent forgetting would show more C->W than W->C and positive margin shift among hard/wrong rows.
    cw = trans.get("C->W", 0)
    wc = trans.get("W->C", 0)
    summary = {
        "name_a": name_a,
        "name_b": name_b,
        "n_shared": len(ids),
        "transitions": dict(trans),
        "net_correct_gain_b_minus_a_rows": wc - cw,
        "same_choice_frac": sum(r["same_choice"] for r in rows) / max(1, len(rows)),
        "rank_delta_counts_b_minus_a": dict(sorted(rank_delta.items())),
        "mean_margin_delta_b_minus_a": statistics.fmean(delta_margins) if delta_margins else None,
        "median_margin_delta_b_minus_a": statistics.median(delta_margins) if delta_margins else None,
        "mean_abs_margin_delta": statistics.fmean(abs_delta_margins) if abs_delta_margins else None,
        "a_summary": summarise_rows(a),
        "b_summary": summarise_rows(b),
    }
    return summary, rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        "scale1p75_80M_live": RAW154 / "scale1p75_live_rows.json",
        "scale1p75_80M_disabled": RAW154 / "scale1p75_disabled_rows.json",
        "scale1p75_80M_scale1p0": RAW154 / "scale1p75_scale1p0_rows.json",
        "scale1p75_80M_scale2p5": RAW154 / "scale1p75_scale2p5_rows.json",
        "a01_legal40k_80M": RAW154 / "anchor_legal40k_80M_rows.json",
        "scale1p75_100M": RAW155 / "scale1p75_100M_rows.json",
        "legal16k_100M": RAW155 / "legal16k_100M_rows.json",
        "a01_legal40k_100M": RAW155 / "a01_legal40k_100M_rows.json",
    }
    missing = {k: rel(p) for k, p in inputs.items() if not p.exists()}
    loaded = {k: load_rows(p) for k, p in inputs.items() if p.exists()}
    comparisons = [
        ("scale1p75_80M_disabled", "scale1p75_80M_live"),
        ("a01_legal40k_80M", "scale1p75_80M_live"),
        ("legal16k_100M", "scale1p75_100M"),
        ("a01_legal40k_100M", "scale1p75_100M"),
        ("scale1p75_80M_live", "scale1p75_100M"),
    ]
    all_summaries: dict[str, Any] = {}
    for a_name, b_name in comparisons:
        if a_name not in loaded or b_name not in loaded:
            all_summaries[f"{a_name}_to_{b_name}"] = {"missing_input": [x for x in [a_name, b_name] if x not in loaded]}
            continue
        for mode in ["parallel", "nonparallel"]:
            if mode not in loaded[a_name] or mode not in loaded[b_name]:
                continue
            summ, rows = compare_rows(loaded[a_name][mode], loaded[b_name][mode], a_name, b_name)
            key = f"{a_name}_to_{b_name}__{mode}"
            all_summaries[key] = summ
            write_csv(OUT / f"{key}_rows.csv", rows)
            # Save the largest worsening and improvements for human inspection.
            worst = sorted(rows, key=lambda r: r["margin_delta_b_minus_a"], reverse=True)[:20]
            best = sorted(rows, key=lambda r: r["margin_delta_b_minus_a"])[:20]
            write_csv(OUT / f"{key}_largest_margin_worsening.csv", worst)
            write_csv(OUT / f"{key}_largest_margin_improvement.csv", best)
    payload = {
        "status": "GLOBALPIQA_ROW_TRANSITION_ANALYSIS_DONE",
        "created_utc": now(),
        "inputs": {k: rel(p) for k, p in inputs.items()},
        "missing_inputs": missing,
        "comparisons": all_summaries,
        "interpretation": "Coherent consolidation would require many correct-to-wrong transitions or systematic margin worsening along the same relation surface; scattered rank changes argue for new natural competition signal rather than replaying endpoint averaging or packet families.",
    }
    out_json = OUT / "globalpiqa_row_transition_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Compose concise note without using final-deliverable framing.
    lines = [
        "# research GlobalPIQA row-transition analysis for scale1.75\n\n",
        f"Summary JSON: `{rel(out_json)}`. Row CSVs are under `{rel(OUT)}`.\n\n",
        "This analysis uses already-produced GlobalPIQA row readouts only; it does not train and is not a source of training examples.\n\n",
    ]
    for key in [
        "scale1p75_80M_disabled_to_scale1p75_80M_live__parallel",
        "legal16k_100M_to_scale1p75_100M__parallel",
        "scale1p75_80M_live_to_scale1p75_100M__parallel",
    ]:
        s = all_summaries.get(key, {})
        lines.append(f"## {key}\n")
        lines.append(json.dumps({
            "n_shared": s.get("n_shared"),
            "transitions": s.get("transitions"),
            "net_correct_gain_b_minus_a_rows": s.get("net_correct_gain_b_minus_a_rows"),
            "same_choice_frac": s.get("same_choice_frac"),
            "mean_margin_delta_b_minus_a": s.get("mean_margin_delta_b_minus_a"),
            "median_margin_delta_b_minus_a": s.get("median_margin_delta_b_minus_a"),
            "rank_delta_counts_b_minus_a": s.get("rank_delta_counts_b_minus_a"),
        }, indent=2) + "\n\n")
    lines.append("## Research implication\n")
    lines.append("If the 80M-to-100M comparison shows mostly W->W rank/margin reshuffling rather than many C->W losses, relational consolidation is weakly supported as a main repair. The stronger next route is to create a natural candidate-contrast or counterfactual credit signal that differs from ordinary MLM and is selected on held-out EWoK/GlobalPIQA movement before endpoint-scale training.\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": rel(out_json), "note": rel(NOTE), "missing_inputs": missing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
