#!/usr/bin/env python3
"""Factorial analysis for operation-preserving expression-transfer records.

The scorer varies update wording and query wording on the same
state maps. This analyzer keeps those factors separate. It is meant for the
six-pair pilot, the full 30-pair screen, and later bridge checkpoints scored
with the same expression-transfer probe.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple


MARGIN_KEYS = [
    "neutral_qa_cross_source", "neutral_qb_cross_source",
    "neutral_qa_correct_over_new", "neutral_qb_correct_over_new",
    "retain_qa_cross_source", "retain_qb_cross_source",
    "retain_qa_correct_over_new", "retain_qb_correct_over_new",
    "self_update_qa_new_over_source", "self_update_qb_new_over_source",
    "self_update_qa_new_over_wrong", "self_update_qb_new_over_wrong",
]

BASE_METRICS = [
    "neutral_both_full_source",
    "retain_both_full_source",
    "self_update_both_full_new",
    "recipient_only_flip_both_queries",
    "neutral_source_over_wrong_both",
    "neutral_source_over_replacement_both",
    "retain_source_over_wrong_both",
    "retain_source_over_replacement_both",
    "self_update_new_over_source_both",
    "self_update_new_over_wrong_both",
    "retain_cross_only_replacement_override",
    "self_update_semantic_inertia",
]

CELL_ORDER = ["orig", "query_only", "update_only", "both_changed"]


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_values(xs: Iterable[Any]) -> List[float]:
    vals: List[float] = []
    for x in xs:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            vals.append(y)
    return vals


def mean(xs: Iterable[Any]) -> float:
    vals = finite_values(xs)
    return sum(vals) / len(vals) if vals else float("nan")


def median(xs: Iterable[Any]) -> float:
    vals = finite_values(xs)
    return statistics.median(vals) if vals else float("nan")


def cell_name(record: Dict[str, Any]) -> str:
    changes = record.get("variant_changes") or {}
    u = bool(changes.get("update_changed"))
    q = bool(changes.get("query_frame_changed"))
    if not u and not q:
        return "orig"
    if not u and q:
        return "query_only"
    if u and not q:
        return "update_only"
    return "both_changed"


def enrich(record: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(record)
    r["factor_cell"] = cell_name(r)
    r["neutral_source_over_wrong_both"] = (float(r["neutral_qa_cross_source"]) > 0.0 and
                                             float(r["neutral_qb_cross_source"]) > 0.0)
    r["neutral_source_over_replacement_both"] = (float(r["neutral_qa_correct_over_new"]) > 0.0 and
                                                   float(r["neutral_qb_correct_over_new"]) > 0.0)
    r["retain_source_over_wrong_both"] = (float(r["retain_qa_cross_source"]) > 0.0 and
                                           float(r["retain_qb_cross_source"]) > 0.0)
    r["retain_source_over_replacement_both"] = (float(r["retain_qa_correct_over_new"]) > 0.0 and
                                                 float(r["retain_qb_correct_over_new"]) > 0.0)
    r["self_update_new_over_source_both"] = (float(r["self_update_qa_new_over_source"]) > 0.0 and
                                              float(r["self_update_qb_new_over_source"]) > 0.0)
    r["self_update_new_over_wrong_both"] = (float(r["self_update_qa_new_over_wrong"]) > 0.0 and
                                             float(r["self_update_qb_new_over_wrong"]) > 0.0)
    # Two separable ways to lose the complete operation even while a useful part survives.
    r["retain_cross_only_replacement_override"] = bool(r["retain_source_over_wrong_both"] and
                                                        not r["retain_source_over_replacement_both"])
    r["self_update_semantic_inertia"] = bool(r["self_update_new_over_wrong_both"] and
                                              not r["self_update_new_over_source_both"])
    r["neutral_selector_available_but_operation_fails"] = bool(r.get("neutral_both_full_source") and
                                                                not r.get("recipient_only_flip_both_queries"))
    return r


def margin_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in MARGIN_KEYS:
        vals = [r.get(k) for r in records]
        out[f"mean_{k}"] = mean(vals)
        out[f"median_{k}"] = median(vals)
    # Averaged two-query views used in the research interpretation.
    derived = {
        "neutral_cross_source_avg": [(r["neutral_qa_cross_source"] + r["neutral_qb_cross_source"]) / 2.0 for r in records],
        "neutral_correct_over_new_avg": [(r["neutral_qa_correct_over_new"] + r["neutral_qb_correct_over_new"]) / 2.0 for r in records],
        "retain_cross_source_avg": [(r["retain_qa_cross_source"] + r["retain_qb_cross_source"]) / 2.0 for r in records],
        "retain_correct_over_new_avg": [(r["retain_qa_correct_over_new"] + r["retain_qb_correct_over_new"]) / 2.0 for r in records],
        "self_update_new_over_source_avg": [(r["self_update_qa_new_over_source"] + r["self_update_qb_new_over_source"]) / 2.0 for r in records],
        "self_update_new_over_wrong_avg": [(r["self_update_qa_new_over_wrong"] + r["self_update_qb_new_over_wrong"]) / 2.0 for r in records],
    }
    for k, vals in derived.items():
        out[f"mean_{k}"] = mean(vals)
        out[f"median_{k}"] = median(vals)
    return out


def group_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    out: Dict[str, Any] = {
        "n": n,
        "pairs": len(set(r["pair_id"] for r in records)),
        "relations": dict(collections.Counter(r.get("relation") for r in records)),
        "variants": dict(collections.Counter(r.get("variant") for r in records)),
    }
    for k in BASE_METRICS + ["neutral_selector_available_but_operation_fails"]:
        c = sum(1 for r in records if bool(r.get(k)))
        out[f"{k}_count"] = c
        out[f"{k}_rate"] = c / n if n else float("nan")
    out["margins"] = margin_summary(records)
    return out


def pair_cell_means(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        grouped[(r["model"], r["pair_id"], r["factor_cell"])].append(r)
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    metrics = list(BASE_METRICS) + [
        "neutral_selector_available_but_operation_fails",
        "neutral_cross_source_avg", "neutral_correct_over_new_avg",
        "retain_cross_source_avg", "retain_correct_over_new_avg",
        "self_update_new_over_source_avg", "self_update_new_over_wrong_avg",
    ]
    for key, rs in grouped.items():
        d: Dict[str, float] = {"n_variants": float(len(rs))}
        for m in BASE_METRICS + ["neutral_selector_available_but_operation_fails"]:
            d[m] = mean([1.0 if r.get(m) else 0.0 for r in rs])
        d["neutral_cross_source_avg"] = mean([(r["neutral_qa_cross_source"] + r["neutral_qb_cross_source"]) / 2.0 for r in rs])
        d["neutral_correct_over_new_avg"] = mean([(r["neutral_qa_correct_over_new"] + r["neutral_qb_correct_over_new"]) / 2.0 for r in rs])
        d["retain_cross_source_avg"] = mean([(r["retain_qa_cross_source"] + r["retain_qb_cross_source"]) / 2.0 for r in rs])
        d["retain_correct_over_new_avg"] = mean([(r["retain_qa_correct_over_new"] + r["retain_qb_correct_over_new"]) / 2.0 for r in rs])
        d["self_update_new_over_source_avg"] = mean([(r["self_update_qa_new_over_source"] + r["self_update_qb_new_over_source"]) / 2.0 for r in rs])
        d["self_update_new_over_wrong_avg"] = mean([(r["self_update_qa_new_over_wrong"] + r["self_update_qb_new_over_wrong"]) / 2.0 for r in rs])
        out[key] = d
    return out


def factorial_effects(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pcm = pair_cell_means(records)
    models = sorted(set(r["model"] for r in records))
    pair_ids_by_model: Dict[str, List[str]] = {m: sorted(set(r["pair_id"] for r in records if r["model"] == m)) for m in models}
    metrics = list(BASE_METRICS) + [
        "neutral_selector_available_but_operation_fails",
        "neutral_cross_source_avg", "neutral_correct_over_new_avg",
        "retain_cross_source_avg", "retain_correct_over_new_avg",
        "self_update_new_over_source_avg", "self_update_new_over_wrong_avg",
    ]
    out: Dict[str, Any] = {}
    for m in models:
        cell_means: Dict[str, Dict[str, float]] = {}
        for cell in CELL_ORDER:
            cell_rows = [pcm[(m, pid, cell)] for pid in pair_ids_by_model[m] if (m, pid, cell) in pcm]
            cell_means[cell] = {metric: mean([row[metric] for row in cell_rows]) for metric in metrics}
            cell_means[cell]["n_pairs_with_cell"] = float(len(cell_rows))
        effects: Dict[str, Dict[str, float]] = {}
        # Per-pair contrasts when all four cells are present.
        complete_pairs = [pid for pid in pair_ids_by_model[m] if all((m, pid, c) in pcm for c in CELL_ORDER)]
        for metric in metrics:
            query_no_update = []
            update_orig_query = []
            both_minus_query = []
            interaction = []
            for pid in complete_pairs:
                o = pcm[(m, pid, "orig")][metric]
                q = pcm[(m, pid, "query_only")][metric]
                u = pcm[(m, pid, "update_only")][metric]
                b = pcm[(m, pid, "both_changed")][metric]
                query_no_update.append(q - o)
                update_orig_query.append(u - o)
                both_minus_query.append(b - q)
                interaction.append(b - q - u + o)
            effects[metric] = {
                "query_effect_at_original_update_mean": mean(query_no_update),
                "update_effect_at_original_query_mean": mean(update_orig_query),
                "additional_effect_when_both_change_mean": mean(interaction),
                "both_changed_minus_query_only_mean": mean(both_minus_query),
                "n_complete_pairs": len(complete_pairs),
            }
        out[m] = {
            "n_pairs": len(pair_ids_by_model[m]),
            "n_pairs_with_all_four_cells": len(complete_pairs),
            "cell_means_over_pairs": cell_means,
            "per_pair_factor_effects": effects,
        }
    return out


def variant_rank(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        grouped[(r["model"], r["variant"])].append(r)
    out: Dict[str, Any] = {}
    for (m, v), rs in sorted(grouped.items()):
        out[f"{m}::{v}"] = group_summary(rs)
    return out


def dissociation_examples(records: Sequence[Dict[str, Any]], model_prefix: str, max_examples: int) -> Dict[str, List[Dict[str, Any]]]:
    rs = [r for r in records if str(r.get("model", "")).startswith(model_prefix)]
    examples: Dict[str, List[Dict[str, Any]]] = {"replacement_override": [], "source_inertia_on_update": [], "neutral_available_operation_fail": []}
    for r in rs:
        base = {
            "model": r["model"],
            "pair_id": r["pair_id"],
            "relation": r.get("relation"),
            "variant": r.get("variant"),
            "factor_cell": r.get("factor_cell"),
            "flip": bool(r.get("recipient_only_flip_both_queries")),
            "neutral_full": bool(r.get("neutral_both_full_source")),
            "retain_full": bool(r.get("retain_both_full_source")),
            "self_update_full": bool(r.get("self_update_both_full_new")),
            "retain_cross_source_avg": (r["retain_qa_cross_source"] + r["retain_qb_cross_source"]) / 2.0,
            "retain_correct_over_new_avg": (r["retain_qa_correct_over_new"] + r["retain_qb_correct_over_new"]) / 2.0,
            "self_update_new_over_source_avg": (r["self_update_qa_new_over_source"] + r["self_update_qb_new_over_source"]) / 2.0,
            "self_update_new_over_wrong_avg": (r["self_update_qa_new_over_wrong"] + r["self_update_qb_new_over_wrong"]) / 2.0,
        }
        if r["retain_cross_only_replacement_override"]:
            examples["replacement_override"].append(base)
        if r["self_update_semantic_inertia"]:
            examples["source_inertia_on_update"].append(base)
        if r["neutral_selector_available_but_operation_fails"]:
            examples["neutral_available_operation_fail"].append(base)
    examples["replacement_override"].sort(key=lambda x: x["retain_correct_over_new_avg"])
    examples["source_inertia_on_update"].sort(key=lambda x: x["self_update_new_over_source_avg"])
    examples["neutral_available_operation_fail"].sort(key=lambda x: (x["factor_cell"], x["pair_id"], x["variant"]))
    return {k: v[:max_examples] for k, v in examples.items()}


def analyze(records: Sequence[Dict[str, Any]], model_prefix_for_examples: str, max_examples: int) -> Dict[str, Any]:
    enriched = [enrich(r) for r in records]
    by_cell: Dict[str, Any] = {}
    for (m, cell), rs in sorted(collections.defaultdict(list, {}).items()):
        pass
    grouped_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in enriched:
        grouped_cell[(r["model"], r["factor_cell"])].append(r)
    for (m, cell), rs in sorted(grouped_cell.items()):
        by_cell[f"{m}::{cell}"] = group_summary(rs)
    model_totals: Dict[str, Any] = {}
    for m in sorted(set(r["model"] for r in enriched)):
        model_totals[m] = group_summary([r for r in enriched if r["model"] == m])
    return {
        "status": "EXPRESSION_FACTORIAL_ANALYSIS_DONE",
        "n_records": len(enriched),
        "models": sorted(set(r["model"] for r in enriched)),
        "n_pairs_by_model": {m: len(set(r["pair_id"] for r in enriched if r["model"] == m)) for m in sorted(set(r["model"] for r in enriched))},
        "cell_order": CELL_ORDER,
        "cell_mean_note": "query_only and both_changed may contain two concrete phrasings per pair; cell means average within pair before effect contrasts.",
        "by_model_total": model_totals,
        "by_model_cell": by_cell,
        "by_model_variant": variant_rank(enriched),
        "factorial_effects": factorial_effects(enriched),
        "dissociation_examples": dissociation_examples(enriched, model_prefix_for_examples, max_examples),
    }


def compact_print(result: Dict[str, Any]) -> None:
    print(json.dumps({"status": result["status"], "n_records": result["n_records"], "models": result["models"], "n_pairs_by_model": result["n_pairs_by_model"]}, ensure_ascii=False))
    for model, info in result["factorial_effects"].items():
        print(f"\nMODEL {model}  pairs={info['n_pairs']} complete_pairs={info['n_pairs_with_all_four_cells']}")
        for cell in CELL_ORDER:
            cm = info["cell_means_over_pairs"][cell]
            print(
                f"  {cell:12s} flip={cm['recipient_only_flip_both_queries']:.3f} "
                f"neutral={cm['neutral_both_full_source']:.3f} retain={cm['retain_both_full_source']:.3f} "
                f"selfupd={cm['self_update_both_full_new']:.3f} repl_override={cm['retain_cross_only_replacement_override']:.3f} "
                f"source_inertia={cm['self_update_semantic_inertia']:.3f}"
            )
        for metric in ["recipient_only_flip_both_queries", "neutral_both_full_source", "retain_both_full_source", "self_update_both_full_new", "retain_cross_only_replacement_override", "self_update_semantic_inertia"]:
            e = info["per_pair_factor_effects"][metric]
            print(
                f"  effect {metric}: query={e['query_effect_at_original_update_mean']:+.3f} "
                f"update={e['update_effect_at_original_query_mean']:+.3f} "
                f"both_extra={e['additional_effect_when_both_change_mean']:+.3f} "
                f"both-query={e['both_changed_minus_query_only_mean']:+.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-prefix-for-examples", default="specialist")
    ap.add_argument("--max-examples", type=int, default=25)
    args = ap.parse_args()
    records = read_jsonl(pathlib.Path(args.records))
    result = analyze(records, args.model_prefix_for_examples, int(args.max_examples))
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact_print(result)


if __name__ == "__main__":
    main()
