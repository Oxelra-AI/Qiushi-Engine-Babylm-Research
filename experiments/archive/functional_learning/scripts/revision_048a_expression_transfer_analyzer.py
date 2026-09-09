#!/usr/bin/env python3
"""Analyze research operation-preserving expression-transfer records.

This is intentionally separate from the scorer so later agents can analyze pilot,
full, or checkpoint-specific expression-transfer outputs in a consistent way.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def agg_bool(rs: Sequence[Dict[str, Any]], key: str) -> int:
    return sum(1 for r in rs if bool(r.get(key)))


def margin_means(rs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "retain_correct_vs_wrong": finite_mean([r["retain_qa_cross_source"] for r in rs] + [r["retain_qb_cross_source"] for r in rs]),
        "retain_correct_vs_replacement": finite_mean([r["retain_qa_correct_over_new"] for r in rs] + [r["retain_qb_correct_over_new"] for r in rs]),
        "self_update_new_vs_source": finite_mean([r["self_update_qa_new_over_source"] for r in rs] + [r["self_update_qb_new_over_source"] for r in rs]),
        "self_update_new_vs_wrong": finite_mean([r["self_update_qa_new_over_wrong"] for r in rs] + [r["self_update_qb_new_over_wrong"] for r in rs]),
        "neutral_correct_vs_wrong": finite_mean([r["neutral_qa_cross_source"] for r in rs] + [r["neutral_qb_cross_source"] for r in rs]),
        "neutral_correct_vs_replacement": finite_mean([r["neutral_qa_correct_over_new"] for r in rs] + [r["neutral_qb_correct_over_new"] for r in rs]),
    }


def summarize(records: Sequence[Dict[str, Any]], orig_variant: str = "orig") -> Dict[str, Any]:
    by_mv: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    by_model_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_mv[(r["model"], r["variant"])].append(r)
        by_model_pair[(r["model"], r["pair_id"])].append(r)
    mv_summary = {}
    for (model, variant), rs in sorted(by_mv.items()):
        mv_summary[f"{model}::{variant}"] = {
            "model": model,
            "variant": variant,
            "n": len(rs),
            "relations": dict(collections.Counter(r["relation"] for r in rs)),
            "neutral_both_full_source": agg_bool(rs, "neutral_both_full_source"),
            "retain_both_full_source": agg_bool(rs, "retain_both_full_source"),
            "self_update_both_full_new": agg_bool(rs, "self_update_both_full_new"),
            "recipient_only_flip_both_queries": agg_bool(rs, "recipient_only_flip_both_queries"),
            "margins": margin_means(rs),
        }
    pair_summary = {}
    for (model, pair_id), rs in sorted(by_model_pair.items()):
        expr = [r for r in rs if r["variant"] != orig_variant]
        orig = [r for r in rs if r["variant"] == orig_variant]
        pair_summary[f"{model}::{pair_id}"] = {
            "model": model,
            "pair_id": pair_id,
            "relation": rs[0].get("relation"),
            "orig_flip": bool(orig[0].get("recipient_only_flip_both_queries")) if orig else None,
            "n_expression_variants": len(expr),
            "expression_flip_count": agg_bool(expr, "recipient_only_flip_both_queries"),
            "expression_retain_count": agg_bool(expr, "retain_both_full_source"),
            "expression_self_update_count": agg_bool(expr, "self_update_both_full_new"),
            "expression_variants_flipped": [r["variant"] for r in expr if r.get("recipient_only_flip_both_queries")],
            "expression_variants_retain_failed": [r["variant"] for r in expr if not r.get("retain_both_full_source")],
            "expression_variants_self_update_failed": [r["variant"] for r in expr if not r.get("self_update_both_full_new")],
            "expression_margins": margin_means(expr) if expr else {},
        }
    model_summary = {}
    for model in sorted({r["model"] for r in records}):
        rs = [r for r in records if r["model"] == model]
        expr = [r for r in rs if r["variant"] != orig_variant]
        model_summary[model] = {
            "n_total_records": len(rs),
            "n_expression_records": len(expr),
            "n_unique_pairs": len(set(r["pair_id"] for r in rs)),
            "orig_flip": agg_bool([r for r in rs if r["variant"] == orig_variant], "recipient_only_flip_both_queries"),
            "expression_flip": agg_bool(expr, "recipient_only_flip_both_queries"),
            "expression_retain": agg_bool(expr, "retain_both_full_source"),
            "expression_self_update": agg_bool(expr, "self_update_both_full_new"),
            "n_pairs_with_any_expression_flip": sum(1 for (m, _), prs in by_model_pair.items() if m == model and any(r.get("recipient_only_flip_both_queries") and r["variant"] != orig_variant for r in prs)),
            "n_pairs_with_all_expression_flip": sum(1 for (m, _), prs in by_model_pair.items() if m == model and expr and all(r.get("recipient_only_flip_both_queries") for r in prs if r["variant"] != orig_variant)),
            "expression_margins": margin_means(expr) if expr else {},
        }
    # Most informative failures for the acquired model: strong source discrimination
    # but failure against replacement, versus failure to self-update after rephrasing.
    acquired_failures = []
    for r in records:
        if not r["model"].startswith("specialist"):
            continue
        if r["variant"] == orig_variant or r.get("recipient_only_flip_both_queries"):
            continue
        acquired_failures.append({
            "pair_id": r["pair_id"],
            "relation": r["relation"],
            "variant": r["variant"],
            "retain_ok": r["retain_both_full_source"],
            "self_update_ok": r["self_update_both_full_new"],
            "avg_retain_correct_vs_wrong": (r["retain_qa_cross_source"] + r["retain_qb_cross_source"]) / 2.0,
            "avg_retain_correct_vs_replacement": (r["retain_qa_correct_over_new"] + r["retain_qb_correct_over_new"]) / 2.0,
            "avg_self_update_new_vs_source": (r["self_update_qa_new_over_source"] + r["self_update_qb_new_over_source"]) / 2.0,
        })
    acquired_failures.sort(key=lambda d: (not d["retain_ok"], not d["self_update_ok"], d["avg_retain_correct_vs_replacement"]))
    return {
        "by_model": model_summary,
        "by_model_variant": mv_summary,
        "by_model_pair": pair_summary,
        "acquired_nonorig_failure_examples": acquired_failures[:30],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = read_jsonl(pathlib.Path(args.records))
    out = {
        "status": "EXPRESSION_TRANSFER_ANALYSIS_DONE",
        "records": args.records,
        "n_records": len(records),
        "summary": summarize(records),
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
