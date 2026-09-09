#!/usr/bin/env python3
"""Select the best available high-anchor FineWeb second view per source.

Combines the research near-copy-preserving simplification run and the stronger
compression run.  The intent is to preserve the scientifically useful region:
faithful enough for a same-window semantic view, but sufficiently different from
source repetition to test a representation/credit-assignment effect rather than
copy exposure.  This is a selection over generation outputs only; it does not use
BabyLM evaluation labels or scores.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
from typing import Any

SIMPLE_ROWS = pathlib.Path("experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_generation_rows.jsonl")
COMP_ROWS = pathlib.Path("experiments/archive/frontier_consolidation/data/high_anchor_fineweb_compression_analysis/fineweb_high_anchor_generation_rows.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/high_anchor_best_view_selection")
OUT_JSONL = OUT_DIR / "fineweb_high_anchor_best_views.jsonl"
OUT_META = OUT_DIR / "fineweb_high_anchor_best_view_selection_metadata.json"
NOTE = pathlib.Path("research/notes/frontier_consolidation/high_anchor_best_view_selection.md")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def usable_compact(r: dict[str, Any]) -> bool:
    if not r.get("accepted_for_next_construction"):
        return False
    if float(r.get("length_ratio") or 0) > 0.85:
        return False
    if float(r.get("content_recall") or 0) < 0.45:
        return False
    risks = set(r.get("source_risks") or [])
    if risks & {"angle_heading", "heading_dash_chain"}:
        return False
    return True


def usable_simple_fallback(r: dict[str, Any]) -> bool:
    if not r.get("accepted_for_next_construction"):
        return False
    if float(r.get("content_recall") or 0) < 0.65:
        return False
    if float(r.get("length_ratio") or 0) >= 1.15:
        return False
    # Exclude exact/near copies as a semantic-view corpus source; they belong in the repetition control.
    if float(r.get("content_overlap") or 0) > 0.94 and float(r.get("length_ratio") or 0) >= 0.85:
        return False
    risks = set(r.get("source_risks") or [])
    if risks & {"angle_heading", "heading_dash_chain"}:
        return False
    return True


def choose(simple: dict[str, Any], comp: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    if comp and usable_compact(comp):
        return "compact_primary", comp
    if usable_simple_fallback(simple):
        return "simple_fallback_noncopy", simple
    return "no_view", None


def main() -> None:
    simple = read_jsonl(SIMPLE_ROWS)
    comp = {str(r.get("sentence_id")): r for r in read_jsonl(COMP_ROWS)}
    chosen = []
    counts = collections.Counter()
    rejection_examples = []
    for s in simple:
        typ, r = choose(s, comp.get(str(s.get("sentence_id"))))
        counts[typ] += 1
        if r is None:
            if len(rejection_examples) < 24:
                rejection_examples.append({
                    "sentence_id": s.get("sentence_id"),
                    "source_text": s.get("source_text"),
                    "simple_reasons": s.get("hard_reasons"),
                    "simple_flags": s.get("soft_flags"),
                    "compression_reasons": (comp.get(str(s.get("sentence_id"))) or {}).get("hard_reasons"),
                    "compression_flags": (comp.get(str(s.get("sentence_id"))) or {}).get("soft_flags"),
                })
            continue
        rec = {
            "selection_type": typ,
            "prompt_id": r.get("prompt_id"),
            "sentence_id": r.get("sentence_id"),
            "doc_id": r.get("doc_id"),
            "source_text": r.get("source_text"),
            "rewrite_text": r.get("rewrite_text"),
            "source_words": int(r.get("source_words") or len(str(r.get("source_text", "")).split())),
            "rewrite_words": int(r.get("output_words") or len(str(r.get("rewrite_text", "")).split())),
            "pair_words": int(r.get("pair_words") or 0),
            "length_ratio": r.get("length_ratio"),
            "content_overlap": r.get("content_overlap"),
            "content_recall": r.get("content_recall"),
            "entity_recall": r.get("entity_recall"),
            "number_recall": r.get("number_recall"),
            "domain_hits": r.get("domain_hits") or [],
            "source_entities": r.get("source_entities") or [],
            "source_numbers": r.get("source_numbers") or [],
        }
        if rec["pair_words"] != rec["source_words"] + rec["rewrite_words"]:
            rec["pair_words"] = rec["source_words"] + rec["rewrite_words"]
        chosen.append(rec)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ratios = [r["rewrite_words"] / r["source_words"] for r in chosen]
    domains = collections.Counter(d for r in chosen for d in (r.get("domain_hits") or ["no_domain"]))
    by_type = collections.Counter(r["selection_type"] for r in chosen)
    meta = {
        "status": "HIGH_ANCHOR_BEST_VIEWS_SELECTED",
        "simple_rows": str(SIMPLE_ROWS),
        "compression_rows": str(COMP_ROWS),
        "output_jsonl": str(OUT_JSONL),
        "input_sources": len(simple),
        "selected_views": len(chosen),
        "selection_counts_all_sources": dict(counts),
        "selected_by_type": dict(by_type),
        "selected_source_words": sum(r["source_words"] for r in chosen),
        "selected_rewrite_words": sum(r["rewrite_words"] for r in chosen),
        "selected_pair_words": sum(r["pair_words"] for r in chosen),
        "rewrite_over_source_weighted": sum(r["rewrite_words"] for r in chosen) / max(1, sum(r["source_words"] for r in chosen)),
        "rewrite_over_source_mean": statistics.fmean(ratios) if ratios else None,
        "rewrite_over_source_median": statistics.median(ratios) if ratios else None,
        "unique_docs": len({r["doc_id"] for r in chosen}),
        "domain_hit_counts": dict(domains.most_common()),
        "rejection_examples": rejection_examples,
        "selection_policy": "Prefer accepted compression outputs with ratio <=0.85 and content_recall >=0.45; otherwise use accepted non-copy simple outputs with content_recall >=0.65 and ratio <1.15. Reject heading-like source risks. No BabyLM eval labels/scores used.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research high-anchor best view selection\n\n"
        "Combined the near-copy-safe and compression prompt outputs. The selected set keeps faithful compact views where possible and uses non-copy simple fallbacks, avoiding exact source repetition as a supposed semantic view.\n\n"
        f"Selected {len(chosen):,} views from {len(simple):,} sources. Source words {meta['selected_source_words']:,}; rewrite words {meta['selected_rewrite_words']:,}; source+view words {meta['selected_pair_words']:,}; weighted rewrite/source ratio {meta['rewrite_over_source_weighted']:.3f}.\n\n"
        f"By selection type: {dict(by_type)}.\n\nJSON: `{OUT_META}`\n\nSelected JSONL: `{OUT_JSONL}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": meta["status"], "selected_views": len(chosen), "selected_pair_words": meta["selected_pair_words"], "ratio": meta["rewrite_over_source_weighted"], "metadata": str(OUT_META)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
