#!/usr/bin/env python3
"""research: inventory legal candidate signals for next BabyLM Strict-Small route.

This CPU-only script quantifies two possible unsaturated, token-prediction-connected
objects before any new training:

1. Directed edit-state structure in legal source -> compact-rewrite pairs.
2. Genuine intra-row sentence/utterance adjacency from natural (non-synthetic) rows.

It does not use evaluation labels or external annotations. It only reads local
corpus-construction artifacts and writes a route-planning inventory.
"""

from __future__ import annotations

import collections
import difflib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/frontier_consolidation")
POOL = STUDY / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
PAIR_FILE = STUDY / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
ROW_META = STUDY / "data/density_core_reinvestment_medium_riskhard/fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
OUT_DIR = STUDY / "data/candidate_signal_inventory"

NATURAL_SOURCES = {"childes", "gutenberg", "open_subtitles", "simple_wiki", "bnc_spoken", "switchboard"}
SYNTHETIC_SOURCES = {"qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest"}
# The 9-word topup is natural open_subtitles text but too tiny for a route; record separately.

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\*\[]|[\"'“‘]))")
CHILDES_MARK_RE = re.compile(r"(?=\*[A-Z]{2,5}:)")
SWB_MARK_RE = re.compile(r"(?=\b[AB]:\s)")
SUBTITLE_DASH_RE = re.compile(r"\s+-\s+(?=[A-Z\[\-])")


def tokenish(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text) if t.strip()]


def words_only(tokens: list[str]) -> list[str]:
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def overlap_stats(a: list[str], b: list[str]) -> dict[str, float]:
    aw = words_only(a)
    bw = words_only(b)
    ca = collections.Counter(aw)
    cb = collections.Counter(bw)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return {
        "source_words": len(aw),
        "rewrite_words": len(bw),
        "overlap_multiset": (inter / union) if union else 0.0,
        "rewrite_recall_from_source": (inter / len(bw)) if bw else 0.0,
        "source_recall_from_rewrite": (inter / len(aw)) if aw else 0.0,
    }


def edit_summary(source: str, rewrite: str) -> dict[str, Any]:
    s = tokenish(source)
    r = tokenish(rewrite)
    sm = difflib.SequenceMatcher(a=s, b=r, autojunk=False)
    ops = sm.get_opcodes()
    counts = collections.Counter(tag for tag, *_ in ops)
    # Changed target spans in rewrite are insert or replace target-side spans.
    changed_target_spans = []
    changed_source_spans = []
    equal_anchor_lengths = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            equal_anchor_lengths.append(i2 - i1)
        if tag in {"insert", "replace"} and j2 > j1:
            changed_target_spans.append(j2 - j1)
        if tag in {"delete", "replace"} and i2 > i1:
            changed_source_spans.append(i2 - i1)
    total_changed_target = sum(changed_target_spans)
    total_changed_source = sum(changed_source_spans)
    n_equal = sum(equal_anchor_lengths)
    stats = overlap_stats(s, r)
    stats.update(
        {
            "source_tokenish": len(s),
            "rewrite_tokenish": len(r),
            "op_equal": counts.get("equal", 0),
            "op_insert": counts.get("insert", 0),
            "op_delete": counts.get("delete", 0),
            "op_replace": counts.get("replace", 0),
            "op_total": len(ops),
            "equal_tokenish": n_equal,
            "changed_target_tokenish": total_changed_target,
            "changed_source_tokenish": total_changed_source,
            "changed_target_frac": (total_changed_target / len(r)) if r else 0.0,
            "equal_anchor_frac_source": (n_equal / len(s)) if s else 0.0,
            "n_changed_target_spans": len(changed_target_spans),
            "max_changed_target_span": max(changed_target_spans) if changed_target_spans else 0,
            "mean_changed_target_span": (sum(changed_target_spans) / len(changed_target_spans)) if changed_target_spans else 0.0,
            "n_equal_anchors_ge3": sum(1 for x in equal_anchor_lengths if x >= 3),
            "n_equal_anchors_ge5": sum(1 for x in equal_anchor_lengths if x >= 5),
            "has_directional_edit_target": bool(total_changed_target >= 3 and len(r) >= 8 and len(s) >= 8),
            "usable_for_changed_span_probe": bool(total_changed_target >= 3 and sum(1 for x in equal_anchor_lengths if x >= 3) >= 1),
        }
    )
    return stats


def load_pairs() -> list[dict[str, Any]]:
    pairs = []
    with PAIR_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Field names verified in research examples; keep fallback names robustly.
            source = rec.get("source_text") or rec.get("source") or rec.get("original") or rec.get("source_sentence")
            rewrite = rec.get("rewrite_text") or rec.get("rewrite") or rec.get("compact") or rec.get("compact_text")
            if source is None or rewrite is None:
                # Common actual fields appear in first lines as source_text/rewrite_text, but record missing if not.
                pairs.append({"_missing_text_fields": True, "raw_keys": sorted(rec.keys())[:30], "pair_id": rec.get("pair_id")})
                continue
            es = edit_summary(source, rewrite)
            es.update(
                {
                    "pair_id": rec.get("pair_id"),
                    "key": rec.get("key"),
                    "source_words_meta": rec.get("source_words"),
                    "rewrite_words_meta": rec.get("rewrite_words"),
                    "content_recall_meta": rec.get("content_recall"),
                    "content_overlap_meta": rec.get("content_overlap"),
                    "entity_recall_meta": rec.get("entity_recall"),
                    "number_recall_meta": rec.get("number_recall"),
                }
            )
            pairs.append(es)
    return pairs


def split_segments(text: str, source: str) -> list[str]:
    txt = " ".join(text.split())
    if not txt:
        return []
    chunks: list[str]
    if source == "childes":
        # Split at speaker marks while preserving mark on segment.
        chunks = [c.strip() for c in CHILDES_MARK_RE.split(txt) if c.strip()]
    elif source == "switchboard":
        chunks = [c.strip() for c in SWB_MARK_RE.split(txt) if c.strip()]
    elif source == "open_subtitles":
        # subtitles often concatenate speakers with dash separators; combine with punctuation split.
        rough = []
        for part in SUBTITLE_DASH_RE.split(txt):
            rough.extend(SENT_SPLIT_RE.split(part))
        chunks = [c.strip(" -") for c in rough if c.strip(" -")]
    else:
        chunks = [c.strip() for c in SENT_SPLIT_RE.split(txt) if c.strip()]
    # Filter too-short fragments; keep medium segments for state tests.
    return [c for c in chunks if len(words_only(tokenish(c))) >= 3]


def natural_adjacency_inventory() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_source: dict[str, Any] = {}
    example_rows = []
    counters = {s: collections.Counter() for s in NATURAL_SOURCES | SYNTHETIC_SOURCES}
    seg_lens = {s: [] for s in NATURAL_SOURCES | SYNTHETIC_SOURCES}
    adj_counts = {s: [] for s in NATURAL_SOURCES | SYNTHETIC_SOURCES}
    source_words = collections.Counter()
    source_rows = collections.Counter()
    source_ids_prefix = collections.defaultdict(list)

    with POOL.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            rec = json.loads(line)
            source = rec.get("source")
            if source not in counters:
                counters[source] = collections.Counter()
                seg_lens[source] = []
                adj_counts[source] = []
            source_rows[source] += 1
            source_words[source] += rec.get("words", 0)
            if len(source_ids_prefix[source]) < 20:
                source_ids_prefix[source].append(rec.get("example_id"))
            segs = split_segments(rec.get("text", ""), source)
            lens = [len(words_only(tokenish(s))) for s in segs]
            seg_lens[source].extend(lens)
            nseg = len(segs)
            nadj = max(0, nseg - 1)
            ntrip = max(0, nseg - 2)
            adj_counts[source].append(nadj)
            counters[source].update({
                "rows": 1,
                "words": rec.get("words", 0),
                "rows_ge2_segments": int(nseg >= 2),
                "rows_ge3_segments": int(nseg >= 3),
                "segments": nseg,
                "adjacent_pairs": nadj,
                "adjacent_triples": ntrip,
            })
            if source in NATURAL_SOURCES and len(example_rows) < 12 and nseg >= 3:
                example_rows.append({
                    "row_index_1based": idx,
                    "source": source,
                    "example_id": rec.get("example_id"),
                    "words": rec.get("words"),
                    "n_segments": nseg,
                    "segments_preview": segs[:5],
                })
    for source, c in counters.items():
        if c.get("rows", 0) == 0:
            continue
        ids = source_ids_prefix[source]
        by_source[source] = {
            "rows": c["rows"],
            "words": c["words"],
            "segments": c["segments"],
            "rows_ge2_segments": c["rows_ge2_segments"],
            "rows_ge3_segments": c["rows_ge3_segments"],
            "adjacent_pairs": c["adjacent_pairs"],
            "adjacent_triples": c["adjacent_triples"],
            "mean_segments_per_row": c["segments"] / c["rows"],
            "median_segment_words": statistics.median(seg_lens[source]) if seg_lens[source] else 0,
            "mean_segment_words": statistics.mean(seg_lens[source]) if seg_lens[source] else 0,
            "mean_adjacent_pairs_per_row": statistics.mean(adj_counts[source]) if adj_counts[source] else 0,
            "example_id_prefix": ids,
            "example_id_monotone_prefix": all(isinstance(ids[i], int) and isinstance(ids[i+1], int) and ids[i] <= ids[i+1] for i in range(len(ids)-1)) if len(ids) >= 2 else None,
            "natural_for_discourse_route": source in NATURAL_SOURCES,
            "exclude_from_natural_discourse_aux": source in SYNTHETIC_SOURCES or source.startswith("cleanqwen") or source.startswith("qwen"),
        }
    return by_source, example_rows


def summarize_numeric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    vals = [r[field] for r in rows if field in r and isinstance(r[field], (int, float)) and not math.isnan(float(r[field]))]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        k = (len(vals_sorted)-1)*p
        lo = math.floor(k); hi = math.ceil(k)
        if lo == hi: return vals_sorted[lo]
        return vals_sorted[lo]*(hi-k) + vals_sorted[hi]*(k-lo)
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    missing = [p for p in pairs if p.get("_missing_text_fields")]
    good = [p for p in pairs if not p.get("_missing_text_fields")]
    pair_summary = {
        "pair_file": str(PAIR_FILE),
        "n_records": len(pairs),
        "n_missing_text_fields": len(missing),
        "missing_examples": missing[:3],
        "n_text_pairs": len(good),
        "usable_for_changed_span_probe": sum(1 for p in good if p.get("usable_for_changed_span_probe")),
        "has_directional_edit_target": sum(1 for p in good if p.get("has_directional_edit_target")),
        "stats": {field: summarize_numeric(good, field) for field in [
            "source_words", "rewrite_words", "overlap_multiset", "rewrite_recall_from_source",
            "changed_target_tokenish", "changed_target_frac", "n_changed_target_spans",
            "max_changed_target_span", "n_equal_anchors_ge3", "equal_anchor_frac_source",
        ]},
        "content_overlap_bins": dict(collections.Counter(
            "lt_0p4" if p.get("overlap_multiset", 0) < 0.4 else
            "0p4_0p6" if p.get("overlap_multiset", 0) < 0.6 else
            "0p6_0p8" if p.get("overlap_multiset", 0) < 0.8 else
            "ge_0p8" for p in good
        )),
        "changed_target_frac_bins": dict(collections.Counter(
            "lt_0p2" if p.get("changed_target_frac", 0) < 0.2 else
            "0p2_0p4" if p.get("changed_target_frac", 0) < 0.4 else
            "0p4_0p6" if p.get("changed_target_frac", 0) < 0.6 else
            "ge_0p6" for p in good
        )),
        "top_changed_target_examples": sorted(good, key=lambda r: r.get("changed_target_tokenish", 0), reverse=True)[:20],
        "low_overlap_usable_examples": [p for p in good if p.get("usable_for_changed_span_probe") and p.get("overlap_multiset", 1) < 0.55][:20],
    }

    adjacency_by_source, adjacency_examples = natural_adjacency_inventory()
    natural_totals = collections.Counter()
    synthetic_totals = collections.Counter()
    for src, rec in adjacency_by_source.items():
        c = natural_totals if rec["natural_for_discourse_route"] and not rec["exclude_from_natural_discourse_aux"] else synthetic_totals
        for key in ["rows", "words", "segments", "rows_ge2_segments", "rows_ge3_segments", "adjacent_pairs", "adjacent_triples"]:
            c[key] += rec[key]

    result = {
        "status": "CANDIDATE_SIGNAL_INVENTORY",
        "purpose": "Quantify legal directed-edit and intra-row-adjacency candidate signals before route commitment or training.",
        "inputs": {
            "pool": str(POOL),
            "pair_file": str(PAIR_FILE),
            "row_meta": str(ROW_META),
        },
        "legal_notes": {
            "uses_eval_labels": False,
            "uses_external_annotations": False,
            "uses_external_text": False,
            "synthetic_rows_excluded_from_natural_discourse_aux": sorted(SYNTHETIC_SOURCES),
            "cross_row_order_available": False,
            "within_row_adjacency_available": True,
        },
        "directed_edit_pairs": pair_summary,
        "intra_row_adjacency_by_source": adjacency_by_source,
        "intra_row_adjacency_natural_totals": dict(natural_totals),
        "intra_row_adjacency_synthetic_or_excluded_totals": dict(synthetic_totals),
        "natural_adjacency_example_rows": adjacency_examples,
        "route_implications": {
            "directed_edit_state_probe_supported_by_inventory": bool(pair_summary["usable_for_changed_span_probe"] >= 5000),
            "natural_discourse_state_probe_supported_by_inventory": bool(natural_totals["adjacent_triples"] >= 20000 and natural_totals["rows_ge3_segments"] >= 1000),
            "must_exclude_synthetic_rows_for_discourse_signal": True,
            "must_use_within_source_length_overlap_controls": True,
            "pair_retrieval_not_target": "research saturated true-pair retrieval; edit-state probe must measure conditional token NLL on changed spans, not pair identity.",
        },
    }
    out_json = OUT_DIR / "candidate_signal_inventory.json"
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/candidate_signal_inventory/candidate_signal_inventory.md')
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("# research — candidate signal inventory")
    lines.append("")
    lines.append("CPU-only inventory of two legal candidate signals before any new training.")
    lines.append("")
    lines.append("## Directed source→compact edit structure")
    lines.append(f"- pair records: `{pair_summary['n_records']}`; text pairs parsed: `{pair_summary['n_text_pairs']}`; missing text fields: `{pair_summary['n_missing_text_fields']}`")
    lines.append(f"- usable changed-span probes (>=3 changed target tokens and >=1 equal anchor length>=3): `{pair_summary['usable_for_changed_span_probe']}`")
    lines.append(f"- directional edit targets: `{pair_summary['has_directional_edit_target']}`")
    for field in ["overlap_multiset", "rewrite_recall_from_source", "changed_target_tokenish", "changed_target_frac", "n_equal_anchors_ge3"]:
        st = pair_summary['stats'][field]
        if st.get('n'):
            lines.append(f"- {field}: mean `{st['mean']:.4f}`, median `{st['median']:.4f}`, p10 `{st['p10']:.4f}`, p90 `{st['p90']:.4f}`")
    lines.append(f"- overlap bins: `{pair_summary['content_overlap_bins']}`")
    lines.append(f"- changed-target fraction bins: `{pair_summary['changed_target_frac_bins']}`")
    lines.append("")
    lines.append("## Intra-row natural adjacency")
    lines.append("Cross-row order is not available from the packed pool; only within-row adjacency is valid for this route.")
    lines.append("")
    lines.append("| source | rows | words | segments | rows>=3seg | adj pairs | adj triples | mean seg/row | median seg words | natural? | exclude? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for src, rec in sorted(adjacency_by_source.items(), key=lambda kv: (-kv[1]['words'], kv[0])):
        lines.append(
            f"| {src} | {rec['rows']} | {rec['words']} | {rec['segments']} | {rec['rows_ge3_segments']} | {rec['adjacent_pairs']} | {rec['adjacent_triples']} | {rec['mean_segments_per_row']:.2f} | {rec['median_segment_words']:.1f} | {rec['natural_for_discourse_route']} | {rec['exclude_from_natural_discourse_aux']} |"
        )
    lines.append("")
    lines.append("Natural totals excluding synthetic rows:")
    lines.append(f"- rows `{natural_totals['rows']}`, words `{natural_totals['words']}`, segments `{natural_totals['segments']}`, adjacent pairs `{natural_totals['adjacent_pairs']}`, adjacent triples `{natural_totals['adjacent_triples']}`")
    lines.append("")
    lines.append("## Route implications")
    ri = result['route_implications']
    lines.append(f"- directed edit-state zero-training probe supported: `{ri['directed_edit_state_probe_supported_by_inventory']}`")
    lines.append(f"- natural discourse-state zero-training probe supported: `{ri['natural_discourse_state_probe_supported_by_inventory']}`")
    lines.append("- Discourse route must exclude `qwen_pair_packed` and FineWeb compact rows from the natural adjacency auxiliary, and use true/reversed/shuffled within-source controls.")
    lines.append("- Transformation route must measure conditional token NLL on changed spans with overlap/edit-matched decoys; pair retrieval or pooled alignment is saturated and not a target.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result['status'],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "usable_edit_pairs": pair_summary['usable_for_changed_span_probe'],
        "natural_adjacent_triples": natural_totals['adjacent_triples'],
        "natural_discourse_supported": ri['natural_discourse_state_probe_supported_by_inventory'],
        "directed_edit_supported": ri['directed_edit_state_probe_supported_by_inventory'],
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
