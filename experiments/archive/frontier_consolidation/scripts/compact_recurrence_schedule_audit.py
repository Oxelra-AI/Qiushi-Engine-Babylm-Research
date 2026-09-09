#!/usr/bin/env python3
"""Audit the current compact-view-reinvest source/view organization.

Purpose: before proposing a real-training schedule intervention, establish what the
existing protected corpus actually does with source/view recurrence: exact text
multiset, current adjacency, row packing, unit lengths, and confounds a schedule
materializer must control.
"""
from __future__ import annotations

import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/frontier_consolidation")
PAIR_PATH = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
ROW_META_PATH = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
POOL_10M_PATH = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
COMMON_FILLER_PATH = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/common_filler_rows.jsonl"
OUT_DIR = ROOT / "data/compact_recurrence_schedule_audit"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def stats(xs):
    xs = list(xs)
    if not xs:
        return {"n": 0}
    q = sorted(xs)
    def pct(p):
        if len(q) == 1:
            return q[0]
        idx = int(round((len(q) - 1) * p))
        return q[max(0, min(len(q) - 1, idx))]
    return {
        "n": len(xs),
        "min": min(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "max": max(xs),
        "sum": sum(xs),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = list(read_jsonl(PAIR_PATH))
    metas = list(read_jsonl(ROW_META_PATH))
    pool_rows = []
    with POOL_10M_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i < len(metas):
                pool_rows.append(obj)
            elif i == len(metas):
                first_filler = obj
                break
        else:
            first_filler = None

    by_id = {p["pair_id"]: p for p in pairs}
    missing_pair_ids = []
    duplicate_pair_ids = [pid for pid, c in collections.Counter(p["pair_id"] for p in pairs).items() if c > 1]
    row_pair_counts = []
    row_word_counts = []
    row_sources = collections.Counter()
    rows_with_all_text_exact = 0
    row_exact_failures = []
    pair_presence = {}
    adjacency_word_gaps = []
    adjacency_char_gaps = []
    pair_order_ok = 0
    current_unit_sequence = []

    for row_obj, meta in zip(pool_rows, metas):
        text = row_obj["text"]
        row_word_counts.append(row_obj.get("words", wc(text)))
        pids = meta.get("pair_ids", [])
        row_pair_counts.append(len(pids))
        ok_row = True
        previous_end = -1
        for pid in pids:
            p = by_id.get(pid)
            if not p:
                missing_pair_ids.append(pid)
                ok_row = False
                continue
            src = p["source_text"]
            rew = p["rewrite_text"]
            src_pos = text.find(src)
            rew_pos = text.find(rew)
            if src_pos < 0 or rew_pos < 0:
                ok_row = False
                row_exact_failures.append({
                    "row_index": meta.get("row_index"),
                    "example_id": row_obj.get("example_id"),
                    "pair_id": pid,
                    "source_found": src_pos >= 0,
                    "rewrite_found": rew_pos >= 0,
                })
                continue
            source_before_rewrite = src_pos < rew_pos
            pair_order_ok += int(source_before_rewrite)
            between = text[src_pos + len(src):rew_pos] if source_before_rewrite else text[rew_pos + len(rew):src_pos]
            adjacency_word_gaps.append(wc(between))
            adjacency_char_gaps.append(len(between))
            pair_presence[pid] = {
                "row_index": meta.get("row_index"),
                "example_id": row_obj.get("example_id"),
                "source_pos": src_pos,
                "rewrite_pos": rew_pos,
                "source_before_rewrite": source_before_rewrite,
                "word_gap_between_source_and_rewrite": wc(between),
                "char_gap_between_source_and_rewrite": len(between),
                "source_words_recorded": p.get("source_words"),
                "rewrite_words_recorded": p.get("rewrite_words"),
                "source_words_recomputed": wc(src),
                "rewrite_words_recomputed": wc(rew),
                "domain_hits": p.get("domain_hits", []),
                "doc_id": p.get("doc_id"),
                "sentence_id": p.get("sentence_id"),
            }
            current_unit_sequence.append({"pair_id": pid, "unit": "source", "words": wc(src), "text_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest()})
            current_unit_sequence.append({"pair_id": pid, "unit": "view", "words": wc(rew), "text_sha256": hashlib.sha256(rew.encode("utf-8")).hexdigest()})
            previous_end = max(previous_end, rew_pos + len(rew), src_pos + len(src))
            for d in p.get("domain_hits", []) or ["no_domain"]:
                row_sources[d] += 1
        if ok_row:
            rows_with_all_text_exact += 1

    not_seen_pairs = [pid for pid in by_id if pid not in pair_presence]
    source_words = [wc(p["source_text"]) for p in pairs]
    rewrite_words = [wc(p["rewrite_text"]) for p in pairs]
    pair_words = [wc(p["source_text"]) + wc(p["rewrite_text"]) for p in pairs]
    rows_if_split_source_view = len(list(read_jsonl(COMMON_FILLER_PATH))) + 2 * len(pairs)
    rows_if_pair_as_one_unit = len(list(read_jsonl(COMMON_FILLER_PATH))) + len(pairs)
    max_unit_words = max(source_words + rewrite_words)

    summary: dict[str, Any] = {
        "status": "COMPACT_RECURRENCE_SCHEDULE_AUDIT",
        "scientific_purpose": "Ground a possible schedule-only intervention on the validated compact-view/reinvest text: same strings, changed recurrence distance.",
        "inputs": {
            "selected_pairs": str(PAIR_PATH),
            "selected_pairs_sha256": sha256_path(PAIR_PATH),
            "changed_block_row_meta": str(ROW_META_PATH),
            "changed_block_row_meta_sha256": sha256_path(ROW_META_PATH),
            "pool_10M": str(POOL_10M_PATH),
            "pool_10M_sha256": sha256_path(POOL_10M_PATH),
            "common_filler_rows": str(COMMON_FILLER_PATH),
            "common_filler_sha256": sha256_path(COMMON_FILLER_PATH),
        },
        "pair_text_multiset": {
            "pairs": len(pairs),
            "source_words": sum(source_words),
            "rewrite_words": sum(rewrite_words),
            "pair_words": sum(pair_words),
            "recorded_pair_words_sum": sum(p.get("pair_words", 0) for p in pairs),
            "source_word_stats": stats(source_words),
            "rewrite_word_stats": stats(rewrite_words),
            "pair_word_stats": stats(pair_words),
            "max_unit_words": max_unit_words,
            "duplicate_pair_ids": duplicate_pair_ids[:20],
            "duplicate_pair_id_count": len(duplicate_pair_ids),
        },
        "current_packing": {
            "changed_rows": len(metas),
            "changed_pool_rows_read": len(pool_rows),
            "rows_with_all_source_view_text_exact": rows_with_all_text_exact,
            "row_exact_failure_count": len(row_exact_failures),
            "row_exact_failures_first20": row_exact_failures[:20],
            "missing_pair_id_refs_count": len(missing_pair_ids),
            "not_seen_pairs_count": len(not_seen_pairs),
            "not_seen_pairs_first20": not_seen_pairs[:20],
            "row_word_stats": stats(row_word_counts),
            "pairs_per_changed_row_distribution": dict(sorted(collections.Counter(row_pair_counts).items())),
            "source_before_rewrite_count": pair_order_ok,
            "adjacency_word_gap_stats": stats(adjacency_word_gaps),
            "adjacency_char_gap_stats": stats(adjacency_char_gaps),
            "first_filler_after_changed_block": first_filler,
        },
        "domain_hit_counts_across_pair_rows": dict(row_sources.most_common()),
        "schedule_feasibility": {
            "common_filler_rows": sum(1 for _ in read_jsonl(COMMON_FILLER_PATH)),
            "rows_if_each_pair_single_row_plus_filler": rows_if_pair_as_one_unit,
            "rows_if_source_and_view_split_plus_filler": rows_if_split_source_view,
            "all_source_and_view_units_below_seq256_words": max_unit_words <= 256,
            "proposed_minimal_arms": [
                "current_adjacent_existing_stream",
                "repacked_adjacent_split_units",
                "delayed_within_epoch_split_units",
            ],
            "must_control": [
                "text multiset equality",
                "word counts and pass counts",
                "row-boundary and max-length changes",
                "tokenizer candidate-token / WWM selected-mass shifts",
                "source-view recurrence distance distribution",
            ],
        },
    }

    (OUT_DIR / "compact_recurrence_schedule_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "current_pair_presence_first200.jsonl").open("w", encoding="utf-8") as f:
        for pid in list(pair_presence)[:200]:
            f.write(json.dumps({"pair_id": pid, **pair_presence[pid]}, ensure_ascii=False) + "\n")
    md = []
    md.append("# research compact recurrence schedule audit\n")
    md.append("## Pair text multiset\n")
    md.append(f"- Pairs: `{len(pairs)}`; source words `{sum(source_words)}`; rewrite words `{sum(rewrite_words)}`; pair words `{sum(pair_words)}`.\n")
    md.append(f"- Max individual source/view unit length: `{max_unit_words}` words; all below 256 words: `{max_unit_words <= 256}`.\n")
    md.append(f"- Source word stats: `{summary['pair_text_multiset']['source_word_stats']}`.\n")
    md.append(f"- Rewrite word stats: `{summary['pair_text_multiset']['rewrite_word_stats']}`.\n")
    md.append("\n## Current packing\n")
    md.append(f"- Changed rows: `{len(metas)}`; rows with exact source+view substrings for every listed pair: `{rows_with_all_text_exact}`.\n")
    md.append(f"- Pair count per changed row: `{summary['current_packing']['pairs_per_changed_row_distribution']}`.\n")
    md.append(f"- Source-before-view count: `{pair_order_ok}/{len(pairs)}`.\n")
    md.append(f"- Word gap between source and its compact view in current stream: `{summary['current_packing']['adjacency_word_gap_stats']}`.\n")
    md.append(f"- Row word stats for changed block: `{summary['current_packing']['row_word_stats']}`.\n")
    md.append("\n## Schedule feasibility\n")
    md.append(f"- Common filler rows: `{summary['schedule_feasibility']['common_filler_rows']}`.\n")
    md.append(f"- If each source/view pair stays one row plus filler: `{rows_if_pair_as_one_unit}` rows.\n")
    md.append(f"- If every source and view becomes a separate row plus filler: `{rows_if_split_source_view}` rows.\n")
    md.append("- The schedule route is mechanically feasible as an exact-text intervention, but any training comparison must include a `repacked_adjacent_split_units` arm so row-boundary effects are not misread as delayed recurrence.\n")
    md.append("\nJSON: `experiments/archive/frontier_consolidation/data/compact_recurrence_schedule_audit/compact_recurrence_schedule_audit.json`\n")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/compact_recurrence_schedule_audit/compact_recurrence_schedule_audit.md')).write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "pairs": len(pairs),
        "pair_words": sum(pair_words),
        "changed_rows": len(metas),
        "exact_rows": rows_with_all_text_exact,
        "source_before_rewrite": pair_order_ok,
        "word_gap_stats": summary["current_packing"]["adjacency_word_gap_stats"],
        "out_json": str(OUT_DIR / "compact_recurrence_schedule_audit.json"),
        "out_md": str((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/compact_recurrence_schedule_audit/compact_recurrence_schedule_audit.md')),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
