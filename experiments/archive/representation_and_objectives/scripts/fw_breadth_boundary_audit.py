#!/usr/bin/env python3
"""research: audit whether the research source-breadth companion stream splits
independent FineWeb sentences across training examples.

The FW compact-vs-breadth comparison is intended to isolate aligned compact
re-expression versus additional independent FineWeb experience. research preserved
row word totals by concatenating the selected breadth sentences into one global
word stream and slicing it by the compact arm's per-row companion-word budgets.
This audit quantifies the resulting sentence-boundary damage before any H100
training is launched.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
META = ROOT / "data/fw_source_breadth_arm/source_breadth_row_meta.jsonl"
SOURCES = ROOT / "data/fw_source_breadth_arm/source_breadth_companion_sources.jsonl"
USABLE_PAIRS = ROOT / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl"
SCRIPT = ROOT / "scripts/fw_source_breadth_arm.py"
OUT_DIR = ROOT / "data/fw_breadth_boundary_audit"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_breadth_boundary_audit.md')


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    arr = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        idx = p * (len(arr) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(arr) - 1)
        frac = idx - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac
    return {
        "n": len(arr),
        "sum": float(sum(arr)),
        "min": arr[0],
        "p01": q(0.01),
        "p05": q(0.05),
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": arr[-1],
        "mean": float(statistics.mean(arr)),
    }


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def norm_hash(text: str) -> str:
    return hashlib.sha256(norm_text(text).lower().encode("utf-8")).hexdigest()[:32]


def load_step102_module():
    spec = importlib.util.spec_from_file_location("fw_source_breadth_arm", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def unbounded_possible(lengths: set[int], cap: int) -> bool:
    poss = [False] * (cap + 1)
    poss[0] = True
    for s in range(cap + 1):
        if not poss[s]:
            continue
        for w in lengths:
            ns = s + w
            if ns <= cap:
                poss[ns] = True
    return poss[cap]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metas = read_jsonl(META)
    sources = read_jsonl(SOURCES)
    budgets = [int(r["breadth_companion_words"]) for r in metas]
    starts = [int(r["breadth_word_start"]) for r in metas]
    ends = [int(r["breadth_word_end"]) for r in metas]
    if starts[0] != 0 or any(starts[i] != ends[i-1] for i in range(1, len(starts))):
        raise RuntimeError("row companion intervals are not contiguous")
    total_companion = sum(budgets)
    if ends[-1] != total_companion:
        raise RuntimeError("row interval end does not match budget sum")

    source_intervals: list[tuple[int, int, dict[str, Any]]] = []
    pos = 0
    for s in sources:
        w = int(s["words"])
        source_intervals.append((pos, pos + w, s))
        pos += w
    if pos != total_companion:
        raise RuntimeError(f"source words {pos} != companion budget {total_companion}")

    row_reports: list[dict[str, Any]] = []
    row_source_counts: list[int] = []
    row_split_flags: list[bool] = []
    row_left_fragment: list[bool] = []
    row_right_fragment: list[bool] = []
    source_to_rows: collections.defaultdict[str, list[tuple[int, int]]] = collections.defaultdict(list)

    src_i = 0
    for i, m in enumerate(metas):
        a, b = starts[i], ends[i]
        while src_i < len(source_intervals) and source_intervals[src_i][1] <= a:
            src_i += 1
        j = src_i
        overlaps: list[dict[str, Any]] = []
        left = False
        right = False
        while j < len(source_intervals) and source_intervals[j][0] < b:
            s0, s1, s = source_intervals[j]
            frag_a, frag_b = max(a, s0), min(b, s1)
            h = str(s["norm_hash"])
            source_to_rows[h].append((i, frag_b - frag_a))
            if s0 < a:
                left = True
            if s1 > b:
                right = True
            overlaps.append({
                "source_hash": h,
                "source_words": int(s["words"]),
                "fragment_words": frag_b - frag_a,
                "starts_inside_source": s0 < a,
                "ends_inside_source": s1 > b,
            })
            j += 1
        split = left or right
        row_split_flags.append(split)
        row_left_fragment.append(left)
        row_right_fragment.append(right)
        row_source_counts.append(len(overlaps))
        if split or i < 5:
            row_reports.append({
                "row_index": i,
                "example_id": int(m["example_id"]),
                "budget": budgets[i],
                "left_fragment": left,
                "right_fragment": right,
                "source_count": len(overlaps),
                "overlaps": overlaps[:8],
            })

    split_sources = {h: frags for h, frags in source_to_rows.items() if len(frags) > 1}
    fragments_per_split_source = [len(v) for v in split_sources.values()]
    split_fragment_lengths = [frag for v in split_sources.values() for _, frag in v]

    # Candidate-pool expressive capacity if one wanted to repair row boundaries.
    breadth_builder = load_step102_module()
    pairs = breadth_builder.load_pairs()
    exclude_hashes = {str(p["norm_hash"]) for p in pairs}
    candidates, candidate_counts = breadth_builder.load_candidate_sources(exclude_hashes)
    length_counts = collections.Counter(int(c["words"]) for c in candidates)
    available_lengths = set(length_counts)
    impossible_by_length = [b for b in budgets if not unbounded_possible(available_lengths, b)]

    result = {
        "status": "FW_BREADTH_BOUNDARY_AUDIT",
        "inputs": {
            "row_meta": str(META),
            "selected_sources": str(SOURCES),
            "usable_pairs": str(USABLE_PAIRS),
        },
        "companion_budget": {
            "rows": len(metas),
            "total_words": total_companion,
            "budget_stats": stats(budgets),
            "budget_histogram_first_80": {str(k): int(v) for k, v in sorted(collections.Counter(budgets).items()) if k <= 80},
        },
        "current_step102_global_stream_slicing": {
            "selected_source_sentences": len(sources),
            "source_word_stats": stats([int(s["words"]) for s in sources]),
            "rows_with_any_split_source_count": int(sum(row_split_flags)),
            "rows_with_any_split_source_frac": float(sum(row_split_flags) / len(row_split_flags)),
            "rows_starting_mid_source_count": int(sum(row_left_fragment)),
            "rows_ending_mid_source_count": int(sum(row_right_fragment)),
            "source_sentences_spanning_multiple_rows_count": len(split_sources),
            "source_sentences_spanning_multiple_rows_frac": float(len(split_sources) / len(sources)),
            "sources_per_row_stats": stats(row_source_counts),
            "fragments_per_split_source_stats": stats(fragments_per_split_source),
            "split_fragment_length_stats": stats(split_fragment_lengths),
            "sample_rows": row_reports[:60],
        },
        "repair_feasibility_length_only": {
            "candidate_sources_after_excluding_compact_pair_sources": len(candidates),
            "candidate_words": int(sum(int(c["words"]) for c in candidates)),
            "candidate_counts_by_pool": candidate_counts,
            "candidate_length_min": min(available_lengths) if available_lengths else None,
            "candidate_length_max": max(available_lengths) if available_lengths else None,
            "candidate_length_histogram_1_80": {str(k): int(length_counts[k]) for k in range(1, 81) if length_counts.get(k)},
            "row_budgets_unrepresentable_by_unbounded_candidate_lengths_count": len(impossible_by_length),
            "row_budgets_unrepresentable_by_unbounded_candidate_lengths_hist": {str(k): int(v) for k, v in sorted(collections.Counter(impossible_by_length).items())},
        },
        "scientific_reading": {
            "boundary_confounds_if_high": "If many breadth rows contain sentence fragments or split sources across examples, a compact>broad result could partly reflect damaged independent-source experience rather than the value of aligned compact re-expression.",
            "repair_target": "A stronger first expensive comparison would keep the same per-row companion-word budgets but fill each row with whole independent FineWeb sentences whenever feasible, preserving exact row word sequence without slicing sentences across examples.",
        },
    }
    out_json = OUT_DIR / "fw_breadth_boundary_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    slicing = result["current_step102_global_stream_slicing"]
    feas = result["repair_feasibility_length_only"]
    NOTE.write_text(
        "# research — FW source-breadth sentence-boundary audit\n\n"
        f"Current research source-breadth arm uses {len(sources):,} selected independent FineWeb sentences "
        f"({total_companion:,} companion words) sliced into {len(metas):,} compact-row companion budgets.\n\n"
        "## Boundary damage in current arm\n\n"
        f"- Rows with any split source sentence: {slicing['rows_with_any_split_source_count']:,} / {len(metas):,} "
        f"({slicing['rows_with_any_split_source_frac']:.3f}).\n"
        f"- Rows starting mid-source: {slicing['rows_starting_mid_source_count']:,}; rows ending mid-source: {slicing['rows_ending_mid_source_count']:,}.\n"
        f"- Source sentences spanning multiple rows: {slicing['source_sentences_spanning_multiple_rows_count']:,} / {len(sources):,} "
        f"({slicing['source_sentences_spanning_multiple_rows_frac']:.3f}).\n"
        f"- Sources per row: mean {slicing['sources_per_row_stats']['mean']:.3f}, max {int(slicing['sources_per_row_stats']['max'])}.\n\n"
        "## Repair feasibility clue\n\n"
        f"Candidate pool after excluding compact-pair sources: {feas['candidate_sources_after_excluding_compact_pair_sources']:,} sentences / "
        f"{feas['candidate_words']:,} words. Candidate lengths span {feas['candidate_length_min']}–{feas['candidate_length_max']} words. "
        f"Unbounded length expressibility failures among the {len(metas):,} row companion budgets: "
        f"{feas['row_budgets_unrepresentable_by_unbounded_candidate_lengths_count']}.\n\n"
        "## Scientific consequence\n\n"
        "Before launching H100 training, repair the source-breadth arm if exact whole-sentence per-row packing is feasible. "
        "The expensive compact-vs-breadth result should not be vulnerable to the objection that the breadth arm was made weaker by arbitrary cross-row sentence slicing.\n\n"
        f"JSON: `{out_json}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "rows_with_split_frac": round(slicing["rows_with_any_split_source_frac"], 4),
        "split_sources_frac": round(slicing["source_sentences_spanning_multiple_rows_frac"], 4),
        "unbounded_unrepresentable_budgets": feas["row_budgets_unrepresentable_by_unbounded_candidate_lengths_count"],
        "json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
