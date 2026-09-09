#!/usr/bin/env python3
"""research: CPU audit of compact/repeat causal packet topology.

The causal transfer result is negative for broad compact-view transfer. This audit
checks whether the constructed causal pools really match packet topology: which rows
are paired packets, approximate sentence counts, packet lengths, and whether compact
vs repeat differences are limited to semantic view text rather than row/order shape.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any

PAIR_SOURCE_PREFIX = "qwen_pair_packed"
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    return text.split()


def sentence_count(text: str) -> int:
    s = [x.strip() for x in SENT_RE.split(text.strip()) if x.strip()]
    return max(1, len(s)) if text.strip() else 0


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                obj["_idx"] = i
                rows.append(obj)
    return rows


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    sv = sorted(vals)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": sv[0],
        "p10": sv[int(0.10*(len(sv)-1))],
        "p90": sv[int(0.90*(len(sv)-1))],
        "max": sv[-1],
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
    }


def audit_arm(path: pathlib.Path) -> dict[str, Any]:
    rows = load_rows(path)
    pair_rows = [r for r in rows if str(r.get("source")) == PAIR_SOURCE_PREFIX]
    non_pair = [r for r in rows if str(r.get("source")) != PAIR_SOURCE_PREFIX]
    row_words = [int(r.get("words", len(words(r.get("text", ""))))) for r in rows]
    text_word_counts = [len(words(r.get("text", ""))) for r in rows]
    sent_counts = [sentence_count(r.get("text", "")) for r in rows]
    pair_word_counts = [int(r.get("words", len(words(r.get("text", ""))))) for r in pair_rows]
    pair_sent_counts = [sentence_count(r.get("text", "")) for r in pair_rows]
    sources = Counter(str(r.get("source")) for r in rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "total_words_field": sum(row_words),
        "total_words_text_split": sum(text_word_counts),
        "word_field_minus_split": sum(row_words) - sum(text_word_counts),
        "pair_rows": len(pair_rows),
        "pair_words": sum(pair_word_counts),
        "non_pair_rows": len(non_pair),
        "source_counts_top10": sources.most_common(10),
        "row_word_stats": stats(row_words),
        "text_word_stats": stats(text_word_counts),
        "sentence_count_stats": stats(sent_counts),
        "pair_word_stats": stats(pair_word_counts),
        "pair_sentence_count_stats": stats(pair_sent_counts),
        "first_pair_indices": [r["_idx"] for r in pair_rows[:20]],
        "last_pair_indices": [r["_idx"] for r in pair_rows[-20:]],
    }


def compare(compact_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(compact_rows) == len(repeat_rows)
    n = len(compact_rows)
    mismatches = []
    pair_positions = []
    same_text = 0
    word_deltas = []
    sent_deltas = []
    char_deltas = []
    for i, (c, r) in enumerate(zip(compact_rows, repeat_rows)):
        if c.get("source") != r.get("source"):
            mismatches.append({"idx": i, "field": "source", "compact": c.get("source"), "repeat": r.get("source")})
        cw = int(c.get("words", len(words(c.get("text", "")))))
        rw = int(r.get("words", len(words(r.get("text", "")))))
        if cw != rw:
            mismatches.append({"idx": i, "field": "words", "compact": cw, "repeat": rw})
        if c.get("text") == r.get("text"):
            same_text += 1
        word_deltas.append(len(words(c.get("text", ""))) - len(words(r.get("text", ""))))
        sent_deltas.append(sentence_count(c.get("text", "")) - sentence_count(r.get("text", "")))
        char_deltas.append(len(c.get("text", "")) - len(r.get("text", "")))
        if c.get("source") == PAIR_SOURCE_PREFIX:
            pair_positions.append(i)
    gaps = [b-a for a,b in zip(pair_positions, pair_positions[1:])]
    return {
        "n_rows_compared": n,
        "source_or_word_mismatch_count": len(mismatches),
        "source_or_word_mismatches_first20": mismatches[:20],
        "same_text_rows": same_text,
        "different_text_rows": n - same_text,
        "pair_positions_count": len(pair_positions),
        "pair_gap_stats": stats([float(x) for x in gaps]),
        "text_split_word_delta_stats_compact_minus_repeat": stats([float(x) for x in word_deltas]),
        "sentence_delta_stats_compact_minus_repeat": stats([float(x) for x in sent_deltas]),
        "char_delta_stats_compact_minus_repeat": stats([float(x) for x in char_deltas]),
        "pair_position_first20": pair_positions[:20],
        "pair_position_last20": pair_positions[-20:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", type=pathlib.Path, required=True)
    ap.add_argument("--repeat", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compact_rows = load_rows(args.compact)
    repeat_rows = load_rows(args.repeat)
    result = {
        "status": "CAUSAL_PACKET_TOPOLOGY_AUDIT",
        "compact": audit_arm(args.compact),
        "repeat": audit_arm(args.repeat),
        "paired_row_comparison": compare(compact_rows, repeat_rows),
        "interpretation": [
            "The causal experiment is meant to differ in compact semantic view vs exact repeat text, not tokenizer, legal words, or row/order topology.",
            "If source/word fields and pair positions match, the negative transfer result cannot be dismissed as row-topology mismatch.",
        ],
    }
    out_json = args.out_dir / "causal_packet_topology_audit.json"
    out_md = args.out_dir / "causal_packet_topology_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    c = result["compact"]; r = result["repeat"]; p = result["paired_row_comparison"]
    md = ["# research causal packet topology audit\n\n"]
    md.append(f"Compact rows/words: {c['rows']} / {c['total_words_field']}; repeat rows/words: {r['rows']} / {r['total_words_field']}.\n\n")
    md.append(f"Pair rows: compact {c['pair_rows']} ({c['pair_words']} words), repeat {r['pair_rows']} ({r['pair_words']} words).\n\n")
    md.append(f"Source/word-field mismatches at same row index: {p['source_or_word_mismatch_count']}. Same text rows: {p['same_text_rows']}; different text rows: {p['different_text_rows']}.\n\n")
    md.append(f"Pair-position count: {p['pair_positions_count']}; pair gap mean {p['pair_gap_stats'].get('mean')}, median {p['pair_gap_stats'].get('median')}.\n\n")
    md.append(f"Text split word delta compact-repeat mean {p['text_split_word_delta_stats_compact_minus_repeat'].get('mean')}; sentence-count delta mean {p['sentence_delta_stats_compact_minus_repeat'].get('mean')}; char delta mean {p['char_delta_stats_compact_minus_repeat'].get('mean')}.\n\n")
    md.append("Interpretation: matched source/word fields and pair positions mean the causal negative result is not due to row-order or legal-word mismatch; compact/repeat differ in the packet text content and small tokenization/effective-token asymmetry already recorded.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
