#!/usr/bin/env python3
"""research: annotate the unchanged inherited Qwen pair tail with segment metadata.

Purpose
-------
research built a compact structural policy stream with `qwen_pair_segments`, allowing
real-stream comparison between ordinary WWM and source-visible view-focused targets.
A cleaner control is now needed: apply the SAME objective comparison to the original
inherited source/current-rewrite pairs, with no generated compaction and no top-up
rows. This separates the value of using existing paired correspondence from the value
of generated shorter second views.

The output preserves the exact text and word count of
`reference_tail/reference_tail_ordinary_wordpaced.jsonl`; it only adds
metadata fields for qwen_pair_packed rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
REFERENCE_TAIL = _public_path('experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl')
SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PACKED_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_pairs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in iter_jsonl(SELECTED_PAIRS):
        out[str(r["pair_id"])] = r
    return out


def pair_current_rewrite(pair: Dict[str, Any]) -> str:
    return str(pair.get("rewrite", pair.get("current_rewrite", ""))).strip()


def pair_piece(pair: Dict[str, Any]) -> str:
    original = str(pair.get("original", "")).strip()
    view = pair_current_rewrite(pair)
    return (original + (" " + view if view else "")).strip()


def build_meta_hash_map(pairs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dup = 0
    for rm in iter_jsonl(PACKED_META):
        text = " ".join(pair_piece(pairs[pid]) for pid in rm.get("pair_ids", []) if pid in pairs)
        h = sha_text(text)
        if h in out:
            dup += 1
        out[h] = {**rm, "canonical_text_sha256": h, "canonical_text_words": wc(text)}
    if dup:
        print(json.dumps({"event": "duplicate_meta_hash_warning", "duplicates": dup}), flush=True)
    return out


def unchanged_segments_for_meta(rm: Dict[str, Any], pairs: Dict[str, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Counter]:
    parts: List[str] = []
    segments: List[Dict[str, Any]] = []
    kind_counts: Counter = Counter()
    cursor = 0
    for pid in rm.get("pair_ids", []):
        if pid not in pairs:
            continue
        pair = dict(pairs[pid])
        pair["pair_id"] = pid
        original = str(pair.get("original", "")).strip()
        view = pair_current_rewrite(pair)
        piece = (original + (" " + view if view else "")).strip()
        if parts:
            cursor += 1
        seg = {
            "pair_id": pid,
            "source_text": original,
            "view_text": view,
            "view_kind": "current_inherited_control",
            "candidate_kind": "unchanged_inherited_current",
            "source_start": cursor,
            "source_end": cursor + len(original),
            "view_start": cursor + len(original) + (1 if view else 0),
            "view_end": cursor + len(piece),
            "source_words": wc(original),
            "view_words": wc(view),
            "current_view_words": wc(view),
            "saved_words": 0,
        }
        parts.append(piece)
        cursor += len(piece)
        segments.append(seg)
        kind_counts["unchanged_inherited_current"] += 1
    return " ".join(parts), segments, kind_counts


def build(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    meta_by_hash = build_meta_hash_map(pairs)
    out_jsonl = out_dir / "reference_tail_unchanged_qwen_segments.jsonl"

    counts: Counter = Counter()
    source_rows: Counter = Counter()
    source_words: Counter = Counter()
    qwen_segment_kind_counts: Counter = Counter()
    first_mismatches: List[Dict[str, Any]] = []
    first_bad_segments: List[Dict[str, Any]] = []
    rows_written = 0
    total_words = 0
    with out_jsonl.open("w", encoding="utf-8") as out:
        for line_i, row0 in enumerate(iter_jsonl(Path(args.reference_tail))):
            row = dict(row0)
            text = str(row.get("text", ""))
            words = int(row.get("words", wc(text)))
            row["reference_tail_position"] = row.get("reference_tail_position", line_i)
            if row.get("source") == "qwen_pair_packed":
                counts["qwen_rows"] += 1
                rm = meta_by_hash.get(sha_text(text))
                if rm is None:
                    counts["missing_qwen_meta"] += 1
                else:
                    counts["matched_qwen_meta"] += 1
                    reconstructed, segments, kinds = unchanged_segments_for_meta(rm, pairs)
                    if reconstructed != text:
                        counts["reconstruction_mismatch"] += 1
                        if len(first_mismatches) < 10:
                            first_mismatches.append({
                                "line_i": line_i,
                                "row_sha": sha_text(text),
                                "row_words": words,
                                "reconstructed_words": wc(reconstructed),
                                "row_head": text[:200],
                                "reconstructed_head": reconstructed[:200],
                            })
                    row["qwen_pair_ids"] = list(rm.get("pair_ids", []))
                    row["qwen_pair_segments"] = segments
                    row["qwen_segment_policy"] = "unchanged inherited source/current-rewrite pairs; source-visible view-focused targetable metadata"
                    qwen_segment_kind_counts.update(kinds)
                    # Verify char offsets against row text for a bounded sample of failures.
                    for seg in segments:
                        s0, s1 = int(seg["source_start"]), int(seg["source_end"])
                        v0, v1 = int(seg["view_start"]), int(seg["view_end"])
                        if text[s0:s1] != seg["source_text"] or text[v0:v1] != seg["view_text"]:
                            counts["bad_segment_offsets"] += 1
                            if len(first_bad_segments) < 10:
                                first_bad_segments.append({
                                    "line_i": line_i,
                                    "pair_id": seg.get("pair_id"),
                                    "source_match": text[s0:s1] == seg["source_text"],
                                    "view_match": text[v0:v1] == seg["view_text"],
                                    "source_slice": text[s0:s1],
                                    "source_text": seg["source_text"],
                                    "view_slice": text[v0:v1],
                                    "view_text": seg["view_text"],
                                })
                            break
            rows_written += 1
            total_words += words
            source_rows[str(row.get("source", ""))] += 1
            source_words[str(row.get("source", ""))] += words
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "UNCHANGED_QWEN_SEGMENTS_TAIL_BUILT" if not first_mismatches and not first_bad_segments else "UNCHANGED_QWEN_SEGMENTS_TAIL_HAS_ISSUES",
        "created_utc": now(),
        "reference_tail": rel(args.reference_tail),
        "output_jsonl": rel(out_jsonl),
        "rows_written": rows_written,
        "total_words": total_words,
        "legal_total_words_if_from_coherent86": 86005295 + total_words,
        "qwen_rows": int(counts.get("qwen_rows", 0)),
        "matched_qwen_meta": int(counts.get("matched_qwen_meta", 0)),
        "missing_qwen_meta": int(counts.get("missing_qwen_meta", 0)),
        "reconstruction_mismatch_count": int(counts.get("reconstruction_mismatch", 0)),
        "bad_segment_offset_count": int(counts.get("bad_segment_offsets", 0)),
        "qwen_segment_kind_counts": dict(qwen_segment_kind_counts),
        "source_rows": dict(source_rows),
        "source_words": dict(source_words),
        "first_mismatches": first_mismatches,
        "first_bad_segments": first_bad_segments,
        "sha256": sha_text(out_jsonl.read_text(encoding="utf-8")),
        "interpretation": "This is the unchanged inherited source/current-rewrite legal tail with qwen segment metadata only. It supports a control comparing WWM with source-visible view-focused learning without generated compaction or reinvested top-ups.",
    }
    (out_dir / "tail_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if first_mismatches or first_bad_segments or counts.get("missing_qwen_meta", 0):
        raise SystemExit(2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-tail", type=Path, default=REFERENCE_TAIL)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
