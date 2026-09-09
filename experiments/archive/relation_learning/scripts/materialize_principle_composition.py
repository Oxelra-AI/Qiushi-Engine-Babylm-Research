#!/usr/bin/env python3
"""research: Materialize principle-guided private-phase composition.

Scientific rationale (from combined relation→column evidence):
  - Correct correspondence (ALN) adds context-supported prediction: Entity, Supplement, EWoK
  - Coherent replay preserves the broad benchmark profile
  - By COMPOSING the private phase from both relation types using standard CE+KL
    (no special answer targeting), the data composition IS the principle.

This script creates a mixed JSONL that interleaves:
  1. Ordinary coherent replay rows (from the same stream as coherent86)
  2. Probe-clean aligned-restatement pair rows (from dose25 repaired pairs)

The output JSONL has uniform format: {"text": ..., "words": ..., "source": ..., "example_id": ...}
and can be fed directly to the research replay trainer with --skip_rows 0.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import sys
import time

ROOT = _public_path('.')

# ── data sources ──
BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DOSE21_PAIRS = _public_path('experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose21_packed_pair_rows_probe_clean.jsonl')
DOSE25_EXTRA_PAIRS = _public_path('experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose25_extra_packed_pair_rows_probe_clean.jsonl')

# ── accounting ──
SKIP_ROWS = 530_944          # rows consumed by chck_82M training
INITIAL_CONSUMED = 82_012_495
TARGET_TOTAL_WORDS = 3_992_918  # coherent86's max_tail_charged_words

def load_pair_rows(dose21_path: pathlib.Path, dose25_extra_path: pathlib.Path):
    """Load all dose25 superset pair rows (dose21 + dose25_extra), convert to standard format."""
    rows = []
    for src_path, src_label in [(dose21_path, "dose25_pair_core"), (dose25_extra_path, "dose25_pair_extra")]:
        with src_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text_pairs_only"]
                words = int(obj["pair_words"])
                # verify word count
                actual_words = len(text.split())
                if abs(actual_words - words) > 1:
                    # use actual count
                    words = actual_words
                rows.append({
                    "text": text,
                    "words": words,
                    "source": src_label,
                    "example_id": -1,  # synthetic
                    "pair_ids": obj.get("pair_ids", []),
                })
    return rows

def load_ordinary_rows(stream_path: pathlib.Path, skip_rows: int, max_words: int):
    """Load ordinary text rows from the coherent replay stream."""
    rows = []
    total_words = 0
    with stream_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if total_words + words > max_words:
                break
            rows.append({
                "text": text,
                "words": words,
                "source": str(obj.get("source", "")),
                "example_id": int(obj.get("example_id", -1)),
                "row_index_in_stream": idx,
            })
            total_words += words
    return rows, total_words

def interleave(ordinary_rows: list, pair_rows: list) -> list:
    """Uniformly interleave pair rows into ordinary rows."""
    result = []
    n_ord = len(ordinary_rows)
    n_pair = len(pair_rows)
    if n_pair == 0:
        return ordinary_rows
    # insertion interval: insert 1 pair row every interval ordinary rows
    interval = n_ord / n_pair
    pair_idx = 0
    next_insert_at = interval / 2  # start half-interval in
    
    for i, row in enumerate(ordinary_rows):
        # check if we should insert a pair row before this ordinary row
        while pair_idx < n_pair and i >= next_insert_at:
            result.append(pair_rows[pair_idx])
            pair_idx += 1
            next_insert_at += interval
        result.append(row)
    
    # append any remaining pair rows
    while pair_idx < n_pair:
        result.append(pair_rows[pair_idx])
        pair_idx += 1
    
    return result

def main():
    t0 = time.time()
    out_dir = _public_path('experiments/archive/relation_learning/data/principle_composition')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load pair rows
    print("Loading pair rows...", flush=True)
    pair_rows = load_pair_rows(DOSE21_PAIRS, DOSE25_EXTRA_PAIRS)
    pair_words = sum(r["words"] for r in pair_rows)
    print(f"  Loaded {len(pair_rows)} pair rows, {pair_words} pair words", flush=True)
    
    # 2. Determine ordinary word budget
    ordinary_budget = TARGET_TOTAL_WORDS - pair_words
    if ordinary_budget < 0:
        raise ValueError(f"pair_words ({pair_words}) exceeds TARGET_TOTAL_WORDS ({TARGET_TOTAL_WORDS})")
    print(f"  Ordinary word budget: {ordinary_budget}", flush=True)
    
    # 3. Load ordinary rows
    print("Loading ordinary rows...", flush=True)
    ordinary_rows, ordinary_words = load_ordinary_rows(BASE_STREAM, SKIP_ROWS, ordinary_budget)
    print(f"  Loaded {len(ordinary_rows)} ordinary rows, {ordinary_words} ordinary words", flush=True)
    
    # 4. Interleave
    print("Interleaving...", flush=True)
    mixed = interleave(ordinary_rows, pair_rows)
    total_words = sum(r["words"] for r in mixed)
    n_pair_in_mixed = sum(1 for r in mixed if r["source"].startswith("dose25_pair"))
    n_ordinary_in_mixed = len(mixed) - n_pair_in_mixed
    
    # 5. Write mixed JSONL (only text, words, source, example_id for research compatibility)
    out_jsonl = out_dir / "principle_composition_mixed.jsonl"
    sha = hashlib.sha256()
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in mixed:
            obj = {
                "text": row["text"],
                "words": row["words"],
                "source": row["source"],
                "example_id": row.get("example_id", -1),
            }
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            f.write(line)
            sha.update(line.encode("utf-8"))
    
    stream_sha = sha.hexdigest()
    
    # 6. Source composition analysis
    source_words = {}
    for r in mixed:
        s = r["source"]
        source_words[s] = source_words.get(s, 0) + r["words"]
    
    # 7. Write metadata
    metadata = {
        "status": "PRINCIPLE_COMPOSITION_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_rationale": "Relation→column map: coherent replay for broad preservation, "
                                "correct-correspondence pairs for context-supported columns. "
                                "Standard CE+KL objective; data composition IS the principle.",
        "pair_source": {
            "dose21_clean": str(DOSE21_PAIRS),
            "dose25_extra_clean": str(DOSE25_EXTRA_PAIRS),
        },
        "ordinary_source": str(BASE_STREAM),
        "ordinary_skip_rows": SKIP_ROWS,
        "pair_rows": len(pair_rows),
        "pair_words": pair_words,
        "pair_word_fraction": round(pair_words / total_words, 4),
        "ordinary_rows": n_ordinary_in_mixed,
        "ordinary_words": ordinary_words,
        "total_rows": len(mixed),
        "total_words": total_words,
        "target_words": TARGET_TOTAL_WORDS,
        "initial_consumed_words": INITIAL_CONSUMED,
        "total_consumed_after_composition": INITIAL_CONSUMED + total_words,
        "output_jsonl": str(out_jsonl),
        "output_sha256": stream_sha,
        "source_word_breakdown": source_words,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "status": metadata["status"],
        "pair_rows": metadata["pair_rows"],
        "pair_words": metadata["pair_words"],
        "pair_fraction": metadata["pair_word_fraction"],
        "ordinary_rows": metadata["ordinary_rows"],
        "ordinary_words": metadata["ordinary_words"],
        "total_rows": metadata["total_rows"],
        "total_words": metadata["total_words"],
        "sha256": stream_sha,
        "elapsed_sec": metadata["elapsed_sec"],
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
