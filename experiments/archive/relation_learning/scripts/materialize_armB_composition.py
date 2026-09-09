#!/usr/bin/env python3
"""research: Materialize Arm B principle-guided composition with isolated-sentence practice.

Three-component private phase:
  1. Coherent replay: broad benchmark preservation (~40%)
  2. Aligned-restatement pairs: context-supported columns (Entity/Supplement/EWoK) (~21%)
  3. Sentence-shuffled text: isolation-supported columns (BLiMP/Reading) (~39%)

The sentence shuffling breaks cross-sentence dependencies within each row while
preserving sentence-internal structure. This is the text-level equivalent of research's
spanbreak_replay mode, applied to a subset of rows before tokenization.

Scientific hypothesis: the combined evidence shows SHUF improved BLiMP/Reading through
wrong correspondence (which effectively isolated sentences from their source context).
Sentence shuffling provides the same isolation benefit without wrong-correspondence damage.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import random
import re
import sys
import time

ROOT = _public_path('.')

# ── data sources (same as Arm A) ──
BASE_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DOSE21_PAIRS = _public_path('experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose21_packed_pair_rows_probe_clean.jsonl')
DOSE25_EXTRA_PAIRS = _public_path('experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose25_extra_packed_pair_rows_probe_clean.jsonl')

SKIP_ROWS = 530_944
TARGET_TOTAL_WORDS = 3_992_918
PAIR_WORDS_TARGET = 843_200  # dose25 fraction
SHUFFLE_SEED = 87001

# Allocation: ~40% coherent, ~21% pairs, ~39% shuffled
COHERENT_FRACTION = 0.40
PAIR_FRACTION = 0.211  # fixed by dose25 data
SHUFFLE_FRACTION = 1.0 - COHERENT_FRACTION - PAIR_FRACTION  # ~0.389

def sentence_split(text: str) -> list[str]:
    """Split text into sentences, preserving speaker markers and dialogue structure."""
    # Handle CHILDES-style markers
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z*])', text)
    return [s.strip() for s in sents if s.strip()]

def shuffle_sentences(text: str, rng: random.Random) -> str:
    """Shuffle sentence order within a text block."""
    sents = sentence_split(text)
    if len(sents) <= 1:
        return text
    rng.shuffle(sents)
    return ' '.join(sents)

def load_pair_rows(dose21_path, dose25_extra_path):
    rows = []
    for src_path, src_label in [(dose21_path, "dose25_pair_core"), (dose25_extra_path, "dose25_pair_extra")]:
        with src_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({
                    "text": obj["text_pairs_only"],
                    "words": int(obj["pair_words"]),
                    "source": src_label,
                    "example_id": -1,
                })
    return rows

def load_ordinary_rows(stream_path, skip_rows, max_words):
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

def interleave_three(coherent_rows, pair_rows, shuffled_rows):
    """Uniformly interleave three types of rows."""
    # Simple approach: merge all into one list with type tags, then interleave
    tagged = []
    for r in coherent_rows:
        tagged.append((0, r))  # 0 = coherent
    for r in pair_rows:
        tagged.append((1, r))  # 1 = pair
    for r in shuffled_rows:
        tagged.append((2, r))  # 2 = shuffled
    
    # Sort by a hash to interleave uniformly
    rng = random.Random(SHUFFLE_SEED + 42)
    rng.shuffle(tagged)
    
    return [r for _, r in tagged]

def main():
    t0 = time.time()
    out_dir = _public_path('experiments/archive/relation_learning/data/principle_composition_armB')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SHUFFLE_SEED)
    
    # 1. Load pair rows (fixed 21.1%)
    print("Loading pair rows...", flush=True)
    pair_rows = load_pair_rows(DOSE21_PAIRS, DOSE25_EXTRA_PAIRS)
    pair_words = sum(r["words"] for r in pair_rows)
    
    # 2. Compute budget for coherent + shuffled
    remaining_words = TARGET_TOTAL_WORDS - pair_words
    coherent_budget = int(remaining_words * COHERENT_FRACTION / (COHERENT_FRACTION + SHUFFLE_FRACTION))
    shuffle_budget = remaining_words - coherent_budget
    
    print(f"  Pair: {pair_words} words ({100*pair_words/TARGET_TOTAL_WORDS:.1f}%)")
    print(f"  Coherent budget: {coherent_budget} words ({100*coherent_budget/TARGET_TOTAL_WORDS:.1f}%)")
    print(f"  Shuffle budget: {shuffle_budget} words ({100*shuffle_budget/TARGET_TOTAL_WORDS:.1f}%)")
    
    # 3. Load ordinary rows for both coherent and shuffled
    total_ordinary_needed = coherent_budget + shuffle_budget
    print("Loading ordinary rows...", flush=True)
    ordinary_rows, ordinary_words = load_ordinary_rows(BASE_STREAM, SKIP_ROWS, total_ordinary_needed)
    
    # 4. Split ordinary rows into coherent and shuffled portions
    split_point = 0
    running_words = 0
    for i, r in enumerate(ordinary_rows):
        running_words += r["words"]
        if running_words >= coherent_budget:
            split_point = i + 1
            break
    
    coherent_rows = ordinary_rows[:split_point]
    coherent_words = sum(r["words"] for r in coherent_rows)
    
    # 5. Create shuffled rows from remaining ordinary rows
    shuffled_rows = []
    shuffle_words = 0
    for r in ordinary_rows[split_point:]:
        shuffled_text = shuffle_sentences(r["text"], rng)
        shuffled_words += r["words"]
        shuffled_rows.append({
            "text": shuffled_text,
            "words": r["words"],  # word count preserved
            "source": r["source"] + "_shuffled",
            "example_id": r.get("example_id", -1),
        })
    
    print(f"  Coherent: {len(coherent_rows)} rows, {coherent_words} words")
    print(f"  Shuffled: {len(shuffled_rows)} rows, {shuffle_words} words")
    
    # 6. Interleave all three types
    mixed = interleave_three(coherent_rows, pair_rows, shuffled_rows)
    total_words = sum(r["words"] for r in mixed)
    
    # 7. Write mixed JSONL
    out_jsonl = out_dir / "principle_composition_armB_mixed.jsonl"
    sha = hashlib.sha256()
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in mixed:
            obj = {"text": row["text"], "words": row["words"], "source": row["source"], "example_id": row.get("example_id", -1)}
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            f.write(line)
            sha.update(line.encode("utf-8"))
    
    metadata = {
        "status": "PRINCIPLE_COMPOSITION_ARMB_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_rationale": "Three-component composition: coherent replay (broad), "
                                "aligned pairs (context-supported), sentence-shuffled (isolation-supported). "
                                "Tests whether isolation practice adds BLiMP/Reading benefit.",
        "pair_rows": len(pair_rows), "pair_words": pair_words,
        "coherent_rows": len(coherent_rows), "coherent_words": coherent_words,
        "shuffled_rows": len(shuffled_rows), "shuffled_words": shuffle_words,
        "total_rows": len(mixed), "total_words": total_words,
        "coherent_fraction": round(coherent_words / total_words, 4),
        "pair_fraction": round(pair_words / total_words, 4),
        "shuffle_fraction": round(shuffle_words / total_words, 4),
        "output_sha256": sha.hexdigest(),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    
    print(json.dumps({k: metadata[k] for k in ["status", "pair_words", "coherent_words", "shuffled_words",
                                                  "total_words", "pair_fraction", "coherent_fraction",
                                                  "shuffle_fraction", "elapsed_sec"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
