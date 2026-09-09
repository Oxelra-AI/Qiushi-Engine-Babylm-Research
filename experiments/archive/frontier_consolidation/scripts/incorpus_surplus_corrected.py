#!/usr/bin/env python3
"""research: corrected surplus count with source-name normalization."""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, collections

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/incorpus_surplus_corrected.py')
ROOT = _PUBLIC_ROOT

BASE_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
OFFICIAL_ONLY = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')

def normalize_src(s):
    """Strip official_lengthmatched:: prefix."""
    if "::" in s:
        return s.split("::", 1)[1]
    return s

# 1. Read base pool text prefixes for matching
base_texts = set()
base_src_words = collections.Counter()
with open(BASE_POOL) as f:
    for line in f:
        row = json.loads(line)
        src = normalize_src(row["source"])
        base_src_words[src] += row["words"]
        base_texts.add(row["text"][:300])

# 2. Read official-only and find surplus adult-prose rows
off_src_words = collections.Counter()
surplus_rows = []
in_base_count = 0
with open(OFFICIAL_ONLY) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        src = normalize_src(row["source"])
        off_src_words[src] += row["words"]
        if src in ("gutenberg", "simple_wiki"):
            prefix = row["text"][:300]
            if prefix in base_texts:
                in_base_count += 1
            else:
                surplus_rows.append({
                    "off_index": i,
                    "source": src,
                    "words": row["words"],
                    "text_prefix": row["text"][:80],
                })

surplus_words = sum(r["words"] for r in surplus_rows)
surplus_by_src = collections.Counter()
for r in surplus_rows:
    surplus_by_src[r["source"]] += r["words"]

print("Source word accounting:")
for s in sorted(set(list(off_src_words.keys()) + list(base_src_words.keys()))):
    off = off_src_words.get(s, 0)
    base = base_src_words.get(s, 0)
    print(f"  {s}: official {off} | base {base} | displaced {off - base}")

print(f"\nAdult-prose matching (Gutenberg + SimpleWiki):")
print(f"  In both pools: {in_base_count} rows")
print(f"  Surplus (in official, not in base): {len(surplus_rows)} rows, {surplus_words} words")
for s in sorted(surplus_by_src):
    n = sum(1 for r in surplus_rows if r["source"] == s)
    print(f"    {s}: {n} rows, {surplus_by_src[s]} words")
print(f"  Target: ~423,559 words (~{423559//160} rows of 160w)")
print(f"  Feasible: {surplus_words >= 423559}")
print(f"  Surplus / target ratio: {surplus_words / 423559:.2f}x")

# Show sample surplus rows
print("\nSample surplus rows:")
for r in surplus_rows[:5]:
    print(f"  [{r['source']}] {r['words']}w: {r['text_prefix']}...")
