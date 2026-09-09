#!/usr/bin/env python3
"""research: count raw Gutenberg/SimpleWiki surplus vs base pool.

Determines how much unused adult-prose text exists for the in-corpus arm.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, collections, hashlib

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/incorpus_surplus_count.py')
ROOT = _PUBLIC_ROOT

RAW_GUT = _public_path('experiments/archive/compact_experience/training/runs/smoke_curriculum_32k/raw_dataset/gutenberg.train.txt')
RAW_WIKI = _public_path('experiments/archive/compact_experience/training/runs/smoke_curriculum_32k/raw_dataset/simple_wiki.train.txt')
BASE_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
OFFICIAL_ONLY = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')

# 1. Count raw words
for name, path in [("gutenberg", RAW_GUT), ("simple_wiki", RAW_WIKI)]:
    text = path.read_text(encoding="utf-8", errors="replace")
    words = text.split()
    print(f"Raw {name}: {len(words)} words, {len(text)} chars")

# 2. Count source distribution in official_only_10M (no Qwen pairs)
off_src_words = collections.Counter()
off_src_rows = collections.Counter()
off_total = 0
with open(OFFICIAL_ONLY) as f:
    for line in f:
        row = json.loads(line)
        w = row["words"]
        s = row["source"]
        off_src_words[s] += w
        off_src_rows[s] += 1
        off_total += w

print(f"\nOfficial-only pool: {sum(off_src_rows.values())} rows, {off_total} words")
for s in sorted(off_src_words.keys()):
    print(f"  {s}: {off_src_rows[s]} rows, {off_src_words[s]} words")

# 3. Count base pool (with Qwen pairs)
base_src_words = collections.Counter()
base_src_rows = collections.Counter()
base_total = 0
base_adult_texts = set()
with open(BASE_POOL) as f:
    for line in f:
        row = json.loads(line)
        w = row["words"]
        s = row["source"]
        base_src_words[s] += w
        base_src_rows[s] += 1
        base_total += w
        if s in ("gutenberg", "simple_wiki"):
            base_adult_texts.add(row["text"][:300])

print(f"\nBase pool (with Qwen): {sum(base_src_rows.values())} rows, {base_total} words")

# 4. The difference is unused official text
off_gut = off_src_words.get("gutenberg", 0)
off_wiki = off_src_words.get("simple_wiki", 0)
base_gut = base_src_words.get("gutenberg", 0)
base_wiki = base_src_words.get("simple_wiki", 0)

displaced_gut = off_gut - base_gut
displaced_wiki = off_wiki - base_wiki
print(f"\nGutenberg: official {off_gut} - base {base_gut} = displaced {displaced_gut}")
print(f"SimpleWiki: official {off_wiki} - base {base_wiki} = displaced {displaced_wiki}")
print(f"Total adult-prose surplus: {displaced_gut + displaced_wiki} words")
print(f"Target need: ~423,559 words")
print(f"Feasible: {(displaced_gut + displaced_wiki) >= 423559}")

# 5. Build index of which official-only rows are NOT in the base pool
# by text prefix matching
off_adult_texts = {}
with open(OFFICIAL_ONLY) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        if row["source"] in ("gutenberg", "simple_wiki"):
            prefix = row["text"][:300]
            off_adult_texts[prefix] = {
                "row_index": i,
                "source": row["source"],
                "words": row["words"],
                "in_base": prefix in base_adult_texts,
            }

n_total = len(off_adult_texts)
n_in_base = sum(1 for v in off_adult_texts.values() if v["in_base"])
n_surplus = sum(1 for v in off_adult_texts.values() if not v["in_base"])
surplus_words = sum(v["words"] for v in off_adult_texts.values() if not v["in_base"])
surplus_by_src = collections.Counter()
for v in off_adult_texts.values():
    if not v["in_base"]:
        surplus_by_src[v["source"]] += v["words"]

print(f"\nAdult-prose row matching:")
print(f"  Total official adult rows: {n_total}")
print(f"  In base pool: {n_in_base}")
print(f"  Surplus (not in base): {n_surplus} rows, {surplus_words} words")
for s in sorted(surplus_by_src.keys()):
    print(f"    {s}: {surplus_by_src[s]} words")
print(f"  Surplus >= target 423,559: {surplus_words >= 423559}")
