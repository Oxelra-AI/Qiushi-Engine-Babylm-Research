#!/usr/bin/env python3
"""research: feasibility check for in-corpus adult-prose arm."""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, collections

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/incorpus_feasibility.py')
ROOT = _PUBLIC_ROOT

BASE_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')

src_words = collections.Counter()
src_rows = collections.Counter()
total_words = 0
total_rows = 0
adult_text_prefixes = set()

with open(BASE_POOL) as f:
    for line in f:
        row = json.loads(line)
        w = row["words"]
        s = row["source"]
        src_words[s] += w
        src_rows[s] += 1
        total_words += w
        total_rows += 1
        if s in ("gutenberg", "simple_wiki"):
            adult_text_prefixes.add(row["text"][:200])

print(f"Base pool: {total_rows} rows, {total_words} words")
for s in sorted(src_words.keys()):
    print(f"  {s}: {src_rows[s]} rows, {src_words[s]} words")

# Count raw files
RAW_DIR = _public_path('experiments/archive/compact_experience/training/runs/hf_raw_and_clean/raw_dataset')
for name in ["gutenberg.train.txt", "simple_wiki.train.txt"]:
    p = RAW_DIR / name
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        words = text.split()
        print(f"\nRaw {name}: {len(words)} words")
    else:
        print(f"\nRaw {name}: NOT FOUND at {p}")

dev_speech = {"childes", "open_subtitles", "bnc_spoken", "switchboard"}
dev_words = sum(src_words[s] for s in dev_speech)
dev_rows = sum(src_rows[s] for s in dev_speech)
adult_pool = src_words["gutenberg"] + src_words["simple_wiki"]
print(f"\nDev/speech in pool: {dev_rows} rows, {dev_words} words")
print(f"Adult prose in pool: {src_rows['gutenberg'] + src_rows['simple_wiki']} rows, {adult_pool} words")
print(f"Qwen paired: {src_rows['qwen_pair_packed']} rows, {src_words['qwen_pair_packed']} words")
print(f"Target: ~423,559 words, ~{423559 // 160} rows of 160w")
print(f"Unique adult prefix hashes in pool: {len(adult_text_prefixes)}")
