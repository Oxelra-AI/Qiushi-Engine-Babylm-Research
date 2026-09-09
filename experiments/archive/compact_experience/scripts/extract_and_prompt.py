#!/usr/bin/env python3
"""research: Extract sentences from official BabyLM pool and create Qwen rewrite prompts.

Strategy:
- Extract individual sentences (15-50 words) from all 62,500 official rows
- Balance across sources proportionally
- Create JSONL prompts for Qwen3.5-9B meaning-preserving rewrites
- Target: ~45,000 sentences (generous buffer over the ~36,000 needed)

The rewrite prompt is designed to produce same-meaning, different-structure output
suitable for creating within-window semantic adjacency in training data.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import hashlib
import pathlib
import random
import re
import sys
import time
from dataclasses import dataclass, field

ROOT = _public_path('experiments/archive/compact_experience')  # 
POOL_PATH = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/qwen_aligned')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target parameters
MIN_SENT_WORDS = 12
MAX_SENT_WORDS = 50
TARGET_SENTENCES = 45000  # generous buffer; we need ~36,000-40,000

# Prompt template
PROMPT_TEMPLATE = (
    "Rewrite the following sentence to convey the same meaning using different "
    "words and sentence structure. Output only the rewritten sentence, nothing else.\n\n"
    "Sentence: {sentence}"
)


@dataclass
class ExtractedSentence:
    text: str
    words: int
    source: str
    example_id: int
    sentence_idx: int


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitting by terminal punctuation."""
    # Split on . ? ! followed by space or end
    parts = re.split(r'(?<=[.?!])\s+', text)
    result = []
    for p in parts:
        p = p.strip()
        if p:
            result.append(p)
    return result


def is_good_sentence(sent: str, n_words: int) -> bool:
    """Filter for sentences suitable for rewriting."""
    if n_words < MIN_SENT_WORDS or n_words > MAX_SENT_WORDS:
        return False
    # Skip sentences that are mostly fragments or fillers
    if sent.count(',') > n_words / 3:  # too many commas relative to words
        return False
    # Must contain at least one verb-like word (heuristic: has lowercase words)
    lower_words = sum(1 for w in sent.split() if w[0].islower() and len(w) > 2)
    if lower_words < 3:
        return False
    # Skip if too many ellipsis or special chars
    if sent.count('...') > 1 or sent.count('erm') > 2 or sent.count('er,') > 2:
        return False
    return True


def main():
    t0 = time.time()
    print(f"Loading official pool from {POOL_PATH}")
    
    all_sentences: list[ExtractedSentence] = []
    source_counts: dict[str, int] = {}
    
    with POOL_PATH.open() as f:
        for line_idx, line in enumerate(f):
            row = json.loads(line)
            source = row["source"]
            example_id = row["example_id"]
            text = row["text"]
            
            source_counts[source] = source_counts.get(source, 0) + 1
            
            sentences = split_into_sentences(text)
            for sent_idx, sent in enumerate(sentences):
                n_words = len(sent.split())
                if is_good_sentence(sent, n_words):
                    all_sentences.append(ExtractedSentence(
                        text=sent, words=n_words, source=source,
                        example_id=example_id, sentence_idx=sent_idx
                    ))
    
    print(f"Extracted {len(all_sentences)} candidate sentences from {line_idx+1} rows")
    print(f"Source distribution: {source_counts}")
    
    # Count per source
    source_sent_counts = {}
    for s in all_sentences:
        source_sent_counts[s.source] = source_sent_counts.get(s.source, 0) + 1
    print(f"Sentences per source: {source_sent_counts}")
    
    # Sample proportionally to source size, with randomization
    random.seed(27042)  # reproducible
    random.shuffle(all_sentences)
    
    # Take up to TARGET_SENTENCES, balanced by source proportion
    total_source_rows = sum(source_counts.values())
    target_per_source = {
        src: int(TARGET_SENTENCES * count / total_source_rows)
        for src, count in source_counts.items()
    }
    # Distribute remainder
    remainder = TARGET_SENTENCES - sum(target_per_source.values())
    for src in sorted(target_per_source, key=lambda s: source_counts[s], reverse=True):
        if remainder <= 0:
            break
        target_per_source[src] += 1
        remainder -= 1
    
    selected: list[ExtractedSentence] = []
    taken_per_source: dict[str, int] = {s: 0 for s in source_counts}
    
    for sent in all_sentences:
        if taken_per_source[sent.source] < target_per_source[sent.source]:
            selected.append(sent)
            taken_per_source[sent.source] += 1
        if len(selected) >= TARGET_SENTENCES:
            break
    
    print(f"Selected {len(selected)} sentences")
    print(f"Selected per source: {taken_per_source}")
    print(f"Word stats: min={min(s.words for s in selected)}, "
          f"max={max(s.words for s in selected)}, "
          f"mean={sum(s.words for s in selected)/len(selected):.1f}")
    
    # Create prompts JSONL
    prompts_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/rewrite_prompts.jsonl')
    with prompts_path.open("w") as f:
        for i, sent in enumerate(selected):
            prompt = PROMPT_TEMPLATE.format(sentence=sent.text)
            record = {
                "id": f"rw_{i:06d}",
                "prompt": prompt,
                "source_sentence": sent.text,
                "source_words": sent.words,
                "source_name": sent.source,
                "source_example_id": sent.example_id,
                "source_sentence_idx": sent.sentence_idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Save source sentences separately for materialization
    sources_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl')
    with sources_path.open("w") as f:
        for i, sent in enumerate(selected):
            f.write(json.dumps({
                "id": f"rw_{i:06d}",
                "text": sent.text,
                "words": sent.words,
                "source": sent.source,
                "example_id": sent.example_id,
            }, ensure_ascii=False) + "\n")
    
    # Compute hash for reproducibility
    with prompts_path.open("rb") as f:
        prompts_hash = hashlib.sha256(f.read()).hexdigest()
    
    metadata = {
        "status": "PROMPTS_CREATED",
        "total_candidates": len(all_sentences),
        "selected": len(selected),
        "target": TARGET_SENTENCES,
        "min_words": MIN_SENT_WORDS,
        "max_words": MAX_SENT_WORDS,
        "selected_per_source": taken_per_source,
        "word_stats": {
            "min": min(s.words for s in selected),
            "max": max(s.words for s in selected),
            "mean": round(sum(s.words for s in selected) / len(selected), 2),
            "total": sum(s.words for s in selected),
        },
        "prompts_path": str(prompts_path),
        "sources_path": str(sources_path),
        "prompts_sha256": prompts_hash,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    meta_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/extraction_metadata.json')
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
