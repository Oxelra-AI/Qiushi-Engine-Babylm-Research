#!/usr/bin/env python3
"""research: Generate entity-state-update prompts from the same Qwen selected originals.

Takes the 37,594 selected originals from COMPACT_EXPERIENCE research Qwen clean-aligned pairs.
For each, creates a prompt asking for an entity-state-update companion instead of
a meaning-preserving rewrite.

Entity-state-update: the companion mentions the same entity but changes its state,
property, location, or condition. This practices the relation that Entity tracking tests.

Output: JSONL prompts ready for LLM generation (Qwen3-8B or similar).
"""
from __future__ import annotations

import json
import hashlib
import pathlib
import re
import sys
import time
from collections import Counter

ROOT = pathlib.Path.cwd()  # user root (run from user root)
PAIRS_PATH = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/state_update_generation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Entity-state-update prompt
PROMPT_TEMPLATE = (
    "You will receive a sentence describing a situation involving one or more entities "
    "(people, objects, animals, etc.). Write a single new sentence that describes a change "
    "to the state, location, or condition of one entity mentioned in the original. "
    "The new sentence must:\n"
    "1. Mention the same entity by name or clear reference\n"
    "2. Describe a concrete change (moved, transformed, given away, broken, etc.)\n"
    "3. Be a complete, self-contained sentence\n"
    "4. Be roughly similar in length to the original\n"
    "5. NOT simply restate the same fact in different words\n\n"
    "Original: {sentence}\n\n"
    "State-update sentence:"
)

# Simpler entity detection heuristic
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
}

def has_trackable_entities(text: str) -> tuple[bool, list[str]]:
    """Check if sentence has entities whose state could change."""
    words = text.split()
    entities = []
    for i, w in enumerate(words):
        stripped = w.strip(".,!?;:()[]{}\"'""''")
        if not stripped:
            continue
        # Named entities: capitalized words not at sentence start, or multi-word names
        if i > 0 and stripped[0].isupper() and len(stripped) > 1:
            if stripped.lower() not in STOPWORDS:
                entities.append(stripped)
        # CHILDES speaker marks
        if stripped.startswith("*") and len(stripped) > 2:
            entities.append(stripped)
    # Also check for concrete nouns using simple heuristics
    concrete_indicators = {
        "ball", "box", "cup", "hat", "book", "table", "chair", "door", "car", "house",
        "dog", "cat", "bird", "fish", "tree", "flower", "water", "food", "room", "bed",
        "bag", "bottle", "key", "stone", "ship", "boat", "bridge", "road", "city", "town",
        "kingdom", "castle", "army", "sword", "horse", "river", "mountain", "island",
        "gold", "silver", "iron", "fire", "light", "child", "baby", "boy", "girl",
        "man", "woman", "king", "queen", "prince", "soldier", "farmer", "teacher",
    }
    lower_words = [w.strip(".,!?;:()[]{}\"'""''").lower() for w in words]
    has_concrete = any(w in concrete_indicators for w in lower_words)
    
    # Sentences with named entities or concrete nouns are state-update amenable
    has_ent = len(entities) >= 1 or has_concrete
    return has_ent, entities


def main():
    t0 = time.time()
    
    # Read selected pairs
    pairs = []
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Read {len(pairs)} selected pairs from {PAIRS_PATH}", flush=True)
    
    # For each pair, take the original and check if entity-state-amenable
    amenable = []
    not_amenable = []
    
    for p in pairs:
        original = p["original"]
        has_ent, ents = has_trackable_entities(original)
        if has_ent:
            amenable.append({
                "pair_id": p["pair_id"],
                "original": original,
                "original_words": len(original.split()),
                "detected_entities": ents,
            })
        else:
            not_amenable.append(p["pair_id"])
    
    print(f"Entity-state amenable: {len(amenable)} / {len(pairs)} "
          f"({100*len(amenable)/len(pairs):.1f}%)", flush=True)
    
    # Create prompts for ALL amenable originals
    prompts_path = OUT_DIR / "state_update_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(amenable):
            prompt = PROMPT_TEMPLATE.format(sentence=item["original"])
            rec = {
                "id": f"su_{item['pair_id']}",
                "pair_id": item["pair_id"],
                "prompt": prompt,
                "source_sentence": item["original"],
                "source_words": item["original_words"],
                "detected_entities": item["detected_entities"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Save non-amenable IDs for potential restatement fallback
    nonam_path = OUT_DIR / "not_amenable_pair_ids.json"
    nonam_path.write_text(json.dumps({
        "count": len(not_amenable),
        "pair_ids": not_amenable,
    }, indent=2, ensure_ascii=False))
    
    # Compute hash
    with prompts_path.open("rb") as f:
        prompts_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Word statistics for amenable originals
    word_counts = [item["original_words"] for item in amenable]
    
    metadata = {
        "status": "STATE_UPDATE_PROMPTS_CREATED",
        "total_pairs": len(pairs),
        "amenable_count": len(amenable),
        "not_amenable_count": len(not_amenable),
        "amenable_fraction": round(len(amenable) / len(pairs), 4),
        "total_amenable_original_words": sum(word_counts),
        "word_stats": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(sum(word_counts) / len(word_counts), 2),
        },
        "prompts_path": str(prompts_path),
        "not_amenable_path": str(nonam_path),
        "prompts_sha256": prompts_hash,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    meta_path = OUT_DIR / "prompt_extraction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
