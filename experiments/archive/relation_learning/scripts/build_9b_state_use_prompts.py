#!/usr/bin/env python3
"""Build state-use prompts for batch generation with Qwen3.5-9B.

Reads the 37,594 selected Qwen originals, extracts entity/state candidates,
builds structured three-part prompts (source/update/use), and writes a JSONL
suitable for batch generation with Qwen3.5-9B.

Prompt design and validation:
  - Natural-domain exemplars (no box/basket/marble/container Entity-format)
  - Uses varied examples across domains (person/location, object/property,
    organization/status, character/role)
  - Longer, more specific instructions to avoid copying exemplar format
  - Uses the batch request format: {"id": ..., "prompt": ...}
  - Algorithmic rejection in the materializer, not the generator
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()
PAIRS_PATH = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 43043
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "may", "might", "must", "not", "of", "on",
    "or", "our", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "under", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    "before", "after", "while", "over", "under", "about", "than", "from", "into", "onto", "also",
}

# ──────────────────────────────────────────────────────────────────────
#  Natural-domain examples -- explicitly NOT Entity-benchmark format
# ──────────────────────────────────────────────────────────────────────
UPDATED_USE_EXAMPLES = [
    '{"target_entity":"Sarah","source_state":"lives in Portland","updated_entity":"Sarah","new_state":"lives in Denver","update_sentence":"After accepting a new position, Sarah moved to Denver.","use_sentence":"Sarah now commutes through the Denver suburbs each morning."}',
    '{"target_entity":"the old workshop","source_state":"painted dark green","updated_entity":"the old workshop","new_state":"painted bright yellow","update_sentence":"Over the weekend, volunteers repainted the old workshop bright yellow.","use_sentence":"The old workshop stands out with its bright yellow walls."}',
    '{"target_entity":"Dr. Patel","source_state":"head of the biology department","updated_entity":"Dr. Patel","new_state":"dean of the science faculty","update_sentence":"The university promoted Dr. Patel to dean of the science faculty.","use_sentence":"As dean, Dr. Patel now oversees all science programs."}',
]

UNCHANGED_DISTRACTOR_EXAMPLES = [
    '{"target_entity":"the cathedral","source_state":"on the hill overlooking the river","updated_entity":"Marcus","new_state":"works at the new hospital","update_sentence":"Marcus left the factory and started working at the new hospital downtown.","use_sentence":"The cathedral still stands on the hill overlooking the river."}',
    '{"target_entity":"Elena","source_state":"teaches piano at the conservatory","updated_entity":"the corner shop","new_state":"sells organic produce","update_sentence":"The corner shop switched from selling hardware to organic produce.","use_sentence":"Elena continues to teach piano at the conservatory."}',
    '{"target_entity":"the lighthouse","source_state":"guides ships along the northern coast","updated_entity":"Captain Torres","new_state":"retired to a small village","update_sentence":"Captain Torres retired from the navy and settled in a small village.","use_sentence":"The lighthouse still guides ships along the northern coast."}',
]

PROMPT_TEMPLATE_UPDATED = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create a state-update training packet from this source sentence.

Source sentence:
{sentence}

Entities found in the source: {entities}

Instructions:
1. Pick one entity from the source that has a clear state, location, role, property, or condition.
2. Invent a plausible update event that changes that entity's state to something new and different.
3. Write a use_sentence that naturally refers to the entity's NEW state after the update.
4. The update and use sentences must feel like natural text from books, articles, or conversation.
5. Do NOT use words like "box", "basket", "container", or "marble". Do NOT write "Now X is in the Y."
6. Keep update_sentence between 8-22 words and use_sentence between 6-16 words.

Output JSON keys: target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence
(target_entity and updated_entity should be the same entity)

Examples:
{example1}
{example2}

JSON:"""

PROMPT_TEMPLATE_DISTRACTOR = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create a distractor-update training packet from this source sentence.

Source sentence:
{sentence}

Entities found in the source: {entities}

Instructions:
1. Pick one entity (the TARGET) whose state you will PRESERVE unchanged.
2. Pick or invent a DIFFERENT entity (the DISTRACTOR) and change its state.
3. The use_sentence must refer to the TARGET entity's ORIGINAL state from the source, unchanged.
4. The update_sentence must change the DISTRACTOR entity, not the target.
5. target_entity and updated_entity MUST be different. This is critical.
6. The sentences must feel like natural text from books, articles, or conversation.
7. Do NOT use words like "box", "basket", "container", or "marble". Do NOT write "Now X is in the Y."
8. Keep update_sentence between 8-22 words and use_sentence between 6-16 words.

Output JSON keys: target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence
(target_entity and updated_entity must be DIFFERENT entities)

Examples:
{example1}
{example2}

JSON:"""


def norm_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def wc(text: str) -> int:
    return len(words(text))


def canonical_ent(ent: str) -> str:
    return norm_ws(ent.strip(".,!?;:()[]{}\"'\u201c\u201d\u2018\u2019"))


def entity_candidates(pair: dict[str, Any]) -> list[str]:
    text = pair["original"]
    cands: list[str] = []
    # Multi-word proper-name chunks and capitalized tokens.
    toks = text.split()
    chunk: list[str] = []
    for i, tok in enumerate(toks + ["."]):
        stripped = canonical_ent(tok)
        letters = re.sub(r"[^A-Za-z0-9]", "", stripped)
        is_name = bool(letters) and (letters[0].isupper() or (len(letters) > 1 and any(ch.isupper() for ch in letters[1:])))
        if i == 0 and is_name and letters.istitle():
            is_name = False  # sentence-initial single cap is weak
        if is_name and stripped.lower() not in STOPWORDS and len(stripped) >= 2:
            chunk.append(stripped)
        else:
            if chunk:
                cands.append(" ".join(chunk))
                chunk = []
    # Also look for common nouns that carry state (not benchmark objects)
    STATEFUL_NOUNS = {
        "city", "town", "house", "school", "church", "bridge", "ship", "boat",
        "army", "company", "team", "village", "country", "castle", "river",
        "mountain", "island", "park", "road", "street", "garden", "museum",
        "hospital", "hotel", "station", "airport", "factory", "theater",
        "library", "temple", "palace", "farm", "college", "university",
    }
    for w in words(text):
        wl = w.lower()
        if wl in STATEFUL_NOUNS:
            cands.append(wl)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for c in cands:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def build_prompt(pair: dict, packet_type: str, rng: random.Random) -> dict | None:
    """Build a single generation prompt."""
    ents = entity_candidates(pair)
    if not ents:
        return None

    source = norm_ws(pair["original"])
    ent_str = json.dumps(ents[:7])  # cap at 7 entities

    if packet_type == "UPDATED_USE":
        exs = rng.sample(UPDATED_USE_EXAMPLES, min(2, len(UPDATED_USE_EXAMPLES)))
        prompt = PROMPT_TEMPLATE_UPDATED.format(
            sentence=source, entities=ent_str,
            example1=exs[0], example2=exs[1] if len(exs) > 1 else exs[0],
        )
    else:
        exs = rng.sample(UNCHANGED_DISTRACTOR_EXAMPLES, min(2, len(UNCHANGED_DISTRACTOR_EXAMPLES)))
        prompt = PROMPT_TEMPLATE_DISTRACTOR.format(
            sentence=source, entities=ent_str,
            example1=exs[0], example2=exs[1] if len(exs) > 1 else exs[0],
        )

    return {
        "id": f"su3_{pair['pair_id']}",
        "prompt": prompt,
        "pair_id": pair["pair_id"],
        "packet_type": packet_type,
        "candidate_entities": ents[:7],
        "source_sentence": source,
        "source_words": wc(source),
        "qwen_rewrite": norm_ws(pair["rewrite"]),
        "rewrite_words": wc(pair["rewrite"]),
        "source": pair.get("source_name") or pair.get("source", ""),
        "example_id": pair.get("source_example_id") or pair.get("example_id", -1),
        "cohort": pair.get("cohort", ""),
    }


def main():
    rng = random.Random(RNG_SEED)

    # Load selected pairs
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} selected pairs", flush=True)

    # Filter: need at least one entity candidate
    amenable = [(p, entity_candidates(p)) for p in pairs]
    amenable = [(p, e) for p, e in amenable if e]
    print(f"Entity-amenable: {len(amenable)}/{len(pairs)}", flush=True)

    # Assign packet types: alternate UPDATED_USE and UNCHANGED_DISTRACTOR_USE
    rng.shuffle(amenable)
    half = len(amenable) // 2
    updated_pairs = amenable[:half]
    distractor_pairs = amenable[half:]

    prompts = []
    skipped = 0
    for p, _ents in updated_pairs:
        result = build_prompt(p, "UPDATED_USE", rng)
        if result:
            prompts.append(result)
        else:
            skipped += 1

    for p, _ents in distractor_pairs:
        result = build_prompt(p, "UNCHANGED_DISTRACTOR_USE", rng)
        if result:
            prompts.append(result)
        else:
            skipped += 1

    # Shuffle to avoid systematic ordering effects in batch generation
    rng.shuffle(prompts)

    # Write the prompt JSONL for batch generation
    # This file has all metadata; the batch generator only reads "id" and "prompt"
    prompt_path = OUT_DIR / "state_use_prompts_9b.jsonl"
    with open(prompt_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_updated = sum(1 for p in prompts if p["packet_type"] == "UPDATED_USE")
    n_distractor = sum(1 for p in prompts if p["packet_type"] == "UNCHANGED_DISTRACTOR_USE")

    metadata = {
        "status": "STATE_USE_PROMPTS_9B_CREATED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_selected_qwen_pairs": len(pairs),
        "prompt_count": len(prompts),
        "updated_use_count": n_updated,
        "distractor_use_count": n_distractor,
        "skipped_count": skipped,
        "prompt_path": str(prompt_path),
        "model_target": "qwen3.5-9b",
        "sha256_prompts": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "note": "Natural-domain exemplars; no Entity-benchmark format; structured JSON output",
    }
    (OUT_DIR / "prompt_extraction_metadata_9b.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
