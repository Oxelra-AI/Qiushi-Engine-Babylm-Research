#!/usr/bin/env python3
"""research: build structured state-update/use prompts from Qwen selected originals.

This repairs the research pair-only intervention.  Each accepted packet should have
three same-window parts:
  source: an original selected sentence;
  update: a sentence that changes one entity's state/location/property;
  use: a sentence that asks the model to represent the current state.

Packet types:
  UPDATED_USE: update the queried/used entity, then use the new state.
  UNCHANGED_DISTRACTOR_USE: update a different entity, then use an untouched
    entity's original state.  This is the training analogue of zero-relevant-update
    Entity items where other entities change while the queried entity must be kept.

The script only creates prompts and metadata.  It does not train or materialize.
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

RNG_SEED = 42042
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

# Extra common nouns that make good state-tracking entities even when no proper name exists.
CONCRETE_NOUNS = {
    "ball", "box", "cup", "hat", "book", "table", "chair", "door", "car", "house", "dog", "cat",
    "bird", "fish", "tree", "flower", "water", "food", "room", "bed", "bag", "bottle", "key",
    "stone", "ship", "boat", "bridge", "road", "city", "town", "castle", "army", "sword", "horse",
    "river", "mountain", "island", "fire", "light", "child", "baby", "boy", "girl", "man", "woman",
    "king", "queen", "prince", "soldier", "farmer", "teacher", "student", "school", "letter", "journal",
    "journal", "page", "paper", "song", "album", "company", "team", "player", "coach", "governor", "state",
    "territory", "foundation", "telescope", "software", "program", "vehicle", "canal", "flower", "park",
    "vitamin", "region", "plant", "village", "country", "church", "college", "troop", "carrier", "gear",
}

PROMPT_TEMPLATE = """You are creating one compact training packet for an entity-state tracking language model.
Return ONE LINE of strict JSON only, with keys exactly:
  target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence
No explanation and no markdown.

Packet type: {packet_type}

Original source sentence:
{sentence}

Candidate entities from the source: {entities}

Requirements:
- The update_sentence must be a complete sentence, 7-20 words, and must name updated_entity.
- The use_sentence must be a complete sentence, 5-14 words, and must name target_entity.
- If Packet type is UPDATED_USE: updated_entity and target_entity should be the same entity; use_sentence states the NEW current state after the update.
- If Packet type is UNCHANGED_DISTRACTOR_USE: updated_entity must be different from target_entity; update_sentence changes updated_entity; use_sentence states target_entity's ORIGINAL state from the source, not the new state of the distractor.
- Prefer concrete states, locations, roles, conditions, possessions, or descriptions already visible in the source.  It is allowed to simplify a long source phrase into a short source_state.
- Do not simply restate the whole original sentence.  Do not introduce harmful or adult content.  Keep the sentences plain and child-readable.

Example for UPDATED_USE:
{{"target_entity":"the marble","source_state":"in the box","updated_entity":"the marble","new_state":"in the basket","update_sentence":"Mia moved the marble from the box to the basket.","use_sentence":"Now the marble is in the basket."}}

Example for UNCHANGED_DISTRACTOR_USE:
{{"target_entity":"the book","source_state":"on the table","updated_entity":"the marble","new_state":"in the basket","update_sentence":"Mia moved the marble from the box to the basket.","use_sentence":"Now the book is on the table."}}

JSON:"""


def norm_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def canonical_ent(ent: str) -> str:
    ent = norm_ws(ent.strip(".,!?;:()[]{}\"'\u201c\u201d\u2018\u2019"))
    return ent


def entity_candidates(pair: dict[str, Any]) -> list[str]:
    text = pair["original"]
    cands: list[str] = []
    # Prefer the cleaner research entity annotations when present.
    for e in pair.get("entity_source") or []:
        e2 = canonical_ent(str(e))
        if len(e2) >= 2 and e2.lower() not in STOPWORDS:
            cands.append(e2)
    toks = text.split()
    # Multi-word proper-name chunks and capitalized tokens not solely sentence-initial.
    chunk: list[str] = []
    for i, tok in enumerate(toks + ["."]):
        stripped = canonical_ent(tok)
        letters = re.sub(r"[^A-Za-z0-9]", "", stripped)
        is_name = bool(letters) and (letters[0].isupper() or (len(letters) > 1 and any(ch.isupper() for ch in letters[1:])))
        # sentence-initial single capitalized common word is weak unless all-caps/internal cap
        if i == 0 and is_name and letters.istitle() and stripped.lower() not in {"I".lower()}:
            is_name = False
        if is_name and stripped.lower() not in STOPWORDS and len(stripped) >= 2:
            chunk.append(stripped)
        else:
            if chunk:
                cands.append(" ".join(chunk))
                chunk = []
    # Concrete common nouns as fallback.
    for w in words(text):
        wl = w.lower()
        if wl in CONCRETE_NOUNS:
            cands.append(wl)
    # Add pronoun-bearing human nouns only if present; useful for open subtitles/CHILDES.
    # Deduplicate by lowercase while preserving the most natural surface form.
    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        c = canonical_ent(c)
        key = c.lower()
        if len(key) < 2 or key in STOPWORDS or key in seen:
            continue
        out.append(c)
        seen.add(key)
    return out[:8]


def choose_type(pair_id: str, n_entities: int, counts: Counter[str]) -> str:
    if n_entities < 2:
        return "UPDATED_USE"
    # Deterministic alternation biased toward balance among emitted prompts.
    h = int(hashlib.sha1(f"{RNG_SEED}:{pair_id}".encode()).hexdigest()[:8], 16)
    preferred = "UNCHANGED_DISTRACTOR_USE" if (h % 2 == 0) else "UPDATED_USE"
    # Keep approximate 50/50 when possible.
    if counts["UNCHANGED_DISTRACTOR_USE"] > counts["UPDATED_USE"] + 200:
        return "UPDATED_USE"
    if counts["UPDATED_USE"] > counts["UNCHANGED_DISTRACTOR_USE"] + 200 and n_entities >= 2:
        return "UNCHANGED_DISTRACTOR_USE"
    return preferred


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    pairs: list[dict[str, Any]] = []
    with PAIRS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    counts: Counter[str] = Counter()
    prompts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    entity_count_hist: Counter[int] = Counter()
    word_count_by_type: Counter[str] = Counter()
    for p in pairs:
        ents = entity_candidates(p)
        entity_count_hist[len(ents)] += 1
        if not ents:
            skipped.append({"pair_id": p["pair_id"], "reason": "no_candidate_entity"})
            continue
        typ = choose_type(str(p["pair_id"]), len(ents), counts)
        counts[typ] += 1
        prompt = PROMPT_TEMPLATE.format(packet_type=typ, sentence=norm_ws(p["original"]), entities=", ".join(ents))
        rec = {
            "id": f"su3_{p['pair_id']}",
            "pair_id": p["pair_id"],
            "packet_type": typ,
            "prompt": prompt,
            "source_sentence": norm_ws(p["original"]),
            "source_words": int(p.get("original_words") or len(p["original"].split())),
            "qwen_rewrite": norm_ws(p["rewrite"]),
            "rewrite_words": int(p.get("rewrite_words") or len(p["rewrite"].split())),
            "pair_words_original_qwen": int(p.get("pair_words") or (len(p["original"].split()) + len(p["rewrite"].split()))),
            "source": p.get("source"),
            "example_id": p.get("example_id"),
            "cohort": p.get("cohort"),
            "candidate_entities": ents,
        }
        word_count_by_type[typ] += rec["source_words"]
        prompts.append(rec)
    prompts_path = OUT_DIR / "state_use_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    skipped_path = OUT_DIR / "skipped_pairs.jsonl"
    with skipped_path.open("w", encoding="utf-8") as f:
        for rec in skipped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "status": "STATE_USE_PROMPTS_CREATED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_selected_qwen_pairs": len(pairs),
        "prompt_count": len(prompts),
        "skipped_count": len(skipped),
        "prompt_fraction": round(len(prompts) / max(1, len(pairs)), 4),
        "packet_type_counts": dict(counts),
        "entity_count_hist": {str(k): v for k, v in sorted(entity_count_hist.items())},
        "source_words_by_type": dict(word_count_by_type),
        "prompt_path": str(prompts_path),
        "skipped_path": str(skipped_path),
        "prompt_sha256": sha256_file(prompts_path),
        "design_change_from_step041": "three-part source/update/use packets with UPDATED_USE and UNCHANGED_DISTRACTOR_USE halves; the pair-only raw_state_updates_qwen3_1p7b.jsonl is not a training stream",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = OUT_DIR / "prompt_extraction_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
