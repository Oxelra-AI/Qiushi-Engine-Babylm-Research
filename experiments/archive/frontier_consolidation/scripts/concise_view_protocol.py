#!/usr/bin/env python3
"""research: Repaired Concise Faithful Second-View Protocol.

Builds length-adaptive, entity-guided prompts for generating concise faithful
second views of official BabyLM sentences. The key innovations over research:

1. Length-adaptive compression targets (no impossible 50% on 12-word sentences)
2. Entity pre-extraction → explicit MUST-INCLUDE constraints in prompt
3. Source-aware handling (CHILDES/BNC speaker labels preserved explicitly)
4. Concrete word-count target in the prompt (not just a ratio)

The information-efficiency hypothesis: by generating concise views at 0.55-0.75×
original length, we can cover ~47k-55k unique source examples within the same
~1.66M pair-word budget (vs current 25,486 unique examples at near-length rewrites).
"""
from __future__ import annotations

import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Optional

# ─── Entity extraction helpers ─────────────────────────────────────────────

# Named entity patterns (proper nouns, speaker labels, titles)
SPEAKER_LABEL_RE = re.compile(r"\*([A-Z]{2,5}):")  # CHILDES/BNC speaker labels
PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+(?:de|von|van|the|of|du|di)\s+)?[A-Z][a-z]+|"
    r"[A-Z][a-z]{2,})\b"
)
QUOTED_RE = re.compile(r'"([^"]+)"|"([^"]+)"|\'([^\']+)\'')
NUMBER_RE = re.compile(
    r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?%?|"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)"
    r"(?:th|st|nd|rd)?)\b",
    re.I
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}?|"
    r"\d{4}\s*(?:AD|BC|CE|BCE)?)\b",
    re.I
)
NEGATION_RE = re.compile(
    r"\b(?:not|n't|never|no\b|neither|nor|nobody|nothing|nowhere|"
    r"without|cannot|can't|won't|don't|doesn't|didn't|wasn't|weren't|"
    r"isn't|aren't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't|mustn't)\b",
    re.I
)
MODALITY_RE = re.compile(
    r"\b(?:must|should|could|would|might|may|shall|ought\s+to|"
    r"have\s+to|had\s+to|has\s+to|need\s+to|supposed\s+to)\b",
    re.I
)


def extract_entities(text: str) -> dict:
    """Extract named entities, numbers, dates, negation, modality from text."""
    entities = {}
    
    # Speaker labels (CHILDES/BNC)
    speakers = SPEAKER_LABEL_RE.findall(text)
    if speakers:
        entities["speakers"] = list(set(speakers))
    
    # Proper nouns / named entities
    # Remove speaker-label prefix before scanning
    clean = SPEAKER_LABEL_RE.sub("", text)
    proper = []
    for m in PROPER_NOUN_RE.finditer(clean):
        word = m.group()
        # Skip sentence-initial capitalization (first word after speaker label or sentence start)
        pos = m.start()
        if pos == 0 or clean[pos-1] in ".!?:;":
            continue
        # Skip common false positives
        if word.lower() in {"the", "this", "that", "these", "those", "there", "then",
                            "they", "their", "them", "here", "however", "also", "still",
                            "well", "just", "only", "even", "very", "much", "most",
                            "some", "such", "other", "another"}:
            continue
        proper.append(word)
    if proper:
        entities["named_entities"] = list(set(proper))
    
    # Quoted phrases
    quoted = []
    for m in QUOTED_RE.finditer(text):
        q = m.group(1) or m.group(2) or m.group(3)
        if q and len(q.split()) <= 8:
            quoted.append(q)
    if quoted:
        entities["quoted"] = quoted
    
    # Numbers and quantities
    numbers = [m.group() for m in NUMBER_RE.finditer(text)]
    dates = [m.group() for m in DATE_RE.finditer(text)]
    if numbers:
        entities["numbers"] = list(set(numbers))
    if dates:
        entities["dates"] = list(set(dates))
    
    # Negation markers
    negations = [m.group() for m in NEGATION_RE.finditer(text)]
    if negations:
        entities["negation"] = list(set(negations))
    
    # Modality markers
    modals = [m.group() for m in MODALITY_RE.finditer(text)]
    if modals:
        entities["modality"] = list(set(modals))
    
    return entities


def word_count(text: str) -> int:
    return len(text.split())


# ─── Length-adaptive target computation ────────────────────────────────────

def target_range(source_words: int) -> tuple[float, float, int, int]:
    """Return (min_ratio, max_ratio, min_words, max_words) for the concise view.
    
    Design principle: short sentences can't be compressed much without losing
    meaning. Only sentences with enough redundancy (25+ words) get aggressive
    compression targets.
    """
    if source_words <= 14:
        # Very short: light paraphrase with some compression
        ratio_lo, ratio_hi = 0.75, 0.95
    elif source_words <= 19:
        # Short: moderate compression
        ratio_lo, ratio_hi = 0.65, 0.85
    elif source_words <= 27:
        # Medium: good compression
        ratio_lo, ratio_hi = 0.55, 0.75
    elif source_words <= 40:
        # Long: strong compression
        ratio_lo, ratio_hi = 0.50, 0.70
    else:
        # Very long: aggressive compression
        ratio_lo, ratio_hi = 0.45, 0.65
    
    min_words = max(6, int(source_words * ratio_lo))
    max_words = max(8, int(source_words * ratio_hi))
    return ratio_lo, ratio_hi, min_words, max_words


# ─── Prompt construction ──────────────────────────────────────────────────

def build_prompt(sentence: str, source: str) -> dict:
    """Build a guided concise paraphrase prompt for one sentence.
    
    Returns dict with prompt text, extracted entities, target range, and metadata.
    """
    n_words = word_count(sentence)
    entities = extract_entities(sentence)
    ratio_lo, ratio_hi, min_w, max_w = target_range(n_words)
    
    # Build constraint section
    constraints = []
    
    # Speaker label constraint
    if "speakers" in entities:
        labels = ", ".join(f"*{s}:" for s in entities["speakers"])
        constraints.append(f"Keep the speaker label(s) exactly: {labels}")
    
    # Named entity constraint
    if "named_entities" in entities:
        ents = ", ".join(entities["named_entities"])
        constraints.append(f"Include these names exactly: {ents}")
    
    # Quoted constraint
    if "quoted" in entities:
        qs = "; ".join(f'"{q}"' for q in entities["quoted"])
        constraints.append(f"Preserve quoted text: {qs}")
    
    # Number/date constraint
    if "numbers" in entities or "dates" in entities:
        nums = entities.get("numbers", []) + entities.get("dates", [])
        constraints.append(f"Keep numbers/dates: {', '.join(nums)}")
    
    # Negation constraint
    if "negation" in entities:
        negs = ", ".join(sorted(set(entities["negation"])))
        constraints.append(f"Preserve negation (do NOT drop): {negs}")
    
    # Modality constraint
    if "modality" in entities:
        mods = ", ".join(sorted(set(entities["modality"])))
        constraints.append(f"Keep modal meaning: {mods}")
    
    constraint_block = "\n".join(f"- {c}" for c in constraints) if constraints else "- No special constraints."
    
    # Determine target word count as a concrete number
    target_words = f"{min_w}-{max_w}"
    
    prompt = (
        f"Rewrite the following sentence as a shorter, complete, natural English sentence.\n"
        f"Target length: {target_words} words ({int(ratio_lo*100)}-{int(ratio_hi*100)}% of original).\n"
        f"Preserve all core meaning: who did what to whom, events, causes, consequences.\n"
        f"\nMUST PRESERVE:\n{constraint_block}\n"
        f"\nDo NOT add facts, explain, continue, or write a title/fragment/list.\n"
        f"Output only the rewritten sentence.\n"
        f"\nSentence: {sentence}"
    )
    
    return {
        "prompt": prompt,
        "source_sentence": sentence,
        "source_words": n_words,
        "source_name": source,
        "target_min_words": min_w,
        "target_max_words": max_w,
        "target_ratio_lo": ratio_lo,
        "target_ratio_hi": ratio_hi,
        "extracted_entities": entities,
    }


# ─── Verifier ─────────────────────────────────────────────────────────────

def verify_output(source: str, output: str, meta: dict) -> dict:
    """Verify a generated concise view against fidelity requirements.
    
    Returns dict with accept/reject decision, reasons, and quality metrics.
    """
    src_words = meta["source_words"]
    out_words = word_count(output)
    entities = meta["extracted_entities"]
    min_w = meta["target_min_words"]
    max_w = meta["target_max_words"]
    
    reasons = []
    metrics = {}
    
    # 1. Length check
    len_ratio = out_words / src_words if src_words > 0 else 1.0
    metrics["len_ratio"] = round(len_ratio, 4)
    metrics["output_words"] = out_words
    
    if out_words > max_w + 2:  # small grace margin
        reasons.append("len_ratio_high")
    elif out_words < max(4, min_w - 2):
        reasons.append("len_ratio_low")
    
    # 2. Entity recall
    output_lower = output.lower()
    output_norm = re.sub(r"[^a-z0-9\s]", " ", output_lower)
    
    # Speaker labels
    if "speakers" in entities:
        for sp in entities["speakers"]:
            # Check for *SPK: or SPK: or SPK pattern
            if f"*{sp.lower()}:" not in output_lower and f"{sp.lower()}:" not in output_lower:
                # Also accept "SPK states/says/..." pattern
                if sp.lower() not in output_lower:
                    reasons.append(f"speaker_loss_{sp}")
    
    # Named entities
    if "named_entities" in entities:
        missing = []
        for ent in entities["named_entities"]:
            # Case-insensitive check
            if ent.lower() not in output_lower:
                # Try partial match (first word of multi-word entity)
                parts = ent.split()
                if not any(p.lower() in output_lower for p in parts if len(p) > 2):
                    missing.append(ent)
        if missing:
            reasons.append(f"entity_loss")
            metrics["missing_entities"] = missing
    
    # Numbers
    if "numbers" in entities:
        missing_nums = []
        for num in entities["numbers"]:
            if num.lower() not in output_lower:
                # Try digit-only version
                digits = re.sub(r"[^\d.]", "", num)
                if digits and digits not in output:
                    missing_nums.append(num)
        if missing_nums:
            reasons.append("number_mismatch")
            metrics["missing_numbers"] = missing_nums
    
    # Negation preservation
    if "negation" in entities:
        src_neg_count = len(entities["negation"])
        out_negs = NEGATION_RE.findall(output)
        if len(out_negs) < src_neg_count:
            reasons.append("negation_loss")
    
    # Modality preservation
    if "modality" in entities:
        src_mod_count = len(entities["modality"])
        out_mods = MODALITY_RE.findall(output)
        if len(out_mods) < src_mod_count:
            reasons.append("modality_loss")
    
    # 3. Copy overlap (reject near-copies)
    src_tokens = set(re.sub(r"[^a-z0-9\s]", " ", source.lower()).split())
    out_tokens = set(output_norm.split())
    if src_tokens and out_tokens:
        jaccard = len(src_tokens & out_tokens) / len(src_tokens | out_tokens)
        overlap = len(src_tokens & out_tokens) / len(src_tokens)
        metrics["jaccard"] = round(jaccard, 4)
        metrics["content_overlap"] = round(overlap, 4)
        if jaccard > 0.90 and len_ratio > 0.85:
            reasons.append("copy_overlap")
    
    # 4. Basic quality checks
    output_stripped = output.strip()
    if not output_stripped:
        reasons.append("empty")
    elif output_stripped[0].islower() and not output_stripped.startswith("*"):
        # Allow lowercase start only for speaker-labeled text
        if "speakers" not in entities:
            reasons.append("bad_start")
    
    # Check for meta-prefixes
    meta_prefixes = ["here is", "here's", "rewritten:", "rewrite:", "output:"]
    if any(output_lower.strip().startswith(p) for p in meta_prefixes):
        reasons.append("meta_prefix")
    
    # Check for multi-sentence when source is one sentence
    # Allow up to 2 sentences in output
    sent_ends = len(re.findall(r"[.!?]+\s+[A-Z]", output))
    if sent_ends > 2:
        reasons.append("too_many_sentences")
    
    accepted = len(reasons) == 0
    return {
        "accepted": accepted,
        "reasons": reasons,
        "metrics": metrics,
    }


# ─── Main: test on a small sample ─────────────────────────────────────────

def main():
    """Build prompts for a test slice of the research panel originals."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", type=str, 
                        default="experiments/archive/compact_experience/data/transform_panel/panel_originals.jsonl")
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--output-dir", type=str,
                        default="experiments/archive/frontier_consolidation/data/concise_view_test")
    parser.add_argument("--seed", type=int, default=80291)
    args = parser.parse_args()
    
    import random
    rng = random.Random(args.seed)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load originals
    originals = []
    with open(args.originals, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                originals.append(json.loads(line))
    
    # Sample
    sample = rng.sample(originals, min(args.n_sample, len(originals)))
    
    # Build prompts
    prompts = []
    for i, orig in enumerate(sample):
        text = orig["text"]
        source = orig.get("source", "unknown")
        prompt_data = build_prompt(text, source)
        prompt_data["original_id"] = orig.get("original_id", f"test_{i:05d}")
        prompt_data["id"] = f"gcv_{i:05d}"
        prompt_data["example_id"] = orig.get("example_id", -1)
        prompts.append(prompt_data)
    
    # Save prompts
    prompts_path = out_dir / "test_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Save the prompt batch with one identifier and prompt text per record.
    ailab_path = out_dir / "ailab_prompts.jsonl"
    with ailab_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({
                "id": p["id"],
                "prompt": p["prompt"],
            }, ensure_ascii=False) + "\n")
    
    # Compute statistics
    target_stats = {
        "n_prompts": len(prompts),
        "source_word_stats": {
            "min": min(p["source_words"] for p in prompts),
            "mean": round(sum(p["source_words"] for p in prompts) / len(prompts), 1),
            "max": max(p["source_words"] for p in prompts),
        },
        "target_word_stats": {
            "mean_min": round(sum(p["target_min_words"] for p in prompts) / len(prompts), 1),
            "mean_max": round(sum(p["target_max_words"] for p in prompts) / len(prompts), 1),
        },
        "entity_coverage": {
            "has_speakers": sum(1 for p in prompts if "speakers" in p["extracted_entities"]),
            "has_named_entities": sum(1 for p in prompts if "named_entities" in p["extracted_entities"]),
            "has_numbers": sum(1 for p in prompts if "numbers" in p["extracted_entities"]),
            "has_negation": sum(1 for p in prompts if "negation" in p["extracted_entities"]),
            "has_modality": sum(1 for p in prompts if "modality" in p["extracted_entities"]),
        },
        "source_distribution": dict(Counter(p["source_name"] for p in prompts)),
        "files": {
            "prompts": str(prompts_path),
            "ailab_prompts": str(ailab_path),
        }
    }
    
    meta_path = out_dir / "test_prompt_metadata.json"
    meta_path.write_text(json.dumps(target_stats, indent=2), encoding="utf-8")
    print(json.dumps(target_stats, indent=2))


if __name__ == "__main__":
    main()
