#!/usr/bin/env python3
"""Assemble the factual training corpus from Qwen generation outputs.

Takes generation outputs (expansions, simplifications, paraphrases) and combines
them with original SimpleWiki and selected official corpus text into a 10M-word
training corpus formatted as simplification/paraphrase pairs.

The format matches the leader's approach: pairs of sentences separated by blank lines,
where each pair contains an "original" (complex) version followed by a "simplified"
(or paraphrased) version.

Our innovation: We include BOTH simplifications (for factual accessibility) and
paraphrases (for linguistic diversity), giving us advantages on both world knowledge
AND linguistic knowledge evaluations.
"""
import json
import random
import os
from pathlib import Path
from collections import defaultdict

# Paths (relative to user root)
WORKSPACE = Path("experiments/archive/representation_and_objectives")
SIMPLEWIKI_PATH = Path("experiments/archive/initial_model_studies/data/mixture_revision_61/official_raw/simple_wiki.train.txt")
OFFICIAL_RAW = Path("experiments/archive/initial_model_studies/data/mixture_revision_61/official_raw")

# Generation output paths (will be created by launch_factual_generation.sh)
EXPAND_OUTPUT = WORKSPACE / "training/runs/gen_expand_v4/generations.jsonl"
SIMPARA_OUTPUT = WORKSPACE / "training/runs/gen_simpara_v4/generations.jsonl"

# Output
OUTPUT_DIR = WORKSPACE / "training/data/training_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WORDS = 9_999_000  # Stay just under 10M

def load_generations(path):
    """Load generation outputs from JSONL."""
    results = []
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return results
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def load_simplewiki_paragraphs():
    """Load SimpleWiki paragraphs for direct use."""
    paragraphs = []
    with open(SIMPLEWIKI_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("= ="):
                continue
            words = line.split()
            if len(words) >= 8:
                paragraphs.append(line)
    return paragraphs

def load_official_source(name, max_words=None):
    """Load lines from an official corpus source."""
    path = OFFICIAL_RAW / f"{name}.train.txt"
    lines = []
    word_count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) < 5:
                continue
            lines.append(line)
            word_count += len(words)
            if max_words and word_count >= max_words:
                break
    return lines

def format_pair(original, simplified):
    """Format a single pair in the leader's format."""
    return f"{original}\n{simplified}\n"

def main():
    print("=== Corpus Assembly ===")
    random.seed(42)
    
    # Load generation outputs
    print("\nLoading generation outputs...")
    expansions = load_generations(EXPAND_OUTPUT)
    simpara = load_generations(SIMPARA_OUTPUT)
    print(f"  Expansions: {len(expansions)}")
    print(f"  Simplifications + Paraphrases: {len(simpara)}")
    
    # Separate simplifications and paraphrases
    simplifications = [g for g in simpara if g.get("id", "").startswith("simp_")]
    paraphrases = [g for g in simpara if g.get("id", "").startswith("para_")]
    print(f"  -> Simplifications: {len(simplifications)}")
    print(f"  -> Paraphrases: {len(paraphrases)}")
    
    # Load original SimpleWiki
    print("\nLoading SimpleWiki...")
    wiki_paras = load_simplewiki_paragraphs()
    print(f"  Paragraphs: {len(wiki_paras)}")
    
    # Assemble corpus sections
    corpus_parts = []
    word_count = 0
    
    # SECTION 1: Expansion pairs (expansion = "original", simplewiki = "simplified")
    # This is our main factual content section
    print("\nAssembling Section 1: Expansion pairs...")
    expansion_pairs = 0
    for gen in expansions:
        if word_count >= TARGET_WORDS * 0.45:  # Cap at 45% for expansions
            break
        generated_text = gen.get("generated_text", gen.get("text", "")).strip()
        source_text = gen.get("source_text", "").strip()
        if not generated_text or not source_text:
            continue
        # Pair: expanded version (complex) + original wiki (simple)
        pair = format_pair(generated_text, source_text)
        pair_words = len(pair.split())
        corpus_parts.append(pair)
        word_count += pair_words
        expansion_pairs += 1
    print(f"  Expansion pairs: {expansion_pairs}, words so far: {word_count:,}")
    
    # SECTION 2: Simplification pairs (original wiki = "original", simplified = "simplified")
    print("\nAssembling Section 2: Simplification pairs...")
    simp_pairs = 0
    for gen in simplifications:
        if word_count >= TARGET_WORDS * 0.65:  # Cap at 65% cumulative
            break
        generated_text = gen.get("generated_text", gen.get("text", "")).strip()
        source_text = gen.get("source_text", "").strip()
        if not generated_text or not source_text:
            continue
        # Pair: original wiki (complex) + simplified (simple)
        pair = format_pair(source_text, generated_text)
        pair_words = len(pair.split())
        corpus_parts.append(pair)
        word_count += pair_words
        simp_pairs += 1
    print(f"  Simplification pairs: {simp_pairs}, words so far: {word_count:,}")
    
    # SECTION 3: Paraphrase pairs (original wiki + paraphrased)
    print("\nAssembling Section 3: Paraphrase pairs...")
    para_pairs = 0
    for gen in paraphrases:
        if word_count >= TARGET_WORDS * 0.80:  # Cap at 80% cumulative
            break
        generated_text = gen.get("generated_text", gen.get("text", "")).strip()
        source_text = gen.get("source_text", "").strip()
        if not generated_text or not source_text:
            continue
        # Pair: original + paraphrased
        pair = format_pair(source_text, generated_text)
        pair_words = len(pair.split())
        corpus_parts.append(pair)
        word_count += pair_words
        para_pairs += 1
    print(f"  Paraphrase pairs: {para_pairs}, words so far: {word_count:,}")
    
    # SECTION 4: Fill remaining budget with CHILDES + Gutenberg for developmental coverage
    print("\nAssembling Section 4: Official corpus fill...")
    remaining = TARGET_WORDS - word_count
    if remaining > 0:
        # Mix of CHILDES (developmental) and Gutenberg (narrative)
        childes_lines = load_official_source("childes", max_words=remaining // 2)
        gutenberg_lines = load_official_source("gutenberg", max_words=remaining // 2)
        
        fill_lines = childes_lines + gutenberg_lines
        random.shuffle(fill_lines)
        
        fill_count = 0
        for line in fill_lines:
            if word_count >= TARGET_WORDS:
                break
            corpus_parts.append(line + "\n")
            word_count += len(line.split())
            fill_count += 1
        print(f"  Official fill lines: {fill_count}, total words: {word_count:,}")
    
    # Shuffle the corpus (within each section, pairs are shuffled)
    print("\nShuffling corpus...")
    random.shuffle(corpus_parts)
    
    # Write final training file
    output_path = OUTPUT_DIR / "factual_training_corpus.train"
    with open(output_path, 'w', encoding='utf-8') as f:
        for part in corpus_parts:
            f.write(part)
            if not part.endswith('\n'):
                f.write('\n')
    
    # Final verification
    with open(output_path) as f:
        final_text = f.read()
    final_words = len(final_text.split())
    final_lines = len(final_text.split('\n'))
    
    summary = {
        "output_path": str(output_path),
        "total_words": final_words,
        "total_lines": final_lines,
        "within_budget": final_words <= 10_000_000,
        "composition": {
            "expansion_pairs": expansion_pairs,
            "simplification_pairs": simp_pairs,
            "paraphrase_pairs": para_pairs,
            "official_fill_lines": fill_count if remaining > 0 else 0,
        },
        "word_budget_used": f"{final_words / 10_000_000 * 100:.1f}%",
    }
    
    summary_path = OUTPUT_DIR / "corpus_assembly_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Assembly Complete ===")
    print(f"  Output: {output_path}")
    print(f"  Words: {final_words:,} / 10,000,000 ({final_words/10_000_000*100:.1f}%)")
    print(f"  Lines: {final_lines:,}")
    print(f"  Within budget: {final_words <= 10_000_000}")
    print(f"  Summary: {summary_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
