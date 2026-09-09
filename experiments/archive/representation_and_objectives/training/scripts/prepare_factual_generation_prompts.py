#!/usr/bin/env python3
"""Prepare prompts for factual content generation using Qwen3.5-9B.

Strategy: Use SimpleWiki (1.5M words of factual encyclopedic text) as seed.
Generate three types of content:
1. EXPANSION: Enrich SimpleWiki facts into more detailed factual paragraphs (like FineWeb-Edu)
2. SIMPLIFICATION: Even simpler rewrites of SimpleWiki for easy learning
3. PARAPHRASE: Linguistic diversity versions

The resulting corpus will be assembled into pairs similar to the leader's format:
  expanded_version (our "original")
  simplewiki_original (our "simplified")
  
Plus paraphrases for additional linguistic diversity.

Target: Generate enough content for a 10M word training corpus with factual density
matching FineWeb-Edu level (which the 41.8 SOTA leader uses).
"""
import json
import random
import hashlib
from pathlib import Path

# Source data
SIMPLEWIKI_PATH = Path("experiments/archive/initial_model_studies/data/mixture_revision_61/official_raw/simple_wiki.train.txt")
OUTPUT_DIR = Path("experiments/archive/representation_and_objectives/training/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target: Generate ~60k prompts across 2 shards (for dual-GPU parallel generation)
# SimpleWiki has ~58.7k lines but many are short. We'll select substantial paragraphs.

def extract_paragraphs(path, min_words=8, max_words=80):
    """Extract usable paragraphs from SimpleWiki."""
    paragraphs = []
    current_article = ""
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip headers
            if line.startswith("= ="):
                current_article = line.replace("= = =", "").replace("= =", "").strip()
                continue
            
            words = line.split()
            if len(words) < min_words or len(words) > max_words:
                continue
                
            paragraphs.append({
                "text": line,
                "words": len(words),
                "article": current_article,
            })
    
    return paragraphs

def create_expansion_prompt(para):
    """Create prompt to expand a simple fact into richer factual content."""
    return (
        "Expand the following simple factual text into a more detailed and informative "
        "paragraph. Add relevant context, explanations, causes, effects, or related facts. "
        "Keep it factual and informative. Output only the expanded paragraph, nothing else.\n\n"
        f"Text: {para['text']}"
    )

def create_simplification_prompt(para):
    """Create prompt to further simplify text."""
    return (
        "Simplify the following text to be very easy to understand. Use short sentences "
        "and common words. Keep the important facts. Output only the simplified version.\n\n"
        f"Text: {para['text']}"
    )

def create_paraphrase_prompt(para):
    """Create prompt for linguistic diversity."""
    return (
        "Rewrite the following sentence using completely different words and sentence "
        "structure while preserving the same meaning. Output only the rewritten sentence.\n\n"
        f"Sentence: {para['text']}"
    )

def main():
    print("Loading SimpleWiki...")
    paragraphs = extract_paragraphs(SIMPLEWIKI_PATH)
    print(f"  Extracted {len(paragraphs)} usable paragraphs")
    print(f"  Total words: {sum(p['words'] for p in paragraphs):,}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(paragraphs)
    
    # Create prompts: balanced across three types
    # For ~10M word corpus:
    #   - Expansions of top paragraphs → ~4M words (as "originals")
    #   - Original SimpleWiki paired with expansions → ~1.5M words
    #   - Simplifications → ~1.5M words
    #   - Paraphrases → ~1.5M words  
    #   - Other official sources (CHILDES, Gutenberg) → ~1.5M words
    # Total: ~10M
    
    # Select top paragraphs for expansion (most informative content)
    # Prioritize paragraphs with 15-60 words (good factual density)
    good_paras = [p for p in paragraphs if 15 <= p['words'] <= 60]
    short_paras = [p for p in paragraphs if 8 <= p['words'] < 15]
    
    print(f"  Good paragraphs (15-60 words): {len(good_paras)}")
    print(f"  Short paragraphs (8-14 words): {len(short_paras)}")
    
    prompts = []
    prompt_id = 0
    
    # Type 1: EXPANSION (target ~20k prompts, max 200 new tokens each → ~4M words)
    n_expand = min(20000, len(good_paras))
    for para in good_paras[:n_expand]:
        prompts.append({
            "id": f"exp_{prompt_id:06d}",
            "prompt": create_expansion_prompt(para),
            "type": "expansion",
            "source_text": para["text"],
            "source_words": para["words"],
            "source_article": para["article"],
        })
        prompt_id += 1
    
    # Type 2: SIMPLIFICATION (target ~15k prompts, max 90 tokens each → ~1.5M words)
    n_simplify = min(15000, len(good_paras))
    for para in good_paras[:n_simplify]:
        prompts.append({
            "id": f"simp_{prompt_id:06d}",
            "prompt": create_simplification_prompt(para),
            "type": "simplification",
            "source_text": para["text"],
            "source_words": para["words"],
            "source_article": para["article"],
        })
        prompt_id += 1
    
    # Type 3: PARAPHRASE (target ~15k prompts, max 90 tokens each → ~1.5M words)
    n_paraphrase = min(15000, len(good_paras) + len(short_paras))
    combined = good_paras + short_paras
    for para in combined[:n_paraphrase]:
        prompts.append({
            "id": f"para_{prompt_id:06d}",
            "prompt": create_paraphrase_prompt(para),
            "type": "paraphrase",
            "source_text": para["text"],
            "source_words": para["words"],
            "source_article": para["article"],
        })
        prompt_id += 1
    
    print(f"\nTotal prompts: {len(prompts)}")
    print(f"  Expansions: {n_expand}")
    print(f"  Simplifications: {n_simplify}")
    print(f"  Paraphrases: {n_paraphrase}")
    
    # Split into shards for dual-GPU generation
    # Shard 0: expansions (need more tokens, will run on GPU 0)
    # Shard 1: simplifications + paraphrases (shorter outputs, GPU 1)
    expansions = [p for p in prompts if p["type"] == "expansion"]
    others = [p for p in prompts if p["type"] != "expansion"]
    
    # Write shard 0 (expansions - longer outputs)
    shard0_path = OUTPUT_DIR / "factual_prompts_shard0_expand.jsonl"
    with open(shard0_path, 'w') as f:
        for p in expansions:
            f.write(json.dumps(p) + '\n')
    
    # Write shard 1 (simplifications + paraphrases - shorter outputs)
    shard1_path = OUTPUT_DIR / "factual_prompts_shard1_simpara.jsonl"
    with open(shard1_path, 'w') as f:
        for p in others:
            f.write(json.dumps(p) + '\n')
    
    # Summary
    summary = {
        "total_prompts": len(prompts),
        "shard0_expansions": len(expansions),
        "shard1_simpara": len(others),
        "source_paragraphs_total": len(paragraphs),
        "source_good_paragraphs": len(good_paras),
        "shard0_path": str(shard0_path),
        "shard1_path": str(shard1_path),
        "generation_config": {
            "shard0_max_new_tokens": 200,  # expansions need more space
            "shard1_max_new_tokens": 90,   # simplifications/paraphrases are shorter
            "model": "qwen3.5-9b",
            "temperature": 0.3,
            "batch_size": 64,
        },
        "expected_output_words": {
            "expansions": "~3-4M words (avg 150-200 words per expansion)",
            "simplifications": "~1-1.5M words",
            "paraphrases": "~1-1.5M words",
            "plus_original_simplewiki": "1.5M words (used directly)",
            "plus_other_official": "~2-3M words from CHILDES/Gutenberg",
            "total_target": "~10M words",
        }
    }
    
    summary_path = OUTPUT_DIR / "prompt_preparation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFiles written:")
    print(f"  {shard0_path} ({len(expansions)} prompts)")
    print(f"  {shard1_path} ({len(others)} prompts)")
    print(f"  {summary_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
