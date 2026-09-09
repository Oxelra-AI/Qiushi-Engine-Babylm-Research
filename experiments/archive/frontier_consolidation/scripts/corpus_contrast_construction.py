#!/usr/bin/env python3
"""research Phase 1: Build frozen corpus-derived contrast families.

Reads held-out corpus rows (never used for training), extracts sentences,
pairs them to create context-target contrasts, and saves the frozen set.
NO model information is used; this must be run BEFORE any scoring.

Two independent families:
  A: within-source pairs (sentences from same source type)
  B: cross-source pairs (sentences from different source types)
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import hashlib
import random
import re
from pathlib import Path
from collections import defaultdict

ROOT = _public_path('experiments/archive/frontier_consolidation')  # workspace root
HELDOUT = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/frozen_corpus_contrasts')


def extract_sentences(text: str, min_words: int = 8, max_words: int = 50) -> list:
    """Split text into clean sentences suitable for contrast pairs."""
    # Clean CHILDES markers
    text = re.sub(r'\*[A-Z]{2,3}:\s*', '', text)
    text = re.sub(r'xxx', '', text)
    
    # Simple sentence splitting on . ! ? followed by space or end
    raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
    
    sents = []
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) < min_words or len(words) > max_words:
            continue
        # Reject sentences that are too fragmented or have too many special chars
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.6:
            continue
        sents.append(s)
    return sents


def split_context_target(sentence: str, target_frac: float = 0.4):
    """Split sentence into context (prefix) and target (suffix)."""
    words = sentence.split()
    n = len(words)
    split_idx = max(3, int(n * (1 - target_frac)))
    context = " ".join(words[:split_idx])
    target = " ".join(words[split_idx:])
    return context, target


def build_contrast_pair(sent_a: str, sent_b: str, uid: str, family: str, meta: dict):
    """Create one contrast pair from two sentences."""
    ctx_a, tgt_a = split_context_target(sent_a)
    ctx_b, tgt_b = split_context_target(sent_b)
    
    # Context1+Target1 is the "correct" combination (sent_a goes with sent_a)
    return {
        "uid": uid,
        "family": family,
        "Context1": ctx_a,
        "Target1": tgt_a,
        "Context2": ctx_b,
        "Target2": tgt_b,
        "source_a": meta.get("source_a", ""),
        "source_b": meta.get("source_b", ""),
        "sentence_a": sent_a,
        "sentence_b": sent_b,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read held-out rows
    rows_by_source = defaultdict(list)
    with open(HELDOUT) as f:
        for line in f:
            row = json.loads(line)
            src = row.get("source", "unknown")
            sents = extract_sentences(row["text"])
            for s in sents:
                rows_by_source[src].append(s)
    
    print("Sentences extracted per source:")
    for src in sorted(rows_by_source.keys()):
        print(f"  {src}: {len(rows_by_source[src])}")
    total_sents = sum(len(v) for v in rows_by_source.values())
    print(f"  TOTAL: {total_sents}")
    
    # Deterministic seed for reproducibility
    rng = random.Random(81081)
    
    # Shuffle within each source
    for src in rows_by_source:
        rng.shuffle(rows_by_source[src])
    
    # === Family A: Within-source pairs ===
    family_a = []
    for src, sents in rows_by_source.items():
        n_pairs = len(sents) // 2
        for i in range(n_pairs):
            uid = f"within_{src}_{i}"
            pair = build_contrast_pair(
                sents[2*i], sents[2*i + 1], uid, "within_source",
                {"source_a": src, "source_b": src}
            )
            family_a.append(pair)
    
    # === Family B: Cross-source pairs ===
    # Pair sentences from different sources
    sources = sorted(rows_by_source.keys())
    family_b = []
    cross_pairs_per_combo = 100  # limit per source combination
    for i, src_a in enumerate(sources):
        for src_b in sources[i+1:]:
            sents_a = rows_by_source[src_a]
            sents_b = rows_by_source[src_b]
            n_pairs = min(cross_pairs_per_combo, len(sents_a), len(sents_b))
            for j in range(n_pairs):
                uid = f"cross_{src_a}_{src_b}_{j}"
                pair = build_contrast_pair(
                    sents_a[j], sents_b[j], uid, "cross_source",
                    {"source_a": src_a, "source_b": src_b}
                )
                family_b.append(pair)
    
    # Combine and save
    all_contrasts = family_a + family_b
    
    # Save frozen contrast set
    out_path = _public_path('experiments/archive/frontier_consolidation/data/frozen_corpus_contrasts/frozen_corpus_contrasts.jsonl')
    with open(out_path, "w") as f:
        for c in all_contrasts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    # Compute SHA for provenance
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    
    # Save manifest
    manifest = {
        "status": "FROZEN_CORPUS_CONTRASTS_BUILT",
        "purpose": "Non-EWoK corpus-derived contrasts for interaction validation",
        "frozen_before_scoring": True,
        "heldout_source": str(HELDOUT),
        "total_contrasts": len(all_contrasts),
        "family_a_within_source": len(family_a),
        "family_b_cross_source": len(family_b),
        "sources": {src: len(sents) for src, sents in sorted(rows_by_source.items())},
        "output_path": str(out_path),
        "output_sha256": sha,
        "seed": 81081,
        "sentence_extraction": {"min_words": 8, "max_words": 50, "target_frac": 0.4},
    }
    
    with open(_public_path('experiments/archive/frontier_consolidation/data/frozen_corpus_contrasts/construction_manifest.json'), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nFrozen contrast set saved: {out_path}")
    print(f"  Family A (within-source): {len(family_a)}")
    print(f"  Family B (cross-source): {len(family_b)}")
    print(f"  Total: {len(all_contrasts)}")
    print(f"  SHA256: {sha}")
    
    # Print sample
    print("\nSample contrasts:")
    for c in all_contrasts[:3]:
        print(f"  [{c['family']}] Context1: {c['Context1'][:60]}... Target1: {c['Target1'][:40]}")
        print(f"               Context2: {c['Context2'][:60]}... Target2: {c['Target2'][:40]}")
        print()


if __name__ == "__main__":
    main()
