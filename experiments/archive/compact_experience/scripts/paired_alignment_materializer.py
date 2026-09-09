#!/usr/bin/env python3
"""research: Paired-corpus materializer with VISIBLE within-sequence alignment.

Critical design:
If pair sides are independent MLM rows, we measure semantic repetition / lexical 
diversity, NOT training-visible alignment. Alignment requires both sides in the 
SAME 160-word sequence, with a random-mismatch control using the EXACT same text 
collection, plus single-view repetition and multi-semantic single-view to separate 
repetition consolidation from rewrite diversity and semantic coverage.

Arms (all budget-matched to the same total word count):
1. ALIGNED: [orig_i][simp_i][orig_j][simp_j]... — alignment visible to attention
2. MISMATCHED: [orig_i][simp_k][orig_j][simp_l]... — same sentences, no alignment  
3. SINGLE_ORIG: diverse originals only — max semantic coverage, no pairing
4. SINGLE_REPEAT: originals from ALIGNED, each appearing 2x — repetition, no diversity

Budget matching:
- ALIGNED uses ~N_orig words + ~N_simp words = T words total
- MISMATCHED uses exactly the same sentence multiset = T words total
- SINGLE_ORIG uses T words of diverse unique originals
- SINGLE_REPEAT uses ~T/2 unique originals × 2 passes = T words total

Key contrasts:
- ALIGNED vs MISMATCHED: pure alignment effect (same text, only structure differs)
- ALIGNED vs SINGLE_REPEAT: multi-view diversity vs repetition consolidation
- ALIGNED vs SINGLE_ORIG: aligned fewer-meanings vs broad coverage
- MISMATCHED vs SINGLE_ORIG: effect of having simplified text (unaligned) vs not
"""

import json
import hashlib
import random
import time
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
WORDS_PER_EXAMPLE = 160
TARGET_POOL_WORDS = 10_000_000  # 10M for Strict-Small budget
SEED = 43
HF_CACHE = "data/external/hf_hub"
OUTPUT_DIR = Path("experiments/archive/compact_experience/data/paired_alignment")
# ──────────────────────────────────────────────────────────────────────────────

import os
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE


@dataclass
class Pair:
    orig: str
    simp: str
    ow: int
    sw: int
    pw: int  # ow + sw
    src: str
    pid: int = 0


def clean(s: str) -> str:
    return " ".join(s.split())


def load_all_pairs() -> list[Pair]:
    from datasets import load_dataset
    
    pairs = []
    
    # WikiLarge
    print("Loading WikiLarge...", flush=True)
    wl = load_dataset("Nechba/wikilarge-text-simplification", split="train",
                      cache_dir=HF_CACHE)
    skip = 0
    for r in wl:
        o, s = clean(r["Normal"]), clean(r["Simple"])
        ow, sw = len(o.split()), len(s.split())
        if ow < 3 or sw < 3 or o == s:
            skip += 1
            continue
        pairs.append(Pair(orig=o, simp=s, ow=ow, sw=sw, pw=ow+sw, src="wikilarge"))
    print(f"  WikiLarge: {len(pairs)} valid, {skip} skipped", flush=True)
    
    # SynCSE
    n_wl = len(pairs)
    print("Loading SynCSE-partial-NLI...", flush=True)
    sc = load_dataset("hkust-nlp/SynCSE-partial-NLI", split="train",
                      cache_dir=HF_CACHE)
    skip = 0
    for r in sc:
        o, s = clean(r["sent0"]), clean(r["sent1"])
        ow, sw = len(o.split()), len(s.split())
        if ow < 3 or sw < 3 or o == s:
            skip += 1
            continue
        pairs.append(Pair(orig=o, simp=s, ow=ow, sw=sw, pw=ow+sw, src="syncse"))
    print(f"  SynCSE: {len(pairs) - n_wl} valid, {skip} skipped", flush=True)
    
    return pairs


def select_pairs(pairs: list[Pair], target: int, seed: int) -> list[Pair]:
    """Select pairs until cumulative pair_words reaches target."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    
    selected = []
    cum = 0
    for p in shuffled:
        if cum + p.pw > target:
            if cum >= target - WORDS_PER_EXAMPLE:
                break
            continue
        p.pid = len(selected)
        selected.append(p)
        cum += p.pw
        if cum >= target:
            break
    return selected


def pack_word_stream(words: list[str], source_label: str) -> list[dict]:
    """Pack a flat word list into exact 160-word examples."""
    n_full = len(words) // WORDS_PER_EXAMPLE
    examples = []
    for i in range(n_full):
        start = i * WORDS_PER_EXAMPLE
        text = " ".join(words[start:start + WORDS_PER_EXAMPLE])
        examples.append({
            "text": text,
            "words": WORDS_PER_EXAMPLE,
            "example_id": i,
            "source": source_label
        })
    return examples


def build_aligned(pairs: list[Pair]) -> list[dict]:
    """[orig_1][simp_1][orig_2][simp_2]... — alignment visible."""
    words = []
    for p in pairs:
        words.extend(p.orig.split())
        words.extend(p.simp.split())
    return pack_word_stream(words, "aligned_pair")


def build_mismatched(pairs: list[Pair], seed: int) -> list[dict]:
    """Same sentences, scrambled alignment: [orig_i][simp_j]..."""
    rng = random.Random(seed)
    
    originals = [p.orig for p in pairs]
    simplifieds = [p.simp for p in pairs]
    
    # Shuffle simplified sides independently
    shuf = list(range(len(simplifieds)))
    rng.shuffle(shuf)
    # Fix accidental re-alignments
    for i in range(len(shuf)):
        if shuf[i] == i:
            j = (i + 1) % len(shuf)
            shuf[i], shuf[j] = shuf[j], shuf[i]
    
    words = []
    for i in range(len(originals)):
        words.extend(originals[i].split())
        words.extend(simplifieds[shuf[i]].split())
    return pack_word_stream(words, "mismatched_pair")


def build_single_orig(pairs: list[Pair], extra_pool: list[Pair], 
                      target_words: int) -> list[dict]:
    """Diverse originals only — max semantic coverage, no repetition."""
    words = []
    # First from selected pairs
    for p in pairs:
        words.extend(p.orig.split())
        if len(words) >= target_words:
            break
    # Then from extra pool if needed
    if len(words) < target_words:
        for p in extra_pool:
            words.extend(p.orig.split())
            if len(words) >= target_words:
                break
    # Truncate to target
    words = words[:target_words]
    return pack_word_stream(words, "single_orig")


def build_single_repeat(pairs: list[Pair], target_words: int, seed: int) -> list[dict]:
    """Originals repeated 2x. Tests repetition consolidation."""
    rng = random.Random(seed)
    
    # Select enough originals so 2x ≈ target
    unique_target = target_words // 2
    orig_words = []
    for p in pairs:
        orig_words.extend(p.orig.split())
        if len(orig_words) >= unique_target:
            break
    orig_words = orig_words[:unique_target]
    
    # Build two passes with different context (different chunking)
    # Pass 1: natural order
    pass1 = list(orig_words)
    # Pass 2: sentence-level shuffle for different contexts
    # Reconstruct sentences from the original pairs used
    sents = []
    cum = 0
    for p in pairs:
        if cum >= unique_target:
            break
        sents.append(p.orig)
        cum += p.ow
    rng.shuffle(sents)
    pass2 = []
    for s in sents:
        pass2.extend(s.split())
    pass2 = pass2[:unique_target]
    
    # Concatenate both passes
    all_words = pass1 + pass2
    all_words = all_words[:target_words]
    return pack_word_stream(all_words, "single_repeat")


def write_jsonl(examples: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_words = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            total_words += ex["words"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return {"path": str(path), "rows": len(examples), "total_words": total_words,
            "sha256_prefix": sha}


def main():
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Load ──────────────────────────────────────────────────────────────────
    all_pairs = load_all_pairs()
    random.Random(SEED).shuffle(all_pairs)
    print(f"\nTotal pairs available: {len(all_pairs)}")
    tot_pw = sum(p.pw for p in all_pairs)
    print(f"Total pair words: {tot_pw:,} (need {TARGET_POOL_WORDS:,})")
    
    # ── Select pairs for ALIGNED/MISMATCHED ───────────────────────────────────
    selected = select_pairs(all_pairs, TARGET_POOL_WORDS, SEED)
    sel_pw = sum(p.pw for p in selected)
    sel_ow = sum(p.ow for p in selected)
    sel_sw = sum(p.sw for p in selected)
    src_counts = Counter(p.src for p in selected)
    
    print(f"\nSelected {len(selected)} pairs")
    print(f"  Pair words: {sel_pw:,}")
    print(f"  Orig words: {sel_ow:,}")
    print(f"  Simp words: {sel_sw:,}")
    print(f"  Sources: {dict(src_counts)}")
    
    # ── Build arms ────────────────────────────────────────────────────────────
    print("\nBuilding ALIGNED...", flush=True)
    aligned_ex = build_aligned(selected)
    aligned_words = len(aligned_ex) * WORDS_PER_EXAMPLE
    print(f"  {len(aligned_ex)} examples, {aligned_words:,} words")
    
    print("Building MISMATCHED...", flush=True)
    mismatched_ex = build_mismatched(selected, SEED + 7)
    mis_words = len(mismatched_ex) * WORDS_PER_EXAMPLE
    print(f"  {len(mismatched_ex)} examples, {mis_words:,} words")
    
    # IMPORTANT: ALIGNED and MISMATCHED must have same example count
    # (they use the same total words, so they should produce same count)
    assert len(aligned_ex) == len(mismatched_ex), \
        f"aligned {len(aligned_ex)} != mismatched {len(mismatched_ex)}"
    
    # Target for other arms: match ALIGNED total words
    target_arm_words = aligned_words
    n_examples_target = len(aligned_ex)
    
    print("Building SINGLE_ORIG...", flush=True)
    # Extra originals from unselected pairs for coverage
    sel_ids = set(id(p) for p in selected)
    extra = [p for p in all_pairs if id(p) not in sel_ids]
    random.Random(SEED + 3).shuffle(extra)
    single_orig_ex = build_single_orig(selected, extra, target_arm_words)
    single_orig_ex = single_orig_ex[:n_examples_target]
    so_words = len(single_orig_ex) * WORDS_PER_EXAMPLE
    print(f"  {len(single_orig_ex)} examples, {so_words:,} words")
    
    print("Building SINGLE_REPEAT...", flush=True)
    single_repeat_ex = build_single_repeat(selected, target_arm_words, SEED + 11)
    single_repeat_ex = single_repeat_ex[:n_examples_target]
    sr_words = len(single_repeat_ex) * WORDS_PER_EXAMPLE
    print(f"  {len(single_repeat_ex)} examples, {sr_words:,} words")
    
    # ── Write JSONL ───────────────────────────────────────────────────────────
    print("\nWriting JSONL files...", flush=True)
    stats = {}
    stats["aligned"] = write_jsonl(aligned_ex, OUTPUT_DIR / "aligned_pool.jsonl")
    stats["mismatched"] = write_jsonl(mismatched_ex, OUTPUT_DIR / "mismatched_pool.jsonl")
    stats["single_orig"] = write_jsonl(single_orig_ex, OUTPUT_DIR / "single_orig_pool.jsonl")
    stats["single_repeat"] = write_jsonl(single_repeat_ex, OUTPUT_DIR / "single_repeat_pool.jsonl")
    
    for name, s in stats.items():
        print(f"  {name}: {s['rows']} rows, {s['total_words']:,} words -> {s['path']}")
    
    # ── Validation ────────────────────────────────────────────────────────────
    print("\nValidation:", flush=True)
    
    # Word multiset check (ALIGNED vs MISMATCHED)
    a_words = []
    for e in aligned_ex:
        a_words.extend(e["text"].split())
    m_words = []
    for e in mismatched_ex:
        m_words.extend(e["text"].split())
    a_counter = Counter(a_words)
    m_counter = Counter(m_words)
    multiset_match = a_counter == m_counter
    if not multiset_match:
        diff = sum((a_counter - m_counter).values()) + sum((m_counter - a_counter).values())
        print(f"  ⚠ Word multiset ALIGNED vs MISMATCHED: differ by {diff} tokens")
        print(f"    (boundary artifact from sentence-order-dependent 160-word packing)")
    else:
        print("  ✓ Word multiset ALIGNED == MISMATCHED")
    
    # Budget check
    max_w = max(s["total_words"] for s in stats.values())
    budget_ok = max_w <= TARGET_POOL_WORDS
    print(f"  {'✓' if budget_ok else '✗'} Max pool words: {max_w:,} (budget {TARGET_POOL_WORDS:,})")
    
    # All arms same size
    sizes = [s["total_words"] for s in stats.values()]
    same_size = len(set(sizes)) == 1
    print(f"  {'✓' if same_size else '⚠'} All arms same total words: {same_size} ({set(sizes)})")
    
    # Sample examples
    print(f"\n  ALIGNED example 0: '{aligned_ex[0]['text'][:120]}...'")
    print(f"  MISMATCHED example 0: '{mismatched_ex[0]['text'][:120]}...'")
    print(f"  SINGLE_ORIG example 0: '{single_orig_ex[0]['text'][:120]}...'")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    summary = {
        "status": "PAIRED_ALIGNMENT_MATERIALIZED",
        "design": {
            "principle": "Alignment visible within same 160-word sequence via self-attention",
            "scientist_requirement": "Compare aligned pairs in same sequence vs random mismatch of same text; separate repetition consolidation, rewrite diversity, and semantic coverage",
            "arms": {
                "aligned": "[orig_i][simp_i][orig_j][simp_j]... both views adjacent, visible to attention",
                "mismatched": "[orig_i][simp_k]... same sentences, scrambled pairing, no alignment",
                "single_orig": "Diverse originals only — max coverage, no pairing or repetition",
                "single_repeat": "Same originals 2x — repetition consolidation, no diversity"
            },
            "contrasts": {
                "aligned_vs_mismatched": "Pure alignment effect (critical control)",
                "aligned_vs_single_orig": "Multi-view alignment vs broad coverage",
                "aligned_vs_single_repeat": "Multi-view diversity vs repetition consolidation",
                "mismatched_vs_single_orig": "Effect of simplified text (unaligned) vs not"
            }
        },
        "pair_pool": {
            "total_pairs_available": len(all_pairs),
            "selected_pairs": len(selected),
            "selected_pair_words": sel_pw,
            "selected_orig_words": sel_ow,
            "selected_simp_words": sel_sw,
            "source_mix": dict(src_counts),
            "mean_pair_words": sel_pw / len(selected) if selected else 0,
            "mean_orig_words": sel_ow / len(selected) if selected else 0,
            "mean_simp_words": sel_sw / len(selected) if selected else 0
        },
        "arms": stats,
        "validation": {
            "word_multiset_aligned_eq_mismatched": multiset_match,
            "max_pool_words": max_w,
            "within_budget": budget_ok,
            "all_arms_same_words": same_size,
            "words_per_example": WORDS_PER_EXAMPLE,
            "n_examples_per_arm": n_examples_target
        },
        "provenance": {
            "wikilarge": {"source": "Nechba/wikilarge-text-simplification", "license": "Apache-2.0"},
            "syncse": {"source": "hkust-nlp/SynCSE-partial-NLI", "license": "MIT"},
            "seed": SEED,
            "target_pool_words": TARGET_POOL_WORDS
        },
        "elapsed_sec": round(elapsed, 1)
    }
    
    sp = OUTPUT_DIR / "screen_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary: {sp}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(json.dumps({"arms": stats, "validation": summary["validation"]}, indent=2))


if __name__ == "__main__":
    main()
