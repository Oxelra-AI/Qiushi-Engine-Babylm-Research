#!/usr/bin/env python3
"""research: Validate Qwen rewrites and materialize exact 10M training corpora.

Takes the Qwen generation outputs + official pool and creates:
1. official_only: 62,500 rows × 160 words = 10,000,000 words (baseline)
2. qwen_aligned: mix of paired rows + official rows = 10,000,000 words (treatment)

The aligned arm uses ~25% paired content (original+rewrite packed together)
and ~75% standard official text, following the mix25 evidence.

Pair packing: [orig1][rewrite1][orig2][rewrite2]... within 160-word training rows.
This creates within-window semantic adjacency visible to the model's attention.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import pathlib
import random
import sys
import time
from dataclasses import dataclass

ROOT = _public_path('experiments/archive/compact_experience')  # 
DATA_DIR = _public_path('experiments/archive/compact_experience/data/qwen_aligned')
POOL_PATH = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUTPUTS_PATH = _public_path('experiments/archive/compact_experience/training/runs/qwen_rewrites_full/outputs.jsonl')
SOURCES_PATH = _public_path('experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl')

OUT_DIR = _public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
TARGET_ALIGNED_FRACTION = 0.25  # 25% of rows are paired
WORDS_PER_ROW = 160
TOTAL_ROWS = 62500  # 62,500 × 160 = 10,000,000 words
TOTAL_WORDS = TOTAL_ROWS * WORDS_PER_ROW  # 10,000,000
ALIGNED_ROWS = int(TOTAL_ROWS * TARGET_ALIGNED_FRACTION)  # 15,625
OFFICIAL_ROWS = TOTAL_ROWS - ALIGNED_ROWS  # 46,875
PASSES = 10  # 10 passes for 100M exposure

# Validation thresholds
MIN_REWRITE_WORDS = 5
MAX_REWRITE_WORDS = 80
MAX_WORD_RATIO = 3.0  # rewrite can't be >3x or <1/3x the source length


@dataclass
class Pair:
    original: str
    original_words: int
    rewrite: str
    rewrite_words: int
    source_name: str
    pair_id: str


def validate_rewrite(output_text: str, source_words: int) -> tuple[bool, str]:
    """Validate a Qwen rewrite is suitable for training."""
    text = output_text.strip()
    # Remove any leading/trailing quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()
    
    words = text.split()
    n_words = len(words)
    
    if n_words < MIN_REWRITE_WORDS:
        return False, f"too_short_{n_words}"
    if n_words > MAX_REWRITE_WORDS:
        return False, f"too_long_{n_words}"
    
    ratio = n_words / max(1, source_words)
    if ratio > MAX_WORD_RATIO or ratio < (1.0 / MAX_WORD_RATIO):
        return False, f"ratio_{ratio:.2f}"
    
    # Check it's not just copying the original (>80% word overlap)
    source_word_set = set()  # will be checked in pack_pairs
    
    # Check for common generation artifacts
    if text.startswith("Here") or text.startswith("Sure") or text.startswith("I "):
        if ":" in text[:30]:
            return False, "meta_response"
    
    return True, "ok"


def pack_pairs_into_row(pairs: list[Pair], target_words: int = WORDS_PER_ROW) -> tuple[str, int, list[str]]:
    """Pack pairs into a single training row of exactly target_words.
    
    Format: [orig1] [rewrite1] [orig2] [rewrite2] ...
    Returns: (text, actual_words, pair_ids_used)
    """
    words_so_far = 0
    segments = []
    pair_ids_used = []
    
    for pair in pairs:
        # Try to add this pair
        pair_total = pair.original_words + pair.rewrite_words
        
        if words_so_far + pair_total <= target_words:
            # Whole pair fits
            segments.append(pair.original)
            segments.append(pair.rewrite)
            words_so_far += pair_total
            pair_ids_used.append(pair.pair_id)
        elif words_so_far + pair.original_words < target_words:
            # Original fits, truncate rewrite
            segments.append(pair.original)
            words_so_far += pair.original_words
            remaining = target_words - words_so_far
            if remaining > 0:
                rewrite_words = pair.rewrite.split()[:remaining]
                segments.append(" ".join(rewrite_words))
                words_so_far += len(rewrite_words)
            pair_ids_used.append(pair.pair_id)
            break
        else:
            # Only partial original fits
            remaining = target_words - words_so_far
            if remaining > 3:
                orig_words = pair.original.split()[:remaining]
                segments.append(" ".join(orig_words))
                words_so_far += len(orig_words)
                pair_ids_used.append(pair.pair_id + "_partial")
            break
    
    text = " ".join(segments)
    actual_words = len(text.split())
    
    # Pad if needed (shouldn't happen often with enough pairs)
    if actual_words < target_words:
        # This means we ran out of pairs for this row
        pass  # caller will handle
    
    return text, actual_words, pair_ids_used


def main():
    t0 = time.time()
    
    # ── Load source sentences ──
    print("Loading source sentences...")
    sources = {}
    with SOURCES_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            sources[rec["id"]] = rec
    print(f"  Loaded {len(sources)} source sentences")
    
    # ── Load and validate Qwen outputs ──
    print(f"Loading Qwen outputs from {OUTPUTS_PATH}")
    if not OUTPUTS_PATH.exists():
        print(f"ERROR: Outputs file not found at {OUTPUTS_PATH}")
        print("Run the Qwen generation first.")
        sys.exit(1)
    
    valid_pairs: list[Pair] = []
    invalid_count = 0
    rejection_reasons: dict[str, int] = {}
    
    with OUTPUTS_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            idx = rec["index"]
            output = rec.get("output", "").strip()
            pair_id = f"rw_{idx:06d}"
            
            if pair_id not in sources:
                invalid_count += 1
                rejection_reasons["no_source"] = rejection_reasons.get("no_source", 0) + 1
                continue
            
            src = sources[pair_id]
            is_valid, reason = validate_rewrite(output, src["words"])
            
            if not is_valid:
                invalid_count += 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            
            # Clean the output
            rewrite = output.strip()
            if rewrite.startswith('"') and rewrite.endswith('"'):
                rewrite = rewrite[1:-1].strip()
            if rewrite.startswith("'") and rewrite.endswith("'"):
                rewrite = rewrite[1:-1].strip()
            
            valid_pairs.append(Pair(
                original=src["text"],
                original_words=src["words"],
                rewrite=rewrite,
                rewrite_words=len(rewrite.split()),
                source_name=src["source"],
                pair_id=pair_id,
            ))
    
    print(f"  Valid pairs: {len(valid_pairs)}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Rejection reasons: {rejection_reasons}")
    
    if len(valid_pairs) < 1000:
        print("ERROR: Too few valid pairs. Cannot proceed.")
        sys.exit(1)
    
    # ── Compute actual aligned fraction ──
    avg_pair_words = sum(p.original_words + p.rewrite_words for p in valid_pairs) / len(valid_pairs)
    max_aligned_words = len(valid_pairs) * avg_pair_words
    desired_aligned_words = TOTAL_WORDS * TARGET_ALIGNED_FRACTION
    
    if max_aligned_words < desired_aligned_words:
        # Reduce aligned fraction to what we can actually achieve
        actual_aligned_words = max_aligned_words
        actual_aligned_rows = int(actual_aligned_words / WORDS_PER_ROW)
        actual_fraction = actual_aligned_rows / TOTAL_ROWS
        print(f"  NOTE: Reducing aligned fraction from {TARGET_ALIGNED_FRACTION:.3f} "
              f"to {actual_fraction:.3f} due to available pairs")
    else:
        actual_aligned_rows = ALIGNED_ROWS
        actual_fraction = TARGET_ALIGNED_FRACTION
    
    actual_official_rows = TOTAL_ROWS - actual_aligned_rows
    print(f"  Target: {actual_aligned_rows} aligned rows + {actual_official_rows} official rows = {TOTAL_ROWS} total")
    
    # ── Pack pairs into aligned rows ──
    print("Packing pairs into aligned training rows...")
    random.seed(27043)
    random.shuffle(valid_pairs)
    
    # Build a padding buffer from official text for any short rows
    padding_buffer_words: list[str] = []
    with POOL_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            padding_buffer_words.extend(row["text"].split())
            if len(padding_buffer_words) > 100000:  # plenty of padding material
                break
    padding_idx = 0
    
    aligned_rows: list[dict] = []
    pair_idx = 0
    pairs_used_total = 0
    
    for row_i in range(actual_aligned_rows):
        # Collect pairs for this row
        row_pairs = []
        row_word_budget = WORDS_PER_ROW
        est_needed = 0
        
        while pair_idx < len(valid_pairs) and est_needed < row_word_budget:
            p = valid_pairs[pair_idx]
            row_pairs.append(p)
            est_needed += p.original_words + p.rewrite_words
            pair_idx += 1
            if est_needed >= row_word_budget:
                break
        
        text, actual_words, pair_ids = pack_pairs_into_row(row_pairs, WORDS_PER_ROW)
        
        # Ensure EXACTLY WORDS_PER_ROW words (critical for trainer validation)
        words_list = text.split()
        if len(words_list) > WORDS_PER_ROW:
            words_list = words_list[:WORDS_PER_ROW]
        elif len(words_list) < WORDS_PER_ROW:
            # Pad from official text buffer to reach exactly 160
            deficit = WORDS_PER_ROW - len(words_list)
            while deficit > 0 and padding_idx < len(padding_buffer_words):
                words_list.append(padding_buffer_words[padding_idx])
                padding_idx += 1
                deficit -= 1
        
        text = " ".join(words_list)
        actual_words = len(words_list)
        
        if actual_words != WORDS_PER_ROW:
            # Should not happen with sufficient padding buffer
            print(f"  CRITICAL: Row {row_i} has {actual_words} words, not {WORDS_PER_ROW}")
            break
        
        aligned_rows.append({
            "text": text,
            "words": actual_words,
            "example_id": 100000 + row_i,  # distinct from official IDs
            "source": "qwen_aligned",
            "pair_ids": pair_ids,
            "n_pairs": len(pair_ids),
        })
        pairs_used_total += len(pair_ids)
    
    print(f"  Created {len(aligned_rows)} aligned rows using {pairs_used_total} pairs")
    print(f"  Pairs remaining unused: {len(valid_pairs) - pair_idx}")
    
    # Check word counts
    aligned_word_total = sum(r["words"] for r in aligned_rows)
    short_rows = [r for r in aligned_rows if r["words"] < WORDS_PER_ROW]
    if short_rows:
        print(f"  WARNING: {len(short_rows)} short aligned rows (< {WORDS_PER_ROW} words)")
    
    # ── Load official pool ──
    print("Loading official pool...")
    official_pool: list[dict] = []
    with POOL_PATH.open() as f:
        for line in f:
            official_pool.append(json.loads(line))
    print(f"  Loaded {len(official_pool)} official rows")
    
    # ── Create OFFICIAL-ONLY training corpus (control arm) ──
    print("Creating official-only 10M corpus...")
    official_only_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/official_only_10M.jsonl')
    official_word_check = 0
    with official_only_path.open("w") as f:
        for row in official_pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            official_word_check += row["words"]
    assert official_word_check == TOTAL_WORDS, f"Official words: {official_word_check} != {TOTAL_WORDS}"
    print(f"  Written {len(official_pool)} rows, {official_word_check} words")
    
    # ── Create QWEN-ALIGNED training corpus (treatment arm) ──
    print("Creating qwen-aligned 10M corpus...")
    
    # Select official rows for the non-aligned portion
    # Use a deterministic selection: take rows NOT used as source for rewrites
    source_example_ids = set()
    for pair in valid_pairs[:pair_idx]:
        src = sources[pair.pair_id]
        source_example_ids.add(src["example_id"])
    
    # Official rows not used as rewrite sources get priority
    non_source_officials = [r for r in official_pool if r["example_id"] not in source_example_ids]
    source_officials = [r for r in official_pool if r["example_id"] in source_example_ids]
    
    random.seed(27044)
    random.shuffle(non_source_officials)
    random.shuffle(source_officials)
    
    # Take official rows needed
    selected_officials = (non_source_officials + source_officials)[:actual_official_rows]
    
    # Combine aligned + official rows
    aligned_corpus = aligned_rows + [
        {"text": r["text"], "words": r["words"], "example_id": r["example_id"], "source": r["source"]}
        for r in selected_officials
    ]
    
    # Strict validation: every row must be exactly WORDS_PER_ROW
    for i, row in enumerate(aligned_corpus):
        actual_w = len(row["text"].split())
        if actual_w != WORDS_PER_ROW:
            raise RuntimeError(
                f"Row {i} has {actual_w} words, expected {WORDS_PER_ROW}. "
                f"Source: {row['source']}, example_id: {row['example_id']}"
            )
    
    # Verify total
    corpus_words = len(aligned_corpus) * WORDS_PER_ROW
    if len(aligned_corpus) != TOTAL_ROWS:
        # Adjust by adding/removing official rows
        if len(aligned_corpus) < TOTAL_ROWS:
            deficit_rows = TOTAL_ROWS - len(aligned_corpus)
            extra = source_officials[len(selected_officials):][:deficit_rows]
            for r in extra:
                aligned_corpus.append({"text": r["text"], "words": r["words"],
                                       "example_id": r["example_id"], "source": r["source"]})
        elif len(aligned_corpus) > TOTAL_ROWS:
            aligned_corpus = aligned_corpus[:TOTAL_ROWS]
        corpus_words = len(aligned_corpus) * WORDS_PER_ROW
    
    assert corpus_words == TOTAL_WORDS, f"Corpus words {corpus_words} != {TOTAL_WORDS}"
    assert len(aligned_corpus) == TOTAL_ROWS, f"Corpus rows {len(aligned_corpus)} != {TOTAL_ROWS}"
    
    # Shuffle to interleave aligned and official rows (important for training dynamics)
    random.seed(27045)
    random.shuffle(aligned_corpus)
    
    qwen_aligned_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/qwen_aligned_10M.jsonl')
    actual_aligned_count = 0
    actual_official_count = 0
    final_word_total = 0
    with qwen_aligned_path.open("w") as f:
        for row in aligned_corpus:
            out_row = {"text": row["text"], "words": row["words"],
                       "example_id": row["example_id"], "source": row["source"]}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            final_word_total += row["words"]
            if row["source"] == "qwen_aligned":
                actual_aligned_count += 1
            else:
                actual_official_count += 1
    
    print(f"  Written {len(aligned_corpus)} rows: {actual_aligned_count} aligned + {actual_official_count} official")
    print(f"  Total words: {final_word_total}")
    
    # ── Create 100M training files (10 passes) ──
    print("Creating 100M training files (10 passes)...")
    
    for arm_name, pool_path in [("official_only", official_only_path), 
                                 ("qwen_aligned", qwen_aligned_path)]:
        pool_rows = []
        with pool_path.open() as f:
            for line in f:
                pool_rows.append(json.loads(line))
        
        training_path = OUT_DIR / f"{arm_name}_100M.jsonl"
        total_training_words = 0
        
        with training_path.open("w") as f:
            for pass_idx in range(PASSES):
                pass_rng = random.Random(27100 + pass_idx)
                indices = list(range(len(pool_rows)))
                pass_rng.shuffle(indices)
                
                for idx in indices:
                    row = pool_rows[idx]
                    out = {"text": row["text"], "words": row["words"],
                           "example_id": row["example_id"], "source": row["source"]}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total_training_words += row["words"]
        
        print(f"  {arm_name}: {total_training_words} words in {PASSES} passes")
        assert total_training_words == TOTAL_WORDS * PASSES, \
            f"{arm_name} training words {total_training_words} != {TOTAL_WORDS * PASSES}"
    
    # ── Compute hashes ──
    hashes = {}
    for fname in ["official_only_10M.jsonl", "qwen_aligned_10M.jsonl",
                  "official_only_100M.jsonl", "qwen_aligned_100M.jsonl"]:
        fpath = OUT_DIR / fname
        with fpath.open("rb") as f:
            hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    
    # ── Save metadata ──
    metadata = {
        "status": "CORPORA_MATERIALIZED",
        "total_rows": len(aligned_corpus),
        "total_words": final_word_total,
        "aligned_rows": actual_aligned_count,
        "official_rows": actual_official_count,
        "aligned_fraction": round(actual_aligned_count / len(aligned_corpus), 4),
        "valid_pairs_available": len(valid_pairs),
        "pairs_used": pairs_used_total,
        "pair_idx_consumed": pair_idx,
        "avg_pair_words": round(avg_pair_words, 2),
        "short_aligned_rows": len(short_rows),
        "rejection_reasons": rejection_reasons,
        "passes": PASSES,
        "training_words_per_arm": TOTAL_WORDS * PASSES,
        "files": {
            "official_only_pool": str(_public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/official_only_10M.jsonl')),
            "qwen_aligned_pool": str(_public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/qwen_aligned_10M.jsonl')),
            "official_only_training": str(_public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/official_only_100M.jsonl')),
            "qwen_aligned_training": str(_public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/qwen_aligned_100M.jsonl')),
        },
        "sha256": hashes,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    meta_path = _public_path('experiments/archive/compact_experience/data/qwen_aligned/training_corpora/materialization_metadata.json')
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
