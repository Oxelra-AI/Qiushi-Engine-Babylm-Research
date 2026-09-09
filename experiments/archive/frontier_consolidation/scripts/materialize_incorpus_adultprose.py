#!/usr/bin/env python3
"""research: materialize in-corpus adult-prose arm at rho≈0.0424.

Scientific role
---------------
This arm fills the missing 2x2 cell: it replaces developmental/speech rows with
UNUSED Gutenberg/SimpleWiki text from the official BabyLM corpus, with NO FineWeb
content.  If this arm reproduces the broad exEntity gain seen in FineWeb arms,
the mechanism is target-distribution proximity (adding adult prose that is closer
to evaluation distributions).  If it is flat, the persistent term comes from
distinct out-of-corpus content or novelty.

Design
------
- rho ≈ 0.0424, matching full_1x (3,005 active rows, ~423,559 FineWeb words)
- Uses MAX 65,313-row geometry (same as all other arms)
- Replaces developmental/speech rows with surplus official adult-prose rows
- Preserves ALL qwen_pair_packed rows
- Preserves exact 10,000,000-word pool, 100,000,000-word stream
- Uses identical research legal tokenizer and DeBERTa config
- Seed 43022 for direct comparison
- Row positions matched to full_1x changed positions where possible

Source of surplus: official_only_10M.jsonl Gutenberg/SimpleWiki rows that are NOT
in the qwen_aligned_10M base pool (text-deduplicated).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY

# Canonical base pool (with Qwen pairs)
BASE_POOL = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
# Official-only pool (all BabyLM text, no Qwen)
OFFICIAL_ONLY = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl"
# MAX geometry clean pool
MAX_CLEAN = WS / "data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
# Full_1x pool metadata for row positions
FULL_1X_POOL = WS / "data/subdose_ladder_maxgeom_pools/subdose_full_1x_view_10M.jsonl"

OUT_DIR = WS / "data/incorpus_adultprose_arm"
TOTAL_WORDS = 10_000_000
TOTAL_ROWS = 65_313
EPOCHS = 10
ADULT_SOURCES = {"gutenberg", "simple_wiki"}
DEV_SPEECH_SOURCES = {"childes", "open_subtitles", "bnc_spoken", "switchboard"}

# Target: match full_1x at rho≈0.0424
TARGET_CHANGED_WORDS = 423_559  # full_1x pair words
TARGET_CHANGED_ROWS = 3_005     # full_1x active rows


def normalize_src(s: str) -> str:
    return s.split("::", 1)[1] if "::" in s else s


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── research: Load MAX-geometry clean pool ──
    print("Loading MAX-geometry clean pool...")
    if not MAX_CLEAN.exists():
        raise FileNotFoundError(MAX_CLEAN)
    max_clean_rows = load_jsonl(MAX_CLEAN)
    assert len(max_clean_rows) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(max_clean_rows)}"
    total_w = sum(r["words"] for r in max_clean_rows)
    assert total_w == TOTAL_WORDS, f"Expected {TOTAL_WORDS} words, got {total_w}"
    print(f"  Loaded {len(max_clean_rows)} rows, {total_w} words")

    # Build text-hash index for deduplication
    base_text_hashes = set()
    for r in max_clean_rows:
        base_text_hashes.add(text_hash(r["text"]))

    # ── research: Load full_1x pool to get changed-row positions ──
    print("Loading full_1x pool for position matching...")
    full_1x_rows = load_jsonl(FULL_1X_POOL)
    assert len(full_1x_rows) == TOTAL_ROWS

    # Identify which rows changed in full_1x vs clean
    full_1x_changed_indices = set()
    for i, (fr, cr) in enumerate(zip(full_1x_rows, max_clean_rows)):
        if fr["text"] != cr["text"]:
            full_1x_changed_indices.add(i)
    print(f"  full_1x has {len(full_1x_changed_indices)} changed rows")

    # ── research: Identify candidate replacement positions ──
    # We want to replace developmental/speech rows with adult-prose rows
    # Use the same positions as full_1x when possible
    candidate_positions = []
    protected_positions = set()  # qwen_pair_packed rows
    for i, r in enumerate(max_clean_rows):
        src = normalize_src(r["source"])
        if src == "qwen_pair_packed":
            protected_positions.add(i)
        elif src in DEV_SPEECH_SOURCES:
            candidate_positions.append(i)

    print(f"  Protected (qwen) positions: {len(protected_positions)}")
    print(f"  Candidate dev/speech positions: {len(candidate_positions)}")

    # Prefer positions that were also changed in full_1x
    positions_in_both = [i for i in candidate_positions if i in full_1x_changed_indices]
    positions_only_here = [i for i in candidate_positions if i not in full_1x_changed_indices]

    # Select TARGET_CHANGED_ROWS positions, preferring shared positions
    rng = random.Random(43022)  # Deterministic
    selected_positions = []
    if len(positions_in_both) >= TARGET_CHANGED_ROWS:
        selected_positions = sorted(rng.sample(positions_in_both, TARGET_CHANGED_ROWS))
    else:
        selected_positions = list(positions_in_both)
        remaining = TARGET_CHANGED_ROWS - len(selected_positions)
        selected_positions += rng.sample(positions_only_here, remaining)
        selected_positions.sort()

    print(f"  Selected {len(selected_positions)} replacement positions")
    print(f"    From full_1x overlap: {sum(1 for i in selected_positions if i in full_1x_changed_indices)}")
    print(f"    Additional dev/speech: {sum(1 for i in selected_positions if i not in full_1x_changed_indices)}")

    # ── research: Load surplus adult-prose rows from official_only ──
    print("Loading surplus adult-prose rows from official_only...")
    surplus_rows = []
    with open(OFFICIAL_ONLY) as f:
        for line in f:
            row = json.loads(line)
            src = normalize_src(row["source"])
            if src not in ADULT_SOURCES:
                continue
            # Deduplicate against base pool
            th = text_hash(row["text"])
            if th in base_text_hashes:
                continue
            surplus_rows.append({
                "text": row["text"],
                "words": row["words"],
                "source": f"incorpus_adultprose::{src}",
                "original_source": src,
            })

    print(f"  Found {len(surplus_rows)} surplus adult-prose rows, {sum(r['words'] for r in surplus_rows)} words")

    # ── research: Select surplus rows to match target word count ──
    # We need exactly the right number of words to maintain 10M total
    # Selected positions have specific word counts; we need replacement rows
    # with the SAME word counts to maintain geometry
    selected_word_counts = [max_clean_rows[i]["words"] for i in selected_positions]
    total_selected_words = sum(selected_word_counts)
    print(f"  Words to replace: {total_selected_words}")

    # Build word-count index of surplus rows
    surplus_by_wordcount: dict[int, list[dict]] = collections.defaultdict(list)
    for r in surplus_rows:
        surplus_by_wordcount[r["words"]].append(r)

    # Shuffle each bucket deterministically
    for wc in surplus_by_wordcount:
        rng.shuffle(surplus_by_wordcount[wc])

    # Match by word count to preserve geometry exactly
    replacement_rows = []
    unmatched = 0
    surplus_used_indices: dict[int, int] = {}  # word_count -> next index to use

    for wc in selected_word_counts:
        available = surplus_by_wordcount.get(wc, [])
        idx = surplus_used_indices.get(wc, 0)
        if idx < len(available):
            replacement_rows.append(available[idx])
            surplus_used_indices[wc] = idx + 1
        else:
            unmatched += 1
            replacement_rows.append(None)

    matched = sum(1 for r in replacement_rows if r is not None)
    print(f"  Matched by word count: {matched}/{len(selected_positions)}")
    if unmatched > 0:
        print(f"  WARNING: {unmatched} positions unmatched — adjusting selection")

    # For unmatched positions, find the closest available word count
    for j, (pos, repl) in enumerate(zip(selected_positions, replacement_rows)):
        if repl is not None:
            continue
        target_wc = max_clean_rows[pos]["words"]
        # Find nearest available word count
        best_wc = None
        best_dist = float("inf")
        for wc, available in surplus_by_wordcount.items():
            idx = surplus_used_indices.get(wc, 0)
            if idx < len(available):
                dist = abs(wc - target_wc)
                if dist < best_dist:
                    best_dist = dist
                    best_wc = wc
        if best_wc is not None:
            idx = surplus_used_indices.get(best_wc, 0)
            replacement_rows[j] = surplus_by_wordcount[best_wc][idx]
            surplus_used_indices[best_wc] = idx + 1
            if best_wc != target_wc:
                print(f"    Position {pos}: target {target_wc}w, used {best_wc}w (diff {best_wc - target_wc})")

    final_matched = sum(1 for r in replacement_rows if r is not None)
    if final_matched < len(selected_positions):
        print(f"  ERROR: Only matched {final_matched}/{len(selected_positions)} — insufficient surplus")
        sys.exit(1)

    # ── research: Build the modified pool ──
    print("Building modified pool...")
    new_pool = list(max_clean_rows)  # copy
    position_to_replacement = dict(zip(selected_positions, replacement_rows))

    replaced_words = 0
    new_words = 0
    source_summary = collections.Counter()

    for pos, repl in position_to_replacement.items():
        old_row = new_pool[pos]
        replaced_words += old_row["words"]
        new_row = {
            "text": repl["text"],
            "words": repl["words"],
            "example_id": old_row.get("example_id", pos + 900000),
            "source": repl["source"],
        }
        new_pool[pos] = new_row
        new_words += repl["words"]
        source_summary[repl["original_source"]] += repl["words"]

    new_total = sum(r["words"] for r in new_pool)
    word_diff = new_total - TOTAL_WORDS

    print(f"  Replaced {replaced_words} dev/speech words with {new_words} adult-prose words")
    print(f"  Word difference: {word_diff} (target: 0)")
    for s in sorted(source_summary):
        print(f"    {s}: {source_summary[s]} words")

    # If word difference is nonzero, adjust by finding a swap that fixes it
    if word_diff != 0:
        print(f"  Adjusting word count by {word_diff}...")
        # Find an unchanged row that can be swapped with a different-size surplus row
        # to make the total exact
        adjusted = False
        for i, r in enumerate(new_pool):
            if i in position_to_replacement or i in protected_positions:
                continue
            src = normalize_src(r["source"])
            if src not in DEV_SPEECH_SOURCES:
                continue
            target_replacement_wc = r["words"] - word_diff
            if target_replacement_wc <= 0:
                continue
            available = surplus_by_wordcount.get(target_replacement_wc, [])
            idx = surplus_used_indices.get(target_replacement_wc, 0)
            if idx < len(available):
                old_wc = r["words"]
                new_repl = available[idx]
                surplus_used_indices[target_replacement_wc] = idx + 1
                new_pool[i] = {
                    "text": new_repl["text"],
                    "words": new_repl["words"],
                    "example_id": r.get("example_id", i + 900000),
                    "source": new_repl["source"],
                }
                print(f"    Swapped row {i} ({old_wc}w -> {new_repl['words']}w) to fix word count")
                adjusted = True
                break
        if not adjusted:
            print(f"  WARNING: Could not exactly fix word count; diff = {word_diff}")

    final_total = sum(r["words"] for r in new_pool)
    print(f"  Final pool: {len(new_pool)} rows, {final_total} words")

    # ── research: Write pool and stream ──
    pool_path = OUT_DIR / "incorpus_adultprose_rho0p042_10M.jsonl"
    with open(pool_path, "w") as f:
        for r in new_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool_hash = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    print(f"  Pool: {pool_path.name}, SHA256: {pool_hash}")

    # 100M stream = 10x concatenation
    stream_path = OUT_DIR / "incorpus_adultprose_rho0p042_100M.jsonl"
    with open(stream_path, "w") as f:
        pool_text = pool_path.read_text()
        for _ in range(EPOCHS):
            f.write(pool_text)
    stream_hash = hashlib.sha256(stream_path.read_bytes()).hexdigest()
    stream_words = final_total * EPOCHS
    print(f"  Stream: {stream_path.name}, {stream_words} words, SHA256: {stream_hash}")

    # ── research: Audit ──
    changed_positions = set()
    changed_src_words = collections.Counter()
    unchanged_src_words = collections.Counter()
    for i, (new_r, old_r) in enumerate(zip(new_pool, max_clean_rows)):
        if new_r["text"] != old_r["text"]:
            changed_positions.add(i)
            changed_src_words[normalize_src(new_r["source"])] += new_r["words"]
        else:
            unchanged_src_words[normalize_src(new_r["source"])] += new_r["words"]

    # Position matching with full_1x
    position_overlap = len(changed_positions & full_1x_changed_indices)
    position_only_incorpus = len(changed_positions - full_1x_changed_indices)
    position_only_full1x = len(full_1x_changed_indices - changed_positions)
    row_index_mean = statistics.mean(changed_positions) if changed_positions else 0
    full1x_index_mean = statistics.mean(full_1x_changed_indices) if full_1x_changed_indices else 0

    rho = sum(new_r["words"] for i, new_r in enumerate(new_pool) if i in changed_positions) / TOTAL_WORDS

    # Check no FineWeb content
    for r in new_pool:
        src = normalize_src(r["source"])
        if "fineweb" in src.lower() or "compact_view" in src.lower():
            raise ValueError(f"FineWeb content found at source={src}")

    # Check qwen protection
    qwen_preserved = 0
    for i in protected_positions:
        if new_pool[i]["text"] == max_clean_rows[i]["text"]:
            qwen_preserved += 1
    assert qwen_preserved == len(protected_positions), f"Qwen rows modified: {len(protected_positions) - qwen_preserved}"

    elapsed = round(time.time() - t0, 1)

    metadata = {
        "status": "INCORPUS_ADULTPROSE_ARM_MATERIALIZED",
        "created_utc": now(),
        "arm_type": "in_corpus_adult_prose_surplus",
        "description": "Replaces dev/speech rows with unused Gutenberg/SimpleWiki from BabyLM official corpus. No FineWeb.",
        "rho": round(rho, 6),
        "pool_rows": len(new_pool),
        "pool_words": final_total,
        "stream_words": stream_words,
        "epochs": EPOCHS,
        "changed_positions": len(changed_positions),
        "changed_words": sum(changed_src_words.values()),
        "changed_source_words": dict(changed_src_words),
        "unchanged_source_words": dict(unchanged_src_words),
        "position_overlap_with_full_1x": position_overlap,
        "position_only_incorpus": position_only_incorpus,
        "position_only_full1x": position_only_full1x,
        "row_index_mean_incorpus": round(row_index_mean, 2),
        "row_index_mean_full1x": round(full1x_index_mean, 2),
        "qwen_rows_preserved": qwen_preserved,
        "pool_sha256": pool_hash,
        "stream_sha256": stream_hash,
        "pool_path": str(pool_path.relative_to(ROOT)),
        "stream_path": str(stream_path.relative_to(ROOT)),
        "no_fineweb_content": True,
        "no_gpu_training_eval_upload": True,
        "elapsed_sec": elapsed,
    }

    meta_path = OUT_DIR / "incorpus_adultprose_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    # Summary markdown
    summary = f"""# research in-corpus adult-prose arm

No FineWeb content. Replaces developmental/speech rows with unused official
Gutenberg/SimpleWiki text at rho≈{rho:.4f}, matching full_1x geometry.

## Key numbers
- Pool: {len(new_pool)} rows, {final_total} words
- Stream: {stream_words} words ({EPOCHS} epochs)
- Changed rows: {len(changed_positions)} ({rho*100:.2f}% of pool words)
- Changed source: {dict(changed_src_words)}
- Position overlap with full_1x: {position_overlap}/{len(full_1x_changed_indices)}
- Row-index mean: incorpus {row_index_mean:.1f}, full_1x {full1x_index_mean:.1f}
- Qwen preserved: {qwen_preserved}/{len(protected_positions)}
- Pool SHA256: {pool_hash}
- Stream SHA256: {stream_hash}

## Scientific purpose
Tests whether broad ex-Entity competence gain requires out-of-corpus FineWeb
content, or whether in-corpus adult prose (Gutenberg/SimpleWiki) suffices.

If incorpus ≈ full_1x (view): target-proximity mechanism.
If incorpus ≈ clean: distinct-content/novelty mechanism.
"""
    (OUT_DIR / "incorpus_adultprose_summary.md").write_text(summary)

    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
