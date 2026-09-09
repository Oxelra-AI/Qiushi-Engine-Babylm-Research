#!/usr/bin/env python3
"""research: Pack structured-experience triplets into official-matched training geometry.

The research corpus stored each SynCSE triplet as its own short document (median 7
words), producing 632,117 documents. Under the document-level loader this yields
2470 optimizer steps vs the official arm's 245 steps at the same 9,999,969-word
budget. That update-count mismatch breaks the strict 2x2 factorial.

This builder packs triplets into ~160-word examples (matching official
words_per_example=160) while preserving within-triplet structure:
  - Each triplet's 3 sentences (anchor + paraphrase + contradiction) stay adjacent
    and are never split across examples.
  - Triplets are greedily concatenated until adding the next would exceed 160 words;
    then a new example starts. A triplet larger than 160 words becomes its own example.
  - Official sources are also packed into 160-word examples the SAME way the official
    trainer does (iter_examples packs word-stream into fixed 160-word chunks), so both
    arms share example count and update trajectory as closely as the word budget allows.

Output: JSONL with {"text","words","source"} per example, exact-order consumable by
the trainer's --example_jsonl path (which forbids partial-example selection).
Total words held at exactly 9,999,969 to match the official arm budget.
"""
from __future__ import annotations
import json, pathlib, random

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RAW_DIR = ROOT / "data/reconstruct_tmp/raw_dataset"
SRC_JSONL = ROOT / "data/structured_experience_corpus/structured_experience_10M.jsonl"
SRC_LEDGER = ROOT / "data/structured_experience_corpus/source_ledger.json"
OUT_DIR = ROOT / "data/structured_packed_corpus"
OUT_JSONL = OUT_DIR / "structured_experience_packed_10M.jsonl"
LEDGER = OUT_DIR / "packing_ledger.json"

WORDS_PER_EXAMPLE = 160
TARGET_WORDS = 9999969  # exact match to official matched arm
SEED = 42

KEEP_SOURCES = ["bnc_spoken.train.txt", "gutenberg.train.txt",
                "open_subtitles.train.txt", "simple_wiki.train.txt",
                "switchboard.train.txt"]


def count_words(text: str) -> int:
    return len(text.split())


def load_triplet_docs() -> tuple[list[str], list[str]]:
    """Reload triplets and official docs, separated, from the research build.

    We re-derive the exact same content: triplets are the syncscratch documents
    from research (identifiable because official docs came from the .train.txt files).
    Rather than guess, we rebuild from raw sources + SynCSE to keep provenance exact.
    """
    # Rebuild triplets from SynCSE-scratch with the SAME seed/budget as research
    import os
    os.environ["HF_DATASETS_CACHE"] = str((ROOT / "training/hf_home/datasets").resolve())
    from datasets import load_dataset
    ledger = json.loads(SRC_LEDGER.read_text())
    childes_words = ledger["childes_words_replaced"]  # 2841101
    ds = load_dataset("sjtu-lit/SynCSE-scratch-NLI", split="train")
    indices = list(range(len(ds)))
    random.Random(42).shuffle(indices)
    triplets = []
    tw = 0
    for idx in indices:
        row = ds[idx]
        t = f"{row['sent0'].strip()} {row['sent1'].strip()} {row['nli_hard'].strip()}"
        w = count_words(t)
        if tw + w > childes_words:
            break
        triplets.append(t)
        tw += w
    return triplets, []


def pack_word_stream(words: list[str], words_per_example: int) -> list[str]:
    """Pack a flat word stream into fixed-size examples (official iter_examples behavior)."""
    examples = []
    for i in range(0, len(words), words_per_example):
        chunk = words[i:i+words_per_example]
        if chunk:
            examples.append(" ".join(chunk))
    return examples


def pack_triplets(triplets: list[str], words_per_example: int) -> list[dict]:
    """Greedily pack triplets into ~words_per_example examples, never splitting a triplet."""
    examples = []
    buf = []
    buf_words = 0
    for t in triplets:
        w = count_words(t)
        if buf and buf_words + w > words_per_example:
            examples.append({"text": " ".join(buf), "words": buf_words, "source": "syncscratch_packed"})
            buf = []
            buf_words = 0
        buf.append(t)
        buf_words += w
    if buf:
        examples.append({"text": " ".join(buf), "words": buf_words, "source": "syncscratch_packed"})
    return examples


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Triplets (structured replacement for CHILDES)
    triplets, _ = load_triplet_docs()
    triplet_words = sum(count_words(t) for t in triplets)
    triplet_examples = pack_triplets(triplets, WORDS_PER_EXAMPLE)
    print(f"Triplets: {len(triplets)}, words={triplet_words}, packed_examples={len(triplet_examples)}")

    # 2. Official kept sources packed into 160-word chunks (same as official arm)
    official_examples = []
    source_words = {}
    for fname in KEEP_SOURCES:
        words = []
        for line in (RAW_DIR / fname).read_text(encoding="utf-8", errors="replace").splitlines():
            words.extend(line.split())
        source_words[fname] = len(words)
        official_examples.extend([{"text": e, "words": count_words(e), "source": fname}
                                  for e in pack_word_stream(words, WORDS_PER_EXAMPLE)])
    official_words = sum(source_words.values())
    print(f"Official kept: words={official_words}, packed_examples={len(official_examples)}")

    # 3. Combine, shuffle at example level (preserves within-example/within-triplet structure)
    all_examples = official_examples + triplet_examples
    random.Random(SEED).shuffle(all_examples)

    # 4. Trim to exactly TARGET_WORDS without splitting an example: drop/trim last example
    total = 0
    final = []
    for ex in all_examples:
        if total + ex["words"] <= TARGET_WORDS:
            final.append(ex)
            total += ex["words"]
        else:
            need = TARGET_WORDS - total
            if need > 0:
                w = ex["text"].split()[:need]
                final.append({"text": " ".join(w), "words": need, "source": ex["source"] + "_trimmed"})
                total += need
            break
    # If we ran out before target (shouldn't happen), report
    print(f"Packed total words={total} (target {TARGET_WORDS}), examples={len(final)}")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps({"text": ex["text"], "words": ex["words"], "source": ex["source"]}, ensure_ascii=False) + "\n")

    est_steps_batch256 = (len(final) + 255) // 256
    ledger = {
        "status": "PACKED_CORPUS_BUILT",
        "output": str(OUT_JSONL),
        "words_per_example": WORDS_PER_EXAMPLE,
        "total_words": total,
        "total_examples": len(final),
        "estimated_steps_batch256": est_steps_batch256,
        "official_arm_steps_reference": 245,
        "triplet_words": triplet_words,
        "triplet_packed_examples": len(triplet_examples),
        "official_words": official_words,
        "source_word_counts": source_words,
        "design": "Pack SynCSE triplets into 160-word examples (never splitting a triplet), "
                  "and pack official kept sources into 160-word examples identically, so the "
                  "structured arm matches the official arm's example count and update trajectory "
                  "at the same 9,999,969-word budget. Within-triplet adjacency preserved.",
        "seed": SEED,
        "source_corpus": str(SRC_JSONL),
        "compliance": {"total_words_leq_10M": total <= 10_000_000},
    }
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: ledger[k] for k in ["total_words","total_examples","estimated_steps_batch256","official_arm_steps_reference","triplet_packed_examples"]}, indent=2))


if __name__ == "__main__":
    main()
