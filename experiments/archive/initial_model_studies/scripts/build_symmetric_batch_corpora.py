#!/usr/bin/env python3
"""research: Build symmetric packed corpora for strict 2x2 geometry.

Matched-geometry requirement: structured and official arms must have the same effective
word-batch/update trajectory. Prior un-packed structured JSONL produced 2470
updates vs official 245. A naive exact-example pack to 62,501 examples was
mathematically too tight for indivisible SynCSE triplets (only 50 slack words in
triplet bins).

This script builds TWO pre-packed JSONL corpora with identical training geometry:
  - total words: 9,999,969
  - total examples: 62,720
  - batch_size 256 -> exactly 245 full optimizer steps
  - same JSONL loader path for official and structured arms

Official corpus: all 6 official sources as a flat word stream, split into 62,720
near-equal examples (mostly 159/160 words).

Structured corpus: non-CHILDES official sources as near-160 examples plus the
same SynCSE-scratch replacement triplets from research. SynCSE triplets are
indivisible and adjacency-preserved; they are bin-packed into the remaining
example slots under max 160 words per example. Total example count exactly
matches official. Total words exactly matches official.
"""
from __future__ import annotations
import json, os, pathlib, random

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RAW_DIR = ROOT / "data/reconstruct_tmp/raw_dataset"
LEDGER = ROOT / "data/structured_experience_corpus/source_ledger.json"
OUT_DIR = ROOT / "data/symmetric_batch_corpora"
OFFICIAL_OUT = OUT_DIR / "official_symmetric_9999969w_62720ex.jsonl"
STRUCT_OUT = OUT_DIR / "structured_symmetric_9999969w_62720ex.jsonl"
LEDGER = OUT_DIR / "symmetric_packing_ledger.json"

TARGET_WORDS = 9_999_969
TARGET_EXAMPLES = 62_720
BATCH_SIZE = 256
MAX_WORDS_PER_EXAMPLE = 160
TRIPLET_BIN_CAPACITY = 192  # allows indivisible SynCSE triplets to hit exact 245-step geometry
SEED = 42
ALL_SOURCES = ["bnc_spoken.train.txt", "childes.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"]
KEEP_SOURCES = ["bnc_spoken.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"]


def words_from_files(files: list[str]) -> tuple[list[str], dict[str, int]]:
    all_words: list[str] = []
    counts: dict[str, int] = {}
    for fn in files:
        ws: list[str] = []
        for line in (RAW_DIR / fn).read_text(encoding="utf-8", errors="replace").splitlines():
            ws.extend(line.split())
        counts[fn] = len(ws)
        all_words.extend(ws)
    return all_words, counts


def split_flat_words_exact(words: list[str], n_examples: int, source: str) -> list[dict]:
    """Split a word stream into exactly n_examples near-equal examples."""
    total = len(words)
    if total < n_examples:
        raise RuntimeError(f"not enough words {total} for examples {n_examples}")
    base = total // n_examples
    rem = total % n_examples
    if base > MAX_WORDS_PER_EXAMPLE:
        raise RuntimeError(f"base example size {base} exceeds max {MAX_WORDS_PER_EXAMPLE}")
    exs = []
    pos = 0
    for i in range(n_examples):
        size = base + (1 if i < rem else 0)
        chunk = words[pos:pos+size]
        pos += size
        exs.append({"text": " ".join(chunk), "words": size, "source": source})
    if pos != total:
        raise RuntimeError("split did not consume all words")
    return exs


def load_step246_triplets() -> list[dict]:
    ledger = json.loads(LEDGER.read_text())
    target_words = int(ledger["syncscratch_words_used"])
    childes_budget = int(ledger["childes_words_replaced"])
    os.environ["HF_DATASETS_CACHE"] = str((ROOT / "training/hf_home/datasets").resolve())
    from datasets import load_dataset
    ds = load_dataset("sjtu-lit/SynCSE-scratch-NLI", split="train")
    idxs = list(range(len(ds)))
    random.Random(SEED).shuffle(idxs)
    out = []
    total = 0
    for idx in idxs:
        row = ds[idx]
        text = f"{row['sent0'].strip()} {row['sent1'].strip()} {row['nli_hard'].strip()}"
        w = len(text.split())
        if total + w > childes_budget:
            break
        if w > TRIPLET_BIN_CAPACITY:
            raise RuntimeError(f"triplet too long for indivisible packing: {w} words")
        out.append({"text": text, "words": w, "source": "syncscratch_triplet"})
        total += w
    if total != target_words:
        raise RuntimeError(f"triplet rebuild mismatch {total} vs {target_words}")
    return out


def bestfit_triplets(triplets: list[dict], n_bins: int, cap: int) -> list[dict]:
    """Pack indivisible triplets into exactly n_bins bins; split bins if under target."""
    total = sum(t["words"] for t in triplets)
    if total > n_bins * cap:
        raise RuntimeError(f"insufficient triplet capacity {n_bins*cap} for {total}")
    bins: list[dict] = []
    # best-fit with sorted descending item sizes
    remainders: list[int] = []
    for t in sorted(triplets, key=lambda x: x["words"], reverse=True):
        w = t["words"]
        best_i = -1
        best_rem_after = cap + 1
        for i, rem in enumerate(remainders):
            if rem >= w and rem - w < best_rem_after:
                best_i = i
                best_rem_after = rem - w
        if best_i < 0:
            if len(bins) >= n_bins:
                raise RuntimeError(f"bin count exceeded target {n_bins}; total={total}, bins_now={len(bins)}")
            bins.append({"texts": [t["text"]], "words": w})
            remainders.append(cap - w)
        else:
            bins[best_i]["texts"].append(t["text"])
            bins[best_i]["words"] += w
            remainders[best_i] -= w
    # If packing used fewer bins, split multi-triplet bins until exact count.
    while len(bins) < n_bins:
        k = max(range(len(bins)), key=lambda i: len(bins[i]["texts"]))
        if len(bins[k]["texts"]) <= 1:
            raise RuntimeError(f"cannot split to target bins {n_bins}; have {len(bins)}")
        moved = bins[k]["texts"].pop()
        mw = len(moved.split())
        bins[k]["words"] -= mw
        bins.append({"texts": [moved], "words": mw})
    return [{"text": " ".join(b["texts"]), "words": b["words"], "source": "syncscratch_triplets_packed"} for b in bins]


def write_jsonl(path: pathlib.Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def summarize(examples: list[dict]) -> dict:
    lens = [int(e["words"]) for e in examples]
    return {
        "total_words": sum(lens),
        "total_examples": len(lens),
        "estimated_steps_batch256": (len(lens) + BATCH_SIZE - 1)//BATCH_SIZE,
        "min_words_per_example": min(lens),
        "max_words_per_example": max(lens),
        "mean_words_per_example": sum(lens)/len(lens),
        "full_batch_steps": len(lens) % BATCH_SIZE == 0,
        "last_batch_size": len(lens) % BATCH_SIZE,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Official symmetric corpus
    official_words, official_counts = words_from_files(ALL_SOURCES)
    if len(official_words) < TARGET_WORDS:
        raise RuntimeError(f"official corpus only {len(official_words)} words")
    official_words = official_words[:TARGET_WORDS]
    official_examples = split_flat_words_exact(official_words, TARGET_EXAMPLES, "official_all_sources_symmetric")
    random.Random(SEED).shuffle(official_examples)
    write_jsonl(OFFICIAL_OUT, official_examples)

    # Structured symmetric corpus
    kept_words, kept_counts = words_from_files(KEEP_SOURCES)
    triplets = load_step246_triplets()
    triplet_words = sum(t["words"] for t in triplets)
    kept_words_target = TARGET_WORDS - triplet_words
    if len(kept_words) != kept_words_target:
        raise RuntimeError(f"kept official words {len(kept_words)} != target {kept_words_target}")
    # Use as many kept-official examples as if they were near-160 chunks, then allocate remaining bins to triplets.
    kept_examples_count = (kept_words_target + MAX_WORDS_PER_EXAMPLE - 1)//MAX_WORDS_PER_EXAMPLE
    triplet_bins = TARGET_EXAMPLES - kept_examples_count
    if triplet_bins <= 0:
        raise RuntimeError("no bins left for triplets")
    kept_examples = split_flat_words_exact(kept_words, kept_examples_count, "official_non_childes_symmetric")
    triplet_examples = bestfit_triplets(triplets, triplet_bins, TRIPLET_BIN_CAPACITY)
    structured_examples = kept_examples + triplet_examples
    if len(structured_examples) != TARGET_EXAMPLES:
        raise RuntimeError(f"structured examples {len(structured_examples)} != {TARGET_EXAMPLES}")
    if sum(e["words"] for e in structured_examples) != TARGET_WORDS:
        raise RuntimeError("structured word mismatch")
    random.Random(SEED).shuffle(structured_examples)
    write_jsonl(STRUCT_OUT, structured_examples)

    ledger = {
        "status": "SYMMETRIC_BATCH_CORPORA_BUILT",
        "design": "Both official and structured corpora are pre-packed JSONL with exactly 9,999,969 words and 62,720 examples, giving exactly 245 full batch-256 steps. Structured SynCSE triplets are indivisible and adjacency-preserved; triplet bins allow up to 192 words to make exact example/update matching feasible while official non-CHILDES examples remain near-160.",
        "target_words": TARGET_WORDS,
        "target_examples": TARGET_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "max_words_per_official_example": MAX_WORDS_PER_EXAMPLE,
        "max_words_per_triplet_bin": TRIPLET_BIN_CAPACITY,
        "official_output": str(OFFICIAL_OUT),
        "structured_output": str(STRUCT_OUT),
        "official_summary": summarize(official_examples),
        "structured_summary": summarize(structured_examples),
        "official_source_counts_raw": official_counts,
        "structured_kept_source_counts_raw": kept_counts,
        "structured_triplet_words": triplet_words,
        "structured_kept_examples": kept_examples_count,
        "structured_triplet_examples": triplet_bins,
        "structured_triplet_max_bin_words": max(e["words"] for e in triplet_examples),
        "structured_triplet_min_bin_words": min(e["words"] for e in triplet_examples),
        "seed": SEED,
    }
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"official": ledger["official_summary"], "structured": ledger["structured_summary"], "triplet_bins": triplet_bins}, indent=2))

if __name__ == "__main__":
    main()
