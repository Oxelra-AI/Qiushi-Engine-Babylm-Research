#!/usr/bin/env python3
"""research strict repair: structured corpus with exact official example/update count.

Official packed JSONL at 9,999,969 words has 62,501 examples -> 245 steps with
batch_size=256. This script rebuilds the structured CHILDES-replacement corpus
so it has the same 62,501 examples and the same 9,999,969 words while preserving
SynCSE triplet adjacency (each triplet is an indivisible unit).

Method:
- Keep official non-CHILDES sources packed exactly as 160-word chunks.
- Rebuild the same SynCSE-scratch triplet set from research.
- Bin-pack triplets into exactly (62501 - official_kept_examples) bins of
  capacity 160 using bucketed first-fit by remaining word capacity. Total
  SynCSE words are only 50 words below this capacity, so feasible if triplets
  are not larger than 160 words; fail loudly otherwise.
- Shuffle final examples at example level with seed 42.
"""
from __future__ import annotations
import json, os, pathlib, random

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RAW_DIR = ROOT / "data/reconstruct_tmp/raw_dataset"
SRC_LEDGER = ROOT / "data/structured_experience_corpus/source_ledger.json"
OFFICIAL_LEDGER = ROOT / "data/official_packed_corpus/packing_ledger.json"
OUT_DIR = ROOT / "data/structured_exactstep_corpus"
OUT = OUT_DIR / "structured_experience_exactstep_10M.jsonl"
LEDGER = OUT_DIR / "packing_ledger.json"

WPE = 160
TARGET_WORDS = 9999969
SEED = 42
KEEP_SOURCES = ["bnc_spoken.train.txt", "gutenberg.train.txt", "open_subtitles.train.txt", "simple_wiki.train.txt", "switchboard.train.txt"]


def count_words(t: str) -> int:
    return len(t.split())


def pack_word_stream(words: list[str], wpe: int) -> list[dict]:
    out = []
    for i in range(0, len(words), wpe):
        chunk = words[i:i+wpe]
        if chunk:
            out.append({"text": " ".join(chunk), "words": len(chunk)})
    return out


def load_triplets() -> list[dict]:
    os.environ["HF_DATASETS_CACHE"] = str((ROOT / "training/hf_home/datasets").resolve())
    from datasets import load_dataset
    target = json.loads(SRC_LEDGER.read_text())["syncscratch_words_used"]
    childes_target = json.loads(SRC_LEDGER.read_text())["childes_words_replaced"]
    ds = load_dataset("sjtu-lit/SynCSE-scratch-NLI", split="train")
    idxs = list(range(len(ds)))
    random.Random(SEED).shuffle(idxs)
    triplets = []
    total = 0
    for idx in idxs:
        row = ds[idx]
        text = f"{row['sent0'].strip()} {row['sent1'].strip()} {row['nli_hard'].strip()}"
        w = count_words(text)
        if total + w > childes_target:
            break
        triplets.append({"text": text, "words": w})
        total += w
    if total != target:
        raise RuntimeError(f"SynCSE rebuild mismatch {total} vs ledger {target}")
    return triplets


def binpack_exact(triplets: list[dict], n_bins: int, capacity: int = 160) -> list[dict]:
    if any(t["words"] > capacity for t in triplets):
        mx = max(t["words"] for t in triplets)
        bad = sum(t["words"] > capacity for t in triplets)
        raise RuntimeError(f"{bad} triplets exceed capacity {capacity}; max={mx}")
    total = sum(t["words"] for t in triplets)
    if total > n_bins * capacity:
        raise RuntimeError(f"insufficient capacity: words={total}, bins={n_bins}, cap={capacity}")
    bins: list[dict] = []
    # buckets[remaining_capacity] -> list of bin indices with that remaining capacity
    buckets: list[list[int]] = [[] for _ in range(capacity + 1)]
    for t in sorted(triplets, key=lambda x: x["words"], reverse=True):
        w = t["words"]
        chosen = None
        for rem in range(w, capacity + 1):
            if buckets[rem]:
                chosen = buckets[rem].pop()
                break
        if chosen is None:
            if len(bins) >= n_bins:
                raise RuntimeError(f"bin count exceeded target {n_bins} while packing")
            chosen = len(bins)
            bins.append({"texts": [], "words": 0, "remaining": capacity})
        bins[chosen]["texts"].append(t["text"])
        bins[chosen]["words"] += w
        bins[chosen]["remaining"] -= w
        buckets[bins[chosen]["remaining"]].append(chosen)
    if len(bins) > n_bins:
        raise RuntimeError(f"packed bins {len(bins)} > target {n_bins}")
    # If under target, split multi-triplet bins until exact count. This preserves triplets.
    while len(bins) < n_bins:
        k = max(range(len(bins)), key=lambda i: len(bins[i]["texts"]))
        if len(bins[k]["texts"]) <= 1:
            raise RuntimeError(f"cannot split further to reach {n_bins}; have {len(bins)}")
        moved = bins[k]["texts"].pop()
        mw = count_words(moved)
        bins[k]["words"] -= mw
        bins[k]["remaining"] += mw
        bins.append({"texts": [moved], "words": mw, "remaining": capacity - mw})
    return [{"text": " ".join(b["texts"]), "words": b["words"], "source": "syncscratch_exactstep_packed"} for b in bins]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_ref = json.loads(OFFICIAL_LEDGER.read_text())
    target_examples = int(official_ref["total_examples"])
    official_examples = []
    source_words = {}
    for fn in KEEP_SOURCES:
        words = []
        for line in (RAW_DIR / fn).read_text(encoding="utf-8", errors="replace").splitlines():
            words.extend(line.split())
        source_words[fn] = len(words)
        official_examples.extend([{**ex, "source": fn} for ex in pack_word_stream(words, WPE)])
    needed_triplet_bins = target_examples - len(official_examples)
    triplets = load_triplets()
    triplet_examples = binpack_exact(triplets, needed_triplet_bins, WPE)
    final = official_examples + triplet_examples
    random.Random(SEED).shuffle(final)
    total_words = sum(e["words"] for e in final)
    if total_words != TARGET_WORDS:
        raise RuntimeError(f"word mismatch {total_words} vs {TARGET_WORDS}")
    if len(final) != target_examples:
        raise RuntimeError(f"example mismatch {len(final)} vs {target_examples}")
    with OUT.open("w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    led = {
        "status": "STRUCTURED_EXACTSTEP_BUILT",
        "output": str(OUT),
        "total_words": total_words,
        "total_examples": len(final),
        "estimated_steps_batch256": (len(final) + 255)//256,
        "official_packed_reference": str(OFFICIAL_LEDGER),
        "official_reference_examples": target_examples,
        "official_kept_examples": len(official_examples),
        "syncscratch_triplet_examples": len(triplet_examples),
        "syncscratch_triplet_words": sum(e["words"] for e in triplet_examples),
        "max_syncscratch_bin_words": max(e["words"] for e in triplet_examples),
        "min_syncscratch_bin_words": min(e["words"] for e in triplet_examples),
        "design": "Exact 62,501 examples / 245-step training geometry matched to packed official arm; SynCSE triplets are indivisible and adjacency-preserved; all non-CHILDES official sources retained.",
        "source_word_counts": source_words,
    }
    LEDGER.write_text(json.dumps(led, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: led[k] for k in ["total_words","total_examples","estimated_steps_batch256","syncscratch_triplet_examples","syncscratch_triplet_words","max_syncscratch_bin_words","min_syncscratch_bin_words"]}, indent=2))

if __name__ == "__main__":
    main()
