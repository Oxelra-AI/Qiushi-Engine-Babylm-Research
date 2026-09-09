#!/usr/bin/env python3
"""research: SynCSE semantic-relation ladder materializer.

Builds the proposed coherent non-synonymous control using the SynCSE
`hard_neg` field.  The goal is to separate:
  - generic local topical/entity coherence, from
  - true same-meaning paraphrase/rewrite correspondence.

Four arms are produced from the same SynCSE row set and exact 160-word packing:
  1. paraphrase_aligned: sent0_i + sent1_i
  2. paraphrase_mismatched: sent0_i + sent1_j (deranged shuffle), same sent0/sent1 inventory
  3. hardneg_coherent: sent0_i + hard_neg_i, topical/near but non-synonymous
  4. hardneg_mismatched: sent0_i + hard_neg_j (deranged shuffle), same sent0/hardneg inventory

Every arm is truncated to the same base-pool size, then expanded to exactly 10
shuffled passes.  This is an exact-epoch mechanism screen; base pool is ~8M words
because SynCSE hard negatives provide only ~8M pair words.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORDS_PER_EXAMPLE = 160
SEED = 43
ROOT = Path("experiments/archive/compact_experience")
OUT_DIR = ROOT / "data/syncse_relation_ladder"
POOL_DIR = OUT_DIR / "pools"
TRAIN_DIR = OUT_DIR / "training_exact10"
HF_CACHE = ROOT / "staging/hf_hub"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)


def clean(s: Any) -> str:
    return " ".join(str(s).split())


def wlen(s: str) -> int:
    return len(s.split())


def load_syncse_rows() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("hkust-nlp/SynCSE-partial-NLI", split="train", cache_dir=str(HF_CACHE))
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    seen = set()
    for i, r in enumerate(ds):
        s0 = clean(r.get("sent0", ""))
        s1 = clean(r.get("sent1", ""))
        hn = clean(r.get("hard_neg", ""))
        if min(wlen(s0), wlen(s1), wlen(hn)) < 3:
            skipped["too_short"] += 1
            continue
        key = (s0.lower(), s1.lower(), hn.lower())
        if key in seen:
            skipped["duplicate_triplet"] += 1
            continue
        seen.add(key)
        if s0.lower() == s1.lower() or s0.lower() == hn.lower():
            skipped["identity"] += 1
            continue
        rows.append({
            "rid": len(rows),
            "orig": s0,
            "para": s1,
            "hardneg": hn,
            "ow": wlen(s0),
            "pw": wlen(s1),
            "hw": wlen(hn),
        })
    print(f"Loaded {len(rows)} rows; skipped={dict(skipped)}", flush=True)
    return rows


def deranged_indices(n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    for i in range(n):
        if idx[i] == i:
            j = (i + 1) % n
            idx[i], idx[j] = idx[j], idx[i]
    assert all(idx[i] != i for i in range(n))
    return idx


def stream_words(rows: List[Dict[str, Any]], mode: str, shuf_para: List[int], shuf_hard: List[int]) -> List[str]:
    words: List[str] = []
    n = len(rows)
    for i, row in enumerate(rows):
        words.extend(row["orig"].split())
        if mode == "paraphrase_aligned":
            words.extend(row["para"].split())
        elif mode == "paraphrase_mismatched":
            words.extend(rows[shuf_para[i]]["para"].split())
        elif mode == "hardneg_coherent":
            words.extend(row["hardneg"].split())
        elif mode == "hardneg_mismatched":
            words.extend(rows[shuf_hard[i]]["hardneg"].split())
        else:
            raise KeyError(mode)
    return words


def pack(words: List[str], mode: str, target_words: int) -> List[Dict[str, Any]]:
    words = words[:target_words]
    assert len(words) == target_words
    assert target_words % WORDS_PER_EXAMPLE == 0
    rows: List[Dict[str, Any]] = []
    for i in range(target_words // WORDS_PER_EXAMPLE):
        chunk = words[i * WORDS_PER_EXAMPLE:(i + 1) * WORDS_PER_EXAMPLE]
        rows.append({
            "text": " ".join(chunk),
            "words": WORDS_PER_EXAMPLE,
            "example_id": i,
            "source": mode,
        })
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": str(path), "rows": len(rows), "words": sum(int(r["words"]) for r in rows)}


def expand_exact10(mode: str, pool_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = TRAIN_DIR / f"{mode}_exact10.jsonl"
    written_words = 0
    written_rows = 0
    with out.open("w", encoding="utf-8") as f:
        for epoch in range(10):
            epoch_rows = list(pool_rows)
            random.Random(SEED + 1000003 * epoch).shuffle(epoch_rows)
            for row in epoch_rows:
                rec = dict(row)
                rec["source"] = f"epoch{epoch+1}::{row.get('source', mode)}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written_rows += 1
                written_words += int(row["words"])
    return {"path": str(out), "rows": written_rows, "words": written_words}


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_syncse_rows()
    random.Random(SEED).shuffle(rows)
    # Reassign rid after shuffle so the row order is the experimental order.
    for i, r in enumerate(rows):
        r["rid"] = i
    shuf_para = deranged_indices(len(rows), SEED + 1009)
    shuf_hard = deranged_indices(len(rows), SEED + 2003)
    modes = ["paraphrase_aligned", "paraphrase_mismatched", "hardneg_coherent", "hardneg_mismatched"]
    streams = {m: stream_words(rows, m, shuf_para, shuf_hard) for m in modes}
    stream_word_counts = {m: len(v) for m, v in streams.items()}
    target_words = min(stream_word_counts.values()) // WORDS_PER_EXAMPLE * WORDS_PER_EXAMPLE
    print("Stream word counts:", stream_word_counts, flush=True)
    print(f"Target base words: {target_words:,}; exact10 exposure: {target_words * 10:,}", flush=True)

    pool_stats: Dict[str, Any] = {}
    train_stats: Dict[str, Any] = {}
    for mode in modes:
        pool_rows = pack(streams[mode], mode, target_words)
        pool_stats[mode] = write_jsonl(pool_rows, POOL_DIR / f"{mode}_pool.jsonl")
        train_stats[mode] = expand_exact10(mode, pool_rows)
        print(f"{mode}: pool={pool_stats[mode]['words']:,} train={train_stats[mode]['words']:,}", flush=True)

    # Exact inventory checks for matched controls.
    # Compare word multisets after truncation: aligned vs mismatched for paraphrase, coherent vs mismatch for hardneg.
    def word_counter(mode: str) -> Counter:
        c = Counter()
        for rec in open(POOL_DIR / f"{mode}_pool.jsonl", encoding="utf-8"):
            obj = json.loads(rec)
            c.update(obj["text"].split())
        return c
    ca = word_counter("paraphrase_aligned")
    cm = word_counter("paraphrase_mismatched")
    ch = word_counter("hardneg_coherent")
    chm = word_counter("hardneg_mismatched")
    para_diff = sum((ca - cm).values()) + sum((cm - ca).values())
    hard_diff = sum((ch - chm).values()) + sum((chm - ch).values())

    summary = {
        "status": "SYNCSE_RELATION_LADDER_MATERIALIZED",
        "purpose": "Coherent non-synonymous adjacency control for research paired-alignment interpretation.",
        "seed": SEED,
        "rows_used": len(rows),
        "stream_word_counts_before_common_truncation": stream_word_counts,
        "target_base_words": target_words,
        "exact10_exposure_words": target_words * 10,
        "pool_stats": pool_stats,
        "training_exact10": train_stats,
        "matched_inventory_boundary_artifact": {
            "paraphrase_aligned_vs_mismatched_word_multiset_diff": para_diff,
            "hardneg_coherent_vs_mismatched_word_multiset_diff": hard_diff,
            "diff_fraction_of_tokens": {
                "paraphrase": para_diff / target_words,
                "hardneg": hard_diff / target_words,
            },
        },
        "contrasts": {
            "paraphrase_aligned_minus_paraphrase_mismatched": "same sent0/sent1 inventory; local paraphrase adjacency plus topical coherence vs random adjacency",
            "hardneg_coherent_minus_hardneg_mismatched": "same sent0/hardneg inventory; local same-topic non-synonymous coherence vs random adjacency",
            "difference_in_differences": "additional effect of same-meaning paraphrase correspondence beyond coherence, subject to sent1/hardneg distribution differences",
        },
        "elapsed_sec": round(time.time() - start, 3),
    }
    out = OUT_DIR / "relation_ladder_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
