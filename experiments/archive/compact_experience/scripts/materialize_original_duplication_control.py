#!/usr/bin/env python3
"""research: materialize an original-duplication control for research.

This control tests whether any benefit of Qwen-pair rows comes from within-row
propositional redundancy rather than from a paraphrastic/generated second view.
It replaces selected (original, Qwen rewrite) pairs with (original, original) pairs,
keeps complete pair boundaries, uses official text only, and fills the remaining
10M pool with complete official 160-word rows.

It is not an exact row-length match to the Qwen treatment because original lengths
differ from rewrite lengths.  Its role is mechanistic: if original duplication
recovers most of a positive Qwen Delta, the effect is likely redundancy/repetition
rather than semantic paraphrase alignment.
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
from dataclasses import dataclass
from typing import Iterable

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
REF_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/original_dup_control')
CORP = _public_path('experiments/archive/compact_experience/data/original_dup_control/training_corpora')
OUT.mkdir(parents=True, exist_ok=True)
CORP.mkdir(parents=True, exist_ok=True)
TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
RNG_SEED = 29143

@dataclass
class Pair:
    pair_id: str
    original: str
    original_words: int
    source: str
    example_id: int
    cohort: str

@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    pair_ids: list[str] | None = None
    n_pairs: int | None = None


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[int | float]) -> dict:
    vals = list(vals)
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    def pct(q: float):
        if not vals_sorted:
            return None
        i = min(len(vals_sorted)-1, max(0, round((len(vals_sorted)-1)*q)))
        return vals_sorted[i]
    return {"n": len(vals), "min": min(vals), "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4), "p95": pct(0.95), "max": max(vals)}


def load_selected() -> list[Pair]:
    pairs = []
    for line in SELECTED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        pairs.append(Pair(
            pair_id=str(o["pair_id"]), original=str(o["original"]),
            original_words=int(o["original_words"]), source=str(o.get("source", "unknown")),
            example_id=int(o.get("example_id", -1)), cohort=str(o.get("cohort", "unknown")),
        ))
    if not pairs:
        raise RuntimeError("no selected pairs")
    return pairs


def select_dup_pairs(pairs: list[Pair], target_pair_words: int) -> list[Pair]:
    # Select deterministically up to the Qwen selected-pair word budget, with duplicate-pair words divisible
    # by 160 so the treatment can be filled with complete official rows.
    rng = random.Random(RNG_SEED)
    order = list(pairs)
    rng.shuffle(order)
    chosen = []
    total = 0
    for p in order:
        w = 2 * p.original_words
        if w > MAX_ROW_WORDS:
            raise RuntimeError(f"unexpected long original duplicate pair {p.pair_id}: {w}")
        if total + w <= target_pair_words:
            chosen.append(p)
            total += w
    while chosen and total % 160 != 0:
        p = chosen.pop()
        total -= 2 * p.original_words
    if total < 0.10 * TOTAL_WORDS:
        raise RuntimeError(f"duplicate-pair words too low: {total}")
    return chosen


def pack_dup_pairs(pairs: list[Pair]) -> list[Row]:
    rows = []
    cur = []
    ids = []
    w = 0
    for p in pairs:
        pw = 2 * p.original_words
        if w and w + pw > MAX_ROW_WORDS:
            rows.append(Row(" ".join(cur), w, "official_original_dup_packed", 900000+len(rows), list(ids), len(ids)))
            cur = []
            ids = []
            w = 0
        cur.extend([p.original, p.original])
        ids.append(p.pair_id)
        w += pw
    if cur:
        rows.append(Row(" ".join(cur), w, "official_original_dup_packed", 900000+len(rows), list(ids), len(ids)))
    expected = sum(2*p.original_words for p in pairs)
    if sum(r.words for r in rows) != expected:
        raise RuntimeError("packed duplicate rows lost words")
    return rows


def load_official_pool() -> list[dict]:
    rows = []
    for line in OFFICIAL_POOL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            o["words"] = int(o.get("words", len(o["text"].split())))
            rows.append(o)
    if sum(int(r["words"]) for r in rows) != TOTAL_WORDS:
        raise RuntimeError("official pool is not 10M")
    return rows


def official_filler(needed_words: int, exclude_ids: set[int]) -> list[Row]:
    if needed_words % 160 != 0:
        raise RuntimeError(f"needed filler words not multiple of 160: {needed_words}")
    official = load_official_pool()
    non = [o for o in official if int(o["example_id"]) not in exclude_ids]
    src = [o for o in official if int(o["example_id"]) in exclude_ids]
    rng = random.Random(RNG_SEED + 1)
    rng.shuffle(non)
    rng.shuffle(src)
    need = needed_words // 160
    chosen = (non + src)[:need]
    if len(chosen) != need:
        raise RuntimeError("not enough official filler")
    return [Row(str(o["text"]), int(o["words"]), str(o["source"]), int(o["example_id"])) for o in chosen]


def write_pool(path: pathlib.Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.words != len(r.text.split()):
                raise RuntimeError("word mismatch")
            obj = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
            if r.pair_ids:
                obj["pair_ids"] = r.pair_ids
                obj["n_pairs"] = r.n_pairs
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_train(path: pathlib.Path, rows: list[Row]) -> None:
    total = 0
    n = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for p in range(PASSES):
            order = list(range(n))
            random.Random(RNG_SEED + 100 + p).shuffle(order)
            for i in order:
                r = rows[i]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training words {total}")


def main() -> None:
    ref = json.loads(REF_META.read_text(encoding="utf-8"))
    target_pair_words = int(ref["selected_pair_words"])
    pairs_all = load_selected()
    pairs = select_dup_pairs(pairs_all, target_pair_words)
    dup_words = sum(2*p.original_words for p in pairs)
    packed = pack_dup_pairs(pairs)
    filler = official_filler(TOTAL_WORDS - dup_words, {p.example_id for p in pairs})
    pool = packed + filler
    random.Random(RNG_SEED + 2).shuffle(pool)
    if sum(r.words for r in pool) != TOTAL_WORDS:
        raise RuntimeError("pool word total wrong")
    pool_path = _public_path('experiments/archive/compact_experience/data/original_dup_control/training_corpora/official_original_dup_10M.jsonl')
    train_path = _public_path('experiments/archive/compact_experience/data/original_dup_control/training_corpora/official_original_dup_100M.jsonl')
    write_pool(pool_path, pool)
    write_train(train_path, pool)
    source_words = collections.Counter()
    for p in pairs:
        source_words[p.source] += 2*p.original_words
    payload = {
        "status": "ORIGINAL_DUPLICATION_CONTROL_MATERIALIZED",
        "purpose": "official-only original+original rows to test redundancy against Qwen original+rewrite rows",
        "reference_selected_pair_words": target_pair_words,
        "available_selected_pairs": len(pairs_all),
        "selected_duplicate_pairs": len(pairs),
        "duplicate_pair_words": dup_words,
        "duplicate_pair_word_fraction": round(dup_words / TOTAL_WORDS, 6),
        "uses_qwen_words": False,
        "uses_official_words_only": True,
        "pair_boundary_preserved": True,
        "pair_truncation": False,
        "packed_pair_rows": len(packed),
        "filler_rows": len(filler),
        "pool_rows": len(pool),
        "duplicate_pair_row_word_stats": stats([r.words for r in packed]),
        "duplicate_pair_source_words": dict(source_words),
        "files": {"pool": str(pool_path), "training": str(train_path)},
        "sha256": {"pool": sha(pool_path), "training": sha(train_path)},
    }
    meta = _public_path('experiments/archive/compact_experience/data/original_dup_control/original_dup_control_metadata.json')
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
