#!/usr/bin/env python3
"""Materialize a shuffled-rewrite control from final research selected pairs.

Use only after `clean_materialize_qwen_pairs.py` has produced
`selected_pairs.jsonl`.  This control keeps the same selected originals, same
selected Qwen rewrite texts, and same total pair-word budget, but permutes rewrites
within source/length bins to break original--rewrite correspondence.  It is a
causal follow-up if the clean-Qwen treatment beats the matched official control.
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
OUT = _public_path('experiments/archive/compact_experience/data/qwen_shuffled_control')
CORP = _public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT.mkdir(parents=True, exist_ok=True)
CORP.mkdir(parents=True, exist_ok=True)
WORDS_TOTAL = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
RNG_SEED = 28900

@dataclass
class Pair:
    pair_id: str
    original: str
    rewrite: str
    original_words: int
    rewrite_words: int
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
    return {"n": len(vals), "min": min(vals), "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4), "max": max(vals)}


def load_pairs() -> list[Pair]:
    if not SELECTED.exists():
        raise FileNotFoundError(SELECTED)
    pairs = []
    for line in SELECTED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        pairs.append(Pair(
            pair_id=o["pair_id"], original=o["original"], rewrite=o["rewrite"],
            original_words=int(o["original_words"]), rewrite_words=int(o["rewrite_words"]),
            source=o.get("source", "unknown"), example_id=int(o.get("example_id", -1)), cohort=o.get("cohort", "unknown"),
        ))
    return pairs


def length_bin(n: int) -> str:
    if n <= 12:
        return "le12"
    if n <= 18:
        return "13_18"
    if n <= 26:
        return "19_26"
    if n <= 38:
        return "27_38"
    return "gt38"


def shuffle_rewrites(pairs: list[Pair]) -> tuple[list[Pair], dict]:
    rng = random.Random(RNG_SEED)
    groups = collections.defaultdict(list)
    for i, p in enumerate(pairs):
        groups[(p.source, length_bin(p.rewrite_words))].append(i)
    shuffled = list(pairs)
    changed = 0
    singleton = 0
    for key, idxs in groups.items():
        if len(idxs) < 2:
            singleton += len(idxs)
            continue
        rewrites = [(pairs[i].rewrite, pairs[i].rewrite_words, pairs[i].pair_id) for i in idxs]
        # Try to derange within the source/length bin so no original keeps its own rewrite.
        # If a rare small group cannot be deranged after repeated shuffles, rotate the group;
        # the later changed_pairs count makes any residual unchanged cases visible.
        orig_ids = [pairs[i].pair_id for i in idxs]
        for _ in range(200):
            rng.shuffle(rewrites)
            if all(src_pid != orig_pid for orig_pid, (_, _, src_pid) in zip(orig_ids, rewrites)):
                break
        else:
            rewrites = rewrites[1:] + rewrites[:1]
        for i, (rw, rw_words, src_pid) in zip(idxs, rewrites):
            p = pairs[i]
            if src_pid != p.pair_id:
                changed += 1
            shuffled[i] = Pair(
                pair_id=p.pair_id + "__rw_from__" + src_pid,
                original=p.original,
                rewrite=rw,
                original_words=p.original_words,
                rewrite_words=rw_words,
                source=p.source,
                example_id=p.example_id,
                cohort=p.cohort,
            )
    return shuffled, {"groups": len(groups), "changed_pairs": changed, "singleton_pairs_not_shuffled": singleton}


def pack_pairs(pairs: list[Pair]) -> list[Row]:
    rows=[]
    cur=[]
    ids=[]
    w=0
    for p in pairs:
        pw=p.original_words+p.rewrite_words
        if w and w+pw > MAX_ROW_WORDS:
            rows.append(Row(" ".join(cur), w, "qwen_shuffled_pair_packed", 700000+len(rows), list(ids), len(ids)))
            cur=[]; ids=[]; w=0
        if pw > MAX_ROW_WORDS:
            rows.append(Row(p.original + " " + p.rewrite, pw, "qwen_shuffled_pair_packed", 700000+len(rows), [p.pair_id], 1))
        else:
            cur += [p.original, p.rewrite]
            ids.append(p.pair_id)
            w += pw
    if cur:
        rows.append(Row(" ".join(cur), w, "qwen_shuffled_pair_packed", 700000+len(rows), list(ids), len(ids)))
    assert sum(r.words for r in rows) == sum(p.original_words+p.rewrite_words for p in pairs)
    return rows


def official_filler(needed_words: int, exclude_ids: set[int]) -> list[Row]:
    rows=[]
    with OFFICIAL_POOL.open("r", encoding="utf-8") as f:
        official=[json.loads(line) for line in f if line.strip()]
    for o in official:
        o["words"]=int(o.get("words", len(o["text"].split())))
    non=[o for o in official if int(o["example_id"]) not in exclude_ids]
    src=[o for o in official if int(o["example_id"]) in exclude_ids]
    rng=random.Random(RNG_SEED+1); rng.shuffle(non); rng.shuffle(src)
    need_rows=needed_words//160
    chosen=(non+src)[:need_rows]
    if len(chosen)!=need_rows:
        raise RuntimeError("not enough official filler")
    return [Row(o["text"], int(o["words"]), str(o["source"]), int(o["example_id"])) for o in chosen]


def write_pool(path: pathlib.Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.words != len(r.text.split()):
                raise RuntimeError(f"word mismatch in {path}")
            f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False)+"\n")


def write_train(path: pathlib.Path, rows: list[Row]) -> None:
    n=len(rows)
    total=0
    with path.open("w", encoding="utf-8") as f:
        for p in range(PASSES):
            order=list(range(n)); random.Random(RNG_SEED+100+p).shuffle(order)
            for i in order:
                r=rows[i]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False)+"\n")
                total += r.words
    assert total == WORDS_TOTAL*PASSES


def main() -> None:
    pairs=load_pairs()
    shuffled, shuf_meta=shuffle_rewrites(pairs)
    pair_words=sum(p.original_words+p.rewrite_words for p in shuffled)
    if pair_words % 160 != 0:
        raise RuntimeError(f"pair_words {pair_words} not divisible by 160; should match selected clean materialization")
    packed=pack_pairs(shuffled)
    filler=official_filler(WORDS_TOTAL-pair_words, {p.example_id for p in shuffled})
    pool=packed+filler
    random.Random(RNG_SEED+2).shuffle(pool)
    assert sum(r.words for r in pool)==WORDS_TOTAL
    pool_path=_public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora/qwen_shuffled_10M.jsonl')
    train_path=_public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora/qwen_shuffled_100M.jsonl')
    write_pool(pool_path, pool)
    write_train(train_path, pool)
    meta={
        "status":"QWEN_SHUFFLED_CONTROL_MATERIALIZED",
        "source_selected_pairs":str(SELECTED),
        "selected_pairs":len(pairs),
        "pair_words":pair_words,
        "pair_word_fraction":round(pair_words/WORDS_TOTAL,6),
        "shuffle_meta":shuf_meta,
        "pool_rows":len(pool),
        "packed_pair_rows":len(packed),
        "filler_rows":len(filler),
        "pair_boundary_preserved":True,
        "pair_truncation":False,
        "same_original_and_rewrite_multisets_as_selected_pairs":True,
        "pair_row_word_stats":stats([r.words for r in packed]),
        "files":{"pool":str(pool_path),"training":str(train_path)},
        "sha256":{"pool":sha(pool_path),"training":sha(train_path)},
    }
    (_public_path('experiments/archive/compact_experience/data/qwen_shuffled_control/shuffled_control_metadata.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(meta,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
