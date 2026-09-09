#!/usr/bin/env python3
"""research: separated-pair control for clean-Qwen original--rewrite data.

This control keeps the same final research selected originals and Qwen rewrite
texts, uses no additional generated text, and preserves the same pair-word budget
as the aligned treatment.  Unlike the aligned treatment, originals and rewrites are
packed into separate rows/windows and row order is shuffled so paired sides are not
in the same self-attention context.  This separates the effect of same-window
original--rewrite correspondence from the mere coexistence of generated rewrites in
the 10M pool.

The pool matches the clean-Qwen treatment's effective source word totals by filling
from official text source streams after assigning pair words back to the original
source.  It is not an exact row-length-sequence control; its purpose is mechanism.
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
OUT = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control')
CORP = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/training_corpora')
OUT.mkdir(parents=True, exist_ok=True)
CORP.mkdir(parents=True, exist_ok=True)

TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
RNG_SEED = 32143
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
SOURCE_SEED_OFFSET = {s: i * 1009 for i, s in enumerate(SOURCES)}


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
class Segment:
    pair_id: str
    side: str
    text: str
    words: int
    source: str
    example_id: int


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    pair_ids: list[str] | None = None
    sides: list[str] | None = None
    component_sources: dict[str, int] | None = None


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
        i = min(len(vals_sorted) - 1, max(0, round((len(vals_sorted) - 1) * q)))
        return vals_sorted[i]

    return {"n": len(vals), "min": min(vals), "mean": round(statistics.mean(vals), 4), "median": round(statistics.median(vals), 4), "p05": pct(0.05), "p95": pct(0.95), "max": max(vals)}


def load_selected() -> list[Pair]:
    pairs: list[Pair] = []
    with SELECTED.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            pairs.append(Pair(
                pair_id=str(o["pair_id"]),
                original=str(o["original"]),
                rewrite=str(o["rewrite"]),
                original_words=int(o["original_words"]),
                rewrite_words=int(o["rewrite_words"]),
                source=str(o.get("source", "unknown")),
                example_id=int(o.get("example_id", -1)),
                cohort=str(o.get("cohort", "unknown")),
            ))
    if not pairs:
        raise RuntimeError("no selected pairs")
    for p in pairs:
        if p.original_words != len(p.original.split()) or p.rewrite_words != len(p.rewrite.split()):
            raise RuntimeError(f"word mismatch in pair {p.pair_id}")
        if max(p.original_words, p.rewrite_words) > MAX_ROW_WORDS:
            raise RuntimeError(f"unexpected too-long segment {p.pair_id}")
    return pairs


def load_official_pool() -> list[dict]:
    rows = []
    with OFFICIAL_POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                o["words"] = int(o.get("words", len(o["text"].split())))
                if o["words"] != len(str(o["text"]).split()):
                    raise RuntimeError(f"official row word mismatch {o.get('example_id')}")
                rows.append(o)
    if sum(int(r["words"]) for r in rows) != TOTAL_WORDS:
        raise RuntimeError("official pool is not 10M words")
    return rows


def effective_treatment_source_targets(ref: dict) -> dict[str, int]:
    target = collections.Counter({s: 0 for s in SOURCES})
    for s in SOURCES:
        target[s] += int(ref.get("treatment_pool_source_words", {}).get(s, 0))
    for s, w in ref.get("selected_pair_source_words", {}).items():
        target[str(s)] += int(w)
    if sum(target.values()) != TOTAL_WORDS:
        raise RuntimeError(f"targets sum {sum(target.values())}")
    return dict(target)


def pack_segments(segments: list[Segment], source_label: str, example_base: int) -> list[Row]:
    rows: list[Row] = []
    cur_text: list[str] = []
    cur_ids: list[str] = []
    cur_sides: list[str] = []
    cur_words = 0
    cur_sources: collections.Counter[str] = collections.Counter()
    for s in segments:
        if cur_words and cur_words + s.words > MAX_ROW_WORDS:
            rows.append(Row(
                text=" ".join(cur_text), words=cur_words, source=source_label,
                example_id=example_base + len(rows), pair_ids=list(cur_ids), sides=list(cur_sides), component_sources=dict(cur_sources),
            ))
            cur_text, cur_ids, cur_sides, cur_words, cur_sources = [], [], [], 0, collections.Counter()
        cur_text.append(s.text)
        cur_ids.append(s.pair_id)
        cur_sides.append(s.side)
        cur_words += s.words
        cur_sources[s.source] += s.words
    if cur_text:
        rows.append(Row(
            text=" ".join(cur_text), words=cur_words, source=source_label,
            example_id=example_base + len(rows), pair_ids=list(cur_ids), sides=list(cur_sides), component_sources=dict(cur_sources),
        ))
    if sum(r.words for r in rows) != sum(s.words for s in segments):
        raise RuntimeError("packed segments lost words")
    return rows


def official_source_streams(official_rows: list[dict], selected_ids: set[int]) -> tuple[dict[str, list[str]], dict[str, list[int]], dict[str, int]]:
    streams: dict[str, list[str]] = {s: [] for s in SOURCES}
    id_streams: dict[str, list[int]] = {s: [] for s in SOURCES}
    rows_by_source: dict[str, tuple[list[dict], list[dict]]] = {s: ([], []) for s in SOURCES}
    for r in official_rows:
        s = str(r["source"])
        if s not in rows_by_source:
            rows_by_source[s] = ([], [])
        bucket = rows_by_source[s][1] if int(r["example_id"]) in selected_ids else rows_by_source[s][0]
        bucket.append(r)
    available: dict[str, int] = {}
    for s in SOURCES:
        non, sel = rows_by_source[s]
        random.Random(RNG_SEED + SOURCE_SEED_OFFSET[s]).shuffle(non)
        random.Random(RNG_SEED + SOURCE_SEED_OFFSET[s] + 17).shuffle(sel)
        for r in non + sel:
            ws = str(r["text"]).split()
            streams[s].extend(ws)
            id_streams[s].extend([int(r["example_id"])] * len(ws))
        available[s] = len(streams[s])
        if not streams[s]:
            raise RuntimeError(f"empty source {s}")
    return streams, id_streams, available


def make_filler_rows(filler_targets: dict[str, int], streams: dict[str, list[str]], id_streams: dict[str, list[int]], selected_ids: set[int]) -> tuple[list[Row], dict[str, int], int]:
    remaining = collections.Counter(filler_targets)
    cursors = {s: 0 for s in SOURCES}
    reuse = {s: max(0, filler_targets.get(s, 0) - len(streams[s])) for s in SOURCES}
    words: list[str] = []
    marks: list[str] = []
    ids: list[int] = []
    rng = random.Random(RNG_SEED + 11)
    while sum(remaining.values()) > 0:
        active = [s for s in SOURCES if remaining[s] > 0]
        weights = [remaining[s] for s in active]
        src = rng.choices(active, weights=weights, k=1)[0]
        take = min(remaining[src], rng.randint(40, 160))
        for j in range(take):
            k = (cursors[src] + j) % len(streams[src])
            words.append(streams[src][k])
            marks.append(src)
            ids.append(id_streams[src][k])
        cursors[src] += take
        remaining[src] -= take
    selected_reuse = sum(1 for eid in ids if eid in selected_ids)
    rows: list[Row] = []
    pos = 0
    while pos < len(words):
        L = min(MAX_ROW_WORDS, len(words) - pos)
        seg = words[pos:pos+L]
        c = collections.Counter(marks[pos:pos+L])
        common = c.most_common(1)[0][0]
        rows.append(Row(" ".join(seg), L, "official_filler_source_matched::" + common, 960000 + len(rows), component_sources=dict(c)))
        pos += L
    return rows, reuse, selected_reuse


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path | None = None) -> None:
    mf = meta_path.open("w", encoding="utf-8") if meta_path else None
    try:
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.words != len(r.text.split()):
                    raise RuntimeError(f"row word mismatch {i}")
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                if mf and (r.pair_ids or r.component_sources):
                    m = {"row_index": i, "example_id": r.example_id, "words": r.words, "source": r.source}
                    if r.pair_ids:
                        m["pair_ids"] = r.pair_ids
                        m["sides"] = r.sides
                    if r.component_sources:
                        m["component_sources"] = r.component_sources
                    mf.write(json.dumps(m, ensure_ascii=False) + "\n")
    finally:
        if mf:
            mf.close()


def write_train(path: pathlib.Path, rows: list[Row]) -> None:
    total = 0
    n = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for p in range(PASSES):
            order = list(range(n))
            random.Random(RNG_SEED + 100 + p).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training total {total}")


def main() -> None:
    ref = json.loads(REF_META.read_text(encoding="utf-8"))
    if ref.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise RuntimeError(f"bad ref status {ref.get('status')}")
    pairs = load_selected()
    selected_ids = {p.example_id for p in pairs}
    # Shuffle segment order within side so pairmates are not adjacent even before final row shuffle.
    rng = random.Random(RNG_SEED)
    originals = [Segment(p.pair_id, "original", p.original, p.original_words, p.source, p.example_id) for p in pairs]
    rewrites = [Segment(p.pair_id, "rewrite", p.rewrite, p.rewrite_words, p.source, p.example_id) for p in pairs]
    rng.shuffle(originals)
    rng.shuffle(rewrites)
    original_rows = pack_segments(originals, "qwen_separated_original_rows", 950000)
    rewrite_rows = pack_segments(rewrites, "qwen_separated_rewrite_rows", 955000)
    pair_rows = original_rows + rewrite_rows
    pair_words = sum(r.words for r in pair_rows)
    ref_pair_words = int(ref["selected_pair_words"])
    if pair_words != ref_pair_words:
        raise RuntimeError(f"pair words {pair_words} != reference {ref_pair_words}")

    targets = effective_treatment_source_targets(ref)
    pair_words_by_source = collections.Counter()
    for p in pairs:
        pair_words_by_source[p.source] += p.original_words + p.rewrite_words
    filler_targets = {s: int(targets[s] - pair_words_by_source.get(s, 0)) for s in SOURCES}
    if any(v < 0 for v in filler_targets.values()):
        raise RuntimeError(f"pair words exceed source target: {filler_targets}")
    official = load_official_pool()
    streams, id_streams, available = official_source_streams(official, selected_ids)
    filler_rows, reuse, selected_reuse = make_filler_rows(filler_targets, streams, id_streams, selected_ids)

    pool = pair_rows + filler_rows
    random.Random(RNG_SEED + 2).shuffle(pool)
    if sum(r.words for r in pool) != TOTAL_WORDS:
        raise RuntimeError("pool word total mismatch")
    observed = collections.Counter()
    for r in pool:
        if r.component_sources:
            observed.update(r.component_sources)
    obs_src = {s: observed[s] for s in SOURCES}
    if obs_src != targets:
        raise RuntimeError(f"observed source mismatch {obs_src} != {targets}")

    pool_path = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/training_corpora/qwen_separated_pair_10M.jsonl')
    train_path = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/training_corpora/qwen_separated_pair_100M.jsonl')
    row_meta_path = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/qwen_separated_pair_rows_meta.jsonl')
    write_pool(pool_path, pool, row_meta_path)
    write_train(train_path, pool)

    payload = {
        "status": "QWEN_SEPARATED_PAIR_CONTROL_MATERIALIZED",
        "purpose": "same selected originals and rewrites as aligned treatment, but original/rewrite sides are in separate rows/windows to remove same-window self-attention correspondence",
        "reference_clean_metadata": str(REF_META),
        "selected_pairs": len(pairs),
        "selected_unique_example_ids": len(selected_ids),
        "pair_words": pair_words,
        "pair_word_fraction": round(pair_words / TOTAL_WORDS, 6),
        "matches_clean_qwen_selected_pair_words": pair_words == ref_pair_words,
        "same_original_and_rewrite_multisets_as_selected_pairs": True,
        "keeps_all_step028_selected_pair_ids": True,
        "pair_boundary_preserved": True,
        "original_and_rewrite_in_same_row": False,
        "uses_qwen_words": True,
        "uses_only_step028_selected_qwen_rewrites": True,
        "row_length_sequence_matched_to_qwen_treatment": False,
        "source_totals_matched_to_qwen_effective_targets": True,
        "target_effective_source_words": targets,
        "observed_source_words": obs_src,
        "pair_source_words": {s: pair_words_by_source[s] for s in SOURCES},
        "filler_source_targets": filler_targets,
        "official_source_words_available": available,
        "filler_source_reuse_words_due_to_target_exceeding_available": reuse,
        "selected_id_words_in_filler_after_nonselected_first_policy": selected_reuse,
        "original_side_rows": len(original_rows),
        "rewrite_side_rows": len(rewrite_rows),
        "filler_rows": len(filler_rows),
        "pool_rows": len(pool),
        "original_row_word_stats": stats([r.words for r in original_rows]),
        "rewrite_row_word_stats": stats([r.words for r in rewrite_rows]),
        "all_pool_word_totals": {"qwen_separated_pair_10M": sum(r.words for r in pool), "qwen_separated_pair_100M": TOTAL_WORDS * PASSES},
        "files": {"pool": str(pool_path), "training": str(train_path), "row_meta": str(row_meta_path)},
        "sha256": {"pool": sha(pool_path), "training": sha(train_path), "row_meta": sha(row_meta_path)},
    }
    meta_path = _public_path('experiments/archive/compact_experience/data/qwen_separated_pair_control/qwen_separated_pair_metadata.json')
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
