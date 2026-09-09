#!/usr/bin/env python3
"""research: selected-original duplicate-all control for the clean-Qwen route.

Scientific purpose
------------------
The first clean-Qwen result and controls support a positive paired-data effect, but
one remaining serious alternative explanation is selected-original exposure: the
Qwen pipeline selects clean, information-dense, rewriteable official sentences and
then gives them an extra semantically related view.  The earlier original-dup
control was useful but imperfect because it selected only 36,687 of 37,594 pairs
in order to match the Qwen pair-word budget exactly.

This control keeps *all* final research selected original IDs and replaces each
(original, Qwen rewrite) pair with (original, original).  It uses no Qwen words,
duplicates every selected original once, preserves complete duplicate-pair
boundaries, and fills the rest of the 10M-word pool with official text while
matching the clean-Qwen treatment's effective source word totals as closely as
possible.  It is not an exact row-length-sequence control; it is a mechanism
control for selection/exposure versus generated second-view correspondence.
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
OUT = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control')
CORP = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora')
OUT.mkdir(parents=True, exist_ok=True)
CORP.mkdir(parents=True, exist_ok=True)

TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
RNG_SEED = 32043
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
SOURCE_SEED_OFFSET = {s: i * 1009 for i, s in enumerate(SOURCES)}


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

    return {
        "n": len(vals),
        "min": min(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "max": max(vals),
    }


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
                original_words=int(o["original_words"]),
                source=str(o.get("source", "unknown")),
                example_id=int(o.get("example_id", -1)),
                cohort=str(o.get("cohort", "unknown")),
            ))
    if not pairs:
        raise RuntimeError("no selected pairs")
    for p in pairs:
        if p.original_words != len(p.original.split()):
            raise RuntimeError(f"word mismatch in selected pair {p.pair_id}")
        if 2 * p.original_words > MAX_ROW_WORDS:
            raise RuntimeError(f"selected original too long for duplicate row {p.pair_id}: {p.original_words}")
    return pairs


def load_official_pool() -> list[dict]:
    rows = []
    with OFFICIAL_POOL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            o["words"] = int(o.get("words", len(o["text"].split())))
            if o["words"] != len(str(o["text"]).split()):
                raise RuntimeError(f"official row word mismatch {o.get('example_id')}")
            rows.append(o)
    if sum(int(r["words"]) for r in rows) != TOTAL_WORDS:
        raise RuntimeError("official pool is not 10M words")
    return rows


def effective_treatment_source_targets(ref: dict) -> dict[str, int]:
    # Same definition as source-matched control: official filler source words plus
    # qwen pair words assigned back to the source of each original sentence.
    target = collections.Counter({s: 0 for s in SOURCES})
    pool_counts = ref.get("treatment_pool_source_words", {})
    for s in SOURCES:
        target[s] += int(pool_counts.get(s, 0))
    for s, w in ref.get("selected_pair_source_words", {}).items():
        target[str(s)] += int(w)
    total = sum(target.values())
    if total != TOTAL_WORDS:
        raise RuntimeError(f"effective treatment targets sum {total} != {TOTAL_WORDS}: {dict(target)}")
    return dict(target)


def pack_duplicate_pairs(pairs: list[Pair]) -> list[Row]:
    rows: list[Row] = []
    cur_segments: list[str] = []
    cur_ids: list[str] = []
    cur_words = 0
    cur_sources: collections.Counter[str] = collections.Counter()
    for p in pairs:
        pw = 2 * p.original_words
        if cur_words and cur_words + pw > MAX_ROW_WORDS:
            rows.append(Row(
                text=" ".join(cur_segments),
                words=cur_words,
                source="official_selected_original_dup_all_packed",
                example_id=930000 + len(rows),
                pair_ids=list(cur_ids),
                n_pairs=len(cur_ids),
                component_sources=dict(cur_sources),
            ))
            cur_segments, cur_ids, cur_words, cur_sources = [], [], 0, collections.Counter()
        cur_segments.extend([p.original, p.original])
        cur_ids.append(p.pair_id)
        cur_words += pw
        cur_sources[p.source] += pw
    if cur_segments:
        rows.append(Row(
            text=" ".join(cur_segments),
            words=cur_words,
            source="official_selected_original_dup_all_packed",
            example_id=930000 + len(rows),
            pair_ids=list(cur_ids),
            n_pairs=len(cur_ids),
            component_sources=dict(cur_sources),
        ))
    expected = sum(2 * p.original_words for p in pairs)
    if sum(r.words for r in rows) != expected:
        raise RuntimeError("packed duplicate rows lost words")
    return rows


def official_source_streams(official_rows: list[dict], selected_ids: set[int]) -> tuple[dict[str, list[str]], dict[str, list[int]], dict[str, int]]:
    # For each source, put non-selected rows before selected rows so filler minimizes
    # additional selected-original exposure beyond the explicit duplicate rows.
    streams: dict[str, list[str]] = {s: [] for s in SOURCES}
    id_streams: dict[str, list[int]] = {s: [] for s in SOURCES}
    available_words: dict[str, int] = {s: 0 for s in SOURCES}
    rows_by_source: dict[str, tuple[list[dict], list[dict]]] = {s: ([], []) for s in SOURCES}
    for r in official_rows:
        s = str(r["source"])
        if s not in rows_by_source:
            rows_by_source[s] = ([], [])
        bucket = rows_by_source[s][1] if int(r["example_id"]) in selected_ids else rows_by_source[s][0]
        bucket.append(r)
    for s in SOURCES:
        non, sel = rows_by_source[s]
        random.Random(RNG_SEED + SOURCE_SEED_OFFSET[s]).shuffle(non)
        random.Random(RNG_SEED + SOURCE_SEED_OFFSET[s] + 17).shuffle(sel)
        for r in non + sel:
            ws = str(r["text"]).split()
            streams[s].extend(ws)
            id_streams[s].extend([int(r["example_id"])] * len(ws))
        available_words[s] = len(streams[s])
        if available_words[s] == 0:
            raise RuntimeError(f"empty source stream {s}")
    return streams, id_streams, available_words


def make_filler_rows(filler_targets: dict[str, int], streams: dict[str, list[str]], id_streams: dict[str, list[int]], selected_ids: set[int]) -> tuple[list[Row], dict[str, int], dict[str, int]]:
    # Interleave source chunks to match source totals, preserving within-source word order.
    remaining = collections.Counter(filler_targets)
    cursors = {s: 0 for s in SOURCES}
    reuse_words = {s: max(0, filler_targets.get(s, 0) - len(streams[s])) for s in SOURCES}
    out_words: list[str] = []
    out_marks: list[str] = []
    out_ids: list[int] = []
    rng = random.Random(RNG_SEED + 11)
    while sum(remaining.values()) > 0:
        active = [s for s in SOURCES if remaining[s] > 0]
        weights = [remaining[s] for s in active]
        s = rng.choices(active, weights=weights, k=1)[0]
        take = min(remaining[s], rng.randint(40, 160))
        sw = streams[s]
        si = id_streams[s]
        start = cursors[s]
        for j in range(take):
            k = (start + j) % len(sw)
            out_words.append(sw[k])
            out_marks.append(s)
            out_ids.append(si[k])
        cursors[s] += take
        remaining[s] -= take
    if len(out_words) != sum(filler_targets.values()):
        raise RuntimeError("filler word sequence length mismatch")
    selected_reuse_words = sum(1 for eid in out_ids if eid in selected_ids)
    rows: list[Row] = []
    pos = 0
    while pos < len(out_words):
        L = min(MAX_ROW_WORDS, len(out_words) - pos)
        seg = out_words[pos:pos+L]
        marks = out_marks[pos:pos+L]
        ids = out_ids[pos:pos+L]
        c = collections.Counter(marks)
        common_source = c.most_common(1)[0][0]
        common_example = collections.Counter(ids).most_common(1)[0][0]
        rows.append(Row(
            text=" ".join(seg),
            words=L,
            source="official_filler_source_matched::" + common_source,
            example_id=940000 + len(rows),
            component_sources=dict(c),
        ))
        pos += L
    return rows, reuse_words, {"selected_id_words_in_filler": selected_reuse_words}


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path | None = None) -> None:
    mf = meta_path.open("w", encoding="utf-8") if meta_path else None
    try:
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.words != len(r.text.split()):
                    raise RuntimeError(f"row word mismatch at {i}: {r.words} != {len(r.text.split())}")
                obj = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if mf and (r.pair_ids or r.component_sources):
                    m = {"row_index": i, "example_id": r.example_id, "words": r.words, "source": r.source}
                    if r.pair_ids:
                        m.update({"pair_ids": r.pair_ids, "n_pairs": r.n_pairs})
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
    targets = effective_treatment_source_targets(ref)
    dup_rows = pack_duplicate_pairs(pairs)
    dup_words_by_source = collections.Counter()
    dup_pairs_by_source = collections.Counter()
    for p in pairs:
        dup_words_by_source[p.source] += 2 * p.original_words
        dup_pairs_by_source[p.source] += 1
    filler_targets = {s: int(targets[s] - dup_words_by_source.get(s, 0)) for s in SOURCES}
    if any(v < 0 for v in filler_targets.values()):
        raise RuntimeError(f"duplicate selected originals exceed source target: {filler_targets}")
    official = load_official_pool()
    streams, id_streams, available = official_source_streams(official, selected_ids)
    filler_rows, reuse_words, filler_extra = make_filler_rows(filler_targets, streams, id_streams, selected_ids)
    pool = dup_rows + filler_rows
    random.Random(RNG_SEED + 2).shuffle(pool)
    if sum(r.words for r in pool) != TOTAL_WORDS:
        raise RuntimeError(f"pool word total {sum(r.words for r in pool)}")
    observed = collections.Counter()
    for r in pool:
        if r.component_sources:
            observed.update(r.component_sources)
        elif r.source in SOURCES:
            observed[r.source] += r.words
    # component_sources on duplicate rows and filler rows should exactly recover targets.
    obs_src = {s: observed[s] for s in SOURCES}
    if obs_src != targets:
        raise RuntimeError(f"observed source mismatch {obs_src} != {targets}")

    pool_path = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_10M.jsonl')
    train_path = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_100M.jsonl')
    row_meta_path = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_rows_meta.jsonl')
    write_pool(pool_path, pool, row_meta_path)
    write_train(train_path, pool)

    payload = {
        "status": "SELECTED_ORIGINAL_DUP_ALL_CONTROL_MATERIALIZED",
        "purpose": "all selected official originals duplicated once, no Qwen words, source totals matched to clean-Qwen effective source mixture; tests selected-original exposure/redundancy vs generated second view",
        "reference_clean_metadata": str(REF_META),
        "selected_pairs": len(pairs),
        "selected_unique_example_ids": len(selected_ids),
        "selected_original_words": sum(p.original_words for p in pairs),
        "duplicate_pair_words": sum(r.words for r in dup_rows),
        "duplicate_pair_word_fraction": round(sum(r.words for r in dup_rows) / TOTAL_WORDS, 6),
        "qwen_reference_pair_words": ref.get("selected_pair_words"),
        "qwen_reference_pair_word_fraction": ref.get("selected_pair_word_fraction"),
        "uses_qwen_words": False,
        "uses_official_words_only": True,
        "keeps_all_step028_selected_pair_ids": True,
        "pair_boundary_preserved": True,
        "pair_truncation": False,
        "row_length_sequence_matched_to_qwen_treatment": False,
        "source_totals_matched_to_qwen_effective_targets": True,
        "target_effective_source_words": targets,
        "observed_source_words": obs_src,
        "duplicate_source_words": {s: dup_words_by_source[s] for s in SOURCES},
        "duplicate_source_pairs": {s: dup_pairs_by_source[s] for s in SOURCES},
        "filler_source_targets": filler_targets,
        "official_source_words_available": available,
        "filler_source_reuse_words_due_to_target_exceeding_available": reuse_words,
        "selected_id_words_in_filler_after_nonselected_first_policy": filler_extra["selected_id_words_in_filler"],
        "duplicate_pair_rows": len(dup_rows),
        "filler_rows": len(filler_rows),
        "pool_rows": len(pool),
        "duplicate_pair_row_word_stats": stats([r.words for r in dup_rows]),
        "filler_row_word_stats": stats([r.words for r in filler_rows]),
        "all_pool_word_totals": {"selected_original_dup_all_10M": sum(r.words for r in pool), "selected_original_dup_all_100M": TOTAL_WORDS * PASSES},
        "files": {"pool": str(pool_path), "training": str(train_path), "row_meta": str(row_meta_path)},
        "sha256": {"pool": sha(pool_path), "training": sha(train_path), "row_meta": sha(row_meta_path)},
    }
    meta_path = _public_path('experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_metadata.json')
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
