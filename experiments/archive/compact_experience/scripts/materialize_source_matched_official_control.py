#!/usr/bin/env python3
"""research: materialize an official-only source-matched control for research.

The existing research official control matches the treatment's row-length sequence but
not its effective source mixture.  The research treatment contains official filler
plus Qwen pair rows; to test whether a score Delta is partly a domain/source-mixture
effect, this control uses only official training text but matches:
  * the Qwen treatment's whitespace row-length sequence; and
  * the Qwen treatment's effective per-source word counts, assigning each Qwen pair
    word back to the source of its original sentence.

Important construction choice: preserve contiguous official text order within each
source stream (row-shuffled but word order inside rows is intact).  Do NOT shuffle
individual words.  If a target exceeds the available official words for one source,
wrap the source stream deterministically and record the reuse; this can happen for
Gutenberg because selected Qwen-pair words are source-assigned and may increase a
source above its original official allocation.  This control is for internal causal
interpretation, not a final data recipe.
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
from dataclasses import dataclass

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
META_PATH = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
QWEN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
OFFICIAL_POOL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/source_matched_control')
CORP = _public_path('experiments/archive/compact_experience/data/source_matched_control/training_corpora')
OUT.mkdir(parents=True, exist_ok=True)
CORP.mkdir(parents=True, exist_ok=True)
TOTAL_WORDS = 10_000_000
PASSES = 10
RNG_SEED = 29043
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
SOURCE_SEED_OFFSET = {s: i * 1009 for i, s in enumerate(SOURCES)}

@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    component_sources: dict[str, int] | None = None


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_metadata() -> dict:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise RuntimeError(f"wrong materialization status: {meta.get('status')}")
    if not meta.get("row_length_sequence_matched"):
        raise RuntimeError("research row-length sequence was not matched")
    return meta


def treatment_lengths() -> list[int]:
    lengths = [int(o.get("words", len(o["text"].split()))) for o in load_jsonl(QWEN_POOL)]
    if sum(lengths) != TOTAL_WORDS:
        raise RuntimeError(f"qwen pool word total {sum(lengths)} != {TOTAL_WORDS}")
    return lengths


def effective_treatment_source_targets(meta: dict) -> dict[str, int]:
    # Official filler words in the treatment are already source-labeled in treatment_pool_source_words.
    # Qwen pair-packed words are assigned back to each original source using selected_pair_source_words.
    target = collections.Counter({s: 0 for s in SOURCES})
    pool_counts = meta.get("treatment_pool_source_words", {})
    for s in SOURCES:
        target[s] += int(pool_counts.get(s, 0))
    for s, w in meta.get("selected_pair_source_words", {}).items():
        target[s] += int(w)
    total = sum(target.values())
    if total != TOTAL_WORDS:
        raise RuntimeError(f"effective treatment source targets sum {total} != {TOTAL_WORDS}: {dict(target)}")
    return dict(target)


def official_source_rows() -> dict[str, list[list[str]]]:
    rows_by_source: dict[str, list[list[str]]] = {s: [] for s in SOURCES}
    for o in load_jsonl(OFFICIAL_POOL):
        s = str(o["source"])
        ws = str(o["text"]).split()
        w = int(o.get("words", len(ws)))
        if w != len(ws):
            raise RuntimeError(f"official row word mismatch {o.get('example_id')}")
        rows_by_source.setdefault(s, []).append(ws)
    if sum(sum(len(row) for row in rows) for rows in rows_by_source.values()) != TOTAL_WORDS:
        raise RuntimeError("official rows do not sum to 10M")
    for s in SOURCES:
        if not rows_by_source.get(s):
            raise RuntimeError(f"empty source {s}")
        rng = random.Random(RNG_SEED + SOURCE_SEED_OFFSET[s])
        rng.shuffle(rows_by_source[s])  # row-level shuffle only; word order inside rows is preserved.
    return rows_by_source


def flatten_source_streams(rows_by_source: dict[str, list[list[str]]]) -> dict[str, list[str]]:
    streams = {}
    for s, rows in rows_by_source.items():
        words: list[str] = []
        for row in rows:
            words.extend(row)
        streams[s] = words
    return streams


def make_source_interleaved_word_sequence(targets: dict[str, int], streams: dict[str, list[str]]) -> tuple[list[str], list[str], dict[str, int]]:
    remaining = collections.Counter(targets)
    cursors = {s: 0 for s in SOURCES}
    reuse_words = {s: max(0, targets[s] - len(streams[s])) for s in SOURCES}
    out_words: list[str] = []
    out_marks: list[str] = []
    rng = random.Random(RNG_SEED + 11)
    while sum(remaining.values()) > 0:
        active = [s for s in SOURCES if remaining[s] > 0]
        weights = [remaining[s] for s in active]
        s = rng.choices(active, weights=weights, k=1)[0]
        take = min(remaining[s], rng.randint(40, 160))
        stream = streams[s]
        start = cursors[s]
        for j in range(take):
            out_words.append(stream[(start + j) % len(stream)])
            out_marks.append(s)
        cursors[s] += take
        remaining[s] -= take
    if len(out_words) != TOTAL_WORDS:
        raise RuntimeError(f"word sequence length {len(out_words)}")
    return out_words, out_marks, reuse_words


def chunk_to_lengths(words: list[str], marks: list[str], lengths: list[int]) -> list[Row]:
    rows: list[Row] = []
    pos = 0
    for i, L in enumerate(lengths):
        seg = words[pos:pos+L]
        seg_marks = marks[pos:pos+L]
        if len(seg) != L:
            raise RuntimeError("length sequence exceeded source-matched stream")
        c = collections.Counter(seg_marks)
        rows.append(Row(" ".join(seg), L, "official_sourcematched::" + c.most_common(1)[0][0], 800000 + i, dict(c)))
        pos += L
    if pos != TOTAL_WORDS:
        raise RuntimeError(f"used {pos} words, not {TOTAL_WORDS}")
    return rows


def write_pool(path: pathlib.Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.words != len(r.text.split()):
                raise RuntimeError("row word mismatch")
            obj = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
            if r.component_sources:
                obj["component_sources"] = r.component_sources
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
        raise RuntimeError(f"train total {total}")


def main() -> None:
    meta = load_metadata()
    lengths = treatment_lengths()
    targets = effective_treatment_source_targets(meta)
    rows_by_source = official_source_rows()
    streams = flatten_source_streams(rows_by_source)
    source_available = {s: len(streams[s]) for s in SOURCES}
    words, marks, reuse_words = make_source_interleaved_word_sequence(targets, streams)
    rows = chunk_to_lengths(words, marks, lengths)
    observed = collections.Counter()
    for r in rows:
        if r.component_sources:
            observed.update(r.component_sources)
    if {s: observed[s] for s in SOURCES} != targets:
        raise RuntimeError(f"observed source counts differ: {dict(observed)} != {targets}")
    pool_path = _public_path('experiments/archive/compact_experience/data/source_matched_control/training_corpora/official_sourcematched_10M.jsonl')
    train_path = _public_path('experiments/archive/compact_experience/data/source_matched_control/training_corpora/official_sourcematched_100M.jsonl')
    write_pool(pool_path, rows)
    write_train(train_path, rows)
    payload = {
        "status": "OFFICIAL_SOURCE_MATCHED_CONTROL_MATERIALIZED",
        "control_purpose": "official-only text matching research qwen treatment row-length sequence and effective source word counts",
        "reference_step028_metadata": str(META_PATH),
        "row_count": len(rows),
        "word_total": sum(r.words for r in rows),
        "row_length_sequence_source": str(QWEN_POOL),
        "row_length_sequence_matched_to_qwen_treatment": True,
        "target_source_words": targets,
        "observed_source_words": {s: observed[s] for s in SOURCES},
        "official_source_words_available": source_available,
        "source_reuse_words_due_to_target_exceeding_available": reuse_words,
        "preserves_contiguous_official_text_within_source_stream": True,
        "word_level_shuffle": False,
        "uses_qwen_words": False,
        "uses_official_words_only": True,
        "files": {"pool": str(pool_path), "training": str(train_path)},
        "sha256": {"pool": sha(pool_path), "training": sha(train_path)},
    }
    meta_out = _public_path('experiments/archive/compact_experience/data/source_matched_control/source_matched_metadata.json')
    meta_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
