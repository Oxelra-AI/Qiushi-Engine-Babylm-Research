#!/usr/bin/env python3
"""Materialize a factorized FineWeb source-breadth / second-view contrast.

This script is intentionally separate from generation.  It consumes accepted
source--rewrite pairs (for example the research high-anchor pilot analysis, or a
larger later generation) and creates matched BabyLM-style 10M pools, optionally
100M ten-pass training JSONL files.

Scientific decomposition:
  A. official_lengthmatched: official BabyLM words chunked to the exact changed-
     block row lengths plus identical official filler.
  B. fineweb_packet_local_repeat: the same FineWeb source sentences, but the
     second-view word budget is filled by packet-local source repetition.
  C. fineweb_source_rewrite: the same FineWeb source sentences with the accepted
     simplified second view.

Then B-A estimates the broad-source substrate under a repetition view, C-B
estimates generated second-view value inside the same FineWeb source mixture, and
C-A estimates the combined change.  The design requires that the
pending SimpleWiki result should guide task-family emphasis, not collapse the
whole second-view program into one binary verdict.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import random
import re
import time
from typing import Any, Iterable

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
OFFICIAL_POOL_DEFAULT = pathlib.Path("experiments/archive/compact_experience/data/mixture/official_pool.jsonl")
ACCEPTED_DEFAULT = WORKSPACE / "data" / "high_anchor_fineweb_generation_analysis" / "fineweb_high_anchor_accepted_rewrites.jsonl"
OUT_DEFAULT = WORKSPACE / "data" / "fineweb_factor_contrast"
NOTE_DEFAULT = (WORKSPACE / 'notes'.parents[3] / 'research/notes/frontier_consolidation/fineweb_factor_contrast_materialization.md')

TOTAL_WORDS = 10_000_000
PASSES = 10
OFFICIAL_ROW_WORDS = 160
MAX_PACKET_WORDS = 160
RNG_SEED = 82912012


@dataclasses.dataclass
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    sentence_id: str
    doc_id: str
    domain_hits: list[str]


@dataclasses.dataclass
class Row:
    text: str
    words: int
    example_id: int
    source: str
    pair_ids: list[str] | None = None
    component_sources: dict[str, int] | None = None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len((text or "").split())


def norm_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def coerce_pairs(path: pathlib.Path) -> list[Pair]:
    rows = read_jsonl(path)
    pairs: list[Pair] = []
    for i, r in enumerate(rows):
        if r.get("accepted_for_next_construction") is False:
            continue
        src = norm_text(str(r.get("source_text") or r.get("original") or ""))
        rew = norm_text(str(r.get("rewrite_text") or r.get("rewrite") or r.get("output") or ""))
        if not src or not rew:
            continue
        sw = wc(src)
        rw = wc(rew)
        if sw <= 0 or rw <= 0:
            continue
        pair_id = str(r.get("prompt_id") or r.get("pair_id") or f"fineweb_pair_{i:06d}")
        pairs.append(Pair(
            pair_id=pair_id,
            source_text=src,
            rewrite_text=rew,
            source_words=sw,
            rewrite_words=rw,
            pair_words=sw + rw,
            sentence_id=str(r.get("sentence_id") or ""),
            doc_id=str(r.get("doc_id") or ""),
            domain_hits=[str(x) for x in (r.get("domain_hits") or [])],
        ))
    return pairs


def select_pairs(pairs: list[Pair], target_pair_words: int, seed: int) -> list[Pair]:
    if target_pair_words <= 0:
        target_pair_words = sum(p.pair_words for p in pairs)
    rng = random.Random(seed)
    # Prefer document diversity and broadly factual anchors, while keeping deterministic
    # randomness so a larger generation does not select only one content type.
    by_doc: dict[str, list[Pair]] = collections.defaultdict(list)
    for p in pairs:
        by_doc[p.doc_id or p.pair_id].append(p)
    docs = list(by_doc)
    rng.shuffle(docs)
    ordered: list[Pair] = []
    for d in docs:
        candidates = by_doc[d]
        candidates.sort(key=lambda p: (-(len(p.domain_hits) > 0), abs(p.source_words - 24), p.pair_id))
        ordered.append(candidates[0])
    extras = [p for ps in by_doc.values() for p in ps[1:]]
    rng.shuffle(extras)
    ordered.extend(extras)

    selected: list[Pair] = []
    total = 0
    for p in ordered:
        if p.pair_words > MAX_PACKET_WORDS:
            continue
        if total + p.pair_words <= target_pair_words:
            selected.append(p)
            total += p.pair_words
    while selected and total % OFFICIAL_ROW_WORDS != 0:
        p = selected.pop()
        total -= p.pair_words
    return selected


def repeat_source_words(source_text: str, n_words: int) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    out: list[str] = []
    while len(out) < n_words:
        need = n_words - len(out)
        out.extend(toks[:need])
    return " ".join(out)


def pack_pair_rows(pairs: list[Pair], use_rewrite: bool) -> list[Row]:
    rows: list[Row] = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    cur_domains: collections.Counter[str] = collections.Counter()
    for p in pairs:
        second = p.rewrite_text if use_rewrite else repeat_source_words(p.source_text, p.rewrite_words)
        text = f"{p.source_text} {second}".strip()
        L = wc(text)
        if L != p.pair_words:
            raise RuntimeError(f"pair {p.pair_id} expected {p.pair_words} words but built {L}")
        if cur_words and cur_words + L > MAX_PACKET_WORDS:
            rows.append(Row(
                text=" ".join(cur_segments), words=cur_words, example_id=700000 + len(rows),
                source="fineweb_source_rewrite" if use_rewrite else "fineweb_packet_local_repeat",
                pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains),
            ))
            cur_segments, cur_words, cur_pair_ids, cur_domains = [], 0, [], collections.Counter()
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(p.pair_id)
        for d in p.domain_hits or ["no_domain"]:
            cur_domains[d] += L
    if cur_segments:
        rows.append(Row(
            text=" ".join(cur_segments), words=cur_words, example_id=700000 + len(rows),
            source="fineweb_source_rewrite" if use_rewrite else "fineweb_packet_local_repeat",
            pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains),
        ))
    for r in rows:
        if r.words != wc(r.text):
            raise RuntimeError("packed row word mismatch")
        if r.words > MAX_PACKET_WORDS:
            raise RuntimeError("packed row exceeds max packet words")
    return rows


def read_official_pool(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for r in rows:
        r["words"] = int(r.get("words") or wc(str(r.get("text") or "")))
        if r["words"] != wc(str(r.get("text") or "")):
            raise RuntimeError(f"official row word mismatch example_id={r.get('example_id')}")
    total = sum(int(r["words"]) for r in rows)
    if total != TOTAL_WORDS:
        raise RuntimeError(f"official pool has {total} words, expected {TOTAL_WORDS}")
    return rows


def official_word_stream_rows(official: list[dict[str, Any]], lengths: list[int], offset_words: int, source_prefix: str, example_base: int) -> list[Row]:
    words: list[str] = []
    sources: list[str] = []
    for r in official:
        ws = str(r["text"]).split()
        words.extend(ws)
        sources.extend([str(r.get("source", "official"))] * len(ws))
    offset_words %= len(words)
    words = words[offset_words:] + words[:offset_words]
    sources = sources[offset_words:] + sources[:offset_words]
    out: list[Row] = []
    pos = 0
    for i, L in enumerate(lengths):
        seg = words[pos:pos+L]
        seg_src = sources[pos:pos+L]
        if len(seg) != L:
            raise RuntimeError("official stream exhausted")
        common = collections.Counter(seg_src).most_common(1)[0][0]
        out.append(Row(text=" ".join(seg), words=L, example_id=example_base + i, source=f"{source_prefix}::{common}"))
        pos += L
    return out


def official_filler_rows(official: list[dict[str, Any]], needed_words: int, seed: int) -> list[Row]:
    if needed_words % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError(f"needed filler words not divisible by {OFFICIAL_ROW_WORDS}: {needed_words}")
    needed = needed_words // OFFICIAL_ROW_WORDS
    rows = list(official)
    random.Random(seed).shuffle(rows)
    chosen = rows[:needed]
    if len(chosen) != needed:
        raise RuntimeError(f"need {needed} filler rows, got {len(chosen)}")
    return [Row(text=str(r["text"]), words=int(r["words"]), example_id=int(r.get("example_id") or (800000+i)), source=str(r.get("source", "official"))) for i, r in enumerate(chosen)]


def row_to_record(r: Row) -> dict[str, Any]:
    rec: dict[str, Any] = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
    return rec


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path | None = None) -> None:
    write_jsonl(path, [row_to_record(r) for r in rows])
    if meta_path is not None:
        with meta_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.pair_ids:
                    f.write(json.dumps({
                        "row_index": i,
                        "example_id": r.example_id,
                        "words": r.words,
                        "pair_ids": r.pair_ids,
                        "component_sources": r.component_sources or {},
                    }, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[Row], pass_orders: list[list[int]]) -> None:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"{path} total {total}")


def source_word_counts(rows: list[Row]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c.most_common())


def rows_equal_after_prefix(a: list[Row], b: list[Row], prefix_len: int) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a[prefix_len:], b[prefix_len:]):
        if x.text != y.text or x.words != y.words or x.example_id != y.example_id or x.source != y.source:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default=str(ACCEPTED_DEFAULT))
    ap.add_argument("--official-pool", default=str(OFFICIAL_POOL_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--target-pair-words", type=int, default=1_600_000)
    ap.add_argument("--min-pair-words", type=int, default=500_000)
    ap.add_argument("--write-training", action="store_true")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    t0 = time.time()
    accepted_path = pathlib.Path(args.accepted)
    official_path = pathlib.Path(args.official_pool)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_all = coerce_pairs(accepted_path)
    selected = select_pairs(pairs_all, args.target_pair_words, args.seed)
    selected_words = sum(p.pair_words for p in selected)
    if selected_words < args.min_pair_words:
        raise RuntimeError(
            f"Only {selected_words} selected pair words from {len(selected)} pairs; below min {args.min_pair_words}. "
            "Do not materialize a pretraining contrast yet; generate more or lower min only for a labeled smoke run."
        )
    if selected_words % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError("selected pair words not divisible by official row size")

    official = read_official_pool(official_path)
    rewrite_rows = pack_pair_rows(selected, use_rewrite=True)
    repeat_rows = pack_pair_rows(selected, use_rewrite=False)
    lengths = [r.words for r in rewrite_rows]
    if lengths != [r.words for r in repeat_rows]:
        raise RuntimeError("rewrite/repeat changed-block row lengths differ")
    official_block = official_word_stream_rows(official, lengths, offset_words=409 * OFFICIAL_ROW_WORDS, source_prefix="official_lengthmatched", example_base=900000)
    filler = official_filler_rows(official, TOTAL_WORDS - selected_words, seed=args.seed + 1)

    arms = {
        "official_lengthmatched": official_block + filler,
        "fineweb_packet_local_repeat": repeat_rows + filler,
        "fineweb_source_rewrite": rewrite_rows + filler,
    }
    for name, rows in arms.items():
        total = sum(r.words for r in rows)
        if total != TOTAL_WORDS:
            raise RuntimeError(f"{name} word total {total}")
        if [r.words for r in rows] != [r.words for r in arms["fineweb_source_rewrite"]]:
            raise RuntimeError(f"{name} row length sequence differs")

    pool_paths: dict[str, str] = {}
    train_paths: dict[str, str] = {}
    row_meta_paths: dict[str, str] = {}
    for name, rows in arms.items():
        p = out_dir / f"{name}_10M.jsonl"
        meta_p = out_dir / f"{name}_changed_block_rows_meta.jsonl"
        write_pool(p, rows, meta_p if name.startswith("fineweb_") else None)
        pool_paths[name] = str(p)
        if name.startswith("fineweb_"):
            row_meta_paths[name] = str(meta_p)

    n_rows = len(arms["fineweb_source_rewrite"])
    pass_orders = []
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(args.seed + 1000 + pass_i).shuffle(order)
        pass_orders.append(order)
    if args.write_training:
        for name, rows in arms.items():
            p = out_dir / f"{name}_100M.jsonl"
            write_training(p, rows, pass_orders)
            train_paths[name] = str(p)

    selected_path = out_dir / "selected_pairs.jsonl"
    write_jsonl(selected_path, [dataclasses.asdict(p) for p in selected])
    hashes = {pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in pool_paths.values()}
    if train_paths:
        hashes.update({pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in train_paths.values()})
    hashes["selected_pairs.jsonl"] = sha256_file(selected_path)

    audit = {
        "row_length_sequence_identical": len({tuple(r.words for r in rows) for rows in arms.values()}) == 1,
        "identical_filler_after_changed_prefix": {
            "repeat_vs_rewrite": rows_equal_after_prefix(arms["fineweb_packet_local_repeat"], arms["fineweb_source_rewrite"], len(rewrite_rows)),
            "official_vs_rewrite": rows_equal_after_prefix(arms["official_lengthmatched"], arms["fineweb_source_rewrite"], len(rewrite_rows)),
        },
        "word_totals": {name: sum(r.words for r in rows) for name, rows in arms.items()},
        "row_counts": {name: len(rows) for name, rows in arms.items()},
        "changed_block_rows": len(rewrite_rows),
        "changed_block_words": selected_words,
        "changed_block_fraction": selected_words / TOTAL_WORDS,
        "filler_rows": len(filler),
        "filler_words": sum(r.words for r in filler),
    }

    by_domain = collections.Counter(d for p in selected for d in (p.domain_hits or ["no_domain"]))
    payload = {
        "status": "FINEWEB_FACTOR_CONTRAST_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Factorized source-breadth and second-view contrast: official lengthmatched vs FineWeb packet-local source repetition vs FineWeb source+rewrite.",
        "accepted_input": str(accepted_path),
        "accepted_input_sha256": sha256_file(accepted_path),
        "official_pool": str(official_path),
        "official_pool_sha256": sha256_file(official_path),
        "total_words_per_pool": TOTAL_WORDS,
        "passes": PASSES,
        "write_training": bool(args.write_training),
        "available_pairs": len(pairs_all),
        "available_pair_words": sum(p.pair_words for p in pairs_all),
        "selected_pairs": len(selected),
        "selected_pair_words": selected_words,
        "selected_source_words": sum(p.source_words for p in selected),
        "selected_rewrite_words": sum(p.rewrite_words for p in selected),
        "selected_unique_docs": len({p.doc_id for p in selected}),
        "selected_domain_hit_counts": dict(by_domain.most_common()),
        "selection_seed": args.seed,
        "target_pair_words": args.target_pair_words,
        "min_pair_words": args.min_pair_words,
        "audit": audit,
        "source_word_counts_by_arm": {name: source_word_counts(rows) for name, rows in arms.items()},
        "files": {
            "selected_pairs": str(selected_path),
            "pools": pool_paths,
            "training": train_paths,
            "row_meta": row_meta_paths,
            "note": str(args.note),
        },
        "sha256": hashes,
        "interpretation": {
            "source_breadth_effect": "fineweb_packet_local_repeat minus official_lengthmatched",
            "second_view_effect_within_fineweb": "fineweb_source_rewrite minus fineweb_packet_local_repeat",
            "combined_effect": "fineweb_source_rewrite minus official_lengthmatched",
            "compare_to_existing_official_second_view": "COMPACT_EXPERIENCE clean-Qwen minus selected_original_dup_all gives the analogous official-source rewrite value under older materialization; use task-family signatures rather than only a scalar mean.",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "fineweb_factor_contrast_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research FineWeb factor contrast materialization\n\n"]
    lines.append("This is a research-facing corpus factorization, not a final submission artifact. It separates broad factual source content from generated view structure.\n\n")
    lines.append(f"Accepted input: `{accepted_path}`. Selected {len(selected):,} pairs / {selected_words:,} source+view words, {selected_words / TOTAL_WORDS:.3%} of the 10M pool.\n\n")
    lines.append("Arms:\n\n")
    lines.append("- `official_lengthmatched`: official words chunked to the same changed-block row lengths plus identical official filler.\n")
    lines.append("- `fineweb_packet_local_repeat`: FineWeb sources with packet-local source repetition filling the second-view word budget plus identical filler.\n")
    lines.append("- `fineweb_source_rewrite`: FineWeb sources with accepted simplified rewrites plus identical filler.\n\n")
    lines.append(f"Changed block rows: {len(rewrite_rows):,}; filler rows: {len(filler):,}. Row-length sequence identical: {audit['row_length_sequence_identical']}. Filler identical after prefix: {audit['identical_filler_after_changed_prefix']}.\n\n")
    lines.append(f"Metadata: `{meta_path}`\n")
    note_path.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "metadata": str(meta_path),
        "note": str(note_path),
        "selected_pairs": len(selected),
        "selected_pair_words": selected_words,
        "changed_block_fraction": audit["changed_block_fraction"],
        "row_length_sequence_identical": audit["row_length_sequence_identical"],
        "write_training": bool(args.write_training),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
