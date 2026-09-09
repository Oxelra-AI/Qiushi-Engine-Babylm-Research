#!/usr/bin/env python3
"""Materialize follow-up controls for contextual one-pair cap-120.

These controls are prepared only after the cap-120 recipe is frozen.  They use the same
row count, row order, row-length sequence, selected pair IDs, and pass-order shuffles as
the cap-120 treatment, but alter one mechanistic factor:

1. unrelated_context: keep original+rewrite pair text, but replace the native left/right
   context with deterministic same-source official words from unrelated rows.
2. original_repeat: keep the true native context and original position, but replace the
   rewrite with a length-matched repeat/truncation of the original sentence tokens.

The scripts use only official training text and the already-counted research selected
Qwen rewrites.  They do not read official AoA/CDI words, child curves, AoA predictions,
AoA scores, or downstream evaluation outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
OFFICIAL = _public_path('experiments/archive/compact_experience/data/mixture/official_pool.jsonl')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
CAP_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120')
CAP_TREAT_10M = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/qwen_context_onepair_cap120_10M.jsonl')
CAP_META = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_pair_rows_cap120_meta.jsonl')
TOTAL_WORDS = 10_000_000
PASSES = 10
RNG_SEED = 282103


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repeat_to_len(words: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if not words:
        raise ValueError("cannot repeat empty original words")
    out: List[str] = []
    while len(out) < n:
        take = min(len(words), n - len(out))
        out.extend(words[:take])
    return out


def load_cap120_treatment_rows() -> List[Row]:
    rows = []
    for r in read_jsonl(CAP_TREAT_10M):
        rows.append(Row(text=str(r["text"]), words=int(r["words"]), source=str(r["source"]), example_id=int(r["example_id"])))
    total = sum(r.words for r in rows)
    if total != TOTAL_WORDS:
        raise RuntimeError(f"cap120 treatment total {total} != {TOTAL_WORDS}")
    return rows


def build_source_streams(official: List[Dict[str, Any]], excluded_examples: set[int]) -> Dict[str, List[str]]:
    by_source: Dict[str, List[str]] = collections.defaultdict(list)
    fallback_by_source: Dict[str, List[str]] = collections.defaultdict(list)
    for r in official:
        src = str(r["source"])
        ws = str(r["text"]).split()
        fallback_by_source[src].extend(ws)
        if int(r["example_id"]) not in excluded_examples:
            by_source[src].extend(ws)
    rng = random.Random(RNG_SEED + 909)
    for src in sorted(fallback_by_source):
        if not by_source[src]:
            by_source[src] = list(fallback_by_source[src])
        # Shuffle in row-sized chunks would be cleaner, but a deterministic token stream is
        # sufficient for a same-source-unrelated control whose row lengths are fixed elsewhere.
        chunks = [by_source[src][i:i+160] for i in range(0, len(by_source[src]), 160)]
        rng.shuffle(chunks)
        by_source[src] = [w for c in chunks for w in c]
    return dict(by_source)


def take_stream_words(streams: Dict[str, List[str]], positions: Dict[str, int], wraps: Dict[str, int], source: str, n: int) -> List[str]:
    if n == 0:
        return []
    stream = streams.get(source)
    if not stream:
        raise RuntimeError(f"no official stream for source {source}")
    out: List[str] = []
    pos = positions.get(source, 0)
    while len(out) < n:
        if pos >= len(stream):
            pos = 0
            wraps[source] += 1
        take = min(n - len(out), len(stream) - pos)
        out.extend(stream[pos:pos+take])
        pos += take
    positions[source] = pos
    return out


def validate_segment(row_words: List[str], meta: Dict[str, Any], pair: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    left_n = int(meta["left_take"])
    ow = int(meta["original_words"])
    rw = int(meta["rewrite_words"])
    right_n = int(meta["right_take"])
    if left_n + ow + rw + right_n != int(meta["words"]):
        raise RuntimeError(f"meta length mismatch for {meta['pair_id']}")
    if len(row_words) != int(meta["words"]):
        raise RuntimeError(f"row length mismatch for {meta['pair_id']}: {len(row_words)} vs {meta['words']}")
    left = row_words[:left_n]
    original = row_words[left_n:left_n+ow]
    rewrite = row_words[left_n+ow:left_n+ow+rw]
    right = row_words[left_n+ow+rw:]
    if original != str(pair["original"]).split():
        raise RuntimeError(f"original segment mismatch for {meta['pair_id']}")
    if rewrite != str(pair["rewrite"]).split():
        raise RuntimeError(f"rewrite segment mismatch for {meta['pair_id']}")
    if len(right) != right_n:
        raise RuntimeError(f"right segment mismatch for {meta['pair_id']}")
    return left, original, rewrite, right


def make_pass_orders(n_rows: int) -> List[List[int]]:
    orders: List[List[int]] = []
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(RNG_SEED + 100 + pass_i).shuffle(order)
        orders.append(order)
    return orders


def write_pool(path: pathlib.Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            ws = r.text.split()
            if len(ws) != r.words:
                raise RuntimeError(f"word mismatch for row {r.example_id}: {len(ws)} vs {r.words}")
            f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
            total += r.words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"pool total {total} != {TOTAL_WORDS}")


def write_training(path: pathlib.Path, rows: List[Row], orders: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for order in orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training total {total} != {TOTAL_WORDS * PASSES}")


def build_variant(variant: str, out_root: pathlib.Path) -> Dict[str, Any]:
    cap_rows = load_cap120_treatment_rows()
    meta_rows = read_jsonl(CAP_META)
    selected = {str(p["pair_id"]): p for p in read_jsonl(SELECTED)}
    official = read_jsonl(OFFICIAL)
    excluded = {int(p["example_id"]) for p in selected.values()}
    meta_by_idx = {int(m["row_index"]): m for m in meta_rows}
    if len(meta_by_idx) != 37_594:
        raise RuntimeError(f"unexpected pair meta rows {len(meta_by_idx)}")
    variant_rows = [Row(r.text, r.words, r.source, r.example_id) for r in cap_rows]
    streams = build_source_streams(official, excluded)
    positions: Dict[str, int] = collections.defaultdict(int)
    wraps: Dict[str, int] = collections.defaultdict(int)
    source_context_words = collections.Counter()
    changed = 0
    for idx in sorted(meta_by_idx):
        meta = meta_by_idx[idx]
        pid = str(meta["pair_id"])
        pair = selected.get(pid)
        if pair is None:
            raise RuntimeError(f"missing selected pair {pid}")
        row_words = cap_rows[idx].text.split()
        left, original, rewrite, right = validate_segment(row_words, meta, pair)
        source = str(meta["source"])
        if variant == "unrelated_context":
            left_new = take_stream_words(streams, positions, wraps, source, int(meta["left_take"]))
            right_new = take_stream_words(streams, positions, wraps, source, int(meta["right_take"]))
            new_words = left_new + original + rewrite + right_new
            new_source = f"qwen_unrelated_context_onepair::{source}"
        elif variant == "original_repeat":
            replacement = repeat_to_len(original, len(rewrite))
            new_words = left + original + replacement + right
            new_source = f"qwen_context_original_repeat::{source}"
        else:
            raise RuntimeError(f"unknown variant {variant}")
        if len(new_words) != int(meta["words"]):
            raise RuntimeError(f"variant length mismatch for {pid}: {len(new_words)} vs {meta['words']}")
        source_context_words[source] += int(meta["left_take"]) + int(meta["right_take"])
        variant_rows[idx] = Row(text=" ".join(new_words), words=len(new_words), source=new_source, example_id=900_000_000 + idx)
        changed += 1
    if changed != len(meta_rows):
        raise RuntimeError(f"changed {changed}, expected {len(meta_rows)}")
    if [r.words for r in variant_rows] != [r.words for r in cap_rows]:
        raise RuntimeError("row-length sequence changed")
    out_dir = out_root / variant
    corpus_dir = out_dir / "training_corpora"
    pool_path = corpus_dir / f"qwen_contextual_{variant}_cap120_10M.jsonl"
    train_path = corpus_dir / f"qwen_contextual_{variant}_cap120_100M.jsonl"
    write_pool(pool_path, variant_rows)
    write_training(train_path, variant_rows, make_pass_orders(len(variant_rows)))
    payload = {
        "status": "CONTEXTUAL_FOLLOWUP_CONTROL_MATERIALIZED",
        "variant": variant,
        "non_leakage_statement": "Uses only BabyLM training text plus research selected Qwen rewrites already counted in training data. No official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs are read or used.",
        "base_cap120_treatment": str(CAP_TREAT_10M),
        "base_cap120_meta": str(CAP_META),
        "rows": len(variant_rows),
        "contextual_pair_rows_changed": changed,
        "row_length_sequence_identical_to_cap120_treatment": True,
        "word_totals": {"pool_10M": sum(r.words for r in variant_rows), "train_100M": TOTAL_WORDS * PASSES},
        "source_context_words_requested": dict(sorted(source_context_words.items())),
        "unrelated_stream_wraps": dict(sorted(wraps.items())),
        "files": {"pool_10M": str(pool_path), "train_100M": str(train_path)},
        "sha256": {"pool_10M": sha_file(pool_path), "train_100M": sha_file(train_path)},
        "intended_contrast": {
            "unrelated_context": "Compare native-context cap120 treatment against same pair texts embedded in same-source unrelated official context with identical row lengths/order.",
            "original_repeat": "Compare native-context original+rewrite against true-context original plus length-matched original-token repeat, preserving row lengths/order and pair position.",
        }[variant],
    }
    meta_path = out_dir / f"contextual_{variant}_cap120_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"variant": variant, "metadata": str(meta_path), "train_100M": str(train_path), "sha256_train_100M": payload["sha256"]["train_100M"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["unrelated_context", "original_repeat"])
    ap.add_argument("--out_root", default=str(_public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120_followup_controls')))
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)
    results = [build_variant(v, out_root) for v in args.variants]
    print(json.dumps({"status": "ok", "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
