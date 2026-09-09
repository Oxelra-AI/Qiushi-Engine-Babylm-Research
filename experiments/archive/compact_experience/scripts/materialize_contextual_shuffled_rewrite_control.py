#!/usr/bin/env python3
"""Materialize a native-context shuffled-rewrite control for cap-120.

Purpose: preserve the cap-120 native official context, original sentence, row length,
selected original inventory, generated rewrite word-length distribution, and training pass
orders, while breaking pair-specific original->rewrite correspondence. This is a cleaner
control for Qwen-text/style/dose than an official-only control and a more direct test of
same-window correspondence than original-repeat.

It reads only BabyLM training-derived corpora/metadata and the already-counted research
selected Qwen rewrites. It never reads official AoA/CDI words, child curves, AoA outputs,
AoA predictions, or downstream evaluation results.
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
from typing import Any, Dict, List, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
CAP_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120')
CAP_TREAT_10M = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/qwen_context_onepair_cap120_10M.jsonl')
CAP_META = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_pair_rows_cap120_meta.jsonl')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/contextual_shuffled_rewrite_control/cap120')
TOTAL_WORDS = 10_000_000
PASSES = 10
RNG_SEED = 382019


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repeat_or_trim(words: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if not words:
        raise ValueError("cannot adapt empty rewrite")
    if len(words) >= n:
        return words[:n]
    out: List[str] = []
    while len(out) < n:
        out.extend(words[: min(len(words), n - len(out))])
    return out


def load_rows(path: pathlib.Path) -> List[Row]:
    out: List[Row] = []
    total = 0
    for r in read_jsonl(path):
        text = str(r["text"])
        words = int(r.get("words", len(text.split())))
        if len(text.split()) != words:
            raise RuntimeError(f"word mismatch in {path}: {len(text.split())} vs {words}")
        out.append(Row(text=text, words=words, source=str(r.get("source", "")), example_id=int(r.get("example_id", -1))))
        total += words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"{path} total {total} != {TOTAL_WORDS}")
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
        raise RuntimeError(f"right length mismatch for {meta['pair_id']}")
    return left, original, rewrite, right


def make_same_length_derangements(pairs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    by_len: Dict[int, List[str]] = collections.defaultdict(list)
    for pid, p in pairs.items():
        by_len[int(p["rewrite_words"])].append(pid)
    mapping: Dict[str, str] = {}
    rng = random.Random(RNG_SEED + 11)
    singleton_lengths: List[int] = []
    for L, pids in sorted(by_len.items()):
        pids = sorted(pids)
        if len(pids) == 1:
            singleton_lengths.append(L)
            continue
        shuffled = list(pids)
        # Try deterministic shuffles; if accidental fixed points remain, rotate them away.
        for _ in range(20):
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(pids, shuffled)):
                break
        if any(a == b for a, b in zip(pids, shuffled)):
            shuffled = pids[1:] + pids[:1]
        for src, dst in zip(pids, shuffled):
            if src == dst:
                raise RuntimeError(f"failed derangement for length {L}")
            mapping[src] = dst
    # Rare singleton rewrite lengths cannot be exactly deranged. Assign them to the nearest
    # non-self donor and record that its words will be adapted to the target length.
    all_pids = sorted(pairs)
    for pid in sorted(p for L in singleton_lengths for p in by_len[L]):
        target_len = int(pairs[pid]["rewrite_words"])
        candidates = [q for q in all_pids if q != pid]
        candidates.sort(key=lambda q: (abs(int(pairs[q]["rewrite_words"]) - target_len), str(pairs[q].get("source", "")) != str(pairs[pid].get("source", "")), q))
        if not candidates:
            raise RuntimeError("no donor rewrite candidates")
        mapping[pid] = candidates[0]
    if set(mapping) != set(pairs):
        missing = sorted(set(pairs) - set(mapping))[:10]
        raise RuntimeError(f"derangement missing {len(set(pairs)-set(mapping))} pids, e.g. {missing}")
    return mapping


def make_pass_orders(n_rows: int) -> List[List[int]]:
    orders: List[List[int]] = []
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(282103 + 100 + pass_i).shuffle(order)  # match cap-120/follow-up controls
        orders.append(order)
    return orders


def write_pool(path: pathlib.Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            ws = r.text.split()
            if len(ws) != r.words:
                raise RuntimeError(f"word mismatch while writing {path}: {len(ws)} vs {r.words}")
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
        raise RuntimeError(f"train total {total} != {TOTAL_WORDS * PASSES}")


def build(out_root: pathlib.Path) -> Dict[str, Any]:
    base_rows = load_rows(CAP_TREAT_10M)
    meta_rows = read_jsonl(CAP_META)
    selected = {str(p["pair_id"]): p for p in read_jsonl(SELECTED)}
    meta_by_idx = {int(m["row_index"]): m for m in meta_rows}
    if len(meta_by_idx) != len(meta_rows) or len(meta_rows) != 37_594:
        raise RuntimeError(f"unexpected meta rows: {len(meta_rows)} unique={len(meta_by_idx)}")
    mapping = make_same_length_derangements(selected)
    variant_rows = [Row(r.text, r.words, r.source, r.example_id) for r in base_rows]
    exact_same_length = 0
    adapted_length = 0
    same_source_donors = 0
    source_pairs = collections.Counter()
    length_delta_counter = collections.Counter()
    for idx in sorted(meta_by_idx):
        meta = meta_by_idx[idx]
        pid = str(meta["pair_id"])
        pair = selected[pid]
        donor = selected[mapping[pid]]
        left, original, rewrite, right = validate_segment(base_rows[idx].text.split(), meta, pair)
        target_len = int(meta["rewrite_words"])
        donor_words_raw = str(donor["rewrite"]).split()
        if len(donor_words_raw) == target_len:
            donor_words = donor_words_raw
            exact_same_length += 1
        else:
            donor_words = repeat_or_trim(donor_words_raw, target_len)
            adapted_length += 1
        if str(donor.get("source")) == str(pair.get("source")):
            same_source_donors += 1
        length_delta_counter[int(donor.get("rewrite_words", len(donor_words_raw))) - target_len] += 1
        source_pairs[f"{pair.get('source')}<-{donor.get('source')}"] += 1
        new_words = left + original + donor_words + right
        if len(new_words) != int(meta["words"]):
            raise RuntimeError(f"length mismatch for {pid}: {len(new_words)} vs {meta['words']}")
        variant_rows[idx] = Row(
            text=" ".join(new_words),
            words=len(new_words),
            source=f"qwen_context_shuffled_rewrite::{meta.get('source')}",
            example_id=910_000_000 + idx,
        )
    if [r.words for r in variant_rows] != [r.words for r in base_rows]:
        raise RuntimeError("row-length sequence changed")
    pool = out_root / "training_corpora" / "qwen_contextual_shuffled_rewrite_cap120_10M.jsonl"
    train = out_root / "training_corpora" / "qwen_contextual_shuffled_rewrite_cap120_100M.jsonl"
    write_pool(pool, variant_rows)
    write_training(train, variant_rows, make_pass_orders(len(variant_rows)))
    payload = {
        "status": "CONTEXTUAL_SHUFFLED_REWRITE_CONTROL_MATERIALIZED",
        "non_leakage_statement": "Uses only training-side cap-120 corpus/metadata and research selected Qwen rewrites. No official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs are read or used.",
        "base_cap120_treatment": str(CAP_TREAT_10M),
        "base_cap120_meta": str(CAP_META),
        "rows": len(variant_rows),
        "contextual_pair_rows_changed": len(meta_rows),
        "row_length_sequence_identical_to_cap120_treatment": True,
        "pair_specific_correspondence_broken": True,
        "rewrite_word_length_preserved_exact_rows": exact_same_length,
        "rewrite_word_length_adapted_fallback_rows": adapted_length,
        "same_source_donor_rows": same_source_donors,
        "source_pair_counts_top50": dict(source_pairs.most_common(50)),
        "donor_minus_target_rewrite_word_delta_counts": dict(sorted((str(k), v) for k, v in length_delta_counter.items())),
        "word_totals": {"pool_10M": sum(r.words for r in variant_rows), "train_100M": TOTAL_WORDS * PASSES},
        "files": {"pool_10M": str(pool), "train_100M": str(train)},
        "sha256": {"pool_10M": sha_file(pool), "train_100M": sha_file(train)},
        "intended_contrast": "Compare native-context correct original-rewrite cap120 against native-context same rewrite dose/style/length distribution with pair-specific correspondence broken.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "contextual_shuffled_rewrite_cap120_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"metadata": str(meta_path), "train_100M": str(train), "sha256_train_100M": payload["sha256"]["train_100M"], "adapted_fallback_rows": adapted_length}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    print(json.dumps({"status": "ok", "result": build(pathlib.Path(args.out_root))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
