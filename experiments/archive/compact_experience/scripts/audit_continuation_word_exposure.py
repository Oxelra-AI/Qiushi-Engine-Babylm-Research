#!/usr/bin/env python3
"""Audit actual whitespace-word exposure bounds for research chunked continuations.

The continuation trainers operate on token chunks but name checkpoints by word
exposure. This script reconstructs the token chunks from the clean 10M JSONL pool,
simulates the same per-pass chunk shuffles and stop rule, and reports word-instance
coverage bounds:
  * any-token word exposure: a word counts if any of its subword tokens is processed;
  * complete-token word exposure: a word counts only if all of its subword tokens are processed.
For a full pass these coincide at the pool word count; for the near-full second
pass the gap is only around words split across skipped/processed chunk boundaries.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import array
import json
import pathlib
import random
import re
import time
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_INIT = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_80M')
DEFAULT_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/exposure_audit/continuation_word_exposure_audit.json')
SEQ_LENGTH = 256
BATCH_SIZE = 256


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def token_word_ids_for_text(tokenizer, text: str, word_id_start: int) -> tuple[list[int], int]:
    spans = word_spans(text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    ids = enc["input_ids"]
    if offsets is None:
        raise RuntimeError("fast tokenizer did not return offset_mapping")
    out: list[int] = []
    wptr = 0
    for lo, hi in offsets:
        if hi <= lo:
            out.append(-1)
            continue
        while wptr < len(spans) and spans[wptr][1] <= lo:
            wptr += 1
        # Assign to the first whitespace word span overlapped by this token.
        if wptr < len(spans) and spans[wptr][0] < hi and spans[wptr][1] > lo:
            out.append(word_id_start + wptr)
        else:
            # Rare byte/normalization edge; count as no word rather than inventing exposure.
            out.append(-1)
    if len(out) != len(ids):
        raise RuntimeError("token/offset length mismatch")
    return out, len(spans)


def build_token_word_stream(tokenizer, pool: pathlib.Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    token_word_ids = array.array("i")
    word_token_counts = array.array("H")
    global_word = 0
    pool_words = 0
    rows = 0
    with pool.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            text = str(obj["text"])
            declared_words = int(obj.get("words", len(text.split())))
            actual_words = len(text.split())
            if declared_words != actual_words:
                raise RuntimeError(f"row {rows} word mismatch: field={declared_words} actual={actual_words}")
            ids, n_words = token_word_ids_for_text(tokenizer, text, global_word)
            if n_words != declared_words:
                raise RuntimeError(f"row {rows} span mismatch: spans={n_words} words={declared_words}")
            token_word_ids.extend(ids)
            word_token_counts.extend([0] * n_words)
            for wid in ids:
                if wid >= 0:
                    # Cap should not be reached for normal BPE wordpiece counts.
                    cur = word_token_counts[wid]
                    word_token_counts[wid] = min(65535, cur + 1)
            global_word += n_words
            pool_words += declared_words
    # Replicate ContinuationDataset final partial chunk behavior.
    rem = len(token_word_ids) % SEQ_LENGTH
    if rem and rem > SEQ_LENGTH // 4:
        token_word_ids.extend([-1] * (SEQ_LENGTH - rem))
    elif rem:
        # The trainer discards a short tail < = seq_length//4.
        del token_word_ids[len(token_word_ids) - rem:]
    tw = np.frombuffer(token_word_ids, dtype=np.int32).copy()
    wtc = np.frombuffer(word_token_counts, dtype=np.uint16).copy()
    if int((wtc > 0).sum()) != pool_words:
        # A few unusual words may tokenize to zero counted pieces only if the tokenizer
        # normalizer drops them; keep the exact count in the report.
        pass
    return tw, wtc, pool_words, rows


def simulate_processed_chunks(n_chunks: int, seed: int, pool_words: int, budget: int, batch_size: int) -> list[list[int]]:
    random.seed(seed)
    processed_by_pass: list[list[int]] = []
    processed_chunks_total = 0
    passes_needed = int(np.ceil(budget / pool_words))
    stop = False
    for _pass_i in range(passes_needed):
        indices = list(range(n_chunks))
        random.shuffle(indices)
        pos = 0
        pass_chunks: list[int] = []
        while pos < len(indices):
            batch: list[int] = []
            while pos < len(indices) and len(batch) < batch_size:
                projected_chunks = processed_chunks_total + len(batch) + 1
                projected_words = int(round((projected_chunks / max(1, n_chunks)) * pool_words))
                if projected_words > budget:
                    break
                batch.append(indices[pos])
                pos += 1
            if not batch:
                stop = True
                break
            pass_chunks.extend(batch)
            processed_chunks_total += len(batch)
            if int(round((processed_chunks_total / max(1, n_chunks)) * pool_words)) >= budget:
                stop = True
                break
        processed_by_pass.append(pass_chunks)
        if stop:
            break
    return processed_by_pass


def count_words_for_chunks(token_word_ids: np.ndarray, word_token_counts: np.ndarray, chunks: Iterable[int]) -> dict:
    n_words = len(word_token_counts)
    token_mask = np.zeros(token_word_ids.shape[0], dtype=bool)
    for c in chunks:
        lo = c * SEQ_LENGTH
        hi = min(lo + SEQ_LENGTH, token_word_ids.shape[0])
        token_mask[lo:hi] = True
    selected = token_word_ids[token_mask]
    selected = selected[selected >= 0]
    counts = np.bincount(selected, minlength=n_words).astype(np.uint16, copy=False)
    any_words = int((counts > 0).sum())
    complete_words = int((counts >= word_token_counts).sum())
    token_instances = int(selected.shape[0])
    return {
        "chunks": int(token_mask.reshape(-1, SEQ_LENGTH).any(axis=1).sum()) if token_word_ids.shape[0] % SEQ_LENGTH == 0 else int(len(list(chunks))),
        "token_instances": token_instances,
        "any_token_word_instances": any_words,
        "complete_token_word_instances": complete_words,
        "partial_word_instances": any_words - complete_words,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(DEFAULT_INIT))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=43044)
    ap.add_argument("--budget", type=int, default=19_996_318)
    ap.add_argument("--parent_actual_exposure", type=int, default=80_003_682)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.init, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This audit requires a fast tokenizer for offset mapping")
    token_word_ids, word_token_counts, pool_words, rows = build_token_word_stream(tokenizer, pathlib.Path(args.pool))
    if token_word_ids.shape[0] % SEQ_LENGTH != 0:
        raise RuntimeError("token stream not chunk-aligned")
    n_chunks = token_word_ids.shape[0] // SEQ_LENGTH
    processed_by_pass = simulate_processed_chunks(n_chunks, args.seed, pool_words, args.budget, args.batch_size)
    pass_counts = [count_words_for_chunks(token_word_ids, word_token_counts, chunks) for chunks in processed_by_pass]
    total_any = sum(x["any_token_word_instances"] for x in pass_counts)
    total_complete = sum(x["complete_token_word_instances"] for x in pass_counts)
    total_chunks = sum(len(x) for x in processed_by_pass)
    estimated_words = int(round((total_chunks / max(1, n_chunks)) * pool_words))
    payload = {
        "status": "CONTINUATION_WORD_EXPOSURE_AUDIT",
        "created_utc": now(),
        "pool": args.pool,
        "pool_rows": rows,
        "pool_words": pool_words,
        "init": args.init,
        "seq_length": SEQ_LENGTH,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "requested_continuation_budget": args.budget,
        "parent_actual_exposure": args.parent_actual_exposure,
        "token_count_in_chunk_stream": int(token_word_ids.shape[0]),
        "dataset_chunks": n_chunks,
        "word_count_with_nonzero_tokens": int((word_token_counts > 0).sum()),
        "processed_chunks_total": total_chunks,
        "estimated_continuation_words_by_chunk_ratio": estimated_words,
        "pass_counts": pass_counts,
        "total_any_token_continuation_word_instances": total_any,
        "total_complete_token_continuation_word_instances": total_complete,
        "total_partial_word_instances": total_any - total_complete,
        "parent_plus_any_token_word_instances": args.parent_actual_exposure + total_any,
        "parent_plus_complete_token_word_instances": args.parent_actual_exposure + total_complete,
        "elapsed_sec": round(time.time() - t0, 3),
        "interpretation": "The trainers' manifest exposure is a chunk-ratio estimate. This audit gives lower/upper bounds from exact tokenizer offsets for the same shuffled chunks. Any-token is a conservative upper count of text words touched by at least one processed subword; complete-token is a lower count of fully visible wordpieces.",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "estimated": estimated_words, "any_total": total_any, "complete_total": total_complete, "parent_plus_any": args.parent_actual_exposure + total_any, "elapsed_sec": payload["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
