#!/usr/bin/env python3
"""Tokenization/truncation audit for research contextual one-pair cap120 corpora.

Uses the submitted-model tokenizer only; no evaluation data or AoA data are read.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections
import json
import pathlib
import statistics
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer

# script lives under data/contextual_one_pair/cap120/
WORKSPACE = _public_path('experiments/archive/compact_experience')
SESSIONS = _public_path('experiments/archive')
DATA = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120')
TREAT10 = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/qwen_context_onepair_cap120_10M.jsonl')
CTRL10 = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_10M.jsonl')
PAIR_META = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_pair_rows_cap120_meta.jsonl')
OUT = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_cap120_tokenization_audit.json')
TOKENIZER = _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model')
MAX_LEN = 256


def stats(xs: Iterable[float]) -> Dict[str, Any]:
    ys = sorted(xs)
    if not ys:
        return {"n": 0}
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        i = p * (len(ys) - 1)
        lo = int(i)
        hi = min(lo + 1, len(ys) - 1)
        a = i - lo
        return ys[lo] * (1 - a) + ys[hi] * a
    return {"n": len(ys), "min": ys[0], "p05": q(0.05), "mean": statistics.fmean(ys), "median": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": ys[-1]}


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tok_len(tok, text: str, special: bool) -> int:
    return len(tok(text, add_special_tokens=special, truncation=False)["input_ids"])


def pool_stats(tok, path: pathlib.Path) -> Dict[str, Any]:
    rows = 0
    words_total = 0
    token_lens: List[int] = []
    subword_ratios: List[float] = []
    over = 0
    source_words = collections.Counter()
    source_rows = collections.Counter()
    by_source_tokens = collections.Counter()
    for r in read_jsonl(path):
        rows += 1
        words = int(r["words"])
        words_total += words
        source = str(r.get("source"))
        source_words[source] += words
        source_rows[source] += 1
        n_tok = tok_len(tok, str(r["text"]), True)
        token_lens.append(n_tok)
        subword_ratios.append(n_tok / words if words else 0.0)
        by_source_tokens[source] += n_tok
        if n_tok > MAX_LEN:
            over += 1
    return {
        "path": str(path),
        "rows": rows,
        "words": words_total,
        "token_length_with_special_stats": stats(token_lens),
        "subword_tokens_per_word_with_special_stats": stats(subword_ratios),
        "rows_token_len_gt_256": over,
        "rows_token_len_gt_256_fraction": over / rows if rows else 0.0,
        "source_words": dict(sorted(source_words.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "source_token_counts_with_special": dict(sorted(by_source_tokens.items())),
    }


def pair_visibility(tok) -> Dict[str, Any]:
    meta_by_row = {int(m["row_index"]): m for m in read_jsonl(PAIR_META)}
    rows = 0
    over = 0
    orig_trunc = 0
    rew_trunc = 0
    orig_visible_fracs: List[float] = []
    rew_visible_fracs: List[float] = []
    token_lens: List[int] = []
    pair_token_lens: List[int] = []
    ctx_token_lens: List[int] = []
    examples = []
    for i, r in enumerate(read_jsonl(TREAT10)):
        m = meta_by_row.get(i)
        if m is None:
            continue
        rows += 1
        words = str(r["text"]).split()
        lt = int(m["left_take"])
        ow = int(m["original_words"])
        rw = int(m["rewrite_words"])
        orig_words = words[lt:lt + ow]
        rew_words = words[lt + ow:lt + ow + rw]
        left_words = words[:lt]
        right_words = words[lt + ow + rw:]
        # Token spans without special tokens; reserve two special tokens as a conservative 254-token content budget.
        left_t = tok_len(tok, " ".join(left_words), False) if left_words else 0
        orig_t = tok_len(tok, " ".join(orig_words), False) if orig_words else 0
        rew_t = tok_len(tok, " ".join(rew_words), False) if rew_words else 0
        pair_token_lens.append(orig_t + rew_t)
        ctx_token_lens.append(left_t + (tok_len(tok, " ".join(right_words), False) if right_words else 0))
        row_tok = tok_len(tok, str(r["text"]), True)
        token_lens.append(row_tok)
        if row_tok > MAX_LEN:
            over += 1
        budget = MAX_LEN - 2
        orig_start = left_t
        orig_end = left_t + orig_t
        rew_start = orig_end
        rew_end = rew_start + rew_t
        def visible_frac(start: int, end: int) -> float:
            if end <= start:
                return 1.0
            vis = max(0, min(end, budget) - start)
            return max(0.0, min(1.0, vis / (end - start)))
        of = visible_frac(orig_start, orig_end)
        rf = visible_frac(rew_start, rew_end)
        orig_visible_fracs.append(of)
        rew_visible_fracs.append(rf)
        if of < 1.0:
            orig_trunc += 1
        if rf < 1.0:
            rew_trunc += 1
        if (of < 1.0 or rf < 1.0) and len(examples) < 20:
            examples.append({"row_index": i, "pair_id": m.get("pair_id"), "words": r.get("words"), "row_tokens_with_special": row_tok, "left_tokens": left_t, "original_tokens": orig_t, "rewrite_tokens": rew_t, "orig_visible_fraction": of, "rewrite_visible_fraction": rf, "source": m.get("source")})
    return {
        "contextual_pair_rows": rows,
        "pair_row_token_length_with_special_stats": stats(token_lens),
        "pair_content_token_length_stats": stats(pair_token_lens),
        "context_token_length_stats": stats(ctx_token_lens),
        "pair_rows_token_len_gt_256": over,
        "pair_rows_token_len_gt_256_fraction": over / rows if rows else 0.0,
        "original_side_truncated_rows": orig_trunc,
        "rewrite_side_truncated_rows": rew_trunc,
        "original_visible_fraction_stats": stats(orig_visible_fracs),
        "rewrite_visible_fraction_stats": stats(rew_visible_fracs),
        "truncated_examples": examples,
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    payload = {
        "status": "CONTEXTUAL_CAP120_TOKENIZATION_AUDIT",
        "non_leakage_statement": "Uses only training corpora and the submission tokenizer; no evaluation/AoA/CDI data are read.",
        "tokenizer": str(TOKENIZER),
        "max_length_assumed": MAX_LEN,
        "treatment_10M": pool_stats(tok, TREAT10),
        "official_lengthmatched_control_10M": pool_stats(tok, CTRL10),
        "contextual_pair_visibility": pair_visibility(tok),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tv = payload["contextual_pair_visibility"]
    print(json.dumps({
        "out": str(OUT),
        "treatment_rows_gt_256": payload["treatment_10M"]["rows_token_len_gt_256"],
        "control_rows_gt_256": payload["official_lengthmatched_control_10M"]["rows_token_len_gt_256"],
        "pair_rows_gt_256": tv["pair_rows_token_len_gt_256"],
        "original_side_truncated_rows": tv["original_side_truncated_rows"],
        "rewrite_side_truncated_rows": tv["rewrite_side_truncated_rows"],
        "rewrite_visible_fraction_mean": tv["rewrite_visible_fraction_stats"].get("mean"),
    }, indent=2))


if __name__ == "__main__":
    main()
