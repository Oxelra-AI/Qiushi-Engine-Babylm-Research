#!/usr/bin/env python3
"""research: token/truncation/pair-dosage audit for optimizer×data 10M screen.

Reads the exact checkpoint cumulative-word counts from the research training verifier
and computes, for official-only and Qwen-aligned corpora, the tokenizer-visible
(non-padding) token exposure, raw token exposure, truncation, and pair-row dosage at
chck_3M/chck_5M/chck_10M under the same tokenizer and seq_length=256 used by the
research optimizer study trainer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

WORKSPACE = _public_path('experiments/archive/compact_experience')
VERIFY_PATH = _public_path('experiments/archive/compact_experience/data/optimizer_10M_training_verify/highlr_10M_training_verify.json')
TOKENIZER_PATH = _public_path('experiments/archive/compact_experience/training/runs/tok16_qwen_20M_b128_seed43022/hf_model/chck_20M')
CORPORA = {
    "official": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_100M.jsonl'),
    "qwen": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl'),
}
CHECKPOINTS = ["chck_3M", "chck_5M", "chck_10M"]
SEQ_LENGTH = 256


def is_pair_row(obj: dict[str, Any]) -> bool:
    return bool(obj.get("pair_id") or obj.get("is_pair_row") or "qwen_pair" in str(obj.get("source", "")))


def empty_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "whitespace_words": 0,
        "raw_tokens": 0,
        "visible_tokens": 0,
        "truncated_rows": 0,
        "truncated_tokens": 0,
    }


def add_stats(s: dict[str, Any], words: int, raw_tokens: int) -> None:
    s["rows"] += 1
    s["whitespace_words"] += words
    s["raw_tokens"] += raw_tokens
    s["visible_tokens"] += min(raw_tokens, SEQ_LENGTH)
    if raw_tokens > SEQ_LENGTH:
        s["truncated_rows"] += 1
        s["truncated_tokens"] += raw_tokens - SEQ_LENGTH


def finalize(s: dict[str, Any]) -> dict[str, Any]:
    out = dict(s)
    rows = max(1, s["rows"])
    words = max(1, s["whitespace_words"])
    raw = max(1, s["raw_tokens"])
    out.update({
        "mean_words_per_row": s["whitespace_words"] / rows,
        "mean_raw_tokens_per_row": s["raw_tokens"] / rows,
        "mean_visible_tokens_per_row": s["visible_tokens"] / rows,
        "raw_tokens_per_word": s["raw_tokens"] / words,
        "visible_tokens_per_word": s["visible_tokens"] / words,
        "truncated_row_fraction": s["truncated_rows"] / rows,
        "truncated_token_fraction_raw": s["truncated_tokens"] / raw,
    })
    return {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v) for k, v in out.items()}


def read_checkpoint_thresholds() -> dict[str, int]:
    verify = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
    ckmap = verify["checkpoint_cumulative_words_by_arm"]
    # All four arms should share identical cumulative words after verifier pass.
    first = next(iter(ckmap.values()))
    return {ck: int(first[ck]) for ck in CHECKPOINTS}


def audit_corpus(label: str, path: Path, tokenizer, thresholds: dict[str, int]) -> dict[str, Any]:
    max_threshold = max(thresholds.values())
    per_ckpt: dict[str, Any] = {}
    total = empty_stats()
    pair = empty_stats()
    nonpair = empty_stats()
    cum_words = 0
    pending = list(CHECKPOINTS)
    next_ck = pending.pop(0)
    texts: list[tuple[str, int, bool]] = []

    # Read rows through max threshold; it should hit exactly 10M.
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"{label} row {line_no} word mismatch field={words} actual={actual}")
            flag = is_pair_row(obj)
            texts.append((text, words, flag))
            cum_words += words
            if cum_words >= max_threshold:
                break

    if cum_words != max_threshold:
        raise RuntimeError(f"{label} ended at cum_words={cum_words}, expected max_threshold={max_threshold}")

    # Tokenize in small batches but update checkpoint snapshots row by row.
    # We need per-row lengths, so request no truncation/padding.
    idx = 0
    batch_size = 512
    cumulative_words = 0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start+batch_size]
        enc = tokenizer([x[0] for x in chunk], add_special_tokens=False, truncation=False, padding=False)
        for (text, words, flag), ids in zip(chunk, enc["input_ids"]):
            raw_tokens = len(ids)
            add_stats(total, words, raw_tokens)
            add_stats(pair if flag else nonpair, words, raw_tokens)
            cumulative_words += words
            idx += 1
            while next_ck is not None and cumulative_words >= thresholds[next_ck]:
                if cumulative_words != thresholds[next_ck]:
                    # This should not happen for actual checkpoint cumulative words; preserve if it does.
                    pass
                t = finalize(total)
                p = finalize(pair)
                n = finalize(nonpair)
                t["pair_visible_token_fraction"] = round(pair["visible_tokens"] / max(1, total["visible_tokens"]), 6)
                t["pair_raw_token_fraction"] = round(pair["raw_tokens"] / max(1, total["raw_tokens"]), 6)
                t["pair_word_fraction"] = round(pair["whitespace_words"] / max(1, total["whitespace_words"]), 6)
                per_ckpt[next_ck] = {
                    "checkpoint_cumulative_words": thresholds[next_ck],
                    "row_index_included": idx,
                    "total": t,
                    "pair_rows": p,
                    "nonpair_rows": n,
                }
                next_ck = pending.pop(0) if pending else None
                if next_ck is None:
                    break
        if next_ck is None:
            break

    return {
        "corpus": label,
        "path": str(path.relative_to(WORKSPACE)),
        "checkpoints": per_ckpt,
    }


def main() -> None:
    out_root = _public_path('experiments/archive/compact_experience/data/token_exposure_prefix_audit')
    out_root.mkdir(parents=True, exist_ok=True)
    thresholds = read_checkpoint_thresholds()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True)
    report = {
        "status": "TOKEN_EXPOSURE_PREFIX_AUDIT",
        "tokenizer_path": str(TOKENIZER_PATH.relative_to(WORKSPACE)),
        "seq_length": SEQ_LENGTH,
        "checkpoint_thresholds": thresholds,
        "corpora": {},
        "interpretation": {
            "purpose": "Quantify actual visible token exposure, truncation, and Qwen pair dosage for interpreting optimizer×data interaction; not a downstream selection signal.",
            "note": "Official corpus has no pair rows by construction. Qwen pair dosage is computed from rows with pair_id/is_pair_row/qwen_pair source. Words are official budget units; visible tokens are what the model actually processes after truncation.",
        },
    }
    for label, path in CORPORA.items():
        report["corpora"][label] = audit_corpus(label, path, tok, thresholds)

    # Cross-corpus deltas for total visible exposure and truncation.
    deltas: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        o = report["corpora"]["official"]["checkpoints"][ck]["total"]
        q = report["corpora"]["qwen"]["checkpoints"][ck]["total"]
        deltas[ck] = {
            "qwen_minus_official_visible_tokens": q["visible_tokens"] - o["visible_tokens"],
            "qwen_over_official_visible_tokens_ratio": round(q["visible_tokens"] / max(1, o["visible_tokens"]), 6),
            "qwen_minus_official_truncated_rows": q["truncated_rows"] - o["truncated_rows"],
            "qwen_minus_official_truncated_tokens": q["truncated_tokens"] - o["truncated_tokens"],
            "qwen_pair_visible_token_fraction": report["corpora"]["qwen"]["checkpoints"][ck]["total"]["pair_visible_token_fraction"],
            "qwen_pair_word_fraction": report["corpora"]["qwen"]["checkpoints"][ck]["total"]["pair_word_fraction"],
        }
    report["cross_corpus_deltas"] = deltas

    out_path = out_root / "token_exposure_prefix_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
