#!/usr/bin/env python3
"""Audit research clean corpora before long training.

Checks that the cleaned Qwen treatment and official control are exact 10M pools,
have identical row-length sequences, preserve pair-row metadata, and are not badly
mismatched in tokenizer length / truncation under the inherited baseline16k tokenizer.
This is a pre-training scientific check, not an evaluator.
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
import sys
from typing import Iterable

from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
CORP = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora')
META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
OUT = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/pretrain_audit.json')
MAX_SEQ = 256
TOTAL_WORDS = 10_000_000


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            w = int(o.get("words", len(o["text"].split())))
            actual = len(o["text"].split())
            if w != actual:
                raise RuntimeError(f"{path} row {i} word mismatch field={w} actual={actual}")
            rows.append({"text": o["text"], "words": w, "source": str(o.get("source", "")), "example_id": int(o.get("example_id", -1))})
            total += w
    if total != TOTAL_WORDS:
        raise RuntimeError(f"{path} total words {total} != {TOTAL_WORDS}")
    return rows


def stats(vals: Iterable[float]) -> dict:
    vals = list(vals)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": min(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p95": round(statistics.quantiles(vals, n=20)[-1], 4) if len(vals) >= 20 else max(vals),
        "p99": round(statistics.quantiles(vals, n=100)[-1], 4) if len(vals) >= 100 else max(vals),
        "max": max(vals),
    }


def token_audit(rows: list[dict], tok) -> dict:
    lens = []
    words = []
    over = 0
    trunc_tokens = 0
    over_examples = []
    for i, r in enumerate(rows):
        ids = tok(r["text"], add_special_tokens=True, truncation=False)["input_ids"]
        L = len(ids)
        lens.append(L)
        words.append(r["words"])
        if L > MAX_SEQ:
            over += 1
            trunc_tokens += L - MAX_SEQ
            if len(over_examples) < 20:
                over_examples.append({"row": i, "source": r["source"], "words": r["words"], "tokens": L, "text_prefix": r["text"][:220]})
    return {
        "row_count": len(rows),
        "word_total": sum(words),
        "token_length_stats": stats(lens),
        "word_length_stats": stats(words),
        "rows_over_seq256": over,
        "fraction_rows_over_seq256": round(over / len(rows), 6),
        "excess_tokens_if_truncated_at_256": trunc_tokens,
        "over_seq256_examples": over_examples,
    }


def main() -> None:
    if not META.exists():
        raise FileNotFoundError(META)
    meta = json.loads(META.read_text(encoding="utf-8"))
    off_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')
    qwen_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
    off_train = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_100M.jsonl')
    qwen_train = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
    for p in [off_path, qwen_path, off_train, qwen_train]:
        if not p.exists():
            raise FileNotFoundError(p)
    off = read_rows(off_path)
    qwen = read_rows(qwen_path)
    if [r["words"] for r in off] != [r["words"] for r in qwen]:
        raise RuntimeError("official and qwen pools do not have identical row-length sequences")

    # Check 100M training files line/word totals cheaply.
    train_info = {}
    for p in [off_train, qwen_train]:
        rows = 0
        words = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                rows += 1
                words += int(o["words"])
        train_info[p.name] = {"rows": rows, "words": words, "sha256": sha256_file(p)}
        if words != TOTAL_WORDS * 10:
            raise RuntimeError(f"{p} words {words}")

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    off_a = token_audit(off, tok)
    qwen_a = token_audit(qwen, tok)

    # Pair metadata and source multiplicity.
    pair_meta_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
    pair_meta_rows = []
    if pair_meta_path.exists():
        with pair_meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pair_meta_rows.append(json.loads(line))
    selected_pairs_path = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
    selected_pairs = []
    if selected_pairs_path.exists():
        with selected_pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    selected_pairs.append(json.loads(line))
    selected_example_counts = collections.Counter(int(p["example_id"]) for p in selected_pairs)
    qwen_official_filler_counts = collections.Counter(r["example_id"] for r in qwen if r["source"] != "qwen_pair_packed")
    duplicated_selected_examples_in_filler = {str(eid): qwen_official_filler_counts[eid] for eid in selected_example_counts if qwen_official_filler_counts[eid]}
    cohort_words = collections.Counter()
    cohort_pairs = collections.Counter()
    for p in selected_pairs:
        cohort_words[p.get("cohort", "unknown")] += int(p["pair_words"])
        cohort_pairs[p.get("cohort", "unknown")] += 1

    report = {
        "status": "CLEAN_CORPUS_PREFLIGHT_AUDIT",
        "metadata_path": str(META),
        "materialization_status": meta.get("status"),
        "pair_boundary_preserved": meta.get("pair_boundary_preserved"),
        "pair_truncation": meta.get("pair_truncation"),
        "word_fragment_padding_in_qwen_pair_rows": meta.get("word_fragment_padding_in_qwen_pair_rows"),
        "selected_pair_words": meta.get("selected_pair_words"),
        "selected_pair_word_fraction": meta.get("selected_pair_word_fraction"),
        "selected_pairs": len(selected_pairs),
        "cohort_pairs": dict(cohort_pairs),
        "cohort_words": dict(cohort_words),
        "pool_row_counts": {"official": len(off), "qwen": len(qwen)},
        "row_length_sequence_matched": True,
        "source_word_counts_official_control": dict(collections.Counter({s: sum(r["words"] for r in off if r["source"] == s) for s in set(r["source"] for r in off)})),
        "source_word_counts_qwen_treatment": dict(collections.Counter({s: sum(r["words"] for r in qwen if r["source"] == s) for s in set(r["source"] for r in qwen)})),
        "selected_example_ids_reused_in_qwen_official_filler_count": len(duplicated_selected_examples_in_filler),
        "selected_example_ids_reused_in_qwen_official_filler_sample": dict(list(duplicated_selected_examples_in_filler.items())[:20]),
        "tokenizer_path": str(TOKENIZER),
        "tokenizer_vocab_size": len(tok),
        "official_control_token_audit": off_a,
        "qwen_treatment_token_audit": qwen_a,
        "token_audit_delta": {
            "mean_tokens_qwen_minus_official": round(qwen_a["token_length_stats"]["mean"] - off_a["token_length_stats"]["mean"], 4),
            "rows_over_256_qwen_minus_official": qwen_a["rows_over_seq256"] - off_a["rows_over_seq256"],
            "excess_tokens_qwen_minus_official": qwen_a["excess_tokens_if_truncated_at_256"] - off_a["excess_tokens_if_truncated_at_256"],
        },
        "training_files": train_info,
        "metadata_sha256_expected": meta.get("sha256", {}),
    }
    # Check expected hashes for materialized files.
    actual_short_hashes = {
        "official_only_10M.jsonl": sha256_file(off_path),
        "qwen_aligned_10M.jsonl": sha256_file(qwen_path),
        "official_only_100M.jsonl": train_info["official_only_100M.jsonl"]["sha256"],
        "qwen_aligned_100M.jsonl": train_info["qwen_aligned_100M.jsonl"]["sha256"],
    }
    report["actual_sha256"] = actual_short_hashes
    report["sha256_match_metadata"] = {k: (meta.get("sha256", {}).get(k) == v) for k, v in actual_short_hashes.items()}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_pair_word_fraction": report["selected_pair_word_fraction"],
        "cohort_words": report["cohort_words"],
        "rows_over_256": {"official": off_a["rows_over_seq256"], "qwen": qwen_a["rows_over_seq256"]},
        "token_mean_delta": report["token_audit_delta"]["mean_tokens_qwen_minus_official"],
        "selected_example_reuse_in_filler": report["selected_example_ids_reused_in_qwen_official_filler_count"],
        "out": str(OUT),
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
