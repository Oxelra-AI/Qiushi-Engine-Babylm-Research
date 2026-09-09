#!/usr/bin/env python3
"""Audit contextual cap corpora for trainer-visible token/source/pair exposure.

This audit uses the exact tokenizer truncation rule in masking_curriculum_trainer.py:
add_special_tokens=False, truncation=True, max_length=256, padding later.  It reports
row-level visible tokens, over-256 rows, approximate source-word composition, and for
contextual treatment/control variants with pair metadata, pair/context token visibility.
No evaluation outputs, AoA/CDI words, child curves, or downstream scores are read.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_CAP_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_one_pair/cap120')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/compact_experience/data/contextual_token_source_audit')
SEQ_LEN = 256


def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        idx = p * (len(ys) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ys) - 1)
        frac = idx - lo
        return ys[lo] * (1 - frac) + ys[hi] * frac
    return {
        "n": len(ys),
        "mean": statistics.fmean(ys),
        "min": ys[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": ys[-1],
        "sum": sum(ys),
    }


def encode_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def prefix_visible_tokens(tok, words: List[str], max_len: int = SEQ_LEN) -> Tuple[int, int]:
    """Return (visible_words_prefix_count, visible_tokens) under right truncation.

    Because tokenizer may split words into multiple tokens, this counts complete words whose
    tokenized prefixes fit before max_len. A partially visible word is conservatively counted
    as not fully visible but its tokens are included in visible_tokens.
    """
    total = 0
    visible_words = 0
    for w in words:
        n = encode_len(tok, w)
        if total + n <= max_len:
            total += n
            visible_words += 1
        else:
            return visible_words, max_len
    return visible_words, total


def audit_pool(tok, path: pathlib.Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    n_rows = 0
    total_words = 0
    token_lengths: List[int] = []
    visible_tokens: List[int] = []
    over = 0
    source_words = collections.Counter()
    source_rows = collections.Counter()
    for r in read_jsonl(path):
        if max_rows is not None and n_rows >= max_rows:
            break
        text = str(r["text"])
        words = int(r.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"word count mismatch in {path}: declared={words} actual={len(text.split())}")
        src = str(r.get("source", ""))
        L = encode_len(tok, text)
        token_lengths.append(L)
        visible_tokens.append(min(L, SEQ_LEN))
        if L > SEQ_LEN:
            over += 1
        total_words += words
        source_words[src] += words
        source_rows[src] += 1
        n_rows += 1
    return {
        "file": str(path),
        "sampled_rows": n_rows,
        "total_words_in_sample": total_words,
        "tokenizer_rule": {"add_special_tokens": False, "truncation_side": "right", "seq_len": SEQ_LEN},
        "token_length_stats": stats(token_lengths),
        "visible_token_stats": stats(visible_tokens),
        "rows_over_seq_len": over,
        "fraction_rows_over_seq_len": over / n_rows if n_rows else None,
        "total_visible_tokens_in_sample": int(sum(visible_tokens)),
        "total_tokens_untruncated_in_sample": int(sum(token_lengths)),
        "truncated_tokens_in_sample": int(sum(max(0, L - SEQ_LEN) for L in token_lengths)),
        "source_words": dict(sorted(source_words.items())),
        "source_rows": dict(sorted(source_rows.items())),
    }


def segment_token_lengths(tok, meta: Dict[str, Any], row_words: List[str]) -> Dict[str, Any]:
    left_n = int(meta["left_take"])
    ow = int(meta["original_words"])
    rw = int(meta["rewrite_words"])
    right_n = int(meta["right_take"])
    if left_n + ow + rw + right_n != int(meta["words"]):
        raise RuntimeError(f"meta word mismatch {meta.get('pair_id')}")
    if len(row_words) != int(meta["words"]):
        raise RuntimeError(f"row word mismatch {meta.get('pair_id')}: {len(row_words)} vs {meta['words']}")
    parts = {
        "left": row_words[:left_n],
        "original": row_words[left_n:left_n+ow],
        "rewrite": row_words[left_n+ow:left_n+ow+rw],
        "right": row_words[left_n+ow+rw:],
    }
    token_parts = {k: encode_len(tok, " ".join(v)) if v else 0 for k, v in parts.items()}
    # Prefix tokenization of concatenated segments is the exact row length; segment token sums are
    # an approximation because BPE whitespace markers can depend on segment starts. For visibility
    # under right truncation, use per-segment cumulative token counts as a close conservative audit.
    cum = 0
    visible_by_part: Dict[str, int] = {}
    for k in ["left", "original", "rewrite", "right"]:
        L = token_parts[k]
        vis = max(0, min(L, SEQ_LEN - cum))
        visible_by_part[k] = vis
        cum += L
    row_L = encode_len(tok, " ".join(row_words))
    return {
        "segment_tokens_approx": token_parts,
        "segment_visible_tokens_approx": visible_by_part,
        "row_tokens_exact": row_L,
        "row_visible_tokens_exact": min(row_L, SEQ_LEN),
        "original_visible_fraction_approx": visible_by_part["original"] / token_parts["original"] if token_parts["original"] else None,
        "rewrite_visible_fraction_approx": visible_by_part["rewrite"] / token_parts["rewrite"] if token_parts["rewrite"] else None,
        "pair_visible_fraction_approx": (visible_by_part["original"] + visible_by_part["rewrite"]) / (token_parts["original"] + token_parts["rewrite"]) if (token_parts["original"] + token_parts["rewrite"]) else None,
        "context_visible_fraction_approx": (visible_by_part["left"] + visible_by_part["right"]) / (token_parts["left"] + token_parts["right"]) if (token_parts["left"] + token_parts["right"]) else None,
    }


def audit_pair_visibility(tok, pool_path: pathlib.Path, meta_path: pathlib.Path, max_pair_rows: Optional[int] = None) -> Dict[str, Any]:
    rows = list(read_jsonl(pool_path))
    meta_rows = list(read_jsonl(meta_path))
    if max_pair_rows is not None:
        meta_rows = meta_rows[:max_pair_rows]
    pair_fracs: List[float] = []
    orig_fracs: List[float] = []
    rew_fracs: List[float] = []
    ctx_fracs: List[float] = []
    row_exact: List[int] = []
    row_over = 0
    original_truncated = 0
    rewrite_truncated = 0
    pair_truncated = 0
    context_truncated = 0
    source_counts = collections.Counter()
    example_counts = collections.Counter()
    for m in meta_rows:
        idx = int(m["row_index"])
        r = rows[idx]
        words = str(r["text"]).split()
        seg = segment_token_lengths(tok, m, words)
        row_exact.append(int(seg["row_tokens_exact"]))
        if int(seg["row_tokens_exact"]) > SEQ_LEN:
            row_over += 1
        of = float(seg["original_visible_fraction_approx"] or 0.0)
        rf = float(seg["rewrite_visible_fraction_approx"] or 0.0)
        pf = float(seg["pair_visible_fraction_approx"] or 0.0)
        cf = seg["context_visible_fraction_approx"]
        orig_fracs.append(of); rew_fracs.append(rf); pair_fracs.append(pf)
        if cf is not None:
            ctx_fracs.append(float(cf))
        if of < 0.999999:
            original_truncated += 1
        if rf < 0.999999:
            rewrite_truncated += 1
        if pf < 0.999999:
            pair_truncated += 1
        if cf is not None and cf < 0.999999:
            context_truncated += 1
        source_counts[str(m.get("source", ""))] += int(m.get("words", 0))
        example_counts[int(m.get("example_id", -1))] += 1
    return {
        "pool_path": str(pool_path),
        "meta_path": str(meta_path),
        "pair_rows_audited": len(meta_rows),
        "row_token_stats_exact": stats(row_exact),
        "rows_over_seq_len": row_over,
        "original_rows_with_truncation_approx": original_truncated,
        "rewrite_rows_with_truncation_approx": rewrite_truncated,
        "pair_rows_with_any_pair_truncation_approx": pair_truncated,
        "context_rows_with_any_context_truncation_approx": context_truncated,
        "original_visible_fraction_stats_approx": stats(orig_fracs),
        "rewrite_visible_fraction_stats_approx": stats(rew_fracs),
        "pair_visible_fraction_stats_approx": stats(pair_fracs),
        "context_visible_fraction_stats_approx": stats(ctx_fracs),
        "source_words_in_pair_rows": dict(sorted(source_counts.items())),
        "selected_example_reuse_stats": stats([float(v) for v in example_counts.values()]),
        "selected_examples_with_multiple_pairs": sum(1 for v in example_counts.values() if v > 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_root", default=str(DEFAULT_CAP_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--max_rows", type=int, default=0, help="Optional row sample cap for quick smoke; 0 means all rows")
    ap.add_argument("--max_pair_rows", type=int, default=0, help="Optional pair-row sample cap; 0 means all pair rows")
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    args = ap.parse_args()
    cap_root = pathlib.Path(args.cap_root)
    out_root = pathlib.Path(args.out_root)
    max_rows = None if args.max_rows <= 0 else args.max_rows
    max_pair_rows = None if args.max_pair_rows <= 0 else args.max_pair_rows
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    meta_files = sorted(cap_root.glob("contextual_one_pair_cap*_metadata.json"))
    if not meta_files:
        # follow-up controls use contextual_<variant>_cap120_metadata.json but may not have pair meta.
        meta_files = sorted(cap_root.glob("contextual_*_cap*_metadata.json"))
    if not meta_files:
        raise SystemExit(f"no metadata files found under {cap_root}")
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    files = meta.get("files") or {}
    pool_paths = {k: pathlib.Path(v) for k, v in files.items() if k.endswith("10M") and pathlib.Path(v).exists()}
    if not pool_paths:
        raise SystemExit(f"no 10M pool files in metadata {meta_files[0]}")
    audits = {name: audit_pool(tok, path, max_rows=max_rows) for name, path in sorted(pool_paths.items())}
    pair_audit = None
    pair_meta_path = pathlib.Path(files.get("contextual_pair_meta", "")) if files.get("contextual_pair_meta") else None
    treat_path = pool_paths.get("treatment_10M")
    if pair_meta_path and pair_meta_path.exists() and treat_path and treat_path.exists():
        pair_audit = audit_pair_visibility(tok, treat_path, pair_meta_path, max_pair_rows=max_pair_rows)
    payload = {
        "status": "CONTEXTUAL_TOKEN_SOURCE_EXPOSURE_AUDIT",
        "non_leakage_statement": "Uses only training corpora, metadata, and tokenizer/truncation rules. No official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs are read or used.",
        "cap_root": str(cap_root),
        "metadata": str(meta_files[0]),
        "tokenizer": str(args.tokenizer),
        "seq_len": SEQ_LEN,
        "max_rows": max_rows,
        "max_pair_rows": max_pair_rows,
        "pool_audits": audits,
        "pair_visibility_audit": pair_audit,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    name = cap_root.name or "cap"
    out = out_root / f"{name}_token_source_exposure_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "status": payload["status"], "cap_root": str(cap_root), "pool_keys": sorted(audits), "pair_rows_audited": None if pair_audit is None else pair_audit.get("pair_rows_audited")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
