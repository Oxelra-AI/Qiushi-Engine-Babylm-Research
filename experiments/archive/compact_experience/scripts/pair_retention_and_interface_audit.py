#!/usr/bin/env python3
"""research: actual trainer-side tokenization/pair-retention audit for the 16k vs 40k interface.

The earlier tokenizer audit measured raw token lengths. This script measures what the
current trainer actually exposes after seq256 truncation/padding, with special attention
to clean-Qwen packed pair rows. It uses only training-corpus metadata, never eval/AoA/CDI
signals.

Outputs quantify:
  * total visible nonpad tokens after truncation by dataset/tokenizer/source
  * packed pair-row full/partial retention of original and rewrite sides
  * visible word and token fractions in pair rows
  * parameter-count difference between the 16k and 40k cells
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
SEQ = 256
TOKENIZERS = {
    "tok16": _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'),
    "tok40": _public_path('experiments/archive/compact_experience/data/shared_tokenizer/hf_tokenizer_40k_shared'),
}
CORPORA = {
    "qwen": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl'),
    "official": _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl'),
}
SELECTED_PAIRS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PAIR_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
SUMMARY2X2_20M = _public_path('experiments/archive/compact_experience/data/tok16_tok40_2x2_summary.json')
SUMMARY2X2_10M = _public_path('experiments/archive/compact_experience/data/tok2x2_noaoa_eval_10M/tok16_tok40_2x2_10M_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/pair_retention_and_interface_audit.json')
WORD_RE = re.compile(r"\S+")


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    vals = list(vals)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p05": statistics.quantiles(vals, n=20)[0] if len(vals) >= 20 else min(vals),
        "p95": statistics.quantiles(vals, n=20)[-1] if len(vals) >= 20 else max(vals),
        "min": min(vals),
        "max": max(vals),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pairs() -> Dict[str, Dict[str, Any]]:
    out = {}
    with SELECTED_PAIRS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            out[str(p["pair_id"])] = p
    return out


def word_spans(text: str, base: int = 0) -> List[Tuple[int, int]]:
    return [(base + m.start(), base + m.end()) for m in WORD_RE.finditer(text)]


def build_pair_segments(row_text: str, pair_ids: List[str], pairs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reconstruct original/rewrite segment spans in a packed row."""
    pos = 0
    segments: List[Dict[str, Any]] = []
    for pid in pair_ids:
        p = pairs[pid]
        for side in ["original", "rewrite"]:
            seg = str(p[side])
            # The materializer joins segments by exactly one space. Advance over spaces.
            while pos < len(row_text) and row_text[pos] == " ":
                pos += 1
            if not row_text.startswith(seg, pos):
                # Fall back to searching nearby; record mismatch while preserving useful approximate spans.
                found = row_text.find(seg, max(0, pos - 5))
                if found < 0:
                    raise RuntimeError(f"Cannot align segment pid={pid} side={side} near char {pos}: {seg[:80]!r}")
                align_mode = "nearby_search"
                start = found
            else:
                align_mode = "exact"
                start = pos
            end = start + len(seg)
            segments.append({
                "pair_id": pid,
                "side": side,
                "start": start,
                "end": end,
                "chars": len(seg),
                "words": len(seg.split()),
                "word_spans": word_spans(seg, start),
                "align_mode": align_mode,
            })
            pos = end
    return segments


def encode_offsets(tok, text: str) -> List[Tuple[int, int]]:
    enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    return [(int(a), int(b)) for a, b in enc["offset_mapping"] if int(b) > int(a)]


def summarize_corpus(tok, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, float]] = {}
    n_rows = len(rows)
    total_words = 0
    total_untrunc = 0
    total_visible = 0
    over256 = 0
    length_vals = []
    visible_vals = []
    for r in rows:
        source = str(r.get("source", ""))
        words = int(r.get("words", len(str(r["text"]).split())))
        ids = tok(str(r["text"]), add_special_tokens=False, truncation=False)["input_ids"]
        n_tok = len(ids)
        visible = min(n_tok, SEQ)
        total_words += words
        total_untrunc += n_tok
        total_visible += visible
        over256 += int(n_tok > SEQ)
        length_vals.append(n_tok)
        visible_vals.append(visible)
        d = by_source.setdefault(source, {"rows": 0, "words": 0, "untruncated_tokens": 0, "visible_tokens": 0, "over256": 0})
        d["rows"] += 1
        d["words"] += words
        d["untruncated_tokens"] += n_tok
        d["visible_tokens"] += visible
        d["over256"] += int(n_tok > SEQ)
    for d in by_source.values():
        d["visible_tokens_per_word"] = d["visible_tokens"] / max(1, d["words"])
        d["untruncated_tokens_per_word"] = d["untruncated_tokens"] / max(1, d["words"])
        d["over256_rate"] = d["over256"] / max(1, d["rows"])
        d["visible_token_share"] = d["visible_tokens"] / max(1, total_visible)
    return {
        "rows": n_rows,
        "words": total_words,
        "untruncated_tokens": total_untrunc,
        "visible_tokens_seq256": total_visible,
        "padding_tokens_seq256": n_rows * SEQ - total_visible,
        "over256_rows": over256,
        "over256_rate": over256 / max(1, n_rows),
        "untruncated_tokens_per_word": total_untrunc / max(1, total_words),
        "visible_tokens_per_word": total_visible / max(1, total_words),
        "token_lengths": stats(length_vals),
        "visible_lengths": stats(visible_vals),
        "by_source": by_source,
    }


def pair_retention(tok, qwen_rows: List[Dict[str, Any]], pair_meta_rows: List[Dict[str, Any]], pairs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row_by_idx = {i: r for i, r in enumerate(qwen_rows)}
    side_records: List[Dict[str, Any]] = []
    row_records: List[Dict[str, Any]] = []
    align_modes = {"exact": 0, "nearby_search": 0}
    for meta in pair_meta_rows:
        idx = int(meta["row_index"])
        row = row_by_idx[idx]
        text = str(row["text"])
        offsets = encode_offsets(tok, text)
        n_tok = len(offsets)
        visible_offsets = offsets[:SEQ]
        if visible_offsets:
            visible_end = max(e for _, e in visible_offsets)
        else:
            visible_end = 0
        segments = build_pair_segments(text, list(meta["pair_ids"]), pairs)
        full_sides = 0
        full_pairs: Dict[str, Dict[str, bool]] = {}
        visible_words_total = 0
        words_total = 0
        visible_segment_tokens_total = 0
        segment_tokens_total = 0
        for s in segments:
            align_modes[s["align_mode"]] = align_modes.get(s["align_mode"], 0) + 1
            seg_start = int(s["start"]); seg_end = int(s["end"])
            words = int(s["words"])
            full = visible_end >= seg_end
            if full:
                full_sides += 1
            vis_words = sum(1 for a, b in s["word_spans"] if b <= visible_end)
            seg_tokens = sum(1 for a, b in offsets if a >= seg_start and b <= seg_end)
            vis_seg_tokens = sum(1 for a, b in visible_offsets if a >= seg_start and b <= seg_end)
            visible_words_total += vis_words
            words_total += words
            visible_segment_tokens_total += vis_seg_tokens
            segment_tokens_total += seg_tokens
            full_pairs.setdefault(s["pair_id"], {})[s["side"]] = full
            side_records.append({
                "row_index": idx,
                "pair_id": s["pair_id"],
                "side": s["side"],
                "words": words,
                "visible_words": vis_words,
                "visible_word_fraction": vis_words / max(1, words),
                "tokens_untruncated": seg_tokens,
                "tokens_visible_seq256": vis_seg_tokens,
                "visible_token_fraction": vis_seg_tokens / max(1, seg_tokens),
                "full_retained": full,
                "row_tokens_untruncated": n_tok,
                "row_over256": n_tok > SEQ,
            })
        both_full_pairs = sum(1 for v in full_pairs.values() if v.get("original") and v.get("rewrite"))
        row_records.append({
            "row_index": idx,
            "words": int(row["words"]),
            "n_pairs": int(meta.get("n_pairs", len(full_pairs))),
            "tokens_untruncated": n_tok,
            "tokens_visible_seq256": min(n_tok, SEQ),
            "over256": n_tok > SEQ,
            "visible_end_char": visible_end,
            "full_sides": full_sides,
            "total_sides": len(segments),
            "both_sides_full_pairs": both_full_pairs,
            "total_pairs": len(full_pairs),
            "visible_word_fraction": visible_words_total / max(1, words_total),
            "visible_token_fraction": visible_segment_tokens_total / max(1, segment_tokens_total),
        })
    def frac(pred, records):
        return sum(1 for r in records if pred(r)) / max(1, len(records))
    return {
        "pair_rows": len(row_records),
        "pair_sides": len(side_records),
        "align_modes": align_modes,
        "row_over256_rate": frac(lambda r: r["over256"], row_records),
        "row_all_pairs_both_sides_full_rate": frac(lambda r: r["both_sides_full_pairs"] == r["total_pairs"], row_records),
        "pair_both_sides_full_fraction": sum(r["both_sides_full_pairs"] for r in row_records) / max(1, sum(r["total_pairs"] for r in row_records)),
        "side_full_retained_rate": frac(lambda r: r["full_retained"], side_records),
        "original_side_full_rate": frac(lambda r: r["full_retained"], [r for r in side_records if r["side"] == "original"]),
        "rewrite_side_full_rate": frac(lambda r: r["full_retained"], [r for r in side_records if r["side"] == "rewrite"]),
        "visible_word_fraction_stats_by_side": {
            side: stats([r["visible_word_fraction"] for r in side_records if r["side"] == side])
            for side in ["original", "rewrite"]
        },
        "visible_token_fraction_stats_by_side": {
            side: stats([r["visible_token_fraction"] for r in side_records if r["side"] == side])
            for side in ["original", "rewrite"]
        },
        "row_token_lengths": stats([r["tokens_untruncated"] for r in row_records]),
        "row_visible_token_lengths": stats([r["tokens_visible_seq256"] for r in row_records]),
        "truncated_row_examples": [r for r in row_records if r["over256"]][:12],
    }


def load_optional_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return {"missing": str(path)}


def main() -> None:
    qwen_rows = load_jsonl(CORPORA["qwen"])
    official_rows = load_jsonl(CORPORA["official"])
    pair_meta_rows = load_jsonl(PAIR_META)
    pairs = load_pairs()
    report: Dict[str, Any] = {
        "status": "PAIR_RETENTION_AND_INTERFACE_AUDIT",
        "purpose": "Measure actual seq256-visible token and pair-side retention for the 16k vs 40k data-tokenizer interface after the matched 2x2 screen.",
        "seq_length": SEQ,
        "inputs": {k: str(v) for k, v in CORPORA.items()},
        "tokenizers": {k: str(v) for k, v in TOKENIZERS.items()},
        "parameter_counts_from_2x2_runs": {
            "tok16_8x480": 34467424,
            "tok40_8x480": 45826720,
            "delta_parameters_tok40_minus_tok16": 11359296,
            "interpretation": "40k is a whole interface change: segmentation + visible subword targets + embedding/output capacity."
        },
        "corpus_tokenization": {},
        "pair_retention": {},
        "screen_refs": {
            "twenty_m_summary": str(SUMMARY2X2_20M),
            "ten_m_summary": str(SUMMARY2X2_10M),
        }
    }
    for name, tok_path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        report["corpus_tokenization"][name] = {
            "qwen": summarize_corpus(tok, qwen_rows),
            "official": summarize_corpus(tok, official_rows),
        }
        report["pair_retention"][name] = pair_retention(tok, qwen_rows, pair_meta_rows, pairs)
    # Derived deltas focused on the proposed fragmentation/retention mechanism.
    pr16 = report["pair_retention"]["tok16"]
    pr40 = report["pair_retention"]["tok40"]
    cq16 = report["corpus_tokenization"]["tok16"]["qwen"]
    cq40 = report["corpus_tokenization"]["tok40"]["qwen"]
    report["derived_deltas_tok40_minus_tok16"] = {
        "qwen_visible_tokens_per_word": cq40["visible_tokens_per_word"] - cq16["visible_tokens_per_word"],
        "qwen_untruncated_tokens_per_word": cq40["untruncated_tokens_per_word"] - cq16["untruncated_tokens_per_word"],
        "qwen_over256_rate": cq40["over256_rate"] - cq16["over256_rate"],
        "pair_row_over256_rate": pr40["row_over256_rate"] - pr16["row_over256_rate"],
        "pair_both_sides_full_fraction": pr40["pair_both_sides_full_fraction"] - pr16["pair_both_sides_full_fraction"],
        "pair_side_full_retained_rate": pr40["side_full_retained_rate"] - pr16["side_full_retained_rate"],
        "pair_row_mean_untruncated_tokens": pr40["row_token_lengths"]["mean"] - pr16["row_token_lengths"]["mean"],
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Print the scientific summary.
    compact = {
        "status": report["status"],
        "out": str(OUT),
        "delta_parameters_tok40_minus_tok16": report["parameter_counts_from_2x2_runs"]["delta_parameters_tok40_minus_tok16"],
        "tok16_qwen_visible_tpw": cq16["visible_tokens_per_word"],
        "tok40_qwen_visible_tpw": cq40["visible_tokens_per_word"],
        "tok16_pair_both_sides_full_fraction": pr16["pair_both_sides_full_fraction"],
        "tok40_pair_both_sides_full_fraction": pr40["pair_both_sides_full_fraction"],
        "tok16_pair_row_over256_rate": pr16["row_over256_rate"],
        "tok40_pair_row_over256_rate": pr40["row_over256_rate"],
        "derived_deltas_tok40_minus_tok16": report["derived_deltas_tok40_minus_tok16"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
