#!/usr/bin/env python3
"""Audit actual seq256 token exposure by original/rewrite side in research Qwen pair rows.

The trainer tokenizes with `add_special_tokens=False`, `truncation=True`, `max_length=256`,
`padding=max_length`.  The preflight audit counted whole-row token overflow but did not
separate original vs rewrite exposure.  This script uses selected pair metadata plus
qwen_pair_packed_rows_meta.jsonl to reconstruct packed pair-row segment order and estimate
how many non-special token pieces from originals and rewrites are visible after right
truncation.  It also audits the shuffled and original-dup controls when their metadata exist.
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
from typing import Iterable

from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
PAIR_ROWS_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
QWEN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
OUT_DIR = _public_path('experiments/archive/compact_experience/data/token_exposure_audit')
OUT = _public_path('experiments/archive/compact_experience/data/token_exposure_audit/pair_side_truncation_audit.json')
MAX_SEQ = 256


def stats(vals: Iterable[float]) -> dict:
    vals = list(vals)
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    def pct(q: float):
        if not vals_sorted:
            return None
        idx = min(len(vals_sorted)-1, max(0, round((len(vals_sorted)-1)*q)))
        return vals_sorted[idx]
    return {
        "n": len(vals),
        "min": min(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(vals),
    }


def load_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_selected() -> dict[str, dict]:
    pairs = {}
    for o in load_jsonl(SELECTED):
        pid = str(o["pair_id"])
        pairs[pid] = o
    return pairs


def load_pool_rows() -> list[dict]:
    return list(load_jsonl(QWEN_POOL))


def load_pair_row_meta() -> list[dict]:
    rows = list(load_jsonl(PAIR_ROWS_META))
    if not rows:
        raise RuntimeError("no qwen pair-row metadata")
    return rows


def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def visible_by_segments(lengths: list[tuple[str, int, str, str]], max_seq: int = MAX_SEQ) -> list[dict]:
    """lengths entries: (side, token_len, pair_id, source)."""
    remaining = max_seq
    out = []
    for side, L, pid, source in lengths:
        vis = min(L, max(0, remaining))
        trunc = L - vis
        out.append({"side": side, "tokens": L, "visible_tokens": vis, "truncated_tokens": trunc, "pair_id": pid, "source": source})
        remaining -= vis
    return out


def audit_qwen_aligned(tok) -> dict:
    selected = load_selected()
    pool = load_pool_rows()
    meta_rows = load_pair_row_meta()
    side_counts = collections.Counter()
    side_visible = collections.Counter()
    side_trunc = collections.Counter()
    source_side_counts = collections.defaultdict(int)
    source_side_visible = collections.defaultdict(int)
    source_side_trunc = collections.defaultdict(int)
    row_total_tokens = []
    row_visible_tokens = []
    row_trunc_tokens = []
    rows_over = 0
    pair_trunc_rows = []
    per_pair = {}
    for mr in meta_rows:
        row_idx = int(mr["row_index"])
        row_text = pool[row_idx]["text"]
        pair_ids = [str(x) for x in mr["pair_ids"]]
        segments = []
        reconstructed_words = []
        for pid in pair_ids:
            p = selected[pid]
            original = str(p["original"])
            rewrite = str(p["rewrite"])
            source = str(p.get("source", "unknown"))
            segments.append(("original", token_len(tok, original), pid, source))
            segments.append(("rewrite", token_len(tok, rewrite), pid, source))
            reconstructed_words += original.split() + rewrite.split()
        if " ".join(reconstructed_words) != row_text:
            # If punctuation/spacing normalization somehow differs, fail loudly; the audit depends on segment order.
            raise RuntimeError(f"row reconstruction mismatch at qwen row {row_idx}")
        vis = visible_by_segments(segments)
        total = sum(x["tokens"] for x in vis)
        visible = sum(x["visible_tokens"] for x in vis)
        trunc = sum(x["truncated_tokens"] for x in vis)
        row_total_tokens.append(total)
        row_visible_tokens.append(visible)
        row_trunc_tokens.append(trunc)
        if trunc:
            rows_over += 1
        for x in vis:
            side = x["side"]
            src = x["source"]
            side_counts[side] += x["tokens"]
            side_visible[side] += x["visible_tokens"]
            side_trunc[side] += x["truncated_tokens"]
            source_side_counts[src + "::" + side] += x["tokens"]
            source_side_visible[src + "::" + side] += x["visible_tokens"]
            source_side_trunc[src + "::" + side] += x["truncated_tokens"]
            rec = per_pair.setdefault(x["pair_id"], {"source": src, "original_tokens": 0, "rewrite_tokens": 0, "original_visible": 0, "rewrite_visible": 0, "original_truncated": 0, "rewrite_truncated": 0})
            rec[f"{side}_tokens"] += x["tokens"]
            rec[f"{side}_visible"] += x["visible_tokens"]
            rec[f"{side}_truncated"] += x["truncated_tokens"]
        if trunc and len(pair_trunc_rows) < 40:
            pair_trunc_rows.append({
                "row_index": row_idx,
                "pair_ids": pair_ids,
                "row_tokens": total,
                "visible_tokens": visible,
                "truncated_tokens": trunc,
                "segment_visibility": vis,
                "text_prefix": row_text[:260],
            })
    def side_table(side: str) -> dict:
        total = side_counts[side]
        vis = side_visible[side]
        trunc = side_trunc[side]
        return {
            "tokens": total,
            "visible_tokens": vis,
            "truncated_tokens": trunc,
            "visible_fraction": None if total == 0 else round(vis / total, 6),
            "truncated_fraction": None if total == 0 else round(trunc / total, 6),
        }
    pair_rows_with_any_trunc = sum(1 for r in per_pair.values() if r["original_truncated"] or r["rewrite_truncated"])
    pair_rows_with_rewrite_trunc = sum(1 for r in per_pair.values() if r["rewrite_truncated"])
    pair_rows_with_original_trunc = sum(1 for r in per_pair.values() if r["original_truncated"])
    return {
        "status": "QWEN_PAIR_SIDE_TRUNCATION_AUDIT",
        "trainer_tokenization_assumption": "masking_curriculum_trainer tokenizes examples with add_special_tokens=False, truncation=True, max_length=seq_length=256, padding=max_length",
        "qwen_pair_rows": len(meta_rows),
        "selected_pairs": len(selected),
        "pair_rows_over_seq256": rows_over,
        "pair_row_fraction_over_seq256": round(rows_over / len(meta_rows), 6),
        "row_token_stats_before_truncation": stats(row_total_tokens),
        "row_visible_token_stats": stats(row_visible_tokens),
        "row_truncated_token_stats": stats(row_trunc_tokens),
        "side_exposure": {"original": side_table("original"), "rewrite": side_table("rewrite")},
        "source_side_truncated_tokens": dict(source_side_trunc),
        "source_side_visible_tokens": dict(source_side_visible),
        "pairs_with_any_side_truncation": pair_rows_with_any_trunc,
        "pairs_with_original_truncation": pair_rows_with_original_trunc,
        "pairs_with_rewrite_truncation": pair_rows_with_rewrite_trunc,
        "pair_truncation_examples": pair_trunc_rows,
        "per_pair_visibility_path": str(_public_path('experiments/archive/compact_experience/data/token_exposure_audit/per_pair_visibility.jsonl')),
    }, per_pair


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    qwen, per_pair = audit_qwen_aligned(tok)
    per_pair_path = _public_path('experiments/archive/compact_experience/data/token_exposure_audit/per_pair_visibility.jsonl')
    with per_pair_path.open("w", encoding="utf-8") as f:
        for pid, rec in sorted(per_pair.items()):
            obj = {"pair_id": pid, **rec}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    qwen["per_pair_visibility_path"] = str(per_pair_path)
    OUT.write_text(json.dumps(qwen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "qwen_pair_rows": qwen["qwen_pair_rows"],
        "pair_rows_over_seq256": qwen["pair_rows_over_seq256"],
        "side_exposure": qwen["side_exposure"],
        "pairs_with_original_truncation": qwen["pairs_with_original_truncation"],
        "pairs_with_rewrite_truncation": qwen["pairs_with_rewrite_truncation"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
