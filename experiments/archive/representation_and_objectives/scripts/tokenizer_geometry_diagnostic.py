#!/usr/bin/env python3
"""research: tokenizer/seq256 geometry diagnostic for FineWeb route choices.

The goal is not to evaluate a model. It measures whether candidate sources and
source+rewrite rows have very different token/word and seq256 visibility geometry
under the protected baseline16k tokenizer and an available 40k tokenizer, so that
future training contrasts can separate content from exposure/packing effects.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import AutoTokenizer

ROOT = Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data" / "tokenizer_geometry"
OUT_JSON = OUT_DIR / "tokenizer_geometry_diagnostic.json"
OUT_NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/tokenizer_geometry_diagnostic.md')

TOKENIZERS = {
    "baseline16k": "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model",
    "available40k": "experiments/archive/frontier_consolidation/training/runs/qwen10_stagewise_pairfit_40k_12x384_lamb_bf16_seed44011/hf_model",
}

POOLS = {
    "official_pool_first5000": {
        "path": "experiments/archive/compact_experience/data/mixture/official_pool.jsonl",
        "limit": 5000,
        "kind": "text_rows",
    },
    "clean_qwen_aligned_first5000": {
        "path": "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
        "limit": 5000,
        "kind": "text_rows",
    },
    "a02_high_anchor_sources_all": {
        "path": "experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_sources.jsonl",
        "limit": None,
        "kind": "text_rows",
    },
    "a02_high_precision_sources_all": {
        "path": "experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_sources.jsonl",
        "limit": None,
        "kind": "text_rows",
    },
    "a02_medium_repaired_sources_first12000": {
        "path": "experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl",
        "limit": 12000,
        "kind": "text_rows",
    },
    "a02_accepted_source_rewrite_rows_all": {
        "path": "experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_accepted_rewrites.jsonl",
        "limit": None,
        "kind": "source_rewrite_rows",
    },
    "a01_priority_unique_sources_first12000": {
        "path": "experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_priority_unique_sources.jsonl",
        "limit": 12000,
        "kind": "text_rows",
    },
    "a01_seqsafe96_treatment_first12000": {
        "path": "experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl",
        "limit": 12000,
        "kind": "text_rows",
    },
}


def word_count(text: str) -> int:
    return len(str(text or "").split())


def stat(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    vals.sort()
    def q(p):
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {
        "n": len(vals),
        "min": vals[0],
        "p05": q(0.05),
        "mean": mean(vals),
        "median": median(vals),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": vals[-1],
        "sum": sum(vals),
    }


def iter_jsonl(path: Path, limit: int | None):
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                n += 1
            except Exception:
                continue


def text_for_row(row: dict[str, Any], kind: str) -> str:
    if kind == "source_rewrite_rows":
        return f"{row.get('source_text', '')} {row.get('rewrite_text', '')}".strip()
    return str(row.get("text") or row.get("source_text") or row.get("rewrite_text") or "")


def entity_tokens(row: dict[str, Any]) -> list[str]:
    ents = row.get("entities") or row.get("source_entities") or []
    if not isinstance(ents, list):
        return []
    return [str(e) for e in ents if str(e).strip()]


def number_tokens(row: dict[str, Any]) -> list[str]:
    nums = row.get("numbers") or row.get("source_numbers") or []
    if not isinstance(nums, list):
        return []
    return [str(x) for x in nums if str(x).strip()]


def visible_word_count_approx(tokenizer, text: str, seq_len: int = 256) -> int:
    words = text.split()
    if not words:
        return 0
    # Binary search largest word prefix whose tokenizer length fits seq_len without special tokens.
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n_tok = len(tokenizer(" ".join(words[:mid]), add_special_tokens=False)["input_ids"])
        if n_tok <= seq_len:
            lo = mid
        else:
            hi = mid - 1
    return lo


def analyze_pool(rows: list[dict[str, Any]], kind: str, tokenizer, seq_len: int = 256) -> dict[str, Any]:
    word_counts = []
    tok_counts = []
    tok_per_word = []
    visible_words = []
    truncated = 0
    source_counter = Counter()
    domain_counter = Counter()
    entity_piece_counts = []
    number_piece_counts = []
    single_piece_entities = 0
    total_entities = 0
    single_piece_numbers = 0
    total_numbers = 0
    examples_long = []
    examples_high_ratio = []

    for i, row in enumerate(rows):
        text = text_for_row(row, kind)
        wc = int(row.get("words") or row.get("pair_words") or word_count(text))
        if kind == "source_rewrite_rows":
            wc = int(row.get("pair_words") or word_count(text))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tc = len(ids)
        vis = wc if tc <= seq_len else visible_word_count_approx(tokenizer, text, seq_len)
        if tc > seq_len:
            truncated += 1
            if len(examples_long) < 5:
                examples_long.append({"row_index": i, "words": wc, "tokens": tc, "visible_words": vis, "text_excerpt": text[:300]})
        ratio = tc / wc if wc else None
        if ratio is not None and ratio >= 1.9 and len(examples_high_ratio) < 5:
            examples_high_ratio.append({"row_index": i, "words": wc, "tokens": tc, "ratio": ratio, "text_excerpt": text[:300]})
        word_counts.append(wc)
        tok_counts.append(tc)
        tok_per_word.append(ratio)
        visible_words.append(vis)
        src = row.get("source") or row.get("pool") or row.get("source_asset") or "unknown"
        source_counter[str(src)] += wc
        for d in row.get("domain_hits") or []:
            domain_counter[str(d)] += 1
        for ent in entity_tokens(row):
            n = len(tokenizer(ent, add_special_tokens=False)["input_ids"])
            entity_piece_counts.append(n)
            total_entities += 1
            if n == 1:
                single_piece_entities += 1
        for num in number_tokens(row):
            n = len(tokenizer(num, add_special_tokens=False)["input_ids"])
            number_piece_counts.append(n)
            total_numbers += 1
            if n == 1:
                single_piece_numbers += 1

    return {
        "rows": len(rows),
        "words": sum(word_counts),
        "tokens": sum(tok_counts),
        "tokens_per_word_stats": stat(tok_per_word),
        "token_count_stats": stat(tok_counts),
        "word_count_stats": stat(word_counts),
        "truncated_rows_seq256": truncated,
        "truncated_fraction_seq256": truncated / len(rows) if rows else None,
        "visible_words_seq256_approx": sum(visible_words),
        "visible_word_fraction_seq256_approx": sum(visible_words) / sum(word_counts) if sum(word_counts) else None,
        "source_word_counts_top": source_counter.most_common(10),
        "domain_hit_counts": dict(domain_counter.most_common()),
        "entity_piece_stats": stat(entity_piece_counts),
        "entity_single_piece_fraction": single_piece_entities / total_entities if total_entities else None,
        "number_piece_stats": stat(number_piece_counts),
        "number_single_piece_fraction": single_piece_numbers / total_numbers if total_numbers else None,
        "examples_truncated": examples_long,
        "examples_high_token_word_ratio": examples_high_ratio,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizers = {}
    for label, path in TOKENIZERS.items():
        p = Path(path)
        if p.exists():
            tokenizers[label] = AutoTokenizer.from_pretrained(str(p), use_fast=True)

    payload = {
        "status": "TOKENIZER_GEOMETRY_DIAGNOSTIC",
        "tokenizers": TOKENIZERS,
        "loaded_tokenizers": list(tokenizers.keys()),
        "seq_len": 256,
        "pools": {},
    }

    for pool_name, spec in POOLS.items():
        path = Path(spec["path"])
        if not path.exists():
            payload["pools"][pool_name] = {"missing": True, "path": str(path)}
            continue
        rows = list(iter_jsonl(path, spec.get("limit")))
        pool_result = {"path": str(path), "limit": spec.get("limit"), "kind": spec.get("kind"), "n_rows_loaded": len(rows), "by_tokenizer": {}}
        for tok_label, tok in tokenizers.items():
            pool_result["by_tokenizer"][tok_label] = analyze_pool(rows, spec.get("kind", "text_rows"), tok)
        payload["pools"][pool_name] = pool_result

    # Pairwise compact comparisons of token/word ratios and visibility under baseline16k.
    comp = {}
    base = "baseline16k"
    if base in tokenizers:
        ref_name = "official_pool_first5000"
        ref = payload["pools"].get(ref_name, {}).get("by_tokenizer", {}).get(base)
        if ref:
            for pool_name, pool_res in payload["pools"].items():
                cur = pool_res.get("by_tokenizer", {}).get(base)
                if not cur:
                    continue
                comp[f"{pool_name}_minus_{ref_name}"] = {
                    "tokens_per_word_mean_delta": cur["tokens_per_word_stats"].get("mean") - ref["tokens_per_word_stats"].get("mean"),
                    "visible_word_fraction_delta": cur["visible_word_fraction_seq256_approx"] - ref["visible_word_fraction_seq256_approx"],
                    "truncated_fraction_delta": cur["truncated_fraction_seq256"] - ref["truncated_fraction_seq256"],
                    "entity_single_piece_fraction": cur["entity_single_piece_fraction"],
                    "number_single_piece_fraction": cur["number_single_piece_fraction"],
                }
    payload["comparisons_vs_official_baseline16k"] = comp

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research tokenizer geometry diagnostic\n\n")
    lines.append("This CPU-only diagnostic measures token/word and approximate seq256 visibility for candidate data pools. It does not evaluate any model.\n\n")
    lines.append("Loaded tokenizers: " + ", ".join(payload["loaded_tokenizers"]) + "\n\n")
    lines.append("## Baseline16k summary\n\n")
    lines.append("| pool | rows | words | tok/word mean | token p95 | trunc rows | visible word frac | entity one-piece | number one-piece |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for pool_name, pool_res in payload["pools"].items():
        r = pool_res.get("by_tokenizer", {}).get("baseline16k")
        if not r:
            continue
        ent = r.get("entity_single_piece_fraction")
        num = r.get("number_single_piece_fraction")
        lines.append(
            f"| {pool_name} | {r['rows']} | {r['words']:,} | {r['tokens_per_word_stats'].get('mean'):.3f} | {r['token_count_stats'].get('p95'):.1f} | {r['truncated_rows_seq256']} | {r['visible_word_fraction_seq256_approx']:.4f} | {'' if ent is None else f'{ent:.3f}'} | {'' if num is None else f'{num:.3f}'} |\n"
        )
    if "available40k" in payload["loaded_tokenizers"]:
        lines.append("\n## 40k tokenizer comparison\n\n")
        lines.append("| pool | tok/word 16k | tok/word 40k | visible frac 16k | visible frac 40k | entity one-piece 16k | entity one-piece 40k |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for pool_name, pool_res in payload["pools"].items():
            r16 = pool_res.get("by_tokenizer", {}).get("baseline16k")
            r40 = pool_res.get("by_tokenizer", {}).get("available40k")
            if not r16 or not r40:
                continue
            e16 = r16.get("entity_single_piece_fraction")
            e40 = r40.get("entity_single_piece_fraction")
            lines.append(
                f"| {pool_name} | {r16['tokens_per_word_stats'].get('mean'):.3f} | {r40['tokens_per_word_stats'].get('mean'):.3f} | {r16['visible_word_fraction_seq256_approx']:.4f} | {r40['visible_word_fraction_seq256_approx']:.4f} | {'' if e16 is None else f'{e16:.3f}'} | {'' if e40 is None else f'{e40:.3f}'} |\n"
            )
    lines.append("\n## Route reading\n\n")
    lines.append("Future FineWeb training arms must match row length and token exposure, not only whitespace words. If source/rewrite pools tokenize much shorter or longer than the protected official rows, any component movement could reflect masked-token opportunity and context truncation as well as content. A 40k tokenizer may reduce fragmentation on some entities/numbers but also changes vocabulary and parameter allocation, so it should be a separate interaction experiment rather than bundled into the next data contrast.\n\n")
    lines.append(f"JSON: `{OUT_JSON}`\n")
    OUT_NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_note": str(OUT_NOTE), "loaded_tokenizers": payload["loaded_tokenizers"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
