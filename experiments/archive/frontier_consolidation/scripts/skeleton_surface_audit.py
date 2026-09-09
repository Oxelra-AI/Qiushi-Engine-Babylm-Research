#!/usr/bin/env python3
"""research: surface/readability audit of extractive skeleton variants.

This measures whether source-only skeleton variants are plausible training text or
mostly unnatural word lists.  It uses simple closed-form surface features and the
neutral/DeBERTa tokenizer if provided; no LM is loaded.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_AUDIT = ROOT / "data/extractive_skeleton_variant_audit"
DEFAULT_TOKENIZER = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"
DEFAULT_OUT = ROOT / "data/skeleton_surface_audit"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "over", "under",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "this", "that", "these",
    "those", "there", "here", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we",
    "us", "our", "you", "your", "i", "me", "my", "who", "which", "what", "where", "when", "why", "how",
    "not", "no", "only", "just", "also", "very", "more", "most", "less", "many", "some", "any", "all",
    "each", "every", "other", "another", "same",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_word(w: str) -> str:
    parts = WORD_RE.findall(w)
    return "".join(parts).lower() if parts else ""


def words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def lcs_len(a: list[str], b: list[str]) -> int:
    # Efficient enough for <=40 words typical.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1] + 1 if x == y and x else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def mean(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(ys) / len(ys) if ys else None


def median(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(ys) if ys else None


def safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def featurize_row(row: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    text = row["view_text"]
    source = row.get("source_text", "")
    compact = row.get("original_rewrite_text", "")
    view_words = words(text)
    source_words = words(source)
    compact_words = words(compact)
    view_norms = [norm_word(w) for w in view_words]
    source_norms = [norm_word(w) for w in source_words]
    compact_norms = [norm_word(w) for w in compact_words]
    function_count = sum(1 for n in view_norms if n in FUNCTION_WORDS)
    punctuation_count = len(PUNCT_RE.findall(text))
    capitalized_noninitial = sum(1 for i, w in enumerate(view_words) if i > 0 and w[:1].isupper())
    # Source order continuity: selected view words are in source order by construction for source-only variants,
    # but the average gap between matched source positions tells whether it is list-like.
    src_positions = []
    start = 0
    for n in view_norms:
        found = None
        for i in range(start, len(source_norms)):
            if source_norms[i] == n and n:
                found = i
                start = i + 1
                break
        if found is not None:
            src_positions.append(found)
    gaps = [b - a for a, b in zip(src_positions, src_positions[1:])]
    lcs_compact = lcs_len(view_norms, compact_norms)
    tok_len = None
    tok_per_word = None
    unk_frac = None
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tok_len = len(ids)
        tok_per_word = len(ids) / len(view_words) if view_words else None
        unk_id = getattr(tokenizer, "unk_token_id", None)
        unk_frac = sum(1 for x in ids if unk_id is not None and int(x) == int(unk_id)) / len(ids) if ids else None
    return {
        "pair_id": row.get("pair_id"),
        "variant": row.get("variant"),
        "view_words": len(view_words),
        "source_words": len(source_words),
        "function_word_fraction": safe_div(function_count, len(view_words)),
        "punctuation_per_word": safe_div(punctuation_count, len(view_words)),
        "capitalized_noninitial_per_word": safe_div(capitalized_noninitial, len(view_words)),
        "mean_source_position_gap": mean(gaps),
        "median_source_position_gap": median(gaps),
        "max_source_position_gap": max(gaps) if gaps else None,
        "source_match_fraction_in_order": safe_div(len(src_positions), len(view_words)),
        "lcs_fraction_with_compact": safe_div(lcs_compact, len(compact_words)),
        "token_len": tok_len,
        "tokens_per_word": tok_per_word,
        "unk_fraction": unk_frac,
        "text": text,
        "compact_text": compact,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    out = {}
    for v, rs in sorted(by.items()):
        def vals(k: str) -> list[float]:
            return [float(r[k]) for r in rs if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
        out[v] = {
            "n": len(rs),
            "mean_view_words": mean(vals("view_words")),
            "mean_function_word_fraction": mean(vals("function_word_fraction")),
            "median_function_word_fraction": median(vals("function_word_fraction")),
            "mean_punctuation_per_word": mean(vals("punctuation_per_word")),
            "mean_capitalized_noninitial_per_word": mean(vals("capitalized_noninitial_per_word")),
            "mean_source_position_gap": mean(vals("mean_source_position_gap")),
            "median_source_position_gap": median(vals("median_source_position_gap")),
            "mean_max_source_position_gap": mean(vals("max_source_position_gap")),
            "mean_source_match_fraction_in_order": mean(vals("source_match_fraction_in_order")),
            "mean_lcs_fraction_with_compact": mean(vals("lcs_fraction_with_compact")),
            "mean_token_len": mean(vals("token_len")),
            "mean_tokens_per_word": mean(vals("tokens_per_word")),
            "mean_unk_fraction": mean(vals("unk_fraction")),
        }
    # Deltas to compact distribution.
    if "compact" in out:
        c = out["compact"]
        for v, s in out.items():
            for k in ["mean_function_word_fraction", "mean_punctuation_per_word", "mean_source_position_gap", "mean_tokens_per_word", "mean_lcs_fraction_with_compact"]:
                if s.get(k) is not None and c.get(k) is not None:
                    s[f"delta_{k}_vs_compact"] = s[k] - c[k]
    return out


def fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return ""
    return f"{100*float(x):.2f}%" if pct else f"{float(x):.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if args.tokenizer and args.tokenizer.exists():
        os.environ["HF_HOME"] = str(args.out_dir / ".hf_cache")
        os.environ["TRANSFORMERS_CACHE"] = str(args.out_dir / ".hf_cache")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    variants = ["compact", "prefix_repeat", "spread_even", "content_spread", "scored_source_skeleton", "oracle_compact_projection"]
    feats = []
    files = {}
    for v in variants:
        path = args.audit_dir / f"{v}_pairs.jsonl"
        if not path.exists():
            continue
        files[v] = {"path": str(path), "sha256": sha256_file(path)}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feats.append(featurize_row(json.loads(line), tokenizer))
    summary = summarize(feats)
    result = {
        "status": "SKELETON_SURFACE_AUDIT",
        "audit_dir": str(args.audit_dir),
        "variant_files": files,
        "tokenizer": str(args.tokenizer) if tokenizer is not None else None,
        "n_rows": len(feats),
        "summary": summary,
        "notes": [
            "function-word fraction and source-position gaps are crude surface measures of telegraphic/list-like text",
            "source-derived skeletons can have high content coverage while being less grammatical than compact generated views",
            "no language model is loaded; tokenizer is used only for token counts/UNK checks",
        ],
    }
    out_json = args.out_dir / "skeleton_surface_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Write examples with largest distribution gaps for source-only variants.
    ex_path = args.out_dir / "surface_gap_examples.jsonl"
    source_only = [r for r in feats if r["variant"] in {"spread_even", "content_spread", "scored_source_skeleton"}]
    examples = sorted(source_only, key=lambda r: ((r.get("mean_source_position_gap") or 0), -(r.get("function_word_fraction") or 0)), reverse=True)[:60]
    with ex_path.open("w", encoding="utf-8") as f:
        for r in examples:
            f.write(json.dumps({k: r.get(k) for k in ["pair_id", "variant", "view_words", "function_word_fraction", "mean_source_position_gap", "lcs_fraction_with_compact", "text", "compact_text"]}, ensure_ascii=False) + "\n")
    out_md = args.out_dir / "skeleton_surface_audit.md"
    lines = ["# research skeleton surface audit", ""]
    lines.append("Surface/tokenizer audit of exact-length skeleton variants; no model scoring.")
    lines.append("")
    lines.append("| variant | n | function-word fraction | punctuation/word | mean source gap | max-gap mean | LCS with compact | tokens/word | Δ function vs compact | Δ gap vs compact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        s = summary.get(v)
        if not s:
            continue
        lines.append(
            f"| {v} | {s['n']} | {fmt(s.get('mean_function_word_fraction'), True)} | {fmt(s.get('mean_punctuation_per_word'))} | "
            f"{fmt(s.get('mean_source_position_gap'))} | {fmt(s.get('mean_max_source_position_gap'))} | {fmt(s.get('mean_lcs_fraction_with_compact'), True)} | "
            f"{fmt(s.get('mean_tokens_per_word'))} | {fmt(s.get('delta_mean_function_word_fraction_vs_compact'), True)} | {fmt(s.get('delta_mean_source_position_gap_vs_compact'))} |"
        )
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("The source-only variants intentionally improve source-tail content coverage, but the scored/content skeletons are more telegraphic than generated compact views: fewer function words and larger source-position gaps indicate a stronger distribution shift. This does not rule them out as a discriminating short screen, but it argues against treating coverage alone as sufficient. A training test should compare at least compact, prefix-repeat, and one source-only skeleton, and should stop early if source-only skeleton harms broad cheap7 despite high coverage.")
    lines.append("")
    lines.append(f"Examples: `{ex_path}`")
    lines.append(f"JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "n_rows": len(feats)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
