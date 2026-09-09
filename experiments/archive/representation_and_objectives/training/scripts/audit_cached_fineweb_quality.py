#!/usr/bin/env python3
"""Audit heuristic text quality of cached INITIAL_MODEL_STUDIES FineWeb-Edu rows.

The cached fallback source is valuable only if it supplies broad factual English text
rather than hidden tokenization waste, mojibake, non-English snippets, list/index
fragments, or document-boundary artifacts. This script measures those risks and
reports how much single-document mass survives transparent filters.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import unicodedata
from typing import Any, Iterable

DEFAULT_FINEWEB = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_quality/fineweb_quality_audit.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/cached_fineweb_quality_audit.md")

MOJIBAKE_PAT = re.compile(r"(?:�|Ã.|Â.|\u00c3|\u00c2|1⁄[24]|o¥|Ç|Ð|Ð|Þ|þ)")
URL_PAT = re.compile(r"https?://|www\.|@\w+\.\w+|\w+@\w+", re.I)
HTML_PAT = re.compile(r"<[^>]{1,40}>|&[a-z]{2,8};", re.I)
BAD_SUBSTR = [
    "lorem ipsum", "click here", "cookie policy", "privacy policy", "terms of use",
    "all rights reserved", "copyright", "subscribe", "login", "sign up", "worksheet",
    "answer key", "download pdf", "buy now", "advertisement", "sponsored",
]


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {"n": len(xs), "min": ys[0], "p05": q(0.05), "mean": statistics.mean(xs), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": ys[-1], "sum": sum(xs)}


def char_script_counts(text: str) -> dict[str, int]:
    c = collections.Counter()
    for ch in text:
        if ch.isspace():
            continue
        oc = ord(ch)
        if oc < 128:
            if ch.isalpha(): c["ascii_alpha"] += 1
            elif ch.isdigit(): c["digit"] += 1
            elif ch.isprintable(): c["ascii_punct_symbol"] += 1
            else: c["control"] += 1
            continue
        name = unicodedata.name(ch, "")
        cat = unicodedata.category(ch)
        if "LATIN" in name:
            c["latin_ext"] += 1
        elif cat.startswith("P") or cat.startswith("S"):
            c["unicode_punct_symbol"] += 1
        elif ch.isdigit():
            c["digit"] += 1
        elif cat.startswith("L"):
            c["nonlatin_letter"] += 1
        elif cat.startswith("N"):
            c["nonlatin_number"] += 1
        elif cat.startswith("C"):
            c["control"] += 1
        else:
            c["other_unicode"] += 1
    return dict(c)


def row_metrics(text: str) -> dict[str, Any]:
    text = norm_text(text)
    chars = len(text)
    words = text.split()
    sc = char_script_counts(text)
    nonspace = sum(sc.values()) or 1
    alpha = sc.get("ascii_alpha", 0) + sc.get("latin_ext", 0) + sc.get("nonlatin_letter", 0)
    latin_alpha = sc.get("ascii_alpha", 0) + sc.get("latin_ext", 0)
    punctsym = sc.get("ascii_punct_symbol", 0) + sc.get("unicode_punct_symbol", 0)
    digits = sc.get("digit", 0) + sc.get("nonlatin_number", 0)
    nonlatin_letter = sc.get("nonlatin_letter", 0)
    bad_hits = [s for s in BAD_SUBSTR if s in text.lower()]
    token_digit_frac = sum(any(ch.isdigit() for ch in w) for w in words) / len(words) if words else 0.0
    short_or_symbol_frac = sum((len(re.sub(r"[^A-Za-z]", "", w)) <= 1) for w in words) / len(words) if words else 0.0
    repeated_token_frac = 0.0
    if words:
        lc = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
        lc = [w for w in lc if w]
        if lc:
            repeated_token_frac = 1.0 - len(set(lc)) / len(lc)
    start = text[:1]
    startish = bool(start and (start.isupper() or start.isdigit() or start in '"“‘\''))
    sentence_endish = bool(re.search(r"[.!?]['\")\]]?$", text.strip()))
    return {
        "words": len(words),
        "chars": chars,
        "script_counts": sc,
        "alpha_frac": alpha / nonspace,
        "latin_alpha_frac_of_letters": latin_alpha / alpha if alpha else 0.0,
        "nonlatin_letter_frac_of_letters": nonlatin_letter / alpha if alpha else 0.0,
        "punct_symbol_frac": punctsym / nonspace,
        "digit_frac": digits / nonspace,
        "token_digit_frac": token_digit_frac,
        "short_or_symbol_token_frac": short_or_symbol_frac,
        "repeated_token_frac": repeated_token_frac,
        "has_mojibake": bool(MOJIBAKE_PAT.search(text)),
        "has_url_email": bool(URL_PAT.search(text)),
        "has_html": bool(HTML_PAT.search(text)),
        "bad_substrings": bad_hits,
        "startish": startish,
        "sentence_endish": sentence_endish,
    }


def row_flags(m: dict[str, Any]) -> list[str]:
    flags = []
    if m["has_mojibake"]: flags.append("mojibake")
    if m["has_html"]: flags.append("html")
    if m["has_url_email"]: flags.append("url_or_email")
    if m["bad_substrings"]: flags.append("bad_substring")
    if m["nonlatin_letter_frac_of_letters"] > 0.03: flags.append("nonlatin")
    if m["alpha_frac"] < 0.55: flags.append("low_alpha")
    if m["punct_symbol_frac"] > 0.28: flags.append("punct_symbol_heavy")
    if m["digit_frac"] > 0.18: flags.append("digit_heavy")
    if m["token_digit_frac"] > 0.30: flags.append("many_digit_tokens")
    if m["short_or_symbol_token_frac"] > 0.32: flags.append("symbol_or_index_like")
    if m["repeated_token_frac"] > 0.65: flags.append("very_repetitive")
    return flags


def tier_ok(flags: list[str], m: dict[str, Any], tier: str) -> bool:
    f = set(flags)
    if tier == "single_doc":
        return True
    if tier == "english_no_mojibake":
        return not (f & {"mojibake", "html", "nonlatin"})
    if tier == "balanced_quality":
        return not (f & {"mojibake", "html", "nonlatin", "low_alpha", "punct_symbol_heavy", "digit_heavy", "many_digit_tokens", "symbol_or_index_like", "very_repetitive"})
    if tier == "balanced_quality_startish":
        return tier_ok(flags, m, "balanced_quality") and m["startish"]
    raise ValueError(tier)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb", default=str(DEFAULT_FINEWEB))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    args = ap.parse_args()
    path = pathlib.Path(args.fineweb)
    rows = []
    flag_counts = collections.Counter()
    tier_counts = collections.Counter()
    tier_words = collections.Counter()
    by_doc = collections.Counter()
    examples_by_flag: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    examples_by_tier: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    all_metrics = []
    total_rows = 0; single_doc_rows = 0; single_doc_words = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            total_rows += 1
            o = json.loads(line)
            ids = o.get("doc_ids") or []
            text = norm_text(o.get("text", ""))
            words = int(o.get("words", len(text.split())))
            if len(ids) != 1:
                continue
            single_doc_rows += 1
            single_doc_words += words
            by_doc[str(ids[0])] += 1
            m = row_metrics(text)
            flags = row_flags(m)
            all_metrics.append(m)
            for fl in flags:
                flag_counts[fl] += 1
                if len(examples_by_flag[fl]) < 5:
                    examples_by_flag[fl].append({"row": i, "doc_id": str(ids[0]), "words": words, "metrics": {k: m[k] for k in ["alpha_frac", "nonlatin_letter_frac_of_letters", "punct_symbol_frac", "digit_frac", "token_digit_frac", "short_or_symbol_token_frac", "repeated_token_frac", "startish"]}, "text_excerpt": text[:700]})
            if not flags:
                flag_counts["no_flags"] += 1
            for tier in ["single_doc", "english_no_mojibake", "balanced_quality", "balanced_quality_startish"]:
                ok = tier_ok(flags, m, tier)
                if ok:
                    tier_counts[tier] += 1
                    tier_words[tier] += words
                    if len(examples_by_tier[tier]) < 5:
                        examples_by_tier[tier].append({"row": i, "doc_id": str(ids[0]), "words": words, "flags": flags, "text_excerpt": text[:700]})
            rows.append({"row": i, "doc_id": str(ids[0]), "words": words, "flags": flags, "metrics": m, "text_excerpt": text[:400]})
    metric_summary = {}
    for key in ["alpha_frac", "latin_alpha_frac_of_letters", "nonlatin_letter_frac_of_letters", "punct_symbol_frac", "digit_frac", "token_digit_frac", "short_or_symbol_token_frac", "repeated_token_frac"]:
        metric_summary[key] = stats([m[key] for m in all_metrics])
    payload = {
        "status": "CACHED_FINEWEB_QUALITY_AUDIT",
        "input": str(path),
        "total_rows": total_rows,
        "single_doc_rows": single_doc_rows,
        "single_doc_words": single_doc_words,
        "unique_single_docs": len(by_doc),
        "rows_per_doc_top10": by_doc.most_common(10),
        "flag_counts": dict(flag_counts),
        "flag_fractions_of_single_doc_rows": {k: v / single_doc_rows for k, v in flag_counts.items()},
        "tier_counts": dict(tier_counts),
        "tier_words": dict(tier_words),
        "tier_word_fraction_of_single_doc": {k: v / single_doc_words for k, v in tier_words.items()},
        "metric_summary": metric_summary,
        "examples_by_flag": dict(examples_by_flag),
        "examples_by_tier": dict(examples_by_tier),
        "interpretation": "The fallback FineWeb cache is usable only as a lower-confound broad-source candidate after quality and seq256 visibility checks. Balanced-quality filters remove obvious mojibake/non-English/listlike rows but also reduce factual mass.",
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = pathlib.Path(args.note); note.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research cached FineWeb quality audit\n\n"]
    lines.append(f"Input: `{path}`. Single-document rows: {single_doc_rows:,}, words: {single_doc_words:,}, unique docs: {len(by_doc):,}.\n\n")
    lines.append("| tier | rows | words | word fraction of single-doc |\n|---|---:|---:|---:|\n")
    for tier in ["single_doc", "english_no_mojibake", "balanced_quality", "balanced_quality_startish"]:
        lines.append(f"| {tier} | {tier_counts[tier]:,} | {tier_words[tier]:,} | {tier_words[tier]/single_doc_words:.4f} |\n")
    lines.append("\n| flag | rows | fraction |\n|---|---:|---:|\n")
    for fl, n in flag_counts.most_common():
        lines.append(f"| {fl} | {n:,} | {n/single_doc_rows:.4f} |\n")
    lines.append("\nInterpretation: cached FineWeb is not leader-quality FineWeb simplification data. A future fallback should prefer a sequence-safe and quality-filtered variant; otherwise the training signal may be diluted by web-fragment and tokenizer-noise artifacts.\n\n")
    lines.append(f"JSON: `{out}`\n")
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "note": str(note), "tier_words": dict(tier_words), "flag_counts_top": flag_counts.most_common(8)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
