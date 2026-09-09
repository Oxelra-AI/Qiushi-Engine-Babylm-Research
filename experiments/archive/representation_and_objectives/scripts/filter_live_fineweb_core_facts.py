#!/usr/bin/env python3
"""Tighten the research live FineWeb sentence pool into core factual rows.

This is a CPU-only source-quality measurement.  It does not generate text, train a
model, or evaluate a checkpoint.  Purpose: prevent the next source+view experiment
from using noisy web boilerplate or advice/contact fragments merely because the
research heuristic counted them as relation-dense.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
IN_STRICT = ROOT / "data/live_fineweb_relation_dense_pool/live_fineweb_strict_anchor_pool.jsonl"
IN_DENSE = ROOT / "data/live_fineweb_relation_dense_pool/live_fineweb_relation_dense_pool.jsonl"
OUT_DIR = ROOT / "data/live_fineweb_core_fact_filter"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_core_fact_filter.md')

WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
RELATION_PAT = re.compile(r"\b(is|are|was|were|became|becomes|became|located|founded|discovered|contains|includes|measured|caused|produced|formed|developed|created|introduced|served|won|died|born|published|built|invented|designed|used|made|named|defined|classified|requires|required|allows|allow|means|refers|consists|belongs|separated|incorporated|orbits|takes|has|have|had)\b", re.I)
DOMAIN_KEYWORDS = {
    "science": ["study", "research", "species", "temperature", "molecule", "cell", "orbit", "planet", "protein", "disease", "climate", "instrument", "observatory", "comet", "solar", "biological", "membrane", "plasma"],
    "history_society": ["century", "war", "government", "law", "population", "city", "country", "king", "president", "court", "rights", "citizens", "settled", "incorporated", "nation", "treaty", "convention"],
    "geography": ["river", "mountain", "coast", "island", "county", "province", "region", "located", "north", "south", "miles", "kilometers", "states", "kingdom"],
    "quant": ["percent", "million", "billion", "km", "kg", "year", "years", "number", "rate", "members", "1,000", "20", "36"],
}
GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It", "Its", "On", "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When", "Where", "While", "With", "You", "Your", "However", "Moreover", "Many", "Most", "Almost", "About", "Are", "Even", "Choosing", "Explore", "Long", "Why", "Read", "Live", "Let", "License", "Good", "Narrow", "Three", "May", "Over", "Comparing"
}
FIRST_SECOND_PRONOUNS = {"i", "you", "we", "me", "my", "your", "our", "us", "we're", "you're", "you've", "ours"}
HARD_PATTERNS = {
    "url_or_email": re.compile(r"https?://|www\.|\w+@\w+", re.I),
    "html_or_entity": re.compile(r"<[^>]{1,80}>|&[a-z]{2,10};", re.I),
    "mojibake": re.compile(r"(?:�|Ã.|Â.|1⁄[24]|o¥|Ç|Ð|Þ|þ)"),
    "explicit_web_action": re.compile(r"\b(click here|check out|sign up|log in|subscribe|download|donation|out of stock|for more information|read more|go to the|web site|website|newsletter|contact us|rsvp|add to cart|shopping cart|explore further|view comments)\b", re.I),
    "copyright_license_boilerplate": re.compile(r"\b(creative commons|all rights reserved|privacy policy|terms of use|cookie policy|published under .*license|noncommercial|noderivs|license\.?$)\b", re.I),
    "event_contact_price": re.compile(r"(?:\$\s*\d|\b\d{3}[-/]\d{3}[-/]\d{4}\b|\b\d{3}/\d{3}-\d{4}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)\b|bag lunch|bring a|rsvp)", re.I),
    "byline_metadata": re.compile(r"^(by:|about the author:|posted by|written by|photo by|copyright\b)", re.I),
    "imperative_advice_start": re.compile(r"^(if you|when you|for more information|to learn more|read more|click|check out|download|sign up|log in|bring\b|choose\b|choosing\b)", re.I),
    "quote_report_fragment": re.compile(r"^['\"“].{0,80}\b(said|says|according to)\b", re.I),
    "trailing_colon_title": re.compile(r"^[A-Z][^.!?]{0,80}:\s+"),
}


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(norm_ws(text))


def clean_word(w: str) -> str:
    return w.lower().strip(".,;:!?()[]{}\"'“”‘’")


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars) if chars else 0.0


def nonlatin_letter_frac(text: str) -> float:
    letters = 0
    nonlatin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                nonlatin += 1
    return nonlatin / letters if letters else 0.0


def domain_hits(text: str) -> list[str]:
    low = text.lower()
    out = []
    for name, kws in DOMAIN_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw.lower()) + r"\b", low) for kw in kws):
            out.append(name)
    return out


def clean_caps(text: str) -> list[str]:
    out: list[str] = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        if s in GENERIC_CAPS:
            continue
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if not s or s in GENERIC_CAPS:
            continue
        # Preserve real all-caps abbreviations of length >=2; discard accidental one-letter tokens.
        if len(s) <= 1:
            continue
        out.append(s)
    # unique while preserving order
    seen = set(); uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float | int:
        return xs[min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))]
    return {"n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4), "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": xs[-1], "sum": round(float(sum(xs)), 4)}


def row_quality(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    text = norm_ws(row.get("text", ""))
    ws = words(text)
    caps = clean_caps(text)
    nums = NUM_RE.findall(text)
    domains = domain_hits(text)
    relation_count = len(RELATION_PAT.findall(text))
    reasons: list[str] = []
    if not (10 <= len(ws) <= 42):
        reasons.append("length_outside_10_42")
    if alpha_frac(text) < 0.62:
        reasons.append("low_alpha")
    if nonlatin_letter_frac(text) > 0.02:
        reasons.append("nonlatin")
    for name, pat in HARD_PATTERNS.items():
        if pat.search(text):
            reasons.append(name)
    if text.count('"') % 2 == 1 or text.count('“') != text.count('”') or text.count("''") % 2 == 1:
        reasons.append("unbalanced_quote")
    pron_frac = sum(clean_word(w) in FIRST_SECOND_PRONOUNS for w in ws) / len(ws) if ws else 0.0
    if pron_frac > 0.035:
        reasons.append("first_second_person_heavy")
    digit_frac = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws) if ws else 0.0
    if digit_frac > 0.25:
        reasons.append("too_digit_dense")
    if not (caps or nums):
        reasons.append("no_entity_or_number_anchor")
    if relation_count < 1:
        reasons.append("no_explicit_relation_verb")
    # Score favors rows where a masked LM can see a reusable factual relation, not only a title or instruction.
    score = min(len(caps), 5) + 0.9 * min(len(nums), 4) + 1.2 * len(domains) + 0.5 * min(relation_count, 3)
    if score < 4.0:
        reasons.append("low_core_fact_score")
    accepted = not reasons
    info = {
        "core_fact_accepted": accepted,
        "core_reject_reasons": reasons,
        "core_clean_capitalized_phrases": caps[:12],
        "core_numbers": nums[:12],
        "core_domain_hits": domains,
        "core_relation_count": relation_count,
        "core_content_score": round(score, 3),
        "core_pronoun_fraction": round(pron_frac, 4),
        "core_digit_fraction": round(digit_frac, 4),
    }
    return accepted, info


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cap_by_doc(rows: list[dict[str, Any]], max_per_doc: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out = []
    for r in rows:
        d = str(r.get("doc_id"))
        if counts[d] >= max_per_doc:
            continue
        counts[d] += 1
        out.append(r)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict = read_jsonl(IN_STRICT)
    dense_ids = {r.get("dedup_hash") for r in read_jsonl(IN_DENSE)}
    enriched: list[dict[str, Any]] = []
    core: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for r in strict:
        ok, info = row_quality(r)
        rr = dict(r)
        rr["was_step015_relation_dense"] = rr.get("dedup_hash") in dense_ids
        rr.update(info)
        enriched.append(rr)
        if ok:
            core.append(rr)
        else:
            for reason in info["core_reject_reasons"]:
                reject_counts[reason] += 1

    core_div8 = cap_by_doc(core, 8)
    core_div4 = cap_by_doc(core, 4)
    rng = random.Random(16016)
    sample_core = rng.sample(core, min(40, len(core))) if core else []
    rejected = [r for r in enriched if not r["core_fact_accepted"]]
    sample_reject = rng.sample(rejected, min(40, len(rejected))) if rejected else []

    paths = {
        "enriched_all_strict": OUT_DIR / "live_fineweb_strict_with_core_filter.jsonl",
        "core_fact_all": OUT_DIR / "live_fineweb_core_fact_pool.jsonl",
        "core_fact_doc_cap8": OUT_DIR / "live_fineweb_core_fact_pool_doc_cap8.jsonl",
        "core_fact_doc_cap4": OUT_DIR / "live_fineweb_core_fact_pool_doc_cap4.jsonl",
        "sample_core": OUT_DIR / "live_fineweb_core_fact_sample.json",
        "sample_reject": OUT_DIR / "live_fineweb_core_reject_sample.json",
        "summary": OUT_DIR / "live_fineweb_core_fact_filter_summary.json",
    }
    write_jsonl(paths["enriched_all_strict"], enriched)
    write_jsonl(paths["core_fact_all"], core)
    write_jsonl(paths["core_fact_doc_cap8"], core_div8)
    write_jsonl(paths["core_fact_doc_cap4"], core_div4)
    paths["sample_core"].write_text(json.dumps(sample_core, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["sample_reject"].write_text(json.dumps(sample_reject, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    doc_counts = Counter(str(r.get("doc_id")) for r in core)
    domain_counts = Counter(d for r in core for d in r.get("core_domain_hits", []))
    summary = {
        "status": "LIVE_FINEWEB_CORE_FACT_FILTER",
        "inputs": {"strict_anchor": str(IN_STRICT), "relation_dense": str(IN_DENSE)},
        "strict_input_rows": len(strict),
        "strict_input_words": sum(int(r.get("words", 0)) for r in strict),
        "relation_dense_rows_inside_strict": sum(1 for r in enriched if r["was_step015_relation_dense"]),
        "core_fact_rows": len(core),
        "core_fact_words": sum(int(r.get("words", 0)) for r in core),
        "core_fact_doc_cap8_rows": len(core_div8),
        "core_fact_doc_cap8_words": sum(int(r.get("words", 0)) for r in core_div8),
        "core_fact_doc_cap4_rows": len(core_div4),
        "core_fact_doc_cap4_words": sum(int(r.get("words", 0)) for r in core_div4),
        "core_retention_by_words_from_strict": sum(int(r.get("words", 0)) for r in core) / max(1, sum(int(r.get("words", 0)) for r in strict)),
        "core_word_stats": stats([int(r.get("words", 0)) for r in core]),
        "core_score_stats": stats([r.get("core_content_score", 0.0) for r in core]),
        "reject_reason_counts": dict(reject_counts.most_common()),
        "core_domain_counts": dict(domain_counts.most_common()),
        "core_unique_docs": len(doc_counts),
        "core_top_doc_counts": doc_counts.most_common(20),
        "paths": {k: str(v) for k, v in paths.items()},
        "interpretation": "CPU-only source-quality filter for future FineWeb source+view arms; not training or evaluation evidence.",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb core factual filter\n\n"]
    lines.append("This CPU-only pass tightens the research live FineWeb sentence pool after sample inspection found web instructions, licenses, and advice/contact fragments in the relation-dense set. It is preparation for a possible source+faithful-view corpus, not model evidence.\n\n")
    lines.append(f"Strict research input: {summary['strict_input_rows']:,} rows / {summary['strict_input_words']:,} words.\n\n")
    lines.append(f"Core-fact retained: {summary['core_fact_rows']:,} rows / {summary['core_fact_words']:,} words ({100*summary['core_retention_by_words_from_strict']:.2f}% of strict words), across {summary['core_unique_docs']:,} docs.\n\n")
    lines.append(f"Doc-cap8 pool: {summary['core_fact_doc_cap8_rows']:,} rows / {summary['core_fact_doc_cap8_words']:,} words; doc-cap4 pool: {summary['core_fact_doc_cap4_rows']:,} rows / {summary['core_fact_doc_cap4_words']:,} words.\n\n")
    lines.append(f"Core domain counts: `{summary['core_domain_counts']}`.\n\n")
    lines.append(f"Main rejection reasons: `{dict(list(reject_counts.most_common(12)))}`.\n\n")
    lines.append("Use: if A01's pending source-breadth trajectory justifies scaling, run a larger extraction with this stricter notion (or stronger) before Qwen generation. Do not use the research relation-dense pool directly as if it were clean factual data.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nCore pool: `{paths['core_fact_all']}`\n\nSamples: `{paths['sample_core']}` and `{paths['sample_reject']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": summary["status"], "summary": str(paths["summary"]), "note": str(NOTE), "core_fact_words": summary["core_fact_words"], "core_doc_cap8_words": summary["core_fact_doc_cap8_words"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
