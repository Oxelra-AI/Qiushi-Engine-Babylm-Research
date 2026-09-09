#!/usr/bin/env python3
"""research: refine live FineWeb core-fact quality tiers.

Reads the larger research live FineWeb core pool and separates high-precision
self-contained factual rows from rows that passed the first heuristic but still
look like fragments, media/fandom text, attribution/citation residue, advice, or
procedural prose. This is CPU-only source preparation for a possible corrected
FineWeb source/source+view family.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
IN_POOL = ROOT / "data/live_fineweb_core_scale_probe/live_fineweb_core_fact_neardedup.jsonl"
OUT_DIR = ROOT / "data/live_fineweb_quality_tiers"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_quality_tiers.md')

WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
STRONG_RELATION_RE = re.compile(
    r"\b(became|becomes|located|founded|discovered|contains|includes|measured|caused|produced|formed|"
    r"developed|created|introduced|served|won|died|born|published|built|invented|designed|used|made|"
    r"named|defined|classified|requires|required|allows|allow|means|refers|consists|belongs|separated|"
    r"incorporated|orbits|takes|comprises|covers|connects|crosses|flows|borders|led|resulted|increased|decreased)\b",
    re.I,
)
WEAK_BE_HAVE_RE = re.compile(r"\b(is|are|was|were|has|have|had)\b", re.I)
GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It", "Its",
    "On", "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When", "Where",
    "While", "With", "You", "Your", "However", "Moreover", "Many", "Most", "Almost", "About", "Are", "Even",
    "Choosing", "Explore", "Long", "Why", "Read", "Live", "Let", "License", "Good", "Narrow", "Three", "May",
    "Over", "Comparing", "One", "Some", "After", "Before", "During", "Although", "Because", "Since", "New",
    "Old", "Rather", "According", "Principle", "Community", "Ancient", "Modern", "Several", "Another", "Other",
    "First", "Second", "National", "International", "American", "British", "European", "North", "South", "East", "West",
}
BAD_PATTERNS = {
    "citation_parenthetical_or_et_al": re.compile(r"\bet al\.?\b|\([A-Z][A-Za-z-]+(?:\s+(?:and|&)?\s*[A-Z][A-Za-z-]+){0,3}\s*,?\s*(?:19|20)\d{2}[a-z]?\)", re.I),
    "starts_with_attribution": re.compile(r"^(according to|it has been estimated|it is estimated|researchers believe|some believe|critics say|supporters say)\b", re.I),
    "uncertain_or_normative": re.compile(r"\b(possibly|probably|may have|might have|could have|should|ought|most effective when|important for people|in order to|principle\s+\d+)\b", re.I),
    "media_or_fandom": re.compile(r"\b(Star Trek|Doctor Who|Pok[eé]mon|anime|manga|episode|season|television series|video game|fictional|character|novel|film|movie|comics?)\b", re.I),
    "web_or_navigation": re.compile(r"\b(click here|follow this link|read more|for more information|website|web site|subscribe|download|newsletter|contact us|privacy policy|terms of use)\b", re.I),
    "list_or_heading_residue": re.compile(r"^(principle\s*\d+\s*-|[A-Z][A-Za-z]+\s+\d+\s*-|\(?[A-Z][^.!?]{2,50}\):)"),
    "malformed_spacing_or_tokenization": re.compile(r"\b[A-Z][a-z]?\.[A-Z]|\s+[.,;:]|[.,;:]\s*[.,;:]"),
    "first_second_person": re.compile(r"\b(I|you|we|your|our|us|my)\b", re.I),
    "quote_fragment": re.compile(r"^['\"“]|['\"”]$"),
    "phone_price_time": re.compile(r"(?:\$\s*\d|\b\d{3}[-/]\d{3}[-/]\d{4}\b|\b\d{3}/\d{3}-\d{4}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)\b)", re.I),
}


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(norm_ws(text))


def clean_cap_phrases(text: str) -> list[str]:
    out: list[str] = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if s in GENERIC_CAPS or len(s) <= 1:
            continue
        # Single all-caps tokens are useful acronyms; single title-case common words are not.
        if len(s.split()) == 1 and len(s) <= 3 and not s.isupper():
            continue
        out.append(s)
    seen = set(); uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars)


def balanced_light_punct(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for a, b in pairs:
        if text.count(a) != text.count(b):
            return False
    if text.count('"') % 2 != 0 or text.count("“") != text.count("”"):
        return False
    return True


def quality_reasons(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    text = norm_ws(row.get("text", ""))
    ws = words(text)
    caps = clean_cap_phrases(text)
    nums = NUM_RE.findall(text)
    reasons: list[str] = []
    if not text.endswith((".", "!", "?")):
        reasons.append("no_terminal_sentence_punctuation")
    if not balanced_light_punct(text):
        reasons.append("unbalanced_parentheses_or_quotes")
    if alpha_frac(text) < 0.66:
        reasons.append("low_alpha_for_v2")
    digit_frac = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws) if ws else 0.0
    if digit_frac > 0.18:
        reasons.append("too_digit_dense_for_v2")
    for name, pat in BAD_PATTERNS.items():
        if pat.search(text):
            reasons.append(name)
    if len(caps) + len(nums) < 2:
        reasons.append("too_few_specific_anchors")
    if not STRONG_RELATION_RE.search(text):
        # Accept weak be/have rows only if they carry several anchors/domains; otherwise likely definition fragments.
        domains = row.get("core_domain_hits") or []
        if len(caps) + len(nums) + len(domains) < 4 or not WEAK_BE_HAVE_RE.search(text):
            reasons.append("no_strong_relation_or_rich_anchor_set")
    # A very common web-text failure is a sentence whose first cap was counted as an entity but all remaining anchors are thin.
    if caps and caps[0] in GENERIC_CAPS:
        reasons.append("generic_sentence_initial_anchor")
    info = {
        "v2_caps": caps[:12],
        "v2_numbers": nums[:12],
        "v2_anchor_count": len(caps) + len(nums),
        "v2_has_strong_relation": bool(STRONG_RELATION_RE.search(text)),
        "v2_digit_frac": round(digit_frac, 4),
        "v2_alpha_frac": round(alpha_frac(text), 4),
    }
    return reasons, info


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cap_by_doc(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    out = []
    for r in rows:
        d = str(r.get("doc_id"))
        if counts[d] >= cap:
            continue
        counts[d] += 1
        out.append(r)
    return out


def stat(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float | int:
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4), "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": xs[-1], "sum": round(float(sum(xs)), 4)}


def word_sum(rows: list[dict[str, Any]]) -> int:
    return int(sum(int(r.get("words", 0)) for r in rows))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(IN_POOL)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for r in rows:
        reasons, info = quality_reasons(r)
        rr = dict(r)
        rr.update(info)
        rr["quality_v2_reasons"] = reasons
        rr["quality_v2_accepted"] = not reasons
        if reasons:
            rejected.append(rr)
            for reason in reasons:
                reason_counts[reason] += 1
        else:
            kept.append(rr)
    cap12 = cap_by_doc(kept, 12)
    cap8 = cap_by_doc(kept, 8)
    cap4 = cap_by_doc(kept, 4)
    cap2 = cap_by_doc(kept, 2)
    rng = random.Random(18019)
    kept_sample = rng.sample(kept, min(80, len(kept))) if kept else []
    rejected_sample = rng.sample(rejected, min(80, len(rejected))) if rejected else []
    paths = {
        "quality_v2_all_enriched": OUT_DIR / "live_fineweb_quality_v2_all_enriched.jsonl",
        "quality_v2_kept": OUT_DIR / "live_fineweb_quality_v2_kept.jsonl",
        "quality_v2_doccap12": OUT_DIR / "live_fineweb_quality_v2_doccap12.jsonl",
        "quality_v2_doccap8": OUT_DIR / "live_fineweb_quality_v2_doccap8.jsonl",
        "quality_v2_doccap4": OUT_DIR / "live_fineweb_quality_v2_doccap4.jsonl",
        "quality_v2_doccap2": OUT_DIR / "live_fineweb_quality_v2_doccap2.jsonl",
        "kept_sample": OUT_DIR / "live_fineweb_quality_v2_kept_sample.json",
        "rejected_sample": OUT_DIR / "live_fineweb_quality_v2_rejected_sample.json",
        "summary": OUT_DIR / "live_fineweb_quality_tiers_summary.json",
    }
    write_jsonl(paths["quality_v2_all_enriched"], kept + rejected)
    write_jsonl(paths["quality_v2_kept"], kept)
    write_jsonl(paths["quality_v2_doccap12"], cap12)
    write_jsonl(paths["quality_v2_doccap8"], cap8)
    write_jsonl(paths["quality_v2_doccap4"], cap4)
    write_jsonl(paths["quality_v2_doccap2"], cap2)
    paths["kept_sample"].write_text(json.dumps(kept_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["rejected_sample"].write_text(json.dumps(rejected_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    doc_counts = Counter(str(r.get("doc_id")) for r in kept)
    domain_counts = Counter(d for r in kept for d in (r.get("core_domain_hits") or []))
    summary = {
        "status": "LIVE_FINEWEB_QUALITY_TIERS",
        "input_pool": str(IN_POOL),
        "input_rows": len(rows),
        "input_words": word_sum(rows),
        "quality_v2_kept_rows": len(kept),
        "quality_v2_kept_words": word_sum(kept),
        "quality_v2_retention_by_words": word_sum(kept) / max(1, word_sum(rows)),
        "quality_v2_doccap12_rows": len(cap12),
        "quality_v2_doccap12_words": word_sum(cap12),
        "quality_v2_doccap8_rows": len(cap8),
        "quality_v2_doccap8_words": word_sum(cap8),
        "quality_v2_doccap4_rows": len(cap4),
        "quality_v2_doccap4_words": word_sum(cap4),
        "quality_v2_doccap2_rows": len(cap2),
        "quality_v2_doccap2_words": word_sum(cap2),
        "quality_v2_doccap_words_retained_from_step018_cap8": word_sum(cap8) / 300030,
        "word_stats_kept": stat([int(r.get("words", 0)) for r in kept]),
        "score_stats_kept": stat([float(r.get("core_content_score", 0.0)) for r in kept]),
        "reason_counts": dict(reason_counts.most_common(30)),
        "domain_counts_kept": dict(domain_counts.most_common()),
        "unique_docs_kept": len(doc_counts),
        "top_doc_counts_kept": doc_counts.most_common(20),
        "paths": {k: str(v) for k, v in paths.items()},
        "interpretation": "CPU-only source-quality tiering for future source and source+view materialization; not training or score evidence.",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb quality tiers\n\n"]
    lines.append("This CPU-only pass refines the research live core pool after sample reading showed residual citation fragments, fiction/media rows, malformed punctuation, and advice-like statements. It does not train or score a model.\n\n")
    lines.append(f"Input research near-dedup core pool: {len(rows):,} rows / {word_sum(rows):,} words.\n\n")
    lines.append(f"Quality-v2 kept: {len(kept):,} rows / {word_sum(kept):,} words ({100*summary['quality_v2_retention_by_words']:.2f}% of input core words).\n\n")
    lines.append(f"Doc caps: cap12 {len(cap12):,} rows / {word_sum(cap12):,} words; cap8 {len(cap8):,} rows / {word_sum(cap8):,} words; cap4 {len(cap4):,} rows / {word_sum(cap4):,} words; cap2 {len(cap2):,} rows / {word_sum(cap2):,} words.\n\n")
    lines.append(f"Main rejection reasons: `{dict(list(reason_counts.most_common(14)))}`.\n\n")
    lines.append(f"Kept domain counts: `{summary['domain_counts_kept']}`.\n\n")
    lines.append("Scientific use: a future large live FineWeb arm should choose between the broader research cap8 pool and this stricter quality-v2 pool after reading their samples and after the repaired cached-source result arrives. The stricter pool raises self-contained factual precision but reduces usable word yield, so it is a source-quality/coverage tradeoff rather than automatic progress.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nKept sample: `{paths['kept_sample']}`\n\nRejected sample: `{paths['rejected_sample']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary": str(paths["summary"]),
        "note": str(NOTE),
        "kept_words": word_sum(kept),
        "doccap8_words": word_sum(cap8),
        "doccap4_words": word_sum(cap4),
        "retention_pct": 100 * summary["quality_v2_retention_by_words"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
