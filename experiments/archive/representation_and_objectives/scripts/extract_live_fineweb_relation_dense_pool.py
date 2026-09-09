#!/usr/bin/env python3
"""Extract a modest relation-dense live FineWeb-Edu sentence pool.

This is CPU/network-only. It deliberately stops at a small accepted-source budget
(default 160k strict-anchor and 80k relation-dense words) rather than downloading
or processing a full 3-4M-word training corpus. Purpose: establish whether live
FineWeb-Edu can supply cleaner factual sentence sources at scale for a future
source+view arm, after the queued source-breadth contrast and compact-view tests
are interpreted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import time
import unicodedata
from collections import Counter
from typing import Any

CACHE = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_live_capacity_probe/hf_cache")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_relation_dense_pool")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/live_fineweb_relation_dense_pool.md")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,4}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
URL_PAT = re.compile(r"https?://|www\.|\w+@\w+", re.I)
HTML_PAT = re.compile(r"<[^>]{1,60}>|&[a-z]{2,8};", re.I)
MOJIBAKE_PAT = re.compile(r"(?:�|Ã.|Â.|1⁄[24]|o¥|Ç|Ð|Þ|þ)")
WEB_ACTION_PAT = re.compile(r"\b(click here|check out|sign up|log in|subscribe|download|donation|out of stock|for more information|read (why|more|the definitions)|go to the|web site|website|volume license|newsletter|contact us|rsvp|bag lunch|used up your annual|this is your task|bring a|add to cart|shopping cart)\b", re.I)
ADVICE_START_PAT = re.compile(r"^(if you|when you|for more information|to learn more|read more|click|check out|download|sign up|log in|bring\b|about the author\b)\b", re.I)
BYLINE_PAT = re.compile(r"^(by:|about the author:|by\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b)", re.I)
PHONE_OR_PRICE_PAT = re.compile(r"(?:\$\s*\d|\b\d{3}[-/]\d{3}[-/]\d{4}\b|\b\d{3}/\d{3}-\d{4}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)\b)", re.I)
RELATION_PAT = re.compile(r"\b(is|are|was|were|became|becomes|located|founded|discovered|contains|includes|measured|caused|produced|formed|developed|created|introduced|served|won|died|born|published|built|invented|designed|used|made|named|defined|classified)\b", re.I)

GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It", "Its",
    "On", "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When", "Where",
    "While", "With", "You", "Your", "Comparing", "But", "So", "One", "Some", "Many", "Most", "About", "However",
    "Almost", "Building", "Bring", "May", "Good", "Narrow", "Three", "Oct",
}
PRONOUNS = {"i", "you", "we", "me", "my", "your", "our", "us", "we're", "you've", "you're"}
DOMAIN_KEYWORDS = {
    "science": ["study", "research", "species", "temperature", "molecule", "cell", "orbit", "planet", "protein", "disease", "climate", "instrument", "observatory", "comet", "solar", "biological"],
    "history_society": ["century", "war", "government", "law", "population", "city", "country", "king", "president", "court", "license", "rights", "citizens"],
    "geography": ["river", "mountain", "coast", "island", "county", "province", "region", "located", "north", "south", "miles", "kilometers"],
    "quant": ["percent", "million", "billion", "km", "kg", "year", "years", "number", "rate", "1,000", "20", "36"],
}


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(norm_ws(text))


def sha_norm(text: str) -> str:
    return hashlib.sha1(re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().encode()).hexdigest()


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars) if chars else 0.0


def nonlatin_letter_frac(text: str) -> float:
    letters = 0; nonlatin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                nonlatin += 1
    return nonlatin / letters if letters else 0.0


def domain_hits(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for name, kws in DOMAIN_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw.lower()) + r"\b", low) for kw in kws):
            hits.append(name)
    return hits


def clean_caps(text: str) -> list[str]:
    out = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        if s in GENERIC_CAPS:
            continue
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if not s or s in GENERIC_CAPS:
            continue
        # Discard all-uppercase one-letter/abbrev-like fragments unless longer than 1.
        if len(s) <= 2 and not s.isupper():
            continue
        out.append(s)
    return out


def flags(text: str) -> list[str]:
    fl = []
    low = text.lower()
    if URL_PAT.search(text): fl.append("urlish")
    if HTML_PAT.search(text): fl.append("html")
    if MOJIBAKE_PAT.search(text): fl.append("mojibake")
    if WEB_ACTION_PAT.search(text) or ADVICE_START_PAT.search(text): fl.append("web_instruction_or_advice")
    if BYLINE_PAT.search(text): fl.append("byline_or_metadata")
    if PHONE_OR_PRICE_PAT.search(text): fl.append("event_listing_or_contact_fragment")
    if any(s in low for s in ["cookie policy", "privacy policy", "terms of use", "all rights reserved", "advertisement"]): fl.append("web_boilerplate")
    if alpha_frac(text) < 0.55: fl.append("low_alpha")
    if nonlatin_letter_frac(text) > 0.03: fl.append("nonlatin")
    if text.count('"') % 2 == 1 or text.count('“') != text.count('”') or text.count("''") % 2 == 1:
        fl.append("unbalanced_or_fragmentary_quote")
    ws = words(text)
    if ws:
        pron_frac = sum(w.lower().strip(".,;:!?()[]{}\"'“”‘’") in PRONOUNS for w in ws) / len(ws)
        digit_frac = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws)
        short_frac = sum(len(re.sub(r"[^A-Za-z]", "", w)) <= 1 for w in ws) / len(ws)
        if pron_frac > 0.10: fl.append("pronoun_instruction_heavy")
        if digit_frac > 0.28: fl.append("many_digit_tokens")
        if short_frac > 0.32: fl.append("symbol_or_index_like")
    return fl


def score_sentence(text: str) -> dict[str, Any]:
    ws = words(text)
    caps = clean_caps(text)
    nums = NUM_RE.findall(text)
    domains = domain_hits(text)
    relation_count = len(RELATION_PAT.findall(text))
    fl = flags(text)
    # Require some relation structure unless numeric/domain evidence is strong.
    if not RELATION_PAT.search(text) and not nums and not domains:
        fl.append("weak_relation_structure")
    score = 1.0 * min(len(set(caps)), 4) + 0.9 * min(len(nums), 4) + 1.2 * len(domains) + 0.5 * min(relation_count, 3)
    accepted_strict = (not fl and 10 <= len(ws) <= 42 and score >= 3.0 and (caps or nums) and (relation_count >= 1 or domains or nums))
    accepted_dense = (not fl and 10 <= len(ws) <= 36 and score >= 4.0 and (len(set(caps)) + len(nums) >= 2) and (relation_count >= 1 or domains))
    return {
        "words": len(ws),
        "clean_capitalized_phrases": caps[:10],
        "numbers": nums[:10],
        "domain_hits": domains,
        "flags": fl,
        "relation_count": relation_count,
        "content_score": round(score, 3),
        "accepted_strict_anchor": accepted_strict,
        "accepted_relation_dense": accepted_dense,
    }


def setup_cache() -> None:
    for sub in ["home", "hub", "datasets", "modules", "transformers", "xdg", "tmp"]:
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE / "home")
    os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(CACHE / "modules")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(CACHE / "xdg")
    os.environ["TMPDIR"] = str(CACHE / "tmp")


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float): return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4), "median": q(0.5), "p90": q(0.9), "max": xs[-1], "sum": round(float(sum(xs)), 4)}


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--max-docs", type=int, default=2500)
    ap.add_argument("--max-doc-words", type=int, default=1_600_000)
    ap.add_argument("--target-strict-words", type=int, default=160_000)
    ap.add_argument("--target-dense-words", type=int, default=80_000)
    ap.add_argument("--max-sentences-written", type=int, default=80_000)
    args = ap.parse_args()
    t0 = time.time(); setup_cache()
    import datasets  # type: ignore
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    note = pathlib.Path(args.note); note.parent.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, cache_dir=str(CACHE / "datasets"))
    all_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    seen = set()
    docs = 0; doc_words = 0; raw_sentences = 0
    flag_counts = Counter(); domain_counts_strict = Counter(); domain_counts_dense = Counter()
    for ex in ds:
        text = norm_ws(ex.get("text", ""))
        if not text:
            continue
        docs += 1
        dw = len(words(text)); doc_words += dw
        doc_id = str(ex.get("id") or ex.get("url") or ex.get("dump") or docs)
        # Keep paragraph boundaries weakly through sentence indices; simple splitter is enough for source candidates.
        parts: list[str] = []
        for para in re.split(r"\n+", str(ex.get("text", ""))):
            para = norm_ws(para)
            if para:
                parts.extend(SENT_SPLIT.split(para))
        for j, sent in enumerate(parts):
            sent = norm_ws(sent)
            if not sent:
                continue
            raw_sentences += 1
            h = sha_norm(sent)
            if h in seen:
                continue
            seen.add(h)
            info = score_sentence(sent)
            for fl in info["flags"]: flag_counts[fl] += 1
            row = {"source_asset": "live_fineweb_edu_sample10BT_step015", "doc_index": docs-1, "doc_id": doc_id, "sentence_index": j, "text": sent, "dedup_hash": h, **info}
            # Avoid unbounded all_rows; keep accepted or flagged examples plus first N.
            if info["accepted_strict_anchor"] or len(all_rows) < args.max_sentences_written:
                all_rows.append(row)
            if info["accepted_strict_anchor"]:
                strict_rows.append(row)
                for d in info["domain_hits"]: domain_counts_strict[d] += 1
            if info["accepted_relation_dense"]:
                dense_rows.append(row)
                for d in info["domain_hits"]: domain_counts_dense[d] += 1
            if len(all_rows) >= args.max_sentences_written and sum(r["words"] for r in strict_rows) >= args.target_strict_words and sum(r["words"] for r in dense_rows) >= args.target_dense_words:
                break
        if (sum(r["words"] for r in strict_rows) >= args.target_strict_words and sum(r["words"] for r in dense_rows) >= args.target_dense_words) or docs >= args.max_docs or doc_words >= args.max_doc_words:
            break

    paths = {
        "all_scored_or_kept": out / "live_fineweb_scored_or_kept_sentences.jsonl",
        "strict_anchor": out / "live_fineweb_strict_anchor_pool.jsonl",
        "relation_dense": out / "live_fineweb_relation_dense_pool.jsonl",
        "summary": out / "live_fineweb_relation_dense_pool_summary.json",
    }
    write_jsonl(paths["all_scored_or_kept"], all_rows)
    write_jsonl(paths["strict_anchor"], strict_rows)
    write_jsonl(paths["relation_dense"], dense_rows)
    strict_words = sum(r["words"] for r in strict_rows); dense_words = sum(r["words"] for r in dense_rows)
    payload = {
        "status": "LIVE_FINEWEB_RELATION_DENSE_POOL",
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT streaming train",
        "docs_scanned": docs,
        "doc_words_scanned": doc_words,
        "raw_sentences_seen": raw_sentences,
        "unique_sentences_seen": len(seen),
        "strict_anchor_rows": len(strict_rows),
        "strict_anchor_words": strict_words,
        "relation_dense_rows": len(dense_rows),
        "relation_dense_words": dense_words,
        "strict_word_yield_per_doc_word": strict_words / doc_words if doc_words else None,
        "dense_word_yield_per_doc_word": dense_words / doc_words if doc_words else None,
        "strict_word_stats": stats([r["words"] for r in strict_rows]),
        "dense_word_stats": stats([r["words"] for r in dense_rows]),
        "strict_score_stats": stats([r["content_score"] for r in strict_rows]),
        "dense_score_stats": stats([r["content_score"] for r in dense_rows]),
        "flag_counts_seen": dict(flag_counts.most_common(30)),
        "strict_domain_counts": dict(domain_counts_strict.most_common()),
        "dense_domain_counts": dict(domain_counts_dense.most_common()),
        "paths": {k: str(v) for k, v in paths.items()},
        "cache_dir": str(CACHE),
        "elapsed_sec": round(time.time() - t0, 3),
        "limits": vars(args),
        "interpretation": "Source-pool feasibility and future Qwen prompt substrate only; not model-training evidence.",
    }
    paths["summary"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research live FineWeb relation-dense source pool\n\n"]
    lines.append(f"Scanned {docs:,} live FineWeb-Edu docs / {doc_words:,} document words / {raw_sentences:,} raw sentence candidates.\n\n")
    lines.append(f"Strict-anchor pool: {len(strict_rows):,} rows / {strict_words:,} words (yield {payload['strict_word_yield_per_doc_word']:.2%} of scanned doc words).\n\n")
    lines.append(f"Relation-dense pool: {len(dense_rows):,} rows / {dense_words:,} words (yield {payload['dense_word_yield_per_doc_word']:.2%} of scanned doc words).\n\n")
    lines.append(f"Dense domain counts: `{payload['dense_domain_counts']}`.\n\n")
    lines.append(f"Flag counts seen: `{payload['flag_counts_seen']}`.\n\n")
    lines.append("Scientific use: this is the minimal non-GPU substrate for a future faithful-rewrite pilot or larger source-breadth corpus if the A01 seqsafe96 and A02 view/compact results warrant it. It does not justify training by itself.\n\n")
    lines.append(f"Relation-dense JSONL: `{paths['relation_dense']}`\n\nStrict-anchor JSONL: `{paths['strict_anchor']}`\n\nSummary: `{paths['summary']}`\n")
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": str(paths["summary"]), "note": str(note), "docs_scanned": docs, "strict_words": strict_words, "dense_words": dense_words, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
