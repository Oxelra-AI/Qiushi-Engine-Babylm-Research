#!/usr/bin/env python3
"""research: refine live FineWeb sentence-pilot filters offline.

The first live pilot proved reachability but inspection showed false anchors from
sentence-initial generic capitalized words and web-instruction sentences. This
script re-scores the saved candidate sentences with stricter factual-anchor rules
without any network or GPU work.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Any

INP = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_sentence_pilot/live_fineweb_sentence_candidates.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_sentence_filter_refined")
NOTE = pathlib.Path("research/notes/representation_and_objectives/live_fineweb_sentence_filter_refined.md")

GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It",
    "Its", "On", "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When",
    "Where", "While", "With", "You", "Your", "Comparing", "But", "So", "One", "Some", "Many", "Most",
}
WEB_ACTION_PAT = re.compile(r"\b(click here|check out|sign up|log in|subscribe|download|donation|out of stock|for more information|read (why|more|the definitions)|go to the|web site|website|volume license|newsletter|contact us|rsvp|bag lunch|used up your annual|this is your task|bring a)\b", re.I)
ADVICE_START_PAT = re.compile(r"^(if you|when you|for more information|to learn more|read more|click|check out|download|sign up|log in|bring\b|about the author\b)\b", re.I)
BYLINE_PAT = re.compile(r"^(by:|about the author:|by\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b)", re.I)
PHONE_OR_PRICE_PAT = re.compile(r"(?:\$\s*\d|\b\d{3}[-/]\d{3}[-/]\d{4}\b|\b\d{3}/\d{3}-\d{4}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)\b)", re.I)
PRONOUNS = {"i", "you", "we", "me", "my", "your", "our", "us", "we're", "you've", "you're"}
RELATION_PAT = re.compile(r"\b(is|are|was|were|became|becomes|located|founded|discovered|contains|includes|measured|caused|produced|formed|developed|created|introduced|served|won|died|born|published|built)\b", re.I)
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,4}\b")


def clean_caps(text: str) -> list[str]:
    caps = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        # Remove pure generic initial tokens and one-word pronoun/entity false positives.
        if s in GENERIC_CAPS:
            continue
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if not s or s in GENERIC_CAPS:
            continue
        # Single common title-cased words are too weak unless accompanied by numbers/domains.
        caps.append(s)
    return caps


def words(text: str) -> list[str]:
    return text.split()


def extra_flags(row: dict[str, Any]) -> list[str]:
    text = row["text"]
    low = text.lower()
    fl = []
    if WEB_ACTION_PAT.search(text) or ADVICE_START_PAT.search(text):
        fl.append("web_instruction_or_advice")
    if BYLINE_PAT.search(text):
        fl.append("byline_or_metadata")
    if PHONE_OR_PRICE_PAT.search(text):
        fl.append("event_listing_or_contact_fragment")
    if text.count('"') % 2 == 1 or text.count('“') != text.count('”') or text.count("''") % 2 == 1:
        fl.append("unbalanced_or_fragmentary_quote")
    ws = words(text)
    if ws:
        pron_frac = sum(w.lower().strip(".,;:!?()[]{}\"'“”‘’") in PRONOUNS for w in ws) / len(ws)
        if pron_frac > 0.10:
            fl.append("pronoun_instruction_heavy")
    # Reject apparent paragraph headers followed by explanatory text absent a verb.
    if not RELATION_PAT.search(text) and len(NUM_RE.findall(text)) == 0 and len(row.get("domain_hits", [])) == 0:
        fl.append("weak_relation_structure")
    return fl


def stricter_score(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    caps = clean_caps(text)
    nums = NUM_RE.findall(text)
    domains = list(row.get("domain_hits", []))
    relation_count = len(RELATION_PAT.findall(text))
    new_flags = list(row.get("flags", [])) + extra_flags(row)
    score = 1.0 * min(len(set(caps)), 4) + 0.9 * min(len(nums), 4) + 1.2 * len(domains) + 0.5 * min(relation_count, 3)
    w = int(row.get("words", len(text.split())))
    accepted_strict = (
        not new_flags
        and 10 <= w <= 42
        and score >= 3.0
        and (len(set(caps)) >= 1 or len(nums) >= 1)
        and (relation_count >= 1 or len(domains) >= 1 or len(nums) >= 1)
    )
    accepted_relation_dense = (
        not new_flags
        and 10 <= w <= 36
        and score >= 4.0
        and (len(set(caps)) + len(nums) >= 2)
        and (relation_count >= 1 or len(domains) >= 1)
    )
    return {
        **row,
        "clean_capitalized_phrases": caps[:8],
        "strict_numbers": nums[:8],
        "strict_flags": new_flags,
        "strict_relation_count": relation_count,
        "strict_content_score": round(score, 3),
        "accepted_strict_anchor": accepted_strict,
        "accepted_relation_dense": accepted_relation_dense,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in INP.open(encoding="utf-8") if line.strip()]
    scored = [stricter_score(r) for r in rows]
    strict = [r for r in scored if r["accepted_strict_anchor"]]
    dense = [r for r in scored if r["accepted_relation_dense"]]
    flag_counts = Counter(fl for r in scored for fl in r["strict_flags"])
    domain_counts = Counter(d for r in strict for d in r.get("domain_hits", []))
    strict_path = OUT_DIR / "live_fineweb_strict_anchor_candidates.jsonl"
    dense_path = OUT_DIR / "live_fineweb_relation_dense_candidates.jsonl"
    scored_path = OUT_DIR / "live_fineweb_rescored_candidates.jsonl"
    for p, xs in [(scored_path, scored), (strict_path, strict), (dense_path, dense)]:
        with p.open("w", encoding="utf-8") as f:
            for r in xs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    payload = {
        "status": "LIVE_FINEWEB_SENTENCE_FILTER_REFINED",
        "input_candidates": len(rows),
        "input_candidate_words": sum(int(r.get("words", 0)) for r in rows),
        "strict_anchor_rows": len(strict),
        "strict_anchor_words": sum(int(r.get("words", 0)) for r in strict),
        "relation_dense_rows": len(dense),
        "relation_dense_words": sum(int(r.get("words", 0)) for r in dense),
        "strict_accept_rate_by_words": (sum(int(r.get("words", 0)) for r in strict) / max(1, sum(int(r.get("words", 0)) for r in rows))),
        "relation_dense_accept_rate_by_words": (sum(int(r.get("words", 0)) for r in dense) / max(1, sum(int(r.get("words", 0)) for r in rows))),
        "strict_flag_counts": dict(flag_counts.most_common()),
        "strict_domain_counts": dict(domain_counts.most_common()),
        "paths": {"rescored": str(scored_path), "strict_anchor": str(strict_path), "relation_dense": str(dense_path)},
        "interpretation": "Offline repair of naive high-anchor heuristic; use strict/relation-dense rules before scaling live FineWeb selection or Qwen rewrite generation.",
    }
    out_json = OUT_DIR / "live_fineweb_sentence_filter_refined_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research refined live FineWeb sentence filter\n\n"]
    lines.append("Manual inspection of the first live high-anchor sample found false anchors from generic sentence-initial capitals (`In`, `If`, `For`, `By`) and web-instruction sentences. This offline filter removes those without another network pass.\n\n")
    lines.append(f"Input basic candidates: {payload['input_candidates']:,} rows / {payload['input_candidate_words']:,} words.\n\n")
    lines.append(f"Strict-anchor retained: {payload['strict_anchor_rows']:,} rows / {payload['strict_anchor_words']:,} words ({payload['strict_accept_rate_by_words']:.2%} of candidate words).\n\n")
    lines.append(f"Relation-dense retained: {payload['relation_dense_rows']:,} rows / {payload['relation_dense_words']:,} words ({payload['relation_dense_accept_rate_by_words']:.2%} of candidate words).\n\n")
    lines.append(f"Flag counts after repair: `{payload['strict_flag_counts']}`.\n\n")
    lines.append("Scientific implication: live FineWeb can provide large volume, but naive capitalized-phrase heuristics are unsafe. A scaled source+view arm should use relation-dense filters, row provenance, and a small Qwen faithfulness pilot before generation at scale.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "strict_words": payload["strict_anchor_words"], "dense_words": payload["relation_dense_words"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
