#!/usr/bin/env python3
"""research live FineWeb-Edu sentence-pool pilot.

Purpose: after confirming live FineWeb-Edu is reachable with workspace-local HF
caches, cheaply inspect whether a larger future source-breadth/source+view arm
can be built from live FineWeb sentence sources rather than only the 1.75M-word
cached block. This is CPU/network-only and does not train.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import time
import unicodedata
from collections import Counter
from typing import Any

OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_sentence_pilot")
NOTE = pathlib.Path("research/notes/representation_and_objectives/live_fineweb_sentence_pilot.md")
CACHE = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_live_capacity_probe/hf_cache")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,4}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
URL_PAT = re.compile(r"https?://|www\.|\w+@\w+", re.I)
HTML_PAT = re.compile(r"<[^>]{1,60}>|&[a-z]{2,8};", re.I)
MOJIBAKE_PAT = re.compile(r"(?:�|Ã.|Â.|1⁄[24]|o¥|Ç|Ð|Þ|þ)")
BAD_SUBSTR = [
    "cookie policy", "privacy policy", "terms of use", "all rights reserved", "subscribe",
    "click here", "advertisement", "download pdf", "sign up", "log in", "newsletter",
]
DOMAIN_KEYWORDS = {
    "science": ["study", "research", "species", "temperature", "molecule", "cell", "orbit", "planet", "protein", "disease", "climate"],
    "history_society": ["century", "war", "government", "law", "population", "city", "country", "king", "president", "court"],
    "geography": ["river", "mountain", "coast", "island", "county", "province", "region", "located", "north", "south"],
    "quant": ["percent", "million", "billion", "km", "kg", "year", "years", "number", "rate"],
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(" ".join((text or "").split()))


def alpha_frac(text: str) -> float:
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 0.0
    return sum(unicodedata.category(c).startswith("L") for c in nonspace) / len(nonspace)


def nonlatin_letter_frac(text: str) -> float:
    letters = 0
    nonlatin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                nonlatin += 1
    return nonlatin / letters if letters else 0.0


def flags(text: str) -> list[str]:
    low = text.lower()
    out = []
    if URL_PAT.search(text): out.append("urlish")
    if HTML_PAT.search(text): out.append("html")
    if MOJIBAKE_PAT.search(text): out.append("mojibake")
    if any(s in low for s in BAD_SUBSTR): out.append("web_boilerplate")
    if alpha_frac(text) < 0.55: out.append("low_alpha")
    if nonlatin_letter_frac(text) > 0.03: out.append("nonlatin")
    ws = words(text)
    if ws:
        pron = sum(w.lower().strip(".,;:!?()[]{}\"'") in {"i", "you", "we", "me", "my", "your", "our"} for w in ws) / len(ws)
        if pron > 0.18: out.append("advice_or_dialogue_heavy")
        digit_tok = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws)
        if digit_tok > 0.28: out.append("many_digit_tokens")
    return out


def domain_hits(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for k, kws in DOMAIN_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in kws):
            hits.append(k)
    return hits


def score_sentence(sent: str) -> dict[str, Any]:
    ws = words(sent)
    caps = [m.group(0) for m in CAP_PHRASE_RE.finditer(sent)]
    nums = NUM_RE.findall(sent)
    dh = domain_hits(sent)
    fl = flags(sent)
    relationish = 0
    low = sent.lower()
    for token in ["is", "are", "was", "were", "became", "located", "caused", "contains", "includes", "founded", "discovered", "measured", "used", "made", "formed", "produced"]:
        if re.search(r"\b" + re.escape(token) + r"\b", low):
            relationish += 1
    content_score = len(set(caps)) * 0.8 + len(nums) * 0.8 + len(dh) * 1.2 + min(relationish, 3) * 0.5
    return {
        "words": len(ws),
        "capitalized_phrases": caps[:8],
        "numbers": nums[:8],
        "domain_hits": dh,
        "flags": fl,
        "relation_count": relationish,
        "content_score": round(content_score, 3),
        "accepted_basic": len(fl) == 0 and 8 <= len(ws) <= 48 and content_score >= 1.5,
        "accepted_high_anchor": len(fl) == 0 and 10 <= len(ws) <= 36 and content_score >= 3.0 and (len(caps) + len(nums) >= 1),
    }


def setup_hf_cache() -> pathlib.Path:
    for sub in ["home", "hub", "datasets", "modules", "transformers", "xdg", "tmp"]:
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE / "home")
    os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(CACHE / "modules")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(CACHE / "xdg")
    os.environ["TMPDIR"] = str(CACHE / "tmp")
    return CACHE


def stat(vals: list[float | int]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float):
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4), "median": q(0.5), "p90": q(0.9), "max": xs[-1], "sum": round(float(sum(xs)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=64)
    ap.add_argument("--max-doc-words", type=int, default=120_000)
    ap.add_argument("--max-candidates", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    setup_hf_cache()
    import datasets  # type: ignore
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, cache_dir=str(CACHE / "datasets"))
    docs = 0
    doc_words = 0
    candidates = []
    high = []
    flag_counts = Counter()
    domain_counts = Counter()
    for ex in ds:
        text = " ".join(str(ex.get("text", "")).split())
        if not text:
            continue
        docs += 1
        doc_id = str(ex.get("id") or ex.get("dump") or docs)
        dw = len(words(text))
        doc_words += dw
        parts = []
        for para in re.split(r"\n+", text):
            para = " ".join(para.split())
            if not para:
                continue
            parts.extend(SENT_SPLIT.split(para))
        for j, sent in enumerate(parts):
            sent = " ".join(sent.split())
            if not sent:
                continue
            info = score_sentence(sent)
            for fl in info["flags"]:
                flag_counts[fl] += 1
            for dh in info["domain_hits"]:
                domain_counts[dh] += 1
            if info["accepted_basic"]:
                row = {"doc_index": docs-1, "doc_id": doc_id, "sentence_index": j, "text": sent, **info}
                candidates.append(row)
                if info["accepted_high_anchor"]:
                    high.append(row)
                if len(candidates) >= args.max_candidates:
                    break
        if docs >= args.max_docs or doc_words >= args.max_doc_words or len(candidates) >= args.max_candidates:
            break
    cand_path = OUT_DIR / "live_fineweb_sentence_candidates.jsonl"
    high_path = OUT_DIR / "live_fineweb_high_anchor_candidates.jsonl"
    with cand_path.open("w", encoding="utf-8") as f:
        for r in candidates:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with high_path.open("w", encoding="utf-8") as f:
        for r in high:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "status": "LIVE_FINEWEB_SENTENCE_PILOT",
        "max_docs": args.max_docs,
        "docs_scanned": docs,
        "doc_words_scanned": doc_words,
        "candidate_rows": len(candidates),
        "candidate_words": sum(r["words"] for r in candidates),
        "high_anchor_rows": len(high),
        "high_anchor_words": sum(r["words"] for r in high),
        "candidate_word_stats": stat([r["words"] for r in candidates]),
        "high_anchor_word_stats": stat([r["words"] for r in high]),
        "content_score_stats": stat([r["content_score"] for r in candidates]),
        "flag_counts_seen_in_all_sentences": dict(flag_counts.most_common()),
        "domain_counts_in_accepted": dict(Counter(d for r in candidates for d in r["domain_hits"]).most_common()),
        "domain_counts_seen_all": dict(domain_counts.most_common()),
        "candidate_path": str(cand_path),
        "high_anchor_path": str(high_path),
        "cache_dir": str(CACHE),
        "elapsed_sec": round(time.time() - t0, 3),
        "interpretation": "Small live-streaming pilot for feasibility and source quality only; not training evidence.",
    }
    out_json = OUT_DIR / "live_fineweb_sentence_pilot_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research live FineWeb sentence-pool pilot\n\n"]
    lines.append(f"Scanned {docs} live FineWeb-Edu docs / {doc_words:,} document words.\n\n")
    lines.append(f"Accepted basic factual sentence candidates: {len(candidates):,} rows / {summary['candidate_words']:,} words.\n\n")
    lines.append(f"Accepted high-anchor candidates: {len(high):,} rows / {summary['high_anchor_words']:,} words.\n\n")
    lines.append(f"Domain counts in accepted: `{summary['domain_counts_in_accepted']}`.\n\n")
    lines.append("Scientific use: if the queued 17.5% seqsafe96 source-breadth result is positive but under-scaled, live FineWeb-Edu can supply a larger and cleaner sentence-level source pool; the next construction must still preserve exact ≤10M/≤100M accounting, source provenance, quality filtering, tokenizer visibility, and matched controls.\n\n")
    lines.append(f"Candidates: `{cand_path}`\n\nHigh-anchor: `{high_path}`\n\nJSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "note": str(NOTE), "docs": docs, "candidate_words": summary["candidate_words"], "high_anchor_words": summary["high_anchor_words"], "elapsed_sec": summary["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
