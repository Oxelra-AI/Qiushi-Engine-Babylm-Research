#!/usr/bin/env python3
"""research: live FineWeb-Edu core-fact scale probe.

CPU/network-only preparation for the corrected FineWeb source/source+view family.
It estimates whether the stricter core-fact filter from research yields enough
self-contained factual source text after near-deduplication and per-document caps
to support a 2.5M--3.5M changed-block experiment, before spending H100 time or
Qwen generation on a large arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict, deque
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
CACHE = ROOT / "data/fineweb_live_capacity_probe/hf_cache"
DEFAULT_OUT = ROOT / "data/live_fineweb_core_scale_probe"
DEFAULT_NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_core_scale_probe.md')

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
RELATION_PAT = re.compile(
    r"\b(is|are|was|were|became|becomes|located|founded|discovered|contains|includes|"
    r"measured|caused|produced|formed|developed|created|introduced|served|won|died|born|"
    r"published|built|invented|designed|used|made|named|defined|classified|requires|required|"
    r"allows|allow|means|refers|consists|belongs|separated|incorporated|orbits|takes|has|have|had)\b",
    re.I,
)
DOMAIN_KEYWORDS = {
    "science": [
        "study", "research", "species", "temperature", "molecule", "cell", "orbit", "planet",
        "protein", "disease", "climate", "instrument", "observatory", "comet", "solar",
        "biological", "membrane", "plasma", "chemical", "genome", "enzyme", "radiation",
        "geology", "mineral", "fossil", "experiment", "satellite", "telescope",
    ],
    "history_society": [
        "century", "war", "government", "law", "population", "city", "country", "king",
        "president", "court", "rights", "citizens", "settled", "incorporated", "nation",
        "treaty", "convention", "empire", "parliament", "election", "dynasty", "republic",
    ],
    "geography": [
        "river", "mountain", "coast", "island", "county", "province", "region", "located",
        "north", "south", "east", "west", "miles", "kilometers", "states", "kingdom",
        "border", "capital", "valley", "lake", "ocean", "sea",
    ],
    "quant": [
        "percent", "million", "billion", "km", "kg", "year", "years", "number", "rate",
        "members", "1,000", "20", "36", "ratio", "average", "population", "area", "height",
    ],
}
GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If",
    "In", "It", "Its", "On", "Or", "She", "That", "The", "Their", "There", "These", "They",
    "This", "To", "We", "When", "Where", "While", "With", "You", "Your", "However", "Moreover",
    "Many", "Most", "Almost", "About", "Are", "Even", "Choosing", "Explore", "Long", "Why",
    "Read", "Live", "Let", "License", "Good", "Narrow", "Three", "May", "Over", "Comparing",
    "One", "Some", "After", "Before", "During", "Although", "Because", "Since", "New", "Old",
}
FIRST_SECOND_PRONOUNS = {"i", "you", "we", "me", "my", "your", "our", "us", "we're", "you're", "you've", "ours"}
HARD_PATTERNS = {
    "url_or_email": re.compile(r"https?://|www\.|\w+@\w+", re.I),
    "html_or_entity": re.compile(r"<[^>]{1,80}>|&[a-z]{2,10};", re.I),
    "mojibake": re.compile(r"(?:�|Ã.|Â.|1⁄[24]|o¥|Ç|Ð|Þ|þ)"),
    "explicit_web_action": re.compile(
        r"\b(click here|check out|sign up|log in|subscribe|download|donation|out of stock|"
        r"for more information|read more|go to the|web site|website|newsletter|contact us|rsvp|"
        r"add to cart|shopping cart|explore further|view comments|follow this link)\b",
        re.I,
    ),
    "copyright_license_boilerplate": re.compile(
        r"\b(creative commons|all rights reserved|privacy policy|terms of use|cookie policy|"
        r"published under .*license|noncommercial|noderivs|license\.?$)\b",
        re.I,
    ),
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


def sha_norm(text: str) -> str:
    return hashlib.sha1(re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().encode()).hexdigest()


def word_ngrams(text: str, n: int = 5) -> set[str]:
    toks = [clean_word(w) for w in words(text) if clean_word(w)]
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)}


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
        if len(s) <= 1:
            continue
        out.append(s)
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def row_quality(text: str) -> tuple[bool, dict[str, Any]]:
    text = norm_ws(text)
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
    score = min(len(caps), 5) + 0.9 * min(len(nums), 4) + 1.2 * len(domains) + 0.5 * min(relation_count, 3)
    if score < 4.0:
        reasons.append("low_core_fact_score")
    accepted = not reasons
    return accepted, {
        "words": len(ws),
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


def stat(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float | int:
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {
        "n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4),
        "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": xs[-1], "sum": round(float(sum(xs)), 4)
    }


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


def near_dedup(rows: list[dict[str, Any]], threshold: float = 0.82, ngram_n: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Simple local near-dedup over normalized word n-grams.

    The goal is not web-scale duplicate detection; it prevents the source pool from
    being inflated by nearly identical sentence variants encountered close together
    in the streaming order.
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    recent: deque[tuple[set[str], int, str]] = deque(maxlen=5000)
    for idx, r in enumerate(rows):
        grams = word_ngrams(r.get("text", ""), n=ngram_n)
        dup_of = None
        for prev_grams, prev_idx, prev_hash in recent:
            if not grams or not prev_grams:
                continue
            inter = len(grams & prev_grams)
            union = len(grams | prev_grams)
            jac = inter / union if union else 0.0
            if jac >= threshold:
                dup_of = {"kept_index": prev_idx, "kept_hash": prev_hash, "jaccard": round(jac, 4)}
                break
        if dup_of is not None:
            rr = dict(r)
            rr["near_duplicate_of_recent"] = dup_of
            rejected.append(rr)
            continue
        r = dict(r)
        r["near_dedup_index"] = len(kept)
        kept.append(r)
        recent.append((grams, len(kept) - 1, str(r.get("dedup_hash"))))
    return kept, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--max-docs", type=int, default=9000)
    ap.add_argument("--max-doc-words", type=int, default=6_000_000)
    ap.add_argument("--target-core-cap8-words", type=int, default=300_000)
    ap.add_argument("--sample-size", type=int, default=80)
    ap.add_argument("--near-dedup-threshold", type=float, default=0.82)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    setup_cache()
    import datasets  # type: ignore

    t0 = time.time()
    ds = datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, cache_dir=str(CACHE / "datasets"))
    seen_hashes: set[str] = set()
    strict_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    reject_sample: list[dict[str, Any]] = []
    rng = random.Random(18018)
    reason_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    docs = 0
    doc_words = 0
    raw_sents = 0
    unique_sents = 0
    report_every_docs = 1000

    for ex in ds:
        text_raw = str(ex.get("text", ""))
        text_norm = norm_ws(text_raw)
        if not text_norm:
            continue
        docs += 1
        dwords = len(words(text_norm))
        doc_words += dwords
        doc_id = str(ex.get("id") or ex.get("url") or ex.get("dump") or docs)
        parts: list[str] = []
        for para in re.split(r"\n+", text_raw):
            para = norm_ws(para)
            if para:
                parts.extend(SENT_SPLIT.split(para))
        for sent_idx, sent in enumerate(parts):
            sent = norm_ws(sent)
            if not sent:
                continue
            raw_sents += 1
            h = sha_norm(sent)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            unique_sents += 1
            ok, info = row_quality(sent)
            row = {
                "source_asset": "live_fineweb_edu_sample10BT_step018_scale_probe",
                "doc_index": docs - 1,
                "doc_id": doc_id,
                "sentence_index": sent_idx,
                "text": sent,
                "dedup_hash": h,
                **info,
            }
            # Strict-like rows include any research-length row with an anchor, even if not core accepted;
            # this lets later checks see what the filter is throwing away.
            if 10 <= info["words"] <= 42 and (info["core_clean_capitalized_phrases"] or info["core_numbers"]):
                strict_rows.append(row)
            if ok:
                for d in info["core_domain_hits"]:
                    domain_counts[d] += 1
                core_rows.append(row)
            else:
                for reason in info["core_reject_reasons"]:
                    reason_counts[reason] += 1
                # Reservoir sample of rejected anchored rows for later reading.
                if len(reject_sample) < args.sample_size:
                    reject_sample.append(row)
                else:
                    j = rng.randint(0, unique_sents - 1)
                    if j < args.sample_size:
                        reject_sample[j] = row
        cap8_now = cap_by_doc(core_rows, 8)
        if docs % report_every_docs == 0:
            print(json.dumps({"event": "progress", "docs": docs, "doc_words": doc_words, "core_words": sum(r["words"] for r in core_rows), "cap8_words": sum(r["words"] for r in cap8_now)}, ensure_ascii=False), flush=True)
        if sum(r["words"] for r in cap8_now) >= args.target_core_cap8_words:
            break
        if docs >= args.max_docs or doc_words >= args.max_doc_words:
            break

    core_dedup, near_rejects = near_dedup(core_rows, threshold=args.near_dedup_threshold)
    cap8 = cap_by_doc(core_dedup, 8)
    cap4 = cap_by_doc(core_dedup, 4)
    cap2 = cap_by_doc(core_dedup, 2)

    rng.shuffle(core_dedup)
    core_sample = core_dedup[: min(args.sample_size, len(core_dedup))]
    rng.shuffle(near_rejects)
    near_reject_sample = near_rejects[: min(args.sample_size, len(near_rejects))]

    paths = {
        "core_all": out_dir / "live_fineweb_core_fact_all.jsonl",
        "core_neardedup": out_dir / "live_fineweb_core_fact_neardedup.jsonl",
        "core_doc_cap8": out_dir / "live_fineweb_core_fact_neardedup_doccap8.jsonl",
        "core_doc_cap4": out_dir / "live_fineweb_core_fact_neardedup_doccap4.jsonl",
        "core_doc_cap2": out_dir / "live_fineweb_core_fact_neardedup_doccap2.jsonl",
        "strict_anchor_like": out_dir / "live_fineweb_strict_anchor_like_all.jsonl",
        "core_sample": out_dir / "live_fineweb_core_fact_scale_sample.json",
        "reject_sample": out_dir / "live_fineweb_core_fact_scale_reject_sample.json",
        "near_reject_sample": out_dir / "live_fineweb_core_fact_neardup_reject_sample.json",
        "summary": out_dir / "live_fineweb_core_scale_probe_summary.json",
    }
    write_jsonl(paths["core_all"], core_rows)
    write_jsonl(paths["core_neardedup"], core_dedup)
    write_jsonl(paths["core_doc_cap8"], cap8)
    write_jsonl(paths["core_doc_cap4"], cap4)
    write_jsonl(paths["core_doc_cap2"], cap2)
    write_jsonl(paths["strict_anchor_like"], strict_rows)
    paths["core_sample"].write_text(json.dumps(core_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["reject_sample"].write_text(json.dumps(reject_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["near_reject_sample"].write_text(json.dumps(near_reject_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def word_sum(rows: list[dict[str, Any]]) -> int:
        return int(sum(int(r.get("words", 0)) for r in rows))

    doc_counts = Counter(str(r.get("doc_id")) for r in core_dedup)
    summary = {
        "status": "LIVE_FINEWEB_CORE_SCALE_PROBE",
        "scientific_purpose": "Estimate clean live FineWeb core-fact yield for a future corrected source/source+view family before GPU training or bulk Qwen generation.",
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT streaming train",
        "limits": vars(args),
        "docs_scanned": docs,
        "doc_words_scanned": doc_words,
        "raw_sentences_seen": raw_sents,
        "unique_sentences_seen": unique_sents,
        "strict_anchor_like_rows": len(strict_rows),
        "strict_anchor_like_words": word_sum(strict_rows),
        "core_all_rows": len(core_rows),
        "core_all_words": word_sum(core_rows),
        "core_neardedup_rows": len(core_dedup),
        "core_neardedup_words": word_sum(core_dedup),
        "near_duplicate_rejected_rows": len(near_rejects),
        "near_duplicate_rejected_words": word_sum(near_rejects),
        "core_doc_cap8_rows": len(cap8),
        "core_doc_cap8_words": word_sum(cap8),
        "core_doc_cap4_rows": len(cap4),
        "core_doc_cap4_words": word_sum(cap4),
        "core_doc_cap2_rows": len(cap2),
        "core_doc_cap2_words": word_sum(cap2),
        "yields_per_doc_word": {
            "strict_anchor_like": word_sum(strict_rows) / doc_words if doc_words else None,
            "core_all": word_sum(core_rows) / doc_words if doc_words else None,
            "core_neardedup": word_sum(core_dedup) / doc_words if doc_words else None,
            "core_doc_cap8": word_sum(cap8) / doc_words if doc_words else None,
            "core_doc_cap4": word_sum(cap4) / doc_words if doc_words else None,
            "core_doc_cap2": word_sum(cap2) / doc_words if doc_words else None,
        },
        "projected_doc_words_needed": {},
        "core_word_stats": stat([int(r.get("words", 0)) for r in core_dedup]),
        "core_score_stats": stat([float(r.get("core_content_score", 0.0)) for r in core_dedup]),
        "core_domain_counts": dict(domain_counts.most_common()),
        "reject_reason_counts": dict(reason_counts.most_common(20)),
        "core_unique_docs_after_neardedup": len(doc_counts),
        "core_top_doc_counts_after_neardedup": doc_counts.most_common(20),
        "paths": {k: str(v) for k, v in paths.items()},
        "elapsed_sec": round(time.time() - t0, 3),
        "interpretation": "CPU/network source-yield evidence only; not model-training or downstream-score evidence.",
    }
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        proj: dict[str, Any] = {}
        for key, y in summary["yields_per_doc_word"].items():
            proj[key] = round(target / y) if y else None
        summary["projected_doc_words_needed"][str(target)] = proj
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb core-fact scale probe\n\n"]
    lines.append("CPU/network-only source-yield measurement for a possible corrected FineWeb source/source+view family; it does not train or evaluate a model.\n\n")
    lines.append(f"Scanned {docs:,} docs / {doc_words:,} document words / {raw_sents:,} raw sentences.\n\n")
    lines.append(f"Core accepted before near-dedup: {len(core_rows):,} rows / {word_sum(core_rows):,} words.\n\n")
    lines.append(f"After local near-dedup (threshold {args.near_dedup_threshold:.2f}): {len(core_dedup):,} rows / {word_sum(core_dedup):,} words; rejected {len(near_rejects):,} rows / {word_sum(near_rejects):,} words.\n\n")
    lines.append(f"Doc caps after near-dedup: cap8 {len(cap8):,} rows / {word_sum(cap8):,} words; cap4 {len(cap4):,} rows / {word_sum(cap4):,} words; cap2 {len(cap2):,} rows / {word_sum(cap2):,} words.\n\n")
    y = summary["yields_per_doc_word"]
    lines.append("Yield per scanned document word: " + ", ".join(f"{k}={v:.3%}" for k, v in y.items() if v is not None) + ".\n\n")
    lines.append("Projected scanned document words needed for target cap8/cap4 core-source budgets:\n\n")
    lines.append("| target source words | cap8 projection | cap4 projection | core-neardedup projection |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        p = summary["projected_doc_words_needed"][str(target)]
        lines.append(f"| {target:,} | {p['core_doc_cap8']:,} | {p['core_doc_cap4']:,} | {p['core_neardedup']:,} |\n")
    lines.append("\nUse: if the repaired seqsafe96 result supports source breadth, this probe estimates the live FineWeb scanning scale and provides a larger candidate source pool for manual/independent_review quality reading and possible faithful-view prompt construction.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nSample accepted rows: `{paths['core_sample']}`\n\nSample rejected rows: `{paths['reject_sample']}`\n")
    note.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary": str(paths["summary"]),
        "note": str(note),
        "docs_scanned": docs,
        "doc_words_scanned": doc_words,
        "core_doc_cap8_words": word_sum(cap8),
        "core_doc_cap4_words": word_sum(cap4),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
