#!/usr/bin/env python3
"""research: repair live FineWeb source selection after sample reading.

This CPU-only script applies source-text cleanup, self-containment checks,
proposition typing, epistemic-status separation, and stronger within-document
redundancy control to the research live FineWeb strict-anchor-like pool.  It
prepares a more interpretable source substrate for a possible corrected
FineWeb source/source+view experiment; it does not train or evaluate a model.
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
IN_ROWS = ROOT / "data/live_fineweb_core_scale_probe/live_fineweb_strict_anchor_like_all.jsonl"
SUMMARY = ROOT / "data/live_fineweb_core_scale_probe/live_fineweb_core_scale_probe_summary.json"
OUT_DIR = ROOT / "data/live_fineweb_source_selector_v3"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_source_selector_v3.md')

WORD_RE = re.compile(r"\S+")
ALNUM_RE = re.compile(r"[a-z0-9]+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
STOP = {
    "the", "and", "that", "with", "from", "this", "were", "have", "has", "had", "for", "are", "was", "its",
    "into", "onto", "over", "under", "about", "between", "through", "also", "than", "then", "when", "where",
    "which", "while", "their", "there", "these", "those", "being", "been", "after", "before", "because", "during",
}
GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It", "Its", "On",
    "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When", "Where", "While",
    "With", "You", "Your", "However", "Moreover", "Many", "Most", "Almost", "About", "Are", "Even", "Choosing",
    "Explore", "Long", "Why", "Read", "Live", "Let", "License", "Good", "Narrow", "Three", "May", "Over",
    "Comparing", "One", "Some", "After", "Before", "During", "Although", "Because", "Since", "New", "Old", "Rather",
    "According", "Accordingly", "Principle", "Community", "Ancient", "Modern", "Several", "Another", "Other", "First",
    "Second", "National", "International", "American", "British", "European", "North", "South", "East", "West", "States",
}
BAD_PATTERNS = {
    "url_or_email": re.compile(r"https?://|www\.|\w+@\w+", re.I),
    "web_or_navigation": re.compile(r"\b(click here|follow this link|read more|for more information|website|web site|subscribe|download|newsletter|contact us|privacy policy|terms of use|shopping cart|out of stock|view comments)\b", re.I),
    "html_or_markup": re.compile(r"<[^>]{1,80}>|&[a-z]{2,10};|`{1,3}|\*\*|__|\|"),
    "contact_or_listing": re.compile(r"(?:\$\s*\d|\b\d{3}[-/]\d{3}[-/]\d{4}\b|\b\d{3}/\d{3}-\d{4}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)\b|bag lunch|rsvp|email:)" , re.I),
    "byline_or_metadata": re.compile(r"^(by:|about the author:|posted by|written by|photo by|copyright\b|source:|credit:)" , re.I),
    "list_heading_residue": re.compile(r"^(principle\s*\d+\s*-|chapter\s*\d+\s*-|\(?[A-Z][^.!?]{2,48}\):|[-•*]\s+)" , re.I),
    "media_fandom": re.compile(r"\b(Star Trek|Doctor Who|Pok[eé]mon|anime|manga|episode|season|television series|video game|fictional|character|comic book|fan fiction)\b", re.I),
    "copyright_license": re.compile(r"\b(creative commons|all rights reserved|privacy policy|terms of use|cookie policy|noncommercial|noderivs)\b", re.I),
    "malformed_spacing": re.compile(r"\b[A-Z][a-z]?\.[A-Z]|\s+[.,;:]|[.,;:]\s*[.,;:]|\bthat\s+that\b", re.I),
    "quote_fragment": re.compile(r"^['\"“]|['\"”]$"),
}
EPISTEMIC_RE = re.compile(r"\b(according to|reported|argued|claimed|believed|estimated|suggested|forecast|worried|concerned|possible|possibly|probably|likely|may|might|could|would|will soon|recently|currently|today|now)\b", re.I)
REL_STRONG = re.compile(
    r"\b(became|becomes|located|founded|discovered|contains|includes|measured|caused|produced|formed|developed|created|introduced|served|won|died|born|published|built|invented|designed|used|made|named|defined|classified|requires|required|allows|allow|means|refers|consists|belongs|separated|incorporated|orbits|takes|comprises|covers|connects|crosses|flows|borders|led|resulted|increased|decreased|converted|established|replaced|joined|opened|closed|owned|controlled)\b",
    re.I,
)
REL_WEAK = re.compile(r"\b(is|are|was|were|has|have|had)\b", re.I)
DEFINITION_RE = re.compile(r"\b(is|are|was|were)\s+(?:a|an|the)?\s*(?:type|kind|form|class|group|family|method|process|measure|term|name|part|member|example)\b|\b(means|refers to|is defined as|are defined as|consists of|is composed of|are composed of)\b", re.I)
CAUSAL_RE = re.compile(r"\b(because|caused?|due to|led to|resulted in|therefore|thus|so that|prevents?|reduces?|increases?|damages?|protects?|enables?|allows?)\b", re.I)
SPATIAL_RE = re.compile(r"\b(located|lies|situated|river|mountain|island|county|province|region|north|south|east|west|border|coast|flows|crosses|kilometers|miles|capital)\b", re.I)
BIO_HIST_RE = re.compile(r"\b(born|died|founded|established|published|won|served|king|president|minister|war|century|empire|treaty|government|court|election)\b", re.I)
QUANT_RE = re.compile(r"\b(percent|million|billion|rate|ratio|average|population|area|height|weight|km|kg|years?|\d+(?:[,.]\d+)*)\b", re.I)
CATALOGUE_RE = re.compile(r"(?:\b[A-Z][A-Za-z]+\b[^.!?]{0,20},\s*){3,}|\b(?:including|such as)\b[^.!?]{0,160}(?:,\s*[^,]+){3,}", re.I)
UNRESOLVED_START_RE = re.compile(r"^(he|she|it|they|this|that|these|those|such|both|around both|the former|the latter)\b", re.I)
UNRESOLVED_NOMINAL_RE = re.compile(r"^(the|this|that|these|those|such)\s+(mission|craft|ship|report|survey|decision|position|country|region|area|group|team|company|organization|system|process|method|project|program|movement|issue|case|court|law|model|study|research|data|result|effect|pipeline|proposal|plan)\b", re.I)
AMBIG_PASSIVE_RE = re.compile(r"\b(were|was|been|be)\s+(affected|involved|associated|concerned|known|found|seen|observed|reported|estimated)\b(?![^.!?]{0,60}\b(by|from|with|in|on|at|during|because|due to)\b)", re.I)


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(norm_ws(text))


def word_count(text: str) -> int:
    return len(words(text))


def clean_token(w: str) -> str:
    return w.lower().strip(".,;:!?()[]{}\"'“”‘’")


def cap_phrases(text: str) -> list[str]:
    out: list[str] = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if not s or s in GENERIC_CAPS:
            continue
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
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars) if chars else 0.0


def repair_text(text: str) -> tuple[str, list[str]]:
    t = norm_ws(text)
    actions: list[str] = []
    orig = t
    # Drop common markdown bullets and loose wrapping punctuation.
    nt = re.sub(r"^[-•*]+\s+", "", t).strip()
    if nt != t:
        actions.append("removed_leading_bullet")
        t = nt
    nt = re.sub(r"[`*_]+", "", t)
    if nt != t:
        actions.append("removed_markdown_marks")
        t = nt
    # Fix missing sentence spaces, e.g. article.Vinegar.
    nt = re.sub(r"(?<=[a-z0-9])\.(?=[A-Z])", ". ", t)
    if nt != t:
        actions.append("inserted_missing_period_space")
        t = nt
    # Remove trailing citation-only tails but keep the factual sentence.
    nt = re.sub(r"\s*\((?:see\s+)?[^()]{0,90}\b(?:19|20)\d{2}[a-z]?[^()]{0,60}\)\.?$", ".", t, flags=re.I)
    if nt != t:
        actions.append("removed_trailing_author_year_citation")
        t = nt
    nt = re.sub(r"\s*\(\d+\)\.?$", ".", t)
    if nt != t:
        actions.append("removed_trailing_numeric_citation")
        t = nt
    # If the splitter produced multiple sentences, keep the longest candidate that is well-sized.
    parts = [p.strip() for p in SENT_SPLIT.split(t) if p.strip()]
    if len(parts) > 1:
        candidates = [p for p in parts if 10 <= word_count(p) <= 42]
        if candidates:
            t2 = max(candidates, key=lambda p: (len(cap_phrases(p)) + len(NUM_RE.findall(p)), word_count(p)))
            if t2 != t:
                actions.append("selected_one_sentence_after_split")
                t = t2
    # Convert double final punctuation and whitespace before punctuation.
    nt = re.sub(r"\s+([.,;:!?])", r"\1", t)
    nt = re.sub(r"([.!?]){2,}$", r"\1", nt)
    if nt != t:
        actions.append("normalized_punctuation_spacing")
        t = nt
    if t and t[-1] not in ".!?":
        # Do not add punctuation to likely fragments; mark only by leaving as-is.
        pass
    return t, actions if actions else ["none"]


def proposition_types(text: str, caps: list[str], nums: list[str]) -> list[str]:
    types: list[str] = []
    if DEFINITION_RE.search(text): types.append("definition_taxonomy")
    if CAUSAL_RE.search(text): types.append("causal_mechanistic")
    if SPATIAL_RE.search(text): types.append("spatial_geographic")
    if BIO_HIST_RE.search(text): types.append("biographical_historical")
    if QUANT_RE.search(text) and nums: types.append("quantitative")
    if CATALOGUE_RE.search(text) or (len(caps) >= 6 and text.count(",") >= 3): types.append("catalogue_list")
    if EPISTEMIC_RE.search(text): types.append("attributed_modal_temporal")
    if not types and (REL_STRONG.search(text) or REL_WEAK.search(text)):
        types.append("stable_relational_other")
    return types or ["untyped"]


def balanced(text: str) -> bool:
    for a, b in [("(", ")"), ("[", "]"), ("{", "}")]:
        if text.count(a) != text.count(b):
            return False
    if text.count('"') % 2 != 0 or text.count("“") != text.count("”"):
        return False
    return True


def content_tokens(text: str) -> set[str]:
    toks = set()
    for m in ALNUM_RE.finditer(text.lower()):
        s = m.group(0)
        if len(s) >= 4 and s not in STOP:
            toks.add(s)
    return toks


def evaluate_text(text: str) -> tuple[bool, dict[str, Any]]:
    text = norm_ws(text)
    ws = words(text)
    caps = cap_phrases(text)
    nums = NUM_RE.findall(text)
    types = proposition_types(text, caps, nums)
    reasons: list[str] = []
    if not (10 <= len(ws) <= 42): reasons.append("length_outside_10_42")
    if text and text[-1] not in ".!?": reasons.append("no_terminal_sentence_punctuation")
    if alpha_frac(text) < 0.66: reasons.append("low_alpha_fraction")
    if not balanced(text): reasons.append("unbalanced_parentheses_or_quotes")
    digit_frac = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws) if ws else 0.0
    if digit_frac > 0.18: reasons.append("too_digit_dense")
    for name, pat in BAD_PATTERNS.items():
        if pat.search(text): reasons.append(name)
    if UNRESOLVED_START_RE.search(text) or UNRESOLVED_NOMINAL_RE.search(text):
        reasons.append("unresolved_sentence_initial_reference")
    if AMBIG_PASSIVE_RE.search(text):
        reasons.append("ambiguous_passive_relation")
    if "catalogue_list" in types and len(caps) >= 6:
        reasons.append("catalogue_or_entity_list")
    if "attributed_modal_temporal" in types:
        # Keep separately but not in stable source pool.
        reasons.append("attributed_modal_or_relative_time")
    has_relation = bool(REL_STRONG.search(text) or DEFINITION_RE.search(text) or CAUSAL_RE.search(text) or SPATIAL_RE.search(text) or (REL_WEAK.search(text) and len(caps) + len(nums) >= 3))
    if not has_relation:
        reasons.append("no_reusable_relation")
    if len(caps) + len(nums) < 1 and "definition_taxonomy" not in types and "causal_mechanistic" not in types:
        reasons.append("too_few_anchors_for_non_definition")
    return not reasons, {
        "selector_v3_repaired_text": text,
        "selector_v3_words": len(ws),
        "selector_v3_caps": caps[:12],
        "selector_v3_numbers": nums[:12],
        "selector_v3_anchor_count": len(caps) + len(nums),
        "selector_v3_types": types,
        "selector_v3_reject_reasons": reasons,
        "selector_v3_digit_frac": round(digit_frac, 4),
        "selector_v3_alpha_frac": round(alpha_frac(text), 4),
    }


def within_doc_redundancy(rows: list[dict[str, Any]], threshold: float = 0.58) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    per_doc_tokens: dict[str, list[tuple[int, set[str]]]] = defaultdict(list)
    for r in rows:
        d = str(r.get("doc_id"))
        toks = content_tokens(r["selector_v3_repaired_text"])
        match = None
        for kept_index, ktoks in per_doc_tokens[d]:
            if not toks or not ktoks:
                continue
            jac = len(toks & ktoks) / max(1, len(toks | ktoks))
            containment = len(toks & ktoks) / max(1, min(len(toks), len(ktoks)))
            if jac >= threshold or containment >= 0.82:
                match = {"kept_index": kept_index, "jaccard": round(jac, 4), "containment": round(containment, 4)}
                break
        if match is not None:
            rr = dict(r)
            rr["selector_v3_redundant_within_doc"] = match
            rejected.append(rr)
        else:
            rr = dict(r)
            rr["selector_v3_keep_index"] = len(kept)
            kept.append(rr)
            per_doc_tokens[d].append((len(kept)-1, toks))
    return kept, rejected


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


def word_sum(rows: list[dict[str, Any]], key: str = "selector_v3_words") -> int:
    return int(sum(int(r.get(key, r.get("words", 0))) for r in rows))


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary18 = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    scanned_doc_words = int(summary18.get("doc_words_scanned") or 0)
    rows = read_jsonl(IN_ROWS)
    accepted_pre: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    type_counts_pre: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    for r in rows:
        repaired, actions = repair_text(r.get("text", ""))
        ok, info = evaluate_text(repaired)
        rr = dict(r)
        rr["selector_v3_original_text"] = rr.get("text", "")
        rr["text"] = repaired
        rr["selector_v3_repair_actions"] = actions
        rr.update(info)
        rr["selector_v3_accepted_pre_redundancy"] = ok
        for a in actions:
            repair_counts[a] += 1
        if ok:
            accepted_pre.append(rr)
            for t in rr["selector_v3_types"]:
                type_counts_pre[t] += 1
        else:
            rejected.append(rr)
            for reason in rr["selector_v3_reject_reasons"]:
                reason_counts[reason] += 1
    stable_rows, redundant_reject = within_doc_redundancy(accepted_pre)
    cap12 = cap_by_doc(stable_rows, 12)
    cap8 = cap_by_doc(stable_rows, 8)
    cap4 = cap_by_doc(stable_rows, 4)
    cap2 = cap_by_doc(stable_rows, 2)
    type_counts_final = Counter(t for r in stable_rows for t in r.get("selector_v3_types", []))
    doc_counts = Counter(str(r.get("doc_id")) for r in stable_rows)
    rng = random.Random(18020)
    paths = {
        "accepted_pre_redundancy": OUT_DIR / "live_fineweb_selector_v3_accepted_pre_redundancy.jsonl",
        "stable": OUT_DIR / "live_fineweb_selector_v3_stable.jsonl",
        "stable_doccap12": OUT_DIR / "live_fineweb_selector_v3_stable_doccap12.jsonl",
        "stable_doccap8": OUT_DIR / "live_fineweb_selector_v3_stable_doccap8.jsonl",
        "stable_doccap4": OUT_DIR / "live_fineweb_selector_v3_stable_doccap4.jsonl",
        "stable_doccap2": OUT_DIR / "live_fineweb_selector_v3_stable_doccap2.jsonl",
        "rejected": OUT_DIR / "live_fineweb_selector_v3_rejected.jsonl",
        "redundant_reject": OUT_DIR / "live_fineweb_selector_v3_withindoc_redundant.jsonl",
        "stable_sample": OUT_DIR / "live_fineweb_selector_v3_stable_sample.json",
        "rejected_sample": OUT_DIR / "live_fineweb_selector_v3_rejected_sample.json",
        "redundant_sample": OUT_DIR / "live_fineweb_selector_v3_redundant_sample.json",
        "summary": OUT_DIR / "live_fineweb_source_selector_v3_summary.json",
    }
    write_jsonl(paths["accepted_pre_redundancy"], accepted_pre)
    write_jsonl(paths["stable"], stable_rows)
    write_jsonl(paths["stable_doccap12"], cap12)
    write_jsonl(paths["stable_doccap8"], cap8)
    write_jsonl(paths["stable_doccap4"], cap4)
    write_jsonl(paths["stable_doccap2"], cap2)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["redundant_reject"], redundant_reject)
    paths["stable_sample"].write_text(json.dumps(rng.sample(stable_rows, min(100, len(stable_rows))), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["rejected_sample"].write_text(json.dumps(rng.sample(rejected, min(100, len(rejected))), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["redundant_sample"].write_text(json.dumps(rng.sample(redundant_reject, min(80, len(redundant_reject))), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "status": "LIVE_FINEWEB_SOURCE_SELECTOR_V3",
        "scientific_purpose": "Repair live FineWeb source selection for a possible corrected source/source+view family by separating cleanup, self-containment, proposition type, epistemic status, and within-document redundancy.",
        "input_rows": len(rows),
        "input_words_original": int(sum(int(r.get("words", 0)) for r in rows)),
        "scanned_doc_words_from_step018": scanned_doc_words,
        "accepted_pre_redundancy_rows": len(accepted_pre),
        "accepted_pre_redundancy_words": word_sum(accepted_pre),
        "stable_rows": len(stable_rows),
        "stable_words": word_sum(stable_rows),
        "redundant_reject_rows": len(redundant_reject),
        "redundant_reject_words": word_sum(redundant_reject),
        "stable_doccap12_rows": len(cap12),
        "stable_doccap12_words": word_sum(cap12),
        "stable_doccap8_rows": len(cap8),
        "stable_doccap8_words": word_sum(cap8),
        "stable_doccap4_rows": len(cap4),
        "stable_doccap4_words": word_sum(cap4),
        "stable_doccap2_rows": len(cap2),
        "stable_doccap2_words": word_sum(cap2),
        "yields_per_scanned_doc_word": {
            "stable": word_sum(stable_rows) / scanned_doc_words if scanned_doc_words else None,
            "stable_doccap12": word_sum(cap12) / scanned_doc_words if scanned_doc_words else None,
            "stable_doccap8": word_sum(cap8) / scanned_doc_words if scanned_doc_words else None,
            "stable_doccap4": word_sum(cap4) / scanned_doc_words if scanned_doc_words else None,
            "stable_doccap2": word_sum(cap2) / scanned_doc_words if scanned_doc_words else None,
        },
        "projected_scanned_doc_words_needed": {},
        "type_counts_pre_redundancy": dict(type_counts_pre.most_common()),
        "type_counts_final": dict(type_counts_final.most_common()),
        "reject_reason_counts": dict(reason_counts.most_common(30)),
        "repair_action_counts": dict(repair_counts.most_common()),
        "stable_word_stats": stat([int(r.get("selector_v3_words", 0)) for r in stable_rows]),
        "stable_anchor_count_stats": stat([int(r.get("selector_v3_anchor_count", 0)) for r in stable_rows]),
        "stable_unique_docs": len(doc_counts),
        "stable_top_doc_counts": doc_counts.most_common(20),
        "paths": {k: str(v) for k, v in paths.items()},
        "interpretation": "CPU-only source-selection evidence; not model-training or downstream-score evidence.",
    }
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        summary["projected_scanned_doc_words_needed"][str(target)] = {
            k: (round(target / v) if v else None) for k, v in summary["yields_per_scanned_doc_word"].items()
        }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb source selector v3\n\n"]
    lines.append("This CPU-only repair responds to source-sample reading: it cleans extraction artifacts, separates attributed or time-relative text from stable factual rows, rejects unresolved antecedents, labels proposition types, and adds within-document redundancy control. It does not train or score a model.\n\n")
    lines.append(f"Input strict-anchor-like pool: {len(rows):,} rows / {summary['input_words_original']:,} original words from {scanned_doc_words:,} scanned document words.\n\n")
    lines.append(f"Accepted before redundancy control: {len(accepted_pre):,} rows / {word_sum(accepted_pre):,} repaired words. Stable after within-doc redundancy: {len(stable_rows):,} rows / {word_sum(stable_rows):,} words.\n\n")
    lines.append(f"Doc caps: cap12 {len(cap12):,} rows / {word_sum(cap12):,} words; cap8 {len(cap8):,} rows / {word_sum(cap8):,} words; cap4 {len(cap4):,} rows / {word_sum(cap4):,} words; cap2 {len(cap2):,} rows / {word_sum(cap2):,} words.\n\n")
    y = summary["yields_per_scanned_doc_word"]
    lines.append("Stable yield per scanned document word: " + ", ".join(f"{k}={v:.3%}" for k, v in y.items() if v is not None) + ".\n\n")
    lines.append("Projected scanned document words needed for future source budgets:\n\n")
    lines.append("| target source words | stable cap8 | stable cap4 | stable uncapped |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        p = summary["projected_scanned_doc_words_needed"][str(target)]
        lines.append(f"| {target:,} | {p['stable_doccap8']:,} | {p['stable_doccap4']:,} | {p['stable']:,} |\n")
    lines.append("\nFinal proposition types: " + json.dumps(summary["type_counts_final"], ensure_ascii=False) + "\n\n")
    lines.append("Main rejection reasons: " + json.dumps(dict(list(reason_counts.most_common(16))), ensure_ascii=False) + "\n\n")
    lines.append("Scientific reading: this selector is more aligned with self-contained, stable, masked-LM-useful factual relations than the earlier anchor-heavy pool, but its yield is lower. It should be sample-read before any bulk rewriting, and the repaired cached-source training result should still decide whether FineWeb source breadth deserves the next H100 allocation.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nStable sample: `{paths['stable_sample']}`\n\nRejected sample: `{paths['rejected_sample']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary": str(paths["summary"]),
        "note": str(NOTE),
        "stable_words": word_sum(stable_rows),
        "stable_doccap8_words": word_sum(cap8),
        "stable_doccap4_words": word_sum(cap4),
        "stable_yield_pct": 100 * summary["yields_per_scanned_doc_word"]["stable"] if scanned_doc_words else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
