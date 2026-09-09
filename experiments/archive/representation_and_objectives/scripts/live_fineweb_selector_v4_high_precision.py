#!/usr/bin/env python3
"""research: high-precision single-sentence live FineWeb source selector.

Refines research selector-v3 stable rows after sample reading showed remaining
antecedent/discourse and attributed-news failures.  The v4 pool is intentionally
stricter: it prioritizes self-contained, stable, single-sentence factual or
relational propositions for possible masked-LM source/source+view materialization.
It is CPU-only evidence, not training or score evidence.
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
IN_ROWS = ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable.jsonl"
V3_SUMMARY = ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_source_selector_v3_summary.json"
OUT_DIR = ROOT / "data/live_fineweb_selector_v4_high_precision"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/live_fineweb_selector_v4_high_precision.md')

WORD_RE = re.compile(r"\S+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\b")
NUM_RE = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b")
GENERIC_CAPS = {
    "A", "An", "And", "As", "At", "By", "For", "From", "He", "Her", "His", "I", "If", "In", "It", "Its", "On",
    "Or", "She", "That", "The", "Their", "There", "These", "They", "This", "To", "We", "When", "Where", "While",
    "With", "You", "Your", "However", "Moreover", "Many", "Most", "Almost", "About", "Are", "Even", "Choosing",
    "Explore", "Long", "Why", "Read", "Live", "Let", "License", "Good", "Narrow", "Three", "May", "Over",
    "Comparing", "One", "Some", "After", "Before", "During", "Although", "Because", "Since", "New", "Old", "Rather",
    "According", "Accordingly", "Principle", "Community", "Ancient", "Modern", "Several", "Another", "Other", "First",
    "Second", "National", "International", "American", "British", "European", "North", "South", "East", "West", "States",
    "Only", "Both", "Around", "Unlike", "Like", "In", "Within", "Among", "Early", "Late", "Former", "Later",
}
# Third-person pronouns are dangerous for single-sentence source+view because the view
# generator may resolve or hallucinate the antecedent.  Relative pronouns (which/that)
# are allowed; they usually bind within the sentence.
THIRD_PERSON_PRONOUN_RE = re.compile(r"\b(he|she|him|her|hers|his|they|them|their|theirs|it|its|itself|themselves)\b", re.I)
DISCOURSE_START_RE = re.compile(r"^(accordingly|however|but|yet|meanwhile|therefore|thus|then|later|previously|instead|in addition|as a result|on the other hand|for example|for instance|around both|unlike|like many|only about)\b", re.I)
DEMONSTRATIVE_NOMINAL_RE = re.compile(r"\b(this|that|these|those|such)\s+(mission|craft|ship|boat|report|survey|decision|position|country|region|area|group|team|company|organization|system|process|method|project|program|movement|issue|case|court|law|model|study|research|data|result|effect|pipeline|proposal|plan|brain region|river|town|city)\b", re.I)
BARE_PLACEHOLDER_RE = re.compile(r"\b(the|this|that|these|those|such)\s+(mission|craft|ship|report|survey|decision|position|group|team|system|process|method|project|program|issue|case|model|result|effect|proposal|plan)\b", re.I)
ATTRIBUTED_OR_UNSTABLE_RE = re.compile(r"\b(according to|estimate|estimated|reported|argued|claimed|believed|suggested|forecast|worried|concerned|alleged|apparently|possibly|probably|likely|may|might|could|would|will soon|recently|currently|today|now|yesterday|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
BAD_RESIDUE_RE = re.compile(r"\b(preferred citation|doi|isbn|appendix|chapter\s+\d+|table\s+\d+|figure\s+\d+|principle\s+\d+|use acetic acid|how to use|click here|follow this link|read more|website|privacy policy|terms of use)\b|[`*_]|\s+[.,;:]|[.,;:]\s*[.,;:]", re.I)
MEDIA_FANDOM_RE = re.compile(r"\b(Star Trek|Doctor Who|Pok[eé]mon|anime|manga|episode|season|television series|video game|fictional|character|comic book|fan fiction|film|movie|novel)\b", re.I)
LISTY_RE = re.compile(r"(?:\b[A-Z][A-Za-z]+\b[^.!?]{0,20},\s*){3,}|\b(?:including|such as)\b[^.!?]{0,160}(?:,\s*[^,]+){3,}|;\s*[^.!?]{2,40};")
AMBIG_PASSIVE_RE = re.compile(r"\b(?:is|are|was|were|has been|have been|had been|been|be)\s+(affected|involved|associated|concerned|known|found|seen|observed|reported|estimated|lost|supported)\b(?![^.!?]{0,80}\b(by|from|with|in|on|at|during|because|due to|through)\b)", re.I)
REL_STRONG = re.compile(
    r"\b(became|becomes|located|founded|discovered|contains|includes|measured|caused|produced|formed|developed|created|introduced|served|won|died|born|published|built|invented|designed|used|made|named|defined|classified|requires|required|allows|allow|means|refers|consists|belongs|separated|incorporated|orbits|takes|comprises|covers|connects|crosses|flows|borders|led|resulted|increased|decreased|converted|established|replaced|joined|opened|closed|owned|controlled|emits|absorbs|contains|drains|feeds|supplies)\b",
    re.I,
)
DEFINITION_RE = re.compile(r"\b(is|are|was|were)\s+(?:a|an|the)?\s*(?:type|kind|form|class|group|family|method|process|measure|term|name|part|member|example|branch|field|disease|condition|material|chemical|element|language)\b|\b(means|refers to|is defined as|are defined as|consists of|is composed of|are composed of)\b", re.I)
CAUSAL_RE = re.compile(r"\b(because|caused?|due to|led to|resulted in|therefore|thus|so that|prevents?|reduces?|increases?|damages?|protects?|enables?|allows?|responsible for|causing)\b", re.I)
SPATIAL_RE = re.compile(r"\b(located|lies|situated|river|mountain|island|county|province|region|north|south|east|west|border|coast|flows|crosses|kilometers|miles|capital|drains|valley|lake|ocean)\b", re.I)
BIO_HIST_RE = re.compile(r"\b(born|died|founded|established|published|won|served|king|president|minister|war|century|empire|treaty|government|court|election|reign|dynasty|incorporated)\b", re.I)
QUANT_RE = re.compile(r"\b(percent|million|billion|rate|ratio|average|population|area|height|weight|km|kg|years?|\d+(?:[,.]\d+)*)\b", re.I)
WEAK_BE_HAVE_RE = re.compile(r"\b(is|are|was|were|has|have|had)\b", re.I)


def norm_ws(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(norm_ws(text))


def word_count(text: str) -> int:
    return len(words(text))


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars) if chars else 0.0


def cap_phrases(text: str) -> list[str]:
    out: list[str] = []
    for m in CAP_PHRASE_RE.finditer(text):
        s = m.group(0).strip()
        parts = s.split()
        if parts and parts[0] in GENERIC_CAPS and len(parts) > 1:
            s = " ".join(parts[1:])
        if s in GENERIC_CAPS or not s:
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


def prop_types(text: str, caps: list[str], nums: list[str]) -> list[str]:
    types: list[str] = []
    if DEFINITION_RE.search(text): types.append("definition_taxonomy")
    if CAUSAL_RE.search(text): types.append("causal_mechanistic")
    if SPATIAL_RE.search(text): types.append("spatial_geographic")
    if BIO_HIST_RE.search(text): types.append("biographical_historical")
    if QUANT_RE.search(text) and nums: types.append("quantitative")
    if LISTY_RE.search(text): types.append("catalogue_list")
    if not types and (REL_STRONG.search(text) or WEAK_BE_HAVE_RE.search(text)):
        types.append("stable_relational_other")
    return types or ["untyped"]


def balanced(text: str) -> bool:
    for a, b in [("(", ")"), ("[", "]"), ("{", "}")]:
        if text.count(a) != text.count(b):
            return False
    if text.count('"') % 2 != 0 or text.count("“") != text.count("”"):
        return False
    return True


def evaluate(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    text = norm_ws(row.get("selector_v3_repaired_text") or row.get("text", ""))
    ws = words(text)
    caps = cap_phrases(text)
    nums = NUM_RE.findall(text)
    types = prop_types(text, caps, nums)
    reasons: list[str] = []
    if not (10 <= len(ws) <= 42): reasons.append("length_outside_10_42")
    if text and text[-1] not in ".!?": reasons.append("no_terminal_sentence_punctuation")
    if alpha_frac(text) < 0.66: reasons.append("low_alpha_fraction")
    digit_frac = sum(any(ch.isdigit() for ch in w) for w in ws) / len(ws) if ws else 0.0
    if digit_frac > 0.18: reasons.append("too_digit_dense")
    if not balanced(text): reasons.append("unbalanced_parentheses_or_quotes")
    if THIRD_PERSON_PRONOUN_RE.search(text): reasons.append("third_person_pronoun_or_possessive")
    if DISCOURSE_START_RE.search(text): reasons.append("discourse_dependent_start")
    if DEMONSTRATIVE_NOMINAL_RE.search(text) or BARE_PLACEHOLDER_RE.search(text): reasons.append("unresolved_nominal_reference")
    if ATTRIBUTED_OR_UNSTABLE_RE.search(text): reasons.append("attributed_modal_relative_time")
    if BAD_RESIDUE_RE.search(text): reasons.append("extraction_or_web_residue")
    if MEDIA_FANDOM_RE.search(text): reasons.append("media_fandom_or_fiction")
    if AMBIG_PASSIVE_RE.search(text): reasons.append("ambiguous_passive_relation")
    if "catalogue_list" in types: reasons.append("catalogue_or_list_structure")
    has_reusable_relation = bool(REL_STRONG.search(text) or DEFINITION_RE.search(text) or CAUSAL_RE.search(text) or SPATIAL_RE.search(text) or BIO_HIST_RE.search(text) or (WEAK_BE_HAVE_RE.search(text) and len(caps) + len(nums) >= 3))
    if not has_reusable_relation: reasons.append("no_reusable_relation")
    # Stable generic definitions/causal mechanisms may have no proper names.  Otherwise require a specific anchor.
    if len(caps) + len(nums) < 1 and not ("definition_taxonomy" in types or "causal_mechanistic" in types):
        reasons.append("too_few_specific_anchors")
    # Bare shallow place lists with no strong relation should not dominate the source pool.
    if not REL_STRONG.search(text) and not DEFINITION_RE.search(text) and not CAUSAL_RE.search(text) and len(caps) >= 4 and len(nums) == 0:
        reasons.append("many_anchors_without_strong_relation")
    return not reasons, {
        "selector_v4_text": text,
        "selector_v4_words": len(ws),
        "selector_v4_caps": caps[:12],
        "selector_v4_numbers": nums[:12],
        "selector_v4_anchor_count": len(caps) + len(nums),
        "selector_v4_types": types,
        "selector_v4_reject_reasons": reasons,
        "selector_v4_digit_frac": round(digit_frac, 4),
        "selector_v4_alpha_frac": round(alpha_frac(text), 4),
    }


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cap_by_doc(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter(); out = []
    for r in rows:
        d = str(r.get("doc_id"))
        if counts[d] >= cap: continue
        counts[d] += 1; out.append(r)
    return out


def stat(vals: list[int | float]) -> dict[str, Any]:
    if not vals: return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float | int: return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "mean": round(float(statistics.mean(xs)), 4), "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": xs[-1], "sum": round(float(sum(xs)), 4)}


def word_sum(rows: list[dict[str, Any]]) -> int:
    return int(sum(int(r.get("selector_v4_words", r.get("selector_v3_words", r.get("words", 0)))) for r in rows))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v3_summary = json.loads(V3_SUMMARY.read_text(encoding="utf-8"))
    scanned_doc_words = int(v3_summary.get("scanned_doc_words_from_step018") or 0)
    rows = read_jsonl(IN_ROWS)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for r in rows:
        ok, info = evaluate(r)
        rr = dict(r)
        rr.update(info)
        rr["selector_v4_accepted"] = ok
        if ok:
            kept.append(rr)
            for t in info["selector_v4_types"]:
                type_counts[t] += 1
        else:
            rejected.append(rr)
            for reason in info["selector_v4_reject_reasons"]:
                reason_counts[reason] += 1
    # Sort retained rows to prefer relation-rich self-contained rows under fixed deterministic selection.
    type_priority = {
        "definition_taxonomy": 0,
        "causal_mechanistic": 1,
        "spatial_geographic": 2,
        "biographical_historical": 3,
        "quantitative": 4,
        "stable_relational_other": 5,
    }
    def priority(r: dict[str, Any]) -> tuple[Any, ...]:
        types = r.get("selector_v4_types", [])
        best = min(type_priority.get(t, 9) for t in types)
        # keep original stream order inside comparable bins through near_dedup_index/keep_index
        return (best, -int(r.get("selector_v4_anchor_count", 0)), int(r.get("near_dedup_index", r.get("selector_v3_keep_index", 0))))
    kept_sorted = sorted(kept, key=priority)
    cap12 = cap_by_doc(kept_sorted, 12)
    cap8 = cap_by_doc(kept_sorted, 8)
    cap4 = cap_by_doc(kept_sorted, 4)
    cap2 = cap_by_doc(kept_sorted, 2)
    rng = random.Random(18021)
    paths = {
        "kept": OUT_DIR / "live_fineweb_selector_v4_high_precision_kept.jsonl",
        "kept_sorted": OUT_DIR / "live_fineweb_selector_v4_high_precision_kept_sorted.jsonl",
        "doccap12": OUT_DIR / "live_fineweb_selector_v4_doccap12.jsonl",
        "doccap8": OUT_DIR / "live_fineweb_selector_v4_doccap8.jsonl",
        "doccap4": OUT_DIR / "live_fineweb_selector_v4_doccap4.jsonl",
        "doccap2": OUT_DIR / "live_fineweb_selector_v4_doccap2.jsonl",
        "rejected": OUT_DIR / "live_fineweb_selector_v4_rejected.jsonl",
        "kept_sample": OUT_DIR / "live_fineweb_selector_v4_kept_sample.json",
        "rejected_sample": OUT_DIR / "live_fineweb_selector_v4_rejected_sample.json",
        "summary": OUT_DIR / "live_fineweb_selector_v4_high_precision_summary.json",
    }
    write_jsonl(paths["kept"], kept)
    write_jsonl(paths["kept_sorted"], kept_sorted)
    write_jsonl(paths["doccap12"], cap12)
    write_jsonl(paths["doccap8"], cap8)
    write_jsonl(paths["doccap4"], cap4)
    write_jsonl(paths["doccap2"], cap2)
    write_jsonl(paths["rejected"], rejected)
    paths["kept_sample"].write_text(json.dumps(rng.sample(kept_sorted, min(120, len(kept_sorted))), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["rejected_sample"].write_text(json.dumps(rng.sample(rejected, min(120, len(rejected))), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    doc_counts = Counter(str(r.get("doc_id")) for r in kept_sorted)
    summary = {
        "status": "LIVE_FINEWEB_SELECTOR_V4_HIGH_PRECISION",
        "scientific_purpose": "High-precision self-contained stable factual source pool for possible corrected FineWeb source/source+view materialization; CPU-only source evidence.",
        "input_v3_stable_rows": len(rows),
        "input_v3_stable_words": int(sum(int(r.get("selector_v3_words", r.get("words", 0))) for r in rows)),
        "scanned_doc_words_from_step018": scanned_doc_words,
        "kept_rows": len(kept_sorted),
        "kept_words": word_sum(kept_sorted),
        "retention_from_v3_by_words": word_sum(kept_sorted) / max(1, int(sum(int(r.get("selector_v3_words", r.get("words", 0))) for r in rows))),
        "doccap12_rows": len(cap12), "doccap12_words": word_sum(cap12),
        "doccap8_rows": len(cap8), "doccap8_words": word_sum(cap8),
        "doccap4_rows": len(cap4), "doccap4_words": word_sum(cap4),
        "doccap2_rows": len(cap2), "doccap2_words": word_sum(cap2),
        "yields_per_scanned_doc_word": {
            "kept": word_sum(kept_sorted) / scanned_doc_words if scanned_doc_words else None,
            "doccap12": word_sum(cap12) / scanned_doc_words if scanned_doc_words else None,
            "doccap8": word_sum(cap8) / scanned_doc_words if scanned_doc_words else None,
            "doccap4": word_sum(cap4) / scanned_doc_words if scanned_doc_words else None,
            "doccap2": word_sum(cap2) / scanned_doc_words if scanned_doc_words else None,
        },
        "projected_scanned_doc_words_needed": {},
        "type_counts": dict(type_counts.most_common()),
        "reject_reason_counts": dict(reason_counts.most_common(30)),
        "word_stats": stat([int(r.get("selector_v4_words", 0)) for r in kept_sorted]),
        "anchor_count_stats": stat([int(r.get("selector_v4_anchor_count", 0)) for r in kept_sorted]),
        "unique_docs_kept": len(doc_counts),
        "top_doc_counts_kept": doc_counts.most_common(20),
        "paths": {k: str(v) for k, v in paths.items()},
        "interpretation": "Source-quality preparation only; downstream value still depends on real training/evaluation.",
    }
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        summary["projected_scanned_doc_words_needed"][str(target)] = {
            k: (round(target / v) if v else None) for k, v in summary["yields_per_scanned_doc_word"].items()
        }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research live FineWeb selector v4 high precision\n\n"]
    lines.append("This CPU-only pass refines selector-v3 after sample reading showed remaining rows with unresolved `them/it/this/their`, discourse-open starts, attribution/news timing, catalogues, and fandom/procedural residue. It does not train or score a model.\n\n")
    lines.append(f"Input v3 stable pool: {len(rows):,} rows / {summary['input_v3_stable_words']:,} words.\n\n")
    lines.append(f"V4 kept: {len(kept_sorted):,} rows / {word_sum(kept_sorted):,} words ({100*summary['retention_from_v3_by_words']:.2f}% of v3 stable words).\n\n")
    lines.append(f"Doc caps: cap12 {len(cap12):,} rows / {word_sum(cap12):,} words; cap8 {len(cap8):,} rows / {word_sum(cap8):,} words; cap4 {len(cap4):,} rows / {word_sum(cap4):,} words; cap2 {len(cap2):,} rows / {word_sum(cap2):,} words.\n\n")
    lines.append("Yield per scanned document word: " + ", ".join(f"{k}={v:.3%}" for k, v in summary["yields_per_scanned_doc_word"].items() if v is not None) + ".\n\n")
    lines.append("Projected scanned document words needed:\n\n")
    lines.append("| target source words | v4 cap8 | v4 cap4 | v4 uncapped |\n|---:|---:|---:|---:|\n")
    for target in [1_000_000, 1_750_000, 2_500_000, 3_500_000]:
        p = summary["projected_scanned_doc_words_needed"][str(target)]
        lines.append(f"| {target:,} | {p['doccap8']:,} | {p['doccap4']:,} | {p['kept']:,} |\n")
    lines.append("\nType counts: " + json.dumps(summary["type_counts"], ensure_ascii=False) + "\n\n")
    lines.append("Main v4 rejection reasons: " + json.dumps(dict(list(reason_counts.most_common(16))), ensure_ascii=False) + "\n\n")
    lines.append("Scientific use: use this as the high-precision single-sentence candidate source pool if the repaired cached-source training supports continuing FineWeb. Compare it with v3/v2 samples before bulk generation; v4 is safer for faithful rewriting but lower-yield and still needs tokenizer exposure and benchmark-overlap checks in any materialized arms.\n\n")
    lines.append(f"Summary JSON: `{paths['summary']}`\n\nKept sample: `{paths['kept_sample']}`\n\nRejected sample: `{paths['rejected_sample']}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary": str(paths["summary"]),
        "note": str(NOTE),
        "kept_words": word_sum(kept_sorted),
        "doccap8_words": word_sum(cap8),
        "doccap4_words": word_sum(cap4),
        "retention_from_v3_pct": 100 * summary["retention_from_v3_by_words"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
