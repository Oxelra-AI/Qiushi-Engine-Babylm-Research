#!/usr/bin/env python3
"""Repair the next FineWeb source-by-rewrite asset before any GPU generation.

The existing factual source file is a useful starting point, but spot checks
showed that it still admits citation strings, journal metadata, literary quoted
fragments, and weak one-clause publication notices.  This CPU-only script builds
stricter source tiers and pilot prompts in the frontier_consolidation workspace, so a later
H100 allocation can first test faithful Qwen simplification on cleaner inputs
rather than spending generation on noisy rows.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import unicodedata
from typing import Any, Iterable

IN_PATH = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/fineweb_factual_complete_sentence_sources_selected.jsonl")
BALANCED_PATH = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/balanced_source_by_rewrite_sources.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair")
NOTE_PATH = pathlib.Path("research/notes/frontier_consolidation/fineweb_source_by_rewrite_repair.md")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z'’.-]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

RELATION_CUES = {
    " is ", " are ", " was ", " were ", " became ", " becomes ", " born ", " died ", " founded ",
    " located ", " called ", " known ", " used ", " uses ", " includes ", " contains ", " because ",
    " therefore ", " caused ", " led ", " replaced ", " developed ", " published ", " built ",
    " served ", " won ", " established ", " created ", " part of ", " member of ", " consists ",
    " belongs ", " formed ", " opened ", " invented ", " discovered ", " produced ", " released ",
    " measured ", " observed ", " named ", " describes ", " provides ", " requires ", " allows ",
}

GOOD_DOMAIN_KEYS = {
    "science_physical", "geography_places", "people_history", "institutions_society", "quant_numeric", "causal_relational"
}

SYSTEM = (
    "You simplify factual English sentences for a small masked language model. Preserve exactly the same meaning, "
    "including every named entity, number, date, quantity, and relation. Do not add facts. Output only one sentence."
)

BAD_SUBSTRINGS = [
    "doi:", " using doi ", "cite or link", "volume ", " issue ", " page ", "pages ", "issn", "isbn",
    "copyright", "all rights reserved", "privacy policy", "terms of use", "subscribe", "download pdf",
    "advertisement", "related articles", "related persons", "click here", "read more", "follow this link",
    "table of contents", "appendix", "chapter ", "section ", "figure ", "fig.", "et al.",
    "multiple choice", "names of", "example:", "example :", "lorem ipsum", "cookie policy",
]
BAD_START_WORDS = {
    "i", "you", "we", "he", "she", "they", "it", "me", "my", "your", "our", "his", "her", "their",
    "of", "and", "but", "or", "so", "then", "because", "though", "although", "while", "which", "who",
    "whom", "whose", "this", "that", "these", "those", "there", "sadly", "unfortunately", "fortunately",
}
DIALOGUE_OR_NARRATIVE = [
    " said ", " replied ", " asked ", " cried ", " shouted ", " exclaimed ", " whispered ", " told ",
    " laid them ", " golden-breasted ", " foam from the mouths ", " beer was cold ", " effervescence",
]
PUBLICATION_META_PATTERNS = [
    re.compile(r"\bvolume\s+\d+\b", re.I),
    re.compile(r"\bissue\s+\d+\b", re.I),
    re.compile(r"\bpage\s+\d+\b", re.I),
    re.compile(r"\bdoi\s*[: ]\s*10\.\d+", re.I),
    re.compile(r"\b10\.\d{4,9}/\S+", re.I),
    re.compile(r"\b[A-Z]\dN\d\b"),
]


def norm_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ").replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def wc(text: str) -> int:
    return len(norm_text(text).split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q_stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = sorted(list(vals))
    if not xs:
        return {"n": 0}
    def q(p: float):
        return xs[min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))]
    return {
        "n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.mean(xs),
        "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": xs[-1], "sum": sum(xs),
    }


def number_tokens(text: str) -> list[str]:
    return sorted(set(re.sub(r"\s+", "", m.group(0)) for m in NUM_RE.finditer(text)))


def conservative_entities(text: str) -> list[str]:
    toks = text.split()
    spans = []
    pos = 0
    for tok in toks:
        start = text.find(tok, pos)
        if start < 0:
            start = pos
        spans.append((start, start + len(tok), tok))
        pos = start + len(tok)

    out: list[str] = []
    for m in ACRONYM_RE.finditer(text):
        out.append(m.group(0))

    cap_positions: list[tuple[int, str]] = []
    for m in CAP_WORD_RE.finditer(text):
        tok_i = 0
        for j, (a, b, _) in enumerate(spans):
            if a <= m.start() < b:
                tok_i = j
                break
        val = m.group(0).strip(".,;:()[]{}\"“”‘’")
        cap_positions.append((tok_i, val))

    i = 0
    while i < len(cap_positions):
        tok_i, val = cap_positions[i]
        seq = [val]
        j = i + 1
        prev_tok_i = tok_i
        while j < len(cap_positions) and cap_positions[j][0] <= prev_tok_i + 1:
            seq.append(cap_positions[j][1])
            prev_tok_i = cap_positions[j][0]
            j += 1
        if len(seq) >= 2:
            out.append(" ".join(seq))
        elif tok_i != 0 and len(seq[0]) >= 4:
            out.append(seq[0])
        i = j

    bad_single = {"The", "This", "That", "These", "Those", "There", "When", "Where", "What", "How", "Why", "Because", "For", "And", "But", "After", "Before", "During", "Then", "Each", "Every", "Through", "Early", "Research", "Volume", "Issue", "Page", "Nature", "Study"}
    cleaned = []
    for e in out:
        e = " ".join(e.split()).strip(" .,;:()[]{}\"“”‘’")
        if not e:
            continue
        if len(e.split()) == 1 and e in bad_single:
            continue
        cleaned.append(e)
    return sorted(set(cleaned))


def relation_count(text: str) -> int:
    low = " " + text.lower() + " "
    return sum(1 for cue in RELATION_CUES if cue in low)


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars)


def nonlatin_letter_frac(text: str) -> float:
    letters = 0
    nonlatin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                nonlatin += 1
    return nonlatin / letters if letters else 0.0


def hard_reject_reasons(row: dict[str, Any]) -> list[str]:
    text = norm_text(row.get("text") or row.get("source_text") or "")
    low = " " + text.lower() + " "
    toks = text.split()
    n = len(toks)
    ents = row.get("entities") or conservative_entities(text)
    nums = row.get("numbers") or number_tokens(text)
    doms = set(row.get("domain_hits") or [])
    rel = int(row.get("relation_count") or relation_count(text))
    reasons: list[str] = []

    if n < 12:
        reasons.append("too_short_for_useful_rewrite_pair")
    if n > 40:
        reasons.append("too_long_or_overpacked_for_high_precision")
    if not text.endswith((".", "!", "?")):
        reasons.append("bad_terminal_punctuation")
    if toks:
        first = re.sub(r"[^A-Za-z]+", "", toks[0]).lower()
        if first in BAD_START_WORDS:
            reasons.append("context_dependent_or_narrative_start")
    if any(s in low for s in BAD_SUBSTRINGS):
        reasons.append("web_or_publication_artifact_substring")
    if any(p.search(text) for p in PUBLICATION_META_PATTERNS):
        reasons.append("publication_metadata_pattern")
    if any(s in low for s in DIALOGUE_OR_NARRATIVE):
        reasons.append("dialogue_or_literary_fragment")
    if any(q in text for q in ['"', "“", "”", "‘", "’"]):
        reasons.append("quoted_material_high_risk")
    if text.count(";") >= 1 and n < 45:
        reasons.append("semicolon_or_clause_chain")
    if text.count(":") >= 1:
        reasons.append("colon_or_metadata_chain")
    if text.count(",") / max(1, n) > 0.18:
        reasons.append("comma_dense_enumeration")
    if len(nums) > 3:
        reasons.append("number_dense_or_citation_like")
    if len(ents) > 6:
        reasons.append("entity_dense_or_list_like")
    if rel < 1:
        reasons.append("missing_relation_cue")
    if not (doms & GOOD_DOMAIN_KEYS or ents or nums):
        reasons.append("low_factual_anchor")
    # Avoid rows whose only domain is generic media/publication: they often become low-learning-value citations.
    if doms and doms <= {"media_culture"} and len(ents) <= 1 and len(nums) == 0:
        reasons.append("media_only_weak_fact")
    digit_words = sum(any(c.isdigit() for c in t) for t in toks)
    if digit_words / max(1, n) > 0.16:
        reasons.append("digit_token_dense")
    if alpha_frac(text) < 0.66 or nonlatin_letter_frac(text) > 0.01:
        reasons.append("character_quality")
    # Very long slash/hyphen compounds are often scraped IDs or references.
    if len(re.findall(r"[A-Za-z0-9]+[/][A-Za-z0-9/.-]+", text)) >= 1:
        reasons.append("slash_identifier_or_reference")
    return reasons


def medium_reject_reasons(row: dict[str, Any]) -> list[str]:
    # Softer tier for breadth: keep some 41-48 word rows, but still remove the obvious artifacts seen in inspection.
    text = norm_text(row.get("text") or row.get("source_text") or "")
    low = " " + text.lower() + " "
    toks = text.split(); n = len(toks)
    nums = row.get("numbers") or number_tokens(text)
    ents = row.get("entities") or conservative_entities(text)
    rel = int(row.get("relation_count") or relation_count(text))
    reasons: list[str] = []
    if n < 10:
        reasons.append("too_short")
    if n > 48:
        reasons.append("too_long")
    if not text.endswith((".", "!", "?")):
        reasons.append("bad_terminal_punctuation")
    if any(s in low for s in BAD_SUBSTRINGS) or any(p.search(text) for p in PUBLICATION_META_PATTERNS):
        reasons.append("web_or_publication_artifact")
    if any(s in low for s in DIALOGUE_OR_NARRATIVE):
        reasons.append("dialogue_or_literary_fragment")
    if text.count(":") >= 2 or text.count(";") >= 2:
        reasons.append("chain_or_list_punctuation")
    if len(nums) > 5:
        reasons.append("number_dense")
    if len(ents) > 9:
        reasons.append("entity_dense")
    if rel < 1:
        reasons.append("missing_relation_cue")
    if alpha_frac(text) < 0.62 or nonlatin_letter_frac(text) > 0.02:
        reasons.append("character_quality")
    return reasons


def load_jsonl(path: pathlib.Path, source_label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = norm_text(r.get("text") or r.get("source_text") or "")
            if not text:
                continue
            rr = dict(r)
            rr["text"] = text
            rr["words"] = int(r.get("words") or wc(text))
            rr["source_asset"] = source_label
            rr["entities"] = list(r.get("entities") or conservative_entities(text))
            rr["numbers"] = list(r.get("numbers") or number_tokens(text))
            rr["relation_count"] = int(r.get("relation_count") or relation_count(text))
            rr["domain_hits"] = list(r.get("domain_hits") or [])
            rows.append(rr)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    doc_counter = collections.Counter(str(r.get("doc_id", "")) for r in rows)
    domain_counter = collections.Counter(h for r in rows for h in (r.get("domain_hits") or []))
    return {
        "rows": len(rows),
        "words": sum(int(r["words"]) for r in rows),
        "unique_docs": len(doc_counter),
        "word_stats": q_stats([int(r["words"]) for r in rows]),
        "entity_count_stats": q_stats([len(r.get("entities") or []) for r in rows]),
        "number_count_stats": q_stats([len(r.get("numbers") or []) for r in rows]),
        "relation_count_stats": q_stats([int(r.get("relation_count") or 0) for r in rows]),
        "domain_hit_counts": dict(domain_counter.most_common()),
        "top_docs": doc_counter.most_common(10),
    }


def stratified_pilot(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_doc[str(r.get("doc_id", ""))].append(r)
    docs = list(by_doc)
    rng.shuffle(docs)
    selected = []
    for d in docs:
        # Prefer moderately short, anchored rows. Extreme long rows and high punctuation are less useful for the generation slice.
        selected.append(max(by_doc[d], key=lambda r: (len(r.get("domain_hits") or []), int(r.get("relation_count") or 0), -abs(int(r["words"]) - 22), -int(r["words"]))))
        if len(selected) >= n:
            break
    if len(selected) < n:
        seen = {r.get("sentence_id") for r in selected}
        for r in sorted(rows, key=lambda r: (-len(r.get("domain_hits") or []), -int(r.get("relation_count") or 0), int(r["words"]))):
            if r.get("sentence_id") in seen:
                continue
            selected.append(r)
            if len(selected) >= n:
                break
    rng.shuffle(selected)
    return selected[:n]


def make_prompt(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    return {
        "prompt_id": f"frontier_consolidation_fwfactsimp_{int(row.get('sentence_id', 0)):06d}",
        "typ": "simplification",
        "source": "frontier_consolidation_repaired_fineweb_factual_sentence_qwen_simplification",
        "sentence_id": row.get("sentence_id"),
        "doc_id": str(row.get("doc_id", "")),
        "source_row": row.get("source_row"),
        "sent_index_in_row": row.get("sent_index_in_row"),
        "source_text": text,
        "source_words": int(row["words"]),
        "source_entities": row.get("entities") or conservative_entities(text),
        "source_numbers": row.get("numbers") or number_tokens(text),
        "domain_hits": row.get("domain_hits") or [],
        "relation_count": int(row.get("relation_count") or relation_count(text)),
        "source_asset": row.get("source_asset"),
        "system": SYSTEM,
        "prompt": (
            "Rewrite the factual sentence below in simpler plain English while preserving the exact same facts. "
            "Keep every named entity, number, date, quantity, and relation. Do not add facts, explanations, headings, lists, or examples. "
            "Output exactly one grammatical English sentence.\n\nSOURCE SENTENCE:\n" + text
        ),
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    factual = load_jsonl(IN_PATH, "peer_step010_strict_factual")
    balanced = load_jsonl(BALANCED_PATH, "peer_step010_balanced") if BALANCED_PATH.exists() else []

    # The high-precision tier begins from the stricter factual file.  The medium tier can also absorb balanced
    # rows if they pass the softer artifact filters, but duplicates are removed by sentence_id/text.
    high_rows = []
    high_reject = collections.Counter()
    high_examples = []
    for r in factual:
        reasons = hard_reject_reasons(r)
        if reasons:
            for x in reasons:
                high_reject[x] += 1
            if len(high_examples) < 24:
                high_examples.append({"sentence_id": r.get("sentence_id"), "doc_id": r.get("doc_id"), "words": r.get("words"), "reasons": reasons, "text": r.get("text")})
        else:
            high_rows.append(r)

    medium_rows = []
    medium_reject = collections.Counter()
    seen = set()
    for r in factual + balanced:
        key = (r.get("sentence_id"), r.get("text"))
        if key in seen:
            continue
        seen.add(key)
        reasons = medium_reject_reasons(r)
        if reasons:
            for x in reasons:
                medium_reject[x] += 1
        else:
            medium_rows.append(r)

    # Sort rows deterministically: broad doc coverage first for generation pilots, content density for all/full file.
    high_sorted = sorted(high_rows, key=lambda r: (str(r.get("doc_id", "")), int(r.get("sentence_id", 0))))
    high_content_sorted = sorted(high_rows, key=lambda r: (-len(r.get("domain_hits") or []), -int(r.get("relation_count") or 0), int(r["words"]), str(r.get("doc_id", "")), int(r.get("sentence_id", 0))))
    med_content_sorted = sorted(medium_rows, key=lambda r: (-len(r.get("domain_hits") or []), -int(r.get("relation_count") or 0), int(r["words"]), str(r.get("doc_id", "")), int(r.get("sentence_id", 0))))

    pilot2048 = stratified_pilot(high_content_sorted, min(2048, len(high_content_sorted)), 829112)
    pilot4096 = stratified_pilot(high_content_sorted, min(4096, len(high_content_sorted)), 829114)

    high_path = OUT_DIR / "fineweb_factual_high_precision_sources.jsonl"
    med_path = OUT_DIR / "fineweb_factual_medium_repaired_sources.jsonl"
    full_prompt_path = OUT_DIR / "fineweb_factual_high_precision_simplification_prompts_all.jsonl"
    pilot2048_path = OUT_DIR / f"fineweb_factual_high_precision_simplification_prompts_pilot{len(pilot2048)}.jsonl"
    pilot4096_path = OUT_DIR / f"fineweb_factual_high_precision_simplification_prompts_pilot{len(pilot4096)}.jsonl"

    write_jsonl(high_path, high_content_sorted)
    write_jsonl(med_path, med_content_sorted)
    write_jsonl(full_prompt_path, [make_prompt(r) for r in high_content_sorted])
    write_jsonl(pilot2048_path, [make_prompt(r) for r in pilot2048])
    write_jsonl(pilot4096_path, [make_prompt(r) for r in pilot4096])

    def pair_words(rows: list[dict[str, Any]]) -> dict[str, int]:
        w = sum(int(r["words"]) for r in rows)
        return {
            "source_words": w,
            "rewrite_words_if_ratio_0p75": round(0.75 * w),
            "rewrite_words_if_ratio_0p90": round(0.90 * w),
            "rewrite_words_if_ratio_1p05": round(1.05 * w),
            "pair_words_ratio_0p75": w + round(0.75 * w),
            "pair_words_ratio_0p90": w + round(0.90 * w),
            "pair_words_ratio_1p05": w + round(1.05 * w),
        }

    sample_rng = random.Random(829115)
    sample_high = sample_rng.sample(high_content_sorted, min(16, len(high_content_sorted))) if high_content_sorted else []
    sample_medium = sample_rng.sample(med_content_sorted, min(16, len(med_content_sorted))) if med_content_sorted else []

    payload = {
        "status": "FINEWEB_SOURCE_BY_REWRITE_ASSET_REPAIRED",
        "purpose": "CPU-only repair of the next FineWeb generation substrate; no model generation or training was launched.",
        "inputs": {
            "strict_factual_sources": str(IN_PATH),
            "strict_factual_sha256": sha256_file(IN_PATH),
            "balanced_sources": str(BALANCED_PATH),
            "balanced_sha256": sha256_file(BALANCED_PATH) if BALANCED_PATH.exists() else None,
        },
        "high_precision_from_strict_factual": summarize_rows(high_content_sorted),
        "medium_repaired_from_strict_plus_balanced": summarize_rows(med_content_sorted),
        "high_precision_pair_word_budget_estimates": pair_words(high_content_sorted),
        "pilot_pair_word_budget_estimates": {
            "pilot2048": pair_words(pilot2048),
            "pilot4096": pair_words(pilot4096),
        },
        "high_precision_rejection_reason_counts": dict(high_reject.most_common()),
        "medium_rejection_reason_counts": dict(medium_reject.most_common(40)),
        "high_precision_reject_examples": high_examples,
        "outputs": {
            "high_precision_sources": str(high_path),
            "medium_repaired_sources": str(med_path),
            "full_high_precision_prompts": str(full_prompt_path),
            "pilot2048_prompts": str(pilot2048_path),
            "pilot4096_prompts": str(pilot4096_path),
            "metadata": str(OUT_DIR / "fineweb_source_by_rewrite_repair_metadata.json"),
            "samples": str(OUT_DIR / "fineweb_source_by_rewrite_repair_samples.json"),
            "note": str(NOTE_PATH),
        },
        "generation_use": {
            "recommended_first_slice": str(pilot2048_path),
            "why": "small enough to test Qwen simplification acceptance and source faithfulness before committing to full FineWeb rewrite generation",
            "acceptance_check_needed": "verify exact preservation of named entities, numbers, dates, negation/modality, and main relation; reject added facts and multi-sentence/template outputs",
        },
    }

    meta_path = OUT_DIR / "fineweb_source_by_rewrite_repair_metadata.json"
    sample_path = OUT_DIR / "fineweb_source_by_rewrite_repair_samples.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sample_payload = {
        "high_precision_random": sample_high,
        "medium_random": sample_medium,
        "high_precision_top": high_content_sorted[:24],
        "high_precision_reject_examples": high_examples,
    }
    sample_path.write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    hp = payload["high_precision_from_strict_factual"]
    med = payload["medium_repaired_from_strict_plus_balanced"]
    note_lines = []
    note_lines.append("# research FineWeb source-by-rewrite asset repair\n\n")
    note_lines.append("No generation, training, or evaluation was launched. The purpose is to make the next source-by-rewrite experiment sharper if the active SimpleWiki semantic-view contrast is weak.\n\n")
    note_lines.append("## Why this repair was needed\n")
    note_lines.append("Peer research produced a useful factual-source asset, but direct inspection in frontier_consolidation still found citation metadata rows (`Volume/Issue/Page/doi`), literary quoted fragments, and weak publication-notice rows. A generation slice on those examples would spend H100 time on source noise rather than on the scientific question: whether broad factual sentences become more learnable when paired with faithful simplifications.\n\n")
    note_lines.append("## Repaired tiers\n")
    note_lines.append("| tier | rows | words | unique docs | mean words | p95 words | use |\n")
    note_lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    note_lines.append(f"| high_precision_from_strict_factual | {hp['rows']:,} | {hp['words']:,} | {hp['unique_docs']:,} | {hp['word_stats'].get('mean', 0):.2f} | {hp['word_stats'].get('p95', 0)} | recommended first Qwen faithfulness slice |\n")
    note_lines.append(f"| medium_repaired_from_strict_plus_balanced | {med['rows']:,} | {med['words']:,} | {med['unique_docs']:,} | {med['word_stats'].get('mean', 0):.2f} | {med['word_stats'].get('p95', 0)} | fallback if high precision is too small |\n\n")
    est = payload["high_precision_pair_word_budget_estimates"]
    note_lines.append("High-precision full source budget: source words {:,}; paired source+rewrite mass about {:,} words at rewrite/source ratio 0.90 (range {:,}--{:,} for ratios 0.75--1.05).\n\n".format(est["source_words"], est["pair_words_ratio_0p90"], est["pair_words_ratio_0p75"], est["pair_words_ratio_1p05"]))
    note_lines.append("## Output files\n")
    for k, v in payload["outputs"].items():
        note_lines.append(f"- {k}: `{v}`\n")
    note_lines.append("\n## Experimental meaning\n")
    note_lines.append("If the pending SimpleWiki semantic-view treatment beats packet-local repetition, this repaired FineWeb asset is secondary. If the SimpleWiki effect is weak, the next discriminating H100 action should be a small generation-and-audit slice from `pilot2048_prompts`, followed only then by a matched materialization: same FineWeb source sentences plus accepted simplifications versus the same source sentences with length-matched same-source repetition and identical official filler. This preserves the source-breadth question while isolating the effect of a faithful second view.\n")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("".join(note_lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "high_precision_rows": hp["rows"],
        "high_precision_words": hp["words"],
        "high_precision_docs": hp["unique_docs"],
        "medium_rows": med["rows"],
        "medium_words": med["words"],
        "metadata": str(meta_path),
        "note": str(NOTE_PATH),
        "pilot2048": str(pilot2048_path),
        "pilot4096": str(pilot4096_path),
        "top_high_rejections": list(high_reject.most_common(12)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
