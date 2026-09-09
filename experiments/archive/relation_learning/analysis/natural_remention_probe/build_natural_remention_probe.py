#!/usr/bin/env python3
"""Build a CPU-only, natural-text re-mention probe.

Candidates come from quality-filtered, single-document FineWeb chunks and are
screened against all FUNCTIONAL_RELATION_STUDIES VIEW/REPEAT/CLEAN training pools.  spaCy supplies
sentence, NER, POS, dependency, and noun-chunk annotations; all acceptance
rules are explicit below.  No generated rewrite or Entity item is used.
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
import time
from typing import Any, Iterable

import spacy

ROOT = pathlib.Path("/workspace")
OUT = ROOT / "experiments/archive/relation_learning/analysis/natural_remention_probe"
SOURCE = ROOT / "experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl"
SOURCE_META = ROOT / "experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/materialization_metadata.json"
POOL_DIR = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools"
TRAIN_POOLS = {
    "VIEW": POOL_DIR / "compact_view_dose2p64x_10M.jsonl",
    "REPEAT": POOL_DIR / "compact_repeat_dose2p64x_10M.jsonl",
    "CLEAN": POOL_DIR / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
}
NOMINALLY_HELDOUT = POOL_DIR / "heldout_cleanqwen_rows.jsonl"
FINEWEB_SOURCE = "fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies"
SEED = 9061002
NGRAM_N = 12
HEURISTIC_VERSION = "natural_remention_v1.1"
OVERLAP_CACHE = OUT / "training_overlap_cache.json"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
SPACE_RE = re.compile(r"\s+")
NAME_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}
NONPERSON_LABELS = {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"}
ACRONYM_LABELS = {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}
PERSON_PRONOUNS = {"he", "she", "him", "her", "his", "hers"}
NONPERSON_PRONOUNS = {"it", "its"}
BAD_SINGLE_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "english", "american",
    "christian", "internet", "government", "state", "university", "company",
    "noun", "verb", "adjective", "adverb", "pronoun", "knight", "char", "group", "tatar", "digital",
}
BAD_NAME_PHRASES = {"i e", "e g", "et al"}
ACRONYM_STOP = {"the", "of", "and", "for", "in", "on", "at", "a", "an", "to", "&"}
SURNAME_BAD = {"library", "survey", "glacier", "foundation", "university", "college", "school", "buddha",
               "institute", "association", "society", "museum", "literature", "center", "centre",
               "company", "department", "committee", "project", "program", "system", "church"}
HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame", "pope", "president"}
PERSON_APPOS_HEADS = {"president", "director", "founder", "scientist", "author", "writer", "actor", "actress",
                      "singer", "professor", "researcher", "physician", "doctor", "minister", "mayor", "governor",
                      "senator", "representative", "king", "queen", "prince", "princess", "leader", "chairman",
                      "chairwoman", "coach", "player", "artist", "composer", "poet", "lawyer", "judge", "engineer"}
ORG_APPOS_HEADS = {"company", "organization", "organisation", "agency", "foundation", "university", "college",
                   "school", "institute", "association", "society", "museum", "department", "committee", "project",
                   "program", "programme", "system", "network", "publisher", "firm", "bank", "corporation"}


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line.strip():
                yield line_index, json.loads(line)


def norm(s: str) -> str:
    return " ".join(x.lower().replace("’", "'") for x in WORD_RE.findall(s))


def words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def span_word_bounds(words: list[tuple[str, int, int]], start: int, end: int) -> tuple[int, int]:
    ids = [i for i, (_, a, b) in enumerate(words) if b > start and a < end]
    if not ids:
        return -1, -1
    return ids[0], ids[-1] + 1


def distance_bin(n: int) -> str:
    if n <= 4: return "00_04"
    if n <= 9: return "05_09"
    if n <= 19: return "10_19"
    if n <= 39: return "20_39"
    if n <= 79: return "40_79"
    return "80_plus"


def length_bin(n: int) -> str:
    return "1" if n == 1 else "2" if n == 2 else "3_plus"


def sentence_index(doc, token_i: int) -> int:
    for i, sent in enumerate(doc.sents):
        if sent.start <= token_i < sent.end:
            return i
    return -1


def sent_bounds(doc, sent_i: int) -> tuple[int, int]:
    sents = list(doc.sents)
    if 0 <= sent_i < len(sents):
        return sents[sent_i].start_char, sents[sent_i].end_char
    return 0, len(doc.text)


def ent_root(ent):
    roots = [t for t in ent if t.head.i < ent.start or t.head.i >= ent.end]
    return roots[0] if roots else ent.root


def valid_named_entity(ent) -> bool:
    if ent.label_ not in NAME_LABELS:
        return False
    toks = [t for t in ent if not t.is_punct]
    if not toks or len(toks) > 7:
        return False
    s = norm(ent.text)
    if not s or s in BAD_SINGLE_NAMES or s in BAD_NAME_PHRASES:
        return False
    if len(toks) == 1:
        return toks[0].pos_ == "PROPN" and len(s) >= 4 and not s.isdigit()
    return any(t.pos_ == "PROPN" for t in toks)


def clean_person_antecedent(ent) -> bool:
    """Reject common spaCy PERSON boundary errors such as ``Wrote Louis``."""
    alpha = [t for t in ent if t.is_alpha and t.lower_ not in HONORIFICS]
    return bool(alpha) and all(t.pos_ == "PROPN" for t in alpha)


def prose_quality_reject_reason(text: str) -> str | None:
    """Reject web artifacts that make reference heuristics especially unreliable."""
    if text.count("|") >= 2:
        return "table_pipe_markup"
    if text.count("•") >= 3:
        return "bullet_list"
    if len(re.findall(r"https?://|www\.", text, flags=re.I)) >= 2:
        return "multiple_urls"
    if ("Look at other dictionaries:" in text or "ISBN " in text or
            text.count("Wikipedia") >= 2 or "(part of speech:" in text.lower()):
        return "dictionary_or_catalog_entry"
    if len(re.findall(r"\[[^\]]{0,80}\]", text)) >= 4:
        return "citation_or_bracket_list"
    if len(re.findall(r"\([a-dA-D]\)", text)) >= 2:
        return "multiple_choice_item"
    if text.count("POV:") >= 2:
        return "outline_or_study_notes"
    if "[intransitive" in text.lower() or "[transitive" in text.lower():
        return "dictionary_or_catalog_entry"
    if re.search(r"\b([A-Z][a-z]{3,})\s+\1\b", text):
        return "duplicated_titlecase_token"
    if re.search(r"\bSELECT\b.+\bFROM\b", text, re.I):
        return "source_code_or_query"
    if re.search(r"\bIn:\s+.+\b(?:ed|eds)\.", text):
        return "bibliography_entry"
    if "scrabble anagrams" in text.lower():
        return "word_puzzle"
    return None


def make_base_record(obj: dict, line_index: int, doc, ant, rem, class_label: str,
                     subtype: str, ant_type: str, why: list[str], quality: int) -> dict | None:
    text = obj["text"]
    if ant.start_char >= rem.start_char or ant.end_char > rem.start_char:
        return None
    woffs = words_with_offsets(text)
    aw0, aw1 = span_word_bounds(woffs, ant.start_char, ant.end_char)
    rw0, rw1 = span_word_bounds(woffs, rem.start_char, rem.end_char)
    if min(aw0, aw1, rw0, rw1) < 0:
        return None
    intervening = max(0, rw0 - aw1)
    start_distance = rw0 - aw0
    asi, rsi = sentence_index(doc, ant.start), sentence_index(doc, rem.start)
    if asi < 0 or rsi < asi or intervening > 90 or start_distance <= 0:
        return None
    cs, _ = sent_bounds(doc, max(0, asi - 1))
    _, ce = sent_bounds(doc, min(len(list(doc.sents)) - 1, rsi + 1))
    record = {
        "probe_id": "", "heuristic_version": HEURISTIC_VERSION,
        "class_label": class_label, "subtype": subtype, "quality_rank": quality,
        "source_file": rel(SOURCE), "source_line_index": line_index,
        "source_line_number": line_index + 1, "row_id": obj.get("example_id"),
        "text_source": obj.get("source"), "doc_id": obj.get("doc_id"),
        "source_row": obj.get("source_row"), "chunk_start": obj.get("chunk_start"),
        "full_text_original": text,
        "antecedent": {
            "text": text[ant.start_char:ant.end_char], "char_start": ant.start_char,
            "char_end": ant.end_char, "word_start": aw0, "word_end": aw1,
            "word_count": aw1-aw0, "sentence_index": asi,
            "entity_type": ant_type, "syntactic_role": ent_root(ant).dep_,
        },
        "remention": {
            "text": text[rem.start_char:rem.end_char], "char_start": rem.start_char,
            "char_end": rem.end_char, "word_start": rw0, "word_end": rw1,
            "word_count": rw1-rw0, "sentence_index": rsi,
            "syntactic_role": ent_root(rem).dep_,
        },
        "mention_distance": {
            "intervening_words": intervening,
            "antecedent_to_remention_start_words": start_distance,
            "sentence_distance": rsi-asi,
            "distance_bin": distance_bin(intervening),
            "remention_length_bin": length_bin(rw1-rw0),
        },
        "local_context": {"text": text[cs:ce], "char_start": cs, "char_end": ce},
        "mask_plan": {
            "operation": "mask_remention_span", "char_start": rem.start_char,
            "char_end": rem.end_char, "target_text": text[rem.start_char:rem.end_char],
            "mask_each_model_subword": True,
        },
        "why_accepted": why,
        "training_overlap_audit": {
            "normalized_ngram_n": NGRAM_N, "arms_with_overlap": [],
            "acceptance_rule": f"reject row if any normalized {NGRAM_N}-word sequence occurs in an arm's 10M pool",
        },
        "corruption_plan": {},
    }
    return record


def extract_candidates(obj: dict, line_index: int, doc) -> tuple[list[dict], collections.Counter]:
    out: list[dict] = []
    stats = collections.Counter()
    reject = prose_quality_reject_reason(obj["text"])
    if reject:
        stats[f"row_rejected_{reject}"] += 1
        return out, stats
    ents = [e for e in doc.ents if valid_named_entity(e)]
    sents = list(doc.sents)

    # 1) Exact repeated named entities recognized at both mentions.
    groups: dict[tuple[str, str], list] = collections.defaultdict(list)
    for ent in ents:
        if ent.label_ == "NORP":
            # Nationality/religion adjectives (e.g. ``Tatar`` in ``Tatar
            # background``) repeat lexically without re-mentioning an entity.
            continue
        groups[(ent.label_, norm(ent.text))].append(ent)
    for (label, key), spans in groups.items():
        if len(spans) < 2:
            continue
        for ant, rem in zip(spans, spans[1:]):
            if ent_root(ant).dep_ in {"compound", "amod"} and ent_root(rem).dep_ in {"compound", "amod"}:
                continue
            rec = make_base_record(obj, line_index, doc, ant, rem, "verbatim_remention",
                                   "exact_named_entity", label,
                                   [f"both spans recognized as {label}", "case-insensitive normalized entity strings are identical"], 5)
            if rec and rec["mention_distance"]["intervening_words"] >= 3 and rec["mention_distance"]["sentence_distance"] <= 3:
                out.append(rec); stats["raw_verbatim_exact_named_entity"] += 1

    # 2) Indefinite-to-definite exact nominal core, e.g. "a device" -> "the device".
    chunks = list(doc.noun_chunks)
    nominal_groups: dict[str, list[tuple[Any, Any]]] = collections.defaultdict(list)
    for chunk in chunks:
        toks = [t for t in chunk if not t.is_punct]
        if len(toks) < 2 or toks[0].lower_ not in {"a", "an", "the"}:
            continue
        core_start = toks[1].i
        core = doc[core_start:chunk.end]
        core_key = norm(core.text)
        if (not core_key or len(core) > 4 or chunk.root.pos_ != "NOUN" or
                chunk.root.morph.get("Number") == ["Plur"]):
            continue
        nominal_groups[core_key].append((chunk, core))
    for key, vals in nominal_groups.items():
        for i, (achunk, acore) in enumerate(vals):
            if achunk[0].lower_ not in {"a", "an"}:
                continue
            for rchunk, rcore in vals[i+1:]:
                if rchunk[0].lower_ != "the":
                    continue
                rec = make_base_record(obj, line_index, doc, acore, rcore, "verbatim_remention",
                                       "indefinite_to_definite_nominal", "NOMINAL",
                                       ["same singular nominal core", "antecedent is indefinite and later mention is definite"], 4)
                if rec and 3 <= rec["mention_distance"]["intervening_words"] <= 60 and rec["mention_distance"]["sentence_distance"] <= 2:
                    out.append(rec); stats["raw_verbatim_indefinite_to_definite_nominal"] += 1
                break

    # 3) Full PERSON name followed by surname only, with no competing PERSON.
    person_ents = [e for e in ents if e.label_ == "PERSON"]
    for ant in person_ents:
        atoks = [t for t in ant if t.pos_ == "PROPN" and t.is_alpha]
        alpha_toks = [t for t in ant if t.is_alpha and t.lower_ not in HONORIFICS]
        if len(atoks) < 2 or len(atoks) > 5:
            continue
        if any(t.pos_ != "PROPN" for t in alpha_toks) or "'s" in ant.text.lower() or "’s" in ant.text.lower():
            continue
        surname = atoks[-1].text
        if len(surname) < 4 or surname.lower() in SURNAME_BAD:
            continue
        later = [t for t in doc[ant.end:] if t.text == surname and t.pos_ == "PROPN"]
        for tok in later[:1]:
            rem = doc[tok.i:tok.i+1]
            covering = [e for e in doc.ents if e.label_ == "PERSON" and e.start <= tok.i < e.end]
            if not covering:
                continue
            valid_cover = False
            for e in covering:
                other = [t for t in e if t.i != tok.i and t.is_alpha]
                if all(t.lower_ in HONORIFICS for t in other):
                    valid_cover = True
            if not valid_cover:
                continue
            competitors = [e for e in person_ents if e.start >= ant.end and e.end <= rem.start and norm(e.text) != norm(rem.text)]
            if competitors:
                continue
            rec = make_base_record(obj, line_index, doc, ant, rem, "nonidentical_remention",
                                   "full_name_to_surname_or_final_token", "PERSON",
                                   ["antecedent is a multi-token PERSON", "later proper token equals its surname", "no intervening competing PERSON"], 5)
            if rec and 3 <= rec["mention_distance"]["intervening_words"] and rec["mention_distance"]["sentence_distance"] <= 3:
                out.append(rec); stats["raw_nonidentical_full_name_to_surname_or_final_token"] += 1

    # 4) Full named entity followed by a matching initialism/acronym.
    for ant in ents:
        if ant.label_ not in ACRONYM_LABELS:
            continue
        alpha = [t.text for t in ant if t.is_alpha and t.lower_ not in ACRONYM_STOP]
        if not (2 <= len(alpha) <= 8):
            continue
        initials = "".join(x[0] for x in alpha).upper()
        if not (2 <= len(initials) <= 8):
            continue
        for tok in doc[ant.end:]:
            raw = re.sub(r"[^A-Za-z]", "", tok.text)
            # Require a clean acronym token.  This excludes tokenizer artifacts
            # such as ``BSE)-`` while retaining dotted forms such as ``U.S.``.
            clean_acronym_token = bool(re.fullmatch(r"(?:[A-Z]\.?){2,8}", tok.text))
            if raw == initials and clean_acronym_token:
                rem = doc[tok.i:tok.i+1]
                rec = make_base_record(obj, line_index, doc, ant, rem, "nonidentical_remention",
                                       "full_name_to_acronym", ant.label_,
                                       [f"initials of {ant.label_} antecedent equal later uppercase token {raw}", "surface forms differ"], 5)
                if rec and rec["mention_distance"]["sentence_distance"] <= 3:
                    out.append(rec); stats["raw_nonidentical_full_name_to_acronym"] += 1
                break

    # 5) Subject-continuity pronouns.  Previous sentence must have exactly one
    # compatible named entity and that entity must be its grammatical subject.
    for remtok in doc:
        low = remtok.lower_
        if low not in PERSON_PRONOUNS | NONPERSON_PRONOUNS:
            continue
        rsi = sentence_index(doc, remtok.i)
        if rsi <= 0:
            continue
        rsent = sents[rsi]
        if remtok.i - rsent.start > 5 or remtok.dep_ not in {"nsubj", "nsubjpass", "poss"}:
            continue
        first_content = next((t for t in rsent if not t.is_punct), None)
        # Subject pronouns embedded in examples, glosses, or quotations are a
        # frequent false-positive mode.  Require them to open their sentence.
        if first_content is None or first_content.i != remtok.i:
            continue
        if low in NONPERSON_PRONOUNS:
            if remtok.text[:1].upper() != remtok.text[:1]:
                continue
            if rsent.text.lstrip().startswith(('"', '“', '”', "'")):
                continue
            if re.match(r"(?i)^it would be\b", rsent.text.strip(' \t\"“”')):
                continue
            if re.match(r"(?i)^it (?:is|was) (?:stated|reported|clear|possible|likely|unlikely)\b",
                        rsent.text.strip(' \t\"“”')):
                continue
        prev = sents[rsi-1]
        if low in PERSON_PRONOUNS:
            compatible = [e for e in person_ents if prev.start <= e.start and e.end <= prev.end
                          and clean_person_antecedent(e)]
            prev_first = next((t for t in prev if not t.is_punct), None)
            if prev_first is not None and prev_first.lower_ in PERSON_PRONOUNS:
                continue
            subtype, ant_type = "person_name_to_pronoun", "PERSON"
        else:
            compatible = [e for e in ents if e.label_ in NONPERSON_LABELS and prev.start <= e.start and e.end <= prev.end]
            subtype, ant_type = "nonperson_name_to_pronoun", "NONPERSON_ENTITY"
            # Dialogue often shifts from the named speaker to an impersonal
            # ``it`` (or to an entity merely mentioned by the speaker).
            if any(q in prev.text for q in {'"', '“', '”'}) and re.search(r"\b(?:said|says|asked|replied|wrote)\b", prev.text, re.I):
                continue
        if len(compatible) != 1:
            continue
        ant = compatible[0]
        if ant_type == "PERSON" and norm(ant.text) in {"venus", "mars", "jupiter", "saturn", "mercury", "neptune", "uranus"}:
            continue
        if ant[0].i < 5:
            continue
        root = ent_root(ant)
        if root.dep_ not in {"nsubj", "nsubjpass", "ROOT"}:
            continue
        # No later compatible entity before the pronoun.
        if any(e.start >= ant.end and e.end <= remtok.i for e in ents if (e.label_ == "PERSON") == (ant_type == "PERSON")):
            continue
        if ant_type == "PERSON":
            # Catch missed PERSON spans (e.g. ``Davie``) without treating proper
            # tokens already covered by an ORG/GPE as rival people.
            untyped_later_propn = [t for t in prev if ant.end <= t.i < prev.end and t.pos_ == "PROPN" and t.is_alpha and
                                   not any(e.start <= t.i < e.end for e in doc.ents)]
            if untyped_later_propn:
                continue
            tail = doc.text[ant.end_char:prev.end_char]
            if any(q in tail for q in {'"', '“', '”'}) and any(t.lower_ in PERSON_PRONOUNS for t in doc[ant.end:prev.end]):
                continue
        rem = doc[remtok.i:remtok.i+1]
        rec = make_base_record(obj, line_index, doc, ant, rem, "nonidentical_remention",
                               subtype, ant_type,
                               ["exactly one compatible named entity in preceding sentence", "antecedent is a grammatical subject",
                                "later pronoun occurs within first six sentence tokens as subject/possessive", "no intervening compatible entity"],
                               4 if ant_type == "PERSON" else 3)
        if rec and 3 <= rec["mention_distance"]["intervening_words"] <= 60:
            out.append(rec); stats[f"raw_nonidentical_{subtype}"] += 1

    # 6) Explicit appositive descriptions after a named entity.
    for ant in ents:
        if ant.label_ not in {"PERSON", "ORG"}:
            continue
        for chunk in chunks:
            if chunk.start < ant.end or chunk.start - ant.end > 5 or chunk.root.dep_ != "appos":
                continue
            if chunk.root.head.i < ant.start or chunk.root.head.i >= ant.end:
                continue
            allowed_heads = PERSON_APPOS_HEADS if ant.label_ == "PERSON" else ORG_APPOS_HEADS
            head = chunk.root.lower_.rstrip("s") if chunk.root.lower_.endswith("s") else chunk.root.lower_
            if head not in allowed_heads:
                continue
            if ant.label_ == "ORG" and chunk[0].lower_ not in {"a", "an", "the"}:
                continue
            if any(e.start < chunk.end and e.end > chunk.start for e in doc.ents):
                continue
            between = doc.text[ant.end_char:chunk.start_char]
            if not re.fullmatch(r"\s*,\s*", between):
                continue
            rem = chunk
            rec = make_base_record(obj, line_index, doc, ant, rem, "nonidentical_remention",
                                   "explicit_appositive_description", ant.label_,
                                   ["later noun phrase has appos dependency directly headed by the named entity", "surface forms differ"], 5)
            if rec and rec["mention_distance"]["intervening_words"] <= 4:
                out.append(rec); stats["raw_nonidentical_explicit_appositive_description"] += 1
            break

    return out, stats


def ngram_tuples(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    toks = norm(text).split()
    return {tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))}


def screen_training_overlap(candidates: list[dict]) -> tuple[list[dict], dict]:
    row_records: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for r in candidates:
        row_records[(r["source_line_index"], int(r["row_id"]))].append(r)
    source_sha = sha256_file(SOURCE)
    pool_shas = {arm: sha256_file(path) for arm, path in TRAIN_POOLS.items()}
    candidate_cache_keys = {f"{line}:{row_id}" for line, row_id in row_records}

    # The full 30M-word overlap pass is intentionally cached.  Cache reuse is
    # allowed only when source/pool digests and every requested candidate row
    # agree, so tightening extraction rules cannot accidentally weaken screening.
    cache = None
    if OVERLAP_CACHE.exists():
        try:
            proposed = json.loads(OVERLAP_CACHE.read_text(encoding="utf-8"))
            if (proposed.get("schema_version") == 1 and proposed.get("ngram_n") == NGRAM_N and
                    proposed.get("source_sha256") == source_sha and proposed.get("pool_sha256") == pool_shas and
                    candidate_cache_keys <= set(proposed.get("line_to_arms", {}))):
                cache = proposed
        except (OSError, ValueError, TypeError):
            cache = None

    if cache is not None:
        overlap = collections.defaultdict(set)
        for key in row_records:
            overlap[key].update(cache["line_to_arms"][f"{key[0]}:{key[1]}"])
        arm_stats = {}
        for arm, path in TRAIN_POOLS.items():
            base = cache["arm_scan_totals"][arm]
            arm_stats[arm] = {
                "path": rel(path), "sha256": pool_shas[arm],
                "rows_scanned": base["rows_scanned"], "words_scanned": base["words_scanned"],
                "candidate_rows_with_overlap": sum(arm in overlap[k] for k in row_records),
            }
        kept = []
        for key, records in row_records.items():
            arms = sorted(overlap.get(key, set()))
            for r in records:
                r["training_overlap_audit"]["arms_with_overlap"] = arms
                if not arms:
                    kept.append(r)
        return kept, {"cache_used": True, "cache_path": rel(OVERLAP_CACHE),
                      "candidate_rows_screened": len(row_records),
                      "candidate_rows_rejected_any_arm": sum(bool(overlap.get(k)) for k in row_records),
                      "candidate_records_before": len(candidates), "candidate_records_after": len(kept),
                      "arm_scans": arm_stats}

    index: dict[tuple[str, ...], set[tuple[int, int]]] = collections.defaultdict(set)
    for key, records in row_records.items():
        for ng in ngram_tuples(records[0]["full_text_original"]):
            index[ng].add(key)
    overlap: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    arm_stats = {}
    for arm, path in TRAIN_POOLS.items():
        hits = 0; rows_scanned = 0; words_scanned = 0
        for _, obj in read_jsonl(path):
            rows_scanned += 1; words_scanned += int(obj.get("words", len(obj["text"].split())))
            for ng in ngram_tuples(obj["text"]):
                keys = index.get(ng)
                if keys:
                    for key in keys:
                        if arm not in overlap[key]:
                            overlap[key].add(arm); hits += 1
        arm_stats[arm] = {"path": rel(path), "sha256": pool_shas[arm], "rows_scanned": rows_scanned,
                          "words_scanned": words_scanned, "candidate_rows_with_overlap": hits}
    cache_payload = {
        "schema_version": 1, "ngram_n": NGRAM_N, "source_path": rel(SOURCE),
        "source_sha256": source_sha, "pool_sha256": pool_shas,
        "line_to_arms": {f"{k[0]}:{k[1]}": sorted(overlap.get(k, set())) for k in row_records},
        "arm_scan_totals": {arm: {"rows_scanned": d["rows_scanned"], "words_scanned": d["words_scanned"]}
                            for arm, d in arm_stats.items()},
    }
    OVERLAP_CACHE.write_text(json.dumps(cache_payload, sort_keys=True) + "\n", encoding="utf-8")
    kept = []
    for key, records in row_records.items():
        arms = sorted(overlap.get(key, set()))
        for r in records:
            r["training_overlap_audit"]["arms_with_overlap"] = arms
            if not arms:
                kept.append(r)
    return kept, {"cache_used": False, "cache_path": rel(OVERLAP_CACHE),
                  "candidate_rows_screened": len(row_records),
                  "candidate_rows_rejected_any_arm": sum(bool(overlap.get(k)) for k in row_records),
                  "candidate_records_before": len(candidates), "candidate_records_after": len(kept),
                  "arm_scans": arm_stats}


def deduplicate(candidates: list[dict]) -> tuple[list[dict], int]:
    best: dict[tuple, dict] = {}
    for r in candidates:
        key = (r["class_label"], r["source_line_index"], r["antecedent"]["char_start"], r["remention"]["char_start"])
        if key not in best or r["quality_rank"] > best[key]["quality_rank"]:
            best[key] = r
    return list(best.values()), len(candidates)-len(best)


def choose_matched(candidates: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    # At most one record per source row prevents a few repetitive documents from
    # dominating.  Greedily pair classes on distance bin, remention-length bin,
    # and sentence distance.  Relax sentence distance, then length, only if the
    # exact match pool would otherwise be very small.
    by_class = collections.defaultdict(list)
    for r in candidates:
        by_class[r["class_label"]].append(r)
    for vals in by_class.values():
        rng.shuffle(vals)
        vals.sort(key=lambda r: r["quality_rank"], reverse=True)
    vpool, npool = by_class["verbatim_remention"], by_class["nonidentical_remention"]

    # Preserve all high-value non-pronoun types where matching support exists,
    # while capping pronouns at 70% of the nonidentical final set.
    def stratum(r, level):
        d = r["mention_distance"]
        if level == 0: return (d["distance_bin"], d["remention_length_bin"], d["sentence_distance"])
        if level == 1: return (d["distance_bin"], d["remention_length_bin"])
        return (d["distance_bin"],)

    used_rows: set[int] = set()
    selected_v: list[dict] = []; selected_n: list[dict] = []
    levels_used = collections.Counter()
    # Nonpronouns first, then pronouns.  Highest quality and longest distance first.
    def nkey(r):
        is_pron = r["subtype"].endswith("to_pronoun")
        return (is_pron, -r["quality_rank"], -r["mention_distance"]["intervening_words"])
    for nr in sorted(npool, key=nkey):
        rown = nr["source_line_index"]
        if rown in used_rows:
            continue
        is_pron = nr["subtype"].endswith("to_pronoun")
        if is_pron and selected_n and sum(x["subtype"].endswith("to_pronoun") for x in selected_n) / len(selected_n) >= .70:
            continue
        match = None; level_found = None
        for level in range(3):
            key = stratum(nr, level)
            choices = [vr for vr in vpool if vr["source_line_index"] not in used_rows and
                       vr["source_line_index"] != rown and stratum(vr, level) == key]
            if choices:
                choices.sort(key=lambda r: (-r["quality_rank"], abs(r["mention_distance"]["intervening_words"]-nr["mention_distance"]["intervening_words"])))
                match = choices[0]; level_found = level; break
        if match is None:
            continue
        pair_id = f"natural_distance_pair:{len(selected_n):04d}"
        level_label = {0:"distance+length+sentence",1:"distance+length",2:"distance"}[level_found]
        nr["class_matching"] = {"pair_id": pair_id, "matching_level": level_label}
        match["class_matching"] = {"pair_id": pair_id, "matching_level": level_label}
        selected_n.append(nr); selected_v.append(match)
        used_rows.add(rown); used_rows.add(match["source_line_index"])
        levels_used[level_label] += 1
        if len(selected_n) >= 1000:
            break
    final = selected_v + selected_n
    rng.shuffle(final)
    return final, {"matched_pairs": len(selected_n), "records": len(final), "one_record_per_source_row": True,
                   "matching_levels": dict(levels_used), "pronoun_cap": 0.70}


def add_replacements(records: list[dict], rng: random.Random) -> dict:
    donors = collections.defaultdict(list)
    for r in records:
        key = (r["antecedent"]["entity_type"], r["antecedent"]["word_count"])
        donors[key].append(r)
    def typed_fallback(entity_type: str, n: int) -> str:
        if entity_type == "PERSON":
            toks = "Taylor Morgan Jordan Avery Casey Riley Cameron".split()
            return " ".join((toks * math.ceil(n / len(toks)))[:n])
        if entity_type == "NOMINAL":
            return " ".join((["separate"] * max(0, n - 1)) + ["object"])
        if entity_type in {"GPE", "LOC", "FAC"}:
            toks = "The Northern Regional Coastal Harbor Territory".split()
        elif entity_type in {"PRODUCT", "WORK_OF_ART", "EVENT"}:
            toks = "The Independent Harbor Research Project".split()
        else:
            toks = "The Independent Regional Public Research Harbor Institute".split()
        # Taking the final n tokens retains a type-appropriate head while
        # guaranteeing exact lexical length.
        if n <= len(toks):
            return " ".join(toks[-n:])
        return " ".join((["Independent"] * (n - len(toks))) + toks)
    exact = 0; fallback = 0
    for r in records:
        ant = r["antecedent"]; key = (ant["entity_type"], ant["word_count"])
        options = [x for x in donors[key] if x["source_line_index"] != r["source_line_index"] and
                   norm(x["antecedent"]["text"]) != norm(ant["text"]) and
                   norm(x["antecedent"]["text"]) not in norm(r["full_text_original"])]
        if options:
            donor = rng.choice(options)
            replacement = donor["antecedent"]["text"]
            donor_ref = {"source_line_index": donor["source_line_index"], "row_id": donor["row_id"],
                         "entity_type": donor["antecedent"]["entity_type"]}
            strategy = "same_entity_type_and_exact_word_count_donor"
            exact += 1
        else:
            replacement = typed_fallback(ant["entity_type"], ant["word_count"])
            donor_ref = None; strategy = "typed_neutral_fallback"
            fallback += 1
        text = r["full_text_original"]
        corrupted = text[:ant["char_start"]] + replacement + text[ant["char_end"]:]
        char_delta = len(replacement) - (ant["char_end"] - ant["char_start"])
        rem = r["remention"]
        r["corruption_plan"] = {
            "condition_present": "full_text_original",
            "condition_replaced": "full_text_antecedent_replaced",
            "operation": "replace antecedent span only",
            "char_start_original": ant["char_start"], "char_end_original": ant["char_end"],
            "replacement_text": replacement, "strategy": strategy, "donor": donor_ref,
            "original_word_count": ant["word_count"], "replacement_word_count": len(WORD_RE.findall(replacement)),
            "exact_word_count_match": ant["word_count"] == len(WORD_RE.findall(replacement)),
            "character_length_delta": char_delta,
            "remention_char_start_replaced": rem["char_start"] + char_delta,
            "remention_char_end_replaced": rem["char_end"] + char_delta,
            "full_text_antecedent_replaced": corrupted,
            "warning": ("For pronoun targets a same-type donor preserves coarse agreement; the contrast tests antecedent-specific "
                        "conditioning weakly and should be analyzed separately from surname/acronym/description targets."),
        }
    return {"same_type_exact_length_donor": exact, "typed_fallback": fallback,
            "exact_word_count_matches": sum(r["corruption_plan"]["exact_word_count_match"] for r in records)}


def counts_by(records: list[dict], fn) -> dict:
    return dict(sorted(collections.Counter(fn(r) for r in records).items()))


def summarize_numeric(records: list[dict], key: str) -> dict:
    xs = [float(r["mention_distance"][key]) for r in records]
    if not xs: return {"n": 0}
    ys = sorted(xs)
    return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs),
            "p10": ys[int(.1*(len(ys)-1))], "p90": ys[int(.9*(len(ys)-1))],
            "min": min(xs), "max": max(xs)}


def distribution_summary(records: list[dict]) -> dict:
    out = {}
    for label in ["ALL", "verbatim_remention", "nonidentical_remention"]:
        rr = records if label == "ALL" else [r for r in records if r["class_label"] == label]
        out[label] = {
            "records": len(rr), "unique_rows": len({r["source_line_index"] for r in rr}),
            "subtype": counts_by(rr, lambda r:r["subtype"]),
            "distance_bin": counts_by(rr, lambda r:r["mention_distance"]["distance_bin"]),
            "remention_length_bin": counts_by(rr, lambda r:r["mention_distance"]["remention_length_bin"]),
            "sentence_distance": counts_by(rr, lambda r:str(r["mention_distance"]["sentence_distance"])),
            "antecedent_entity_type": counts_by(rr, lambda r:r["antecedent"]["entity_type"]),
            "intervening_words": summarize_numeric(rr, "intervening_words"),
            "antecedent_to_remention_start_words": summarize_numeric(rr, "antecedent_to_remention_start_words"),
        }
    return out


def contamination_of_nominal_holdout() -> dict:
    held = list(read_jsonl(NOMINALLY_HELDOUT))
    held_tokens = []
    for _, r in held: held_tokens.extend(r["text"].split())
    clean_tokens = []
    for _, r in read_jsonl(TRAIN_POOLS["CLEAN"]):
        clean_tokens.extend(r["text"].split())
        if len(clean_tokens) >= len(held_tokens): break
    clean_tokens = clean_tokens[:len(held_tokens)]
    # The first 133 words form the VIEW/REPEAT top-up.
    first_view = next(read_jsonl(TRAIN_POOLS["VIEW"]))[1]["text"].split()
    return {
        "path": rel(NOMINALLY_HELDOUT), "rows": len(held), "words": len(held_tokens),
        "sha256": sha256_file(NOMINALLY_HELDOUT),
        "entire_word_stream_equals_clean_training_prefix": held_tokens == clean_tokens,
        "note": "Rejected as a source: the research materializer uses this stream for CLEAN changed text and a 133-word VIEW/REPEAT top-up.",
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_sample(records: list[dict], rng: random.Random, n: int = 100) -> list[dict]:
    # Round-robin over class/subtype/distance for inspectability.
    buckets = collections.defaultdict(list)
    for r in records:
        buckets[(r["class_label"], r["subtype"], r["mention_distance"]["distance_bin"])].append(r)
    for vals in buckets.values(): rng.shuffle(vals)
    sample = []
    keys = sorted(buckets)
    while len(sample) < min(n, len(records)) and keys:
        next_keys = []
        for k in keys:
            if buckets[k] and len(sample) < n:
                sample.append(buckets[k].pop())
            if buckets[k]: next_keys.append(k)
        keys = next_keys
    return sample


def marked_context(r: dict) -> str:
    text = r["full_text_original"]
    a, b = r["antecedent"], r["remention"]
    return (text[:a["char_start"]] + "**[ANT " + text[a["char_start"]:a["char_end"]] + "]**" +
            text[a["char_end"]:b["char_start"]] + "**[REM " + text[b["char_start"]:b["char_end"]] + "]**" + text[b["char_end"]:])


def write_sample_md(path: pathlib.Path, sample: list[dict]) -> None:
    lines = ["# Natural re-mention probe: 100-record audit sample", "",
             "`[ANT ...]` marks the antecedent and `[REM ...]` the later span that will be masked. "
             "This file is stratified for inspection; inclusion here is not a human correctness label.", ""]
    for i, r in enumerate(sample, 1):
        d = r["mention_distance"]
        lines += [f"## {i}. {r['probe_id']}", "",
                  f"Class/subtype: `{r['class_label']}` / `{r['subtype']}`. "
                  f"Distance: {d['intervening_words']} intervening words, {d['sentence_distance']} sentences. "
                  f"Replacement: `{r['corruption_plan']['replacement_text']}`.", "",
                  marked_context(r), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    source_meta = json.loads(SOURCE_META.read_text(encoding="utf-8"))
    source_rows: list[tuple[int, dict]] = []
    total_lines = 0
    for line_index, obj in read_jsonl(SOURCE):
        total_lines += 1
        if obj.get("source") == FINEWEB_SOURCE:
            source_rows.append((line_index, obj))
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    raw: list[dict] = []; extraction = collections.Counter()
    docs = nlp.pipe((obj["text"] for _, obj in source_rows), batch_size=128)
    for (line_index, obj), doc in zip(source_rows, docs):
        rr, ss = extract_candidates(obj, line_index, doc)
        raw.extend(rr); extraction.update(ss)
    deduped, dup_n = deduplicate(raw)
    screened, overlap_summary = screen_training_overlap(deduped)
    final, matching = choose_matched(screened, rng)
    final.sort(key=lambda r:(r["class_label"], r["source_line_index"], r["antecedent"]["char_start"], r["remention"]["char_start"]))
    for i, r in enumerate(final):
        prefix = "vr" if r["class_label"] == "verbatim_remention" else "nr"
        r["probe_id"] = f"natural_mention:{prefix}:{i:05d}"
    replacement_summary = add_replacements(final, rng)
    output_path = OUT / "natural_remention_probe.jsonl"
    write_jsonl(output_path, final)
    sample = make_sample(final, random.Random(SEED+1), 100)
    write_sample_md(OUT / "natural_remention_probe_sample.md", sample)
    nominal_audit = contamination_of_nominal_holdout()
    summary = {
        "status": "NATURAL_REMENTION_PROBE_CONSTRUCTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heuristic_version": HEURISTIC_VERSION, "seed": SEED,
        "source": {
            "path": rel(SOURCE), "sha256": sha256_file(SOURCE), "total_pool_lines": total_lines,
            "selected_natural_source_label": FINEWEB_SOURCE, "natural_rows_scanned": len(source_rows),
            "source_words_from_metadata": source_meta["verification"]["source_word_counts_treatment"][FINEWEB_SOURCE],
            "provenance": source_meta["inputs"]["fineweb"],
            "generated_rewrites_used": False, "entity_benchmark_items_used": False,
        },
        "rejected_preferred_source_audit": nominal_audit,
        "extraction": {"raw_records": len(raw), "deduplicated_records": len(deduped),
                       "duplicate_span_records_removed": dup_n, "raw_by_rule": dict(sorted(extraction.items())),
                       "after_training_overlap_screen": len(screened)},
        "training_overlap_screen": overlap_summary,
        "selection_and_matching": matching,
        "replacement_plans": replacement_summary,
        "final_distribution": distribution_summary(final),
        "heuristics": {
            "verbatim_exact_named_entity": "Both mentions are spaCy NER spans of the same accepted entity type with identical normalized strings; gap >=3 words, sentence distance <=3.",
            "verbatim_indefinite_to_definite_nominal": "Same <=4-token singular common-noun core changes from a/an to the; gap 3-60, sentence distance <=2.",
            "full_name_to_surname_or_final_token": "Multi-token PERSON-NER span followed by its last proper-name token, no intervening competing PERSON, sentence distance <=3; rare NER boundary errors can yield valid final-head shortening rather than a personal surname.",
            "full_name_to_acronym": "Initials of a 2-8 word named entity exactly equal a later uppercase token, sentence distance <=3.",
            "name_to_pronoun": "Exactly one type-compatible named entity in prior sentence; antecedent is subject; pronoun starts within six tokens as subject/possessive; no intervening compatible entity.",
            "explicit_appositive_description": "Later noun chunk is a direct appos dependency of the PERSON/ORG named entity.",
            "training_disjointness": f"Reject the entire candidate row if any normalized {NGRAM_N}-word sequence is found in any VIEW/REPEAT/CLEAN 10M pool.",
        },
        "output": {"probe": rel(output_path), "probe_sha256": sha256_file(output_path),
                   "sample": rel(OUT / "natural_remention_probe_sample.md"), "sample_records": len(sample)},
        "elapsed_seconds": time.time()-t0,
    }
    (OUT / "natural_remention_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"records": len(final), "distribution": summary["final_distribution"],
                      "overlap": overlap_summary, "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
