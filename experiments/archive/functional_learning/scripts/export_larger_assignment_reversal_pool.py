#!/usr/bin/env python3
"""Export a larger literal-value assignment-reversal pool.

Earlier descriptor-based packets did not put the candidate state
values as literal strings in context.  The research export has the right contract
but only 120 base maps, mostly death_place.  This script builds a larger, relation-
diverse pool from existing source-grounded triples, preserving the same A/B answer-
slot schema:

  * each base map has two exact BabyLM source sentences for the same relation with
    distinct literal source values;
  * update_a and update_b use the same shared_new_value but update opposite entities;
  * each assignment-frame pair has exactly two rows: unchanged/source_state and
    updated/new_state, with answer spans validated in the final context text.

The pool is still a controlled counterfactual construction over extracted relations,
not a natural corpus claim.  It is designed for mechanistic binding/operation tests and
must be evaluated with no-update, wrong-recipient, candidate-type and broad-competence
controls before any practical BabyLM conclusion.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
NARROW_TRIPLES = _public_path('experiments/archive/functional_learning/data/relation_first_packets/extracted_triples.jsonl')
BROAD_TRIPLES = _public_path('experiments/archive/functional_learning/data/broad_extraction/broad_triples.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/larger_assignment_reversal_export')

REPLACEMENT_CITIES = [
    "Amsterdam", "Barcelona", "Dublin", "Milan", "Tokyo", "Sydney", "Vienna",
    "Copenhagen", "Stockholm", "Athens", "Lisbon", "Prague", "Budapest",
    "Helsinki", "Cairo", "Montreal", "Zurich", "Brisbane", "Edinburgh",
    "Florence", "Geneva", "Hamburg", "Kyoto", "Munich", "Osaka", "Salzburg",
    "Venice", "Brussels", "Oslo", "Marseille", "Ankara", "Lima", "Bogota",
    "Manila", "Jakarta", "Havana", "Nairobi", "Doha", "Riyadh", "Bangalore",
    "Reykjavik", "Tallinn", "Vilnius", "Krakow", "Valencia", "Porto",
]
REPLACEMENT_YEARS = [str(y) for y in range(1880, 2015, 5)]
REPLACEMENT_NATIONALITIES = [
    "Canadian", "Australian", "Irish", "Swedish", "Japanese", "Brazilian",
    "Norwegian", "Spanish", "Italian", "German", "French", "Dutch", "Finnish",
    "Mexican", "Argentine", "Kenyan", "Indian", "Singaporean", "Nigerian",
]
REPLACEMENT_OCCUPATIONS = [
    "teacher", "engineer", "lawyer", "composer", "scientist", "journalist",
    "architect", "historian", "economist", "poet", "artist", "director",
    "coach", "pilot", "doctor", "writer", "musician", "mathematician",
]

RELATION_CAPS = {
    "death_place": 160,
    "birthplace": 150,
    "birth_year": 130,
    "located_in": 20,
    "founded_year": 25,
    "nationality": 18,
    "occupation": 18,
}

RELATION_TEMPLATES = {
    "birthplace": {
        "update": "However, recent records show that {ENTITY} was actually born in {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, the birthplace of {entity} is {answer}.",
            "f01_after_records": "After these records, {entity} was born in {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity}'s birthplace is {answer}.",
            "f03_using_details": "Using the details above, {entity} was born in {answer}.",
        },
    },
    "death_place": {
        "update": "However, updated records confirm that {ENTITY} actually died in {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, the place where {entity} died is {answer}.",
            "f01_after_records": "After these records, {entity} died in {answer}.",
            "f02_passage_indicates": "The passage indicates that the place where {entity} died is {answer}.",
            "f03_using_details": "Using the details above, {entity}'s death place is {answer}.",
        },
    },
    "birth_year": {
        "update": "However, recent records show that {ENTITY} was actually born in {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, {entity} was born in {answer}.",
            "f01_after_records": "After these records, {entity}'s birth year is {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity} was born in {answer}.",
            "f03_using_details": "Using the details above, {entity}'s year of birth is {answer}.",
        },
    },
    "located_in": {
        "update": "However, after a boundary change, {ENTITY} is now located in {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, {entity} is located in {answer}.",
            "f01_after_records": "After these records, {entity} belongs in {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity} is in {answer}.",
            "f03_using_details": "Using the details above, the location for {entity} is {answer}.",
        },
    },
    "founded_year": {
        "update": "However, newly discovered documents show that {ENTITY} was actually founded in {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to the latest records, {entity} was founded in {answer}.",
            "f01_after_records": "After these records, {entity}'s founding year is {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity} was founded in {answer}.",
            "f03_using_details": "Using the details above, the founding year for {entity} is {answer}.",
        },
    },
    "nationality": {
        "update": "However, recent records show that {ENTITY} is actually {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, {entity} is {answer}.",
            "f01_after_records": "After these records, {entity}'s nationality is {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity} is {answer}.",
            "f03_using_details": "Using the details above, the nationality for {entity} is {answer}.",
        },
    },
    "occupation": {
        "update": "However, recent records show that {ENTITY} is actually a {NEW_VALUE}.",
        "queries": {
            "f00_original_a01": "According to this information, {entity} is a {answer}.",
            "f01_after_records": "After these records, {entity}'s occupation is {answer}.",
            "f02_passage_indicates": "The passage indicates that {entity} works as a {answer}.",
            "f03_using_details": "Using the details above, the occupation for {entity} is {answer}.",
        },
    },
}
FRAMES = [
    {"frame_id": "f00_original_a01", "frame_split": "train_seen", "style": "original_a01_query"},
    {"frame_id": "f01_after_records", "frame_split": "train_seen", "style": "after_records"},
    {"frame_id": "f02_passage_indicates", "frame_split": "train_seen", "style": "evidence_report"},
    {"frame_id": "f03_using_details", "frame_split": "eval_unseen", "style": "using_details"},
]

BAD_ENTITY_WORDS = {
    "He", "She", "It", "They", "This", "That", "These", "Those", "The", "A", "An",
    "His", "Her", "Their", "Some", "Many", "Most", "All", "Each", "Every", "Who",
}
BAD_SUBSTRINGS = [" who ", " and ", " from ", " with ", " during ", " because ", " after ", " before "]
VALUE_BAD_SUBSTRINGS = [" and ", " from ", " with ", " during ", " because ", " after ", " before ", " at the age", " who ", " which ", " that "]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def valid_entity(ent: str) -> bool:
    ent = clean_space(ent)
    if not ent or len(ent) < 3 or len(ent) > 60:
        return False
    if ent.split()[0] in BAD_ENTITY_WORDS:
        return False
    low = f" {ent.lower()} "
    if any(b in low for b in BAD_SUBSTRINGS):
        return False
    if re.search(r"[=\[\]{}()<>/\\]", ent):
        return False
    if not re.search(r"[A-Z]", ent):
        return False
    return True


def valid_value(relname: str, val: str) -> bool:
    val = clean_space(val).strip('"“”')
    if not val or len(val) > 50:
        return False
    low = f" {val.lower()} "
    if relname in {"birth_year", "founded_year"}:
        return bool(re.fullmatch(r"\d{4}", val)) and 1500 <= int(val) <= 2025
    if relname == "occupation":
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z-]{2,24}", val))
    if relname == "nationality":
        return bool(re.fullmatch(r"[A-Z][a-z]{3,24}", val))
    if any(b in low for b in VALUE_BAD_SUBSTRINGS):
        return False
    if re.search(r"[=\[\]{}()<>/\\]", val):
        return False
    if len(val.split()) > 4:
        return False
    if not re.search(r"[A-Z0-9]", val):
        return False
    return True


def normalize_triple(t: Dict[str, Any], provenance: str) -> Dict[str, Any] | None:
    r = str(t.get("relation"))
    if r not in RELATION_TEMPLATES:
        return None
    ent = clean_space(t.get("entity", ""))
    val = clean_space(t.get("value", "")).rstrip(",.")
    sent = clean_space(t.get("sentence", ""))
    if not valid_entity(ent) or not valid_value(r, val):
        return None
    if wc(sent) > 70 or wc(sent) < 5:
        return None
    vi = sent.find(val)
    if vi < 0:
        return None
    if ent.lower() not in sent.lower():
        return None
    return {
        "sentence_id": t.get("sentence_id"),
        "example_id": t.get("example_id"),
        "entity": ent,
        "relation": r,
        "value": val,
        "value_span": [vi, vi + len(val)],
        "sentence": sent,
        "source_type": t.get("source_type", "simple_wiki"),
        "grounding": t.get("grounding", "raw_source_exact_span"),
        "triple_provenance": provenance,
    }


def load_triples() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Use narrow extractor for place/location relations where broad regex overcaptures.
    rows = []
    for t in load_jsonl(NARROW_TRIPLES):
        nt = normalize_triple(t, "narrow_exact")
        if nt is not None:
            rows.append(nt)
    # Add reliable non-place relations from broader extraction.
    for t in load_jsonl(BROAD_TRIPLES):
        if t.get("relation") in {"birth_year", "nationality", "occupation"}:
            nt = normalize_triple(t, "step039c_broad_filtered")
            if nt is not None:
                rows.append(nt)
    # Deduplicate exact entity/relation/value/sentence.
    seen = set()
    uniq = []
    for t in rows:
        key = (t["entity"].lower(), t["relation"], t["value"].lower(), t["sentence_id"])
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq, {"raw_loaded_after_filter": len(uniq), "by_relation": dict(Counter(t["relation"] for t in uniq))}


def replacement_pool(relation: str) -> List[str]:
    if relation in {"birth_year", "founded_year"}:
        return REPLACEMENT_YEARS
    if relation == "nationality":
        return REPLACEMENT_NATIONALITIES
    if relation == "occupation":
        return REPLACEMENT_OCCUPATIONS
    return REPLACEMENT_CITIES


def choose_shared_new(relation: str, values: List[str], source_context: str, rng: random.Random) -> str | None:
    current = {v.lower() for v in values}
    source_low = source_context.lower()
    candidates = []
    for v in replacement_pool(relation):
        vl = v.lower()
        if vl in current or vl in source_low:
            continue
        candidates.append(v)
    if not candidates:
        return None
    return rng.choice(candidates[: min(20, len(candidates))])


def candidate_pairs(triples: List[Dict[str, Any]], seed: int, caps: Dict[str, int]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # Deduplicate to one sentence per entity/relation; keep shorter sentence.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in triples:
        key = (t["relation"], t["entity"].lower())
        old = best.get(key)
        if old is None or wc(t["sentence"]) < wc(old["sentence"]):
            best[key] = t
    for t in best.values():
        by_rel[t["relation"]].append(t)
    maps: List[Dict[str, Any]] = []
    for relation, group in sorted(by_rel.items()):
        group = sorted(group, key=lambda x: (x["entity"].lower(), x["value"].lower(), str(x.get("sentence_id"))))
        all_pairs = []
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a["value"].lower() == b["value"].lower():
                    continue
                if a["entity"].lower() == b["entity"].lower():
                    continue
                # Avoid direct cross-contamination: the other entity/value should not
                # already appear in the partner's source sentence.
                if a["entity"].lower() in b["sentence"].lower() or b["entity"].lower() in a["sentence"].lower():
                    continue
                if a["value"].lower() in b["sentence"].lower() or b["value"].lower() in a["sentence"].lower():
                    continue
                if wc(a["sentence"]) + wc(b["sentence"]) > 120:
                    continue
                all_pairs.append((a, b))
        rng.shuffle(all_pairs)
        cap = int(caps.get(relation, 0))
        for k, (a, b) in enumerate(all_pairs[:cap]):
            source_context = a["sentence"].rstrip() + " " + b["sentence"].rstrip()
            shared = choose_shared_new(relation, [a["value"], b["value"]], source_context, rng)
            if shared is None:
                continue
            tmpl = RELATION_TEMPLATES[relation]["update"]
            map_id = f"xrf_{relation}_{k:04d}"
            maps.append({
                "base_pair_id": map_id,
                "relation": relation,
                "source_context": source_context,
                "entity_a": a["entity"],
                "entity_b": b["entity"],
                "value_a": a["value"],
                "value_b": b["value"],
                "shared_new_value": shared,
                "update_a_sentence": tmpl.format(ENTITY=a["entity"], NEW_VALUE=shared),
                "update_b_sentence": tmpl.format(ENTITY=b["entity"], NEW_VALUE=shared),
                "source_a_sentence": a["sentence"],
                "source_b_sentence": b["sentence"],
                "source_a_id": a.get("sentence_id"),
                "source_b_id": b.get("sentence_id"),
                "source_a_value_span": a.get("value_span"),
                "source_b_value_span": b.get("value_span"),
                "triple_provenance_a": a.get("triple_provenance"),
                "triple_provenance_b": b.get("triple_provenance"),
                "augmentation_note": "Update sentences are controlled counterfactual augmentation; source sentences are exact BabyLM text from extracted relation triples.",
                "operation_contract": "update_a and update_b are assignment reversals with the same source_context and shared_new_value; retained source answer depends on queried entity.",
            })
    # Balanced train/held split within relation.
    out = []
    for relation in sorted(set(m["relation"] for m in maps)):
        rel_maps = [m for m in maps if m["relation"] == relation]
        rng.shuffle(rel_maps)
        held_n = max(1, int(round(len(rel_maps) * 0.2))) if len(rel_maps) >= 5 else 0
        for i, m in enumerate(rel_maps):
            mm = dict(m)
            mm["split"] = "heldout" if i < held_n else "train"
            out.append(mm)
    out.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))
    return out


def query_template(relation: str, frame_id: str) -> str:
    return RELATION_TEMPLATES[relation]["queries"].get(frame_id, "According to this information, the relevant {relation} value for {entity} is {answer}.")


def render_query(relation: str, frame_id: str, entity: str, answer: str) -> Tuple[str, int, int]:
    tmpl = query_template(relation, frame_id)
    text = tmpl.format(relation=relation.replace("_", " "), entity=entity, answer=answer)
    a = text.find(answer)
    if a < 0:
        raise RuntimeError(f"answer {answer!r} not found in query {text!r}")
    return text, a, a + len(answer)


def assignment_specs(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "assignment": "update_a",
            "updated_original_side": "entity_a",
            "unchanged_original_side": "entity_b",
            "updated_entity": m["entity_a"],
            "unchanged_entity": m["entity_b"],
            "update_sentence": m["update_a_sentence"],
            "source_state": m["value_b"],
            "new_state": m["shared_new_value"],
            "wrong_other_source": m["value_a"],
        },
        {
            "assignment": "update_b",
            "updated_original_side": "entity_b",
            "unchanged_original_side": "entity_a",
            "updated_entity": m["entity_b"],
            "unchanged_entity": m["entity_a"],
            "update_sentence": m["update_b_sentence"],
            "source_state": m["value_a"],
            "new_state": m["shared_new_value"],
            "wrong_other_source": m["value_b"],
        },
    ]


def make_two_rows(m: Dict[str, Any], spec: Dict[str, Any], frame: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pair_id = f"{m['base_pair_id']}::{spec['assignment']}::{frame['frame_id']}"
    source_context = m["source_context"].strip()
    update_sentence = spec["update_sentence"].strip()
    context_prefix = source_context + " " + update_sentence
    rows = []
    for half, role, query_entity, answer_kind, answer in [
        ("A", "unchanged_entity", spec["unchanged_entity"], "source_state", spec["source_state"]),
        ("B", "updated_entity", spec["updated_entity"], "new_state", spec["new_state"]),
    ]:
        query, qa0, qb0 = render_query(m["relation"], frame["frame_id"], query_entity, answer)
        context_text = context_prefix + " " + query
        answer_char_start = len(context_prefix) + 1 + qa0
        answer_char_end = len(context_prefix) + 1 + qb0
        if context_text[answer_char_start:answer_char_end] != answer:
            raise RuntimeError("answer span mismatch")
        rows.append({
            "row_id": f"{m['split']}:{pair_id}:{role}",
            "pair_id": pair_id,
            "split": m["split"],
            "packet_type": "A01_LARGER_ASSIGNMENT_REVERSAL_USE",
            "query_entity": query_entity,
            "answer_text": answer,
            "answer_kind": answer_kind,
            "role": role,
            "pair_half": half,
            "source_sentence": source_context,
            "update_sentence": update_sentence,
            "source_state": spec["source_state"],
            "new_state": spec["new_state"],
            "target_entity": spec["unchanged_entity"],
            "updated_entity": spec["updated_entity"],
            "context_text": context_text,
            "answer_char_start": answer_char_start,
            "answer_char_end": answer_char_end,
            "training_contract": "larger assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; mask/supervise answer_text tokens only.",
            "base_pair_id": m["base_pair_id"],
            "assignment": spec["assignment"],
            "relation": m["relation"],
            "frame_id": frame["frame_id"],
            "frame_split": frame["frame_split"],
            "frame_style": frame["style"],
            "query_template": query_template(m["relation"], frame["frame_id"]),
            "original_entity_a": m["entity_a"],
            "original_entity_b": m["entity_b"],
            "original_value_a": m["value_a"],
            "original_value_b": m["value_b"],
            "shared_new_value": m["shared_new_value"],
            "wrong_other_source": spec["wrong_other_source"],
            "source_a_id": m.get("source_a_id"),
            "source_b_id": m.get("source_b_id"),
            "map_policy": "filtered_extracted_relation_literals",
        })
    binding_pair = {
        "pair_id": pair_id,
        "row_a_id": rows[0]["row_id"],
        "row_b_id": rows[1]["row_id"],
        "entity_a": rows[0]["query_entity"],
        "entity_b": rows[1]["query_entity"],
        "answer_a": rows[0]["answer_text"],
        "answer_b": rows[1]["answer_text"],
        "base_pair_id": m["base_pair_id"],
        "assignment": spec["assignment"],
        "relation": m["relation"],
        "frame_id": frame["frame_id"],
        "frame_split": frame["frame_split"],
        "source_state": spec["source_state"],
        "new_state": spec["new_state"],
        "wrong_other_source": spec["wrong_other_source"],
        "updated_entity": spec["updated_entity"],
        "unchanged_entity": spec["unchanged_entity"],
    }
    return rows, binding_pair


def filter_rows(rows: List[Dict[str, Any]], split: str | None = None, frame_split: str | None = None, original_only: bool = False) -> List[Dict[str, Any]]:
    out = rows
    if split is not None:
        out = [r for r in out if r.get("split") == split]
    if original_only:
        out = [r for r in out if r.get("frame_id") == "f00_original_a01"]
    if frame_split is not None:
        out = [r for r in out if r.get("frame_split") == frame_split]
    return out


def pair_records_for(rows: List[Dict[str, Any]], pair_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pids = sorted({r["pair_id"] for r in rows})
    return [pair_meta[pid] for pid in pids]


def static_check(rows: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    row_ids = [r["row_id"] for r in rows]
    pids = [p["pair_id"] for p in pairs]
    group = defaultdict(list)
    span_errors = []
    long_rows = []
    for r in rows:
        group[r["pair_id"]].append(r)
        if r["context_text"][int(r["answer_char_start"]):int(r["answer_char_end"])] != r["answer_text"]:
            span_errors.append(r["row_id"])
        if wc(r["context_text"]) > 220:
            long_rows.append({"row_id": r["row_id"], "words": wc(r["context_text"])})
    bad_groups = {pid: [r.get("role") for r in rs] for pid, rs in group.items() if len(rs) != 2 or sorted(r.get("role") for r in rs) != ["unchanged_entity", "updated_entity"]}
    return {
        "rows": len(rows),
        "pairs": len(pairs),
        "duplicate_row_ids": len(row_ids) - len(set(row_ids)),
        "duplicate_pair_ids": len(pids) - len(set(pids)),
        "n_span_errors": len(span_errors),
        "span_errors": span_errors[:20],
        "n_bad_pair_groups": len(bad_groups),
        "bad_pair_groups": dict(list(bad_groups.items())[:20]),
        "row_words": sum(wc(r["context_text"]) for r in rows),
        "max_row_words": max([wc(r["context_text"]) for r in rows], default=0),
        "n_rows_over_220_words": len(long_rows),
        "first_long_rows": long_rows[:10],
        "by_relation": dict(Counter(r.get("relation") for r in rows)),
        "by_frame": dict(Counter(r.get("frame_id") for r in rows)),
        "by_assignment": dict(Counter(r.get("assignment") for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=64064)
    ap.add_argument("--max-maps", type=int, default=520)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    triples, triple_summary = load_triples()
    caps = dict(RELATION_CAPS)
    maps = candidate_pairs(triples, int(args.seed), caps)
    if int(args.max_maps) > 0 and len(maps) > int(args.max_maps):
        # Preserve relation balance by round-robin over already split maps.
        by_rel = defaultdict(list)
        for m in maps:
            by_rel[m["relation"]].append(m)
        rr = []
        rels = sorted(by_rel.keys())
        while len(rr) < int(args.max_maps) and any(by_rel.values()):
            for r in rels:
                if by_rel[r] and len(rr) < int(args.max_maps):
                    rr.append(by_rel[r].pop(0))
        maps = rr
    # Re-sort after truncation.
    maps.sort(key=lambda x: (x["split"], x["relation"], x["base_pair_id"]))

    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for m in maps:
        for spec in assignment_specs(m):
            for frame in FRAMES:
                rows, bp = make_two_rows(m, spec, frame)
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp

    files: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for split, split_label in [("train", "train"), ("heldout", "heldout")]:
        rows_all = filter_rows(all_rows, split=split)
        rows_seen = filter_rows(all_rows, split=split, frame_split="train_seen")
        rows_unseen = filter_rows(all_rows, split=split, frame_split="eval_unseen")
        rows_original = filter_rows(all_rows, split=split, original_only=True)
        files[f"{split_label}_frame_all"] = (rows_all, pair_records_for(rows_all, pair_meta))
        files[f"{split_label}_frame_seen"] = (rows_seen, pair_records_for(rows_seen, pair_meta))
        files[f"{split_label}_frame_unseen"] = (rows_unseen, pair_records_for(rows_unseen, pair_meta))
        files[f"{split_label}_original"] = (rows_original, pair_records_for(rows_original, pair_meta))

    output_files: Dict[str, Dict[str, Any]] = {}
    checks: Dict[str, Any] = {}
    for name, (rows, pairs) in files.items():
        row_path = args.out_dir / f"a01_larger_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = static_check(rows, pairs)
        checks[name] = chk
        output_files[name] = {"rows_path": rel(row_path), "binding_pairs_path": rel(pair_path), "rows": len(rows), "pairs": len(pairs)}

    op_path = args.out_dir / "a01_larger_assignment_reversal_operation_maps.jsonl"
    write_jsonl(op_path, maps)

    # Static sample for qualitative inspection.
    sample = []
    for relation in sorted(set(m["relation"] for m in maps)):
        sample.extend([m for m in maps if m["relation"] == relation][:3])
    (args.out_dir / "operation_map_sample.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fatal_counts = sum(ch["duplicate_row_ids"] + ch["duplicate_pair_ids"] + ch["n_span_errors"] + ch["n_bad_pair_groups"] for ch in checks.values())
    summary = {
        "status": "LARGER_ASSIGNMENT_REVERSAL_EXPORT_READY" if fatal_counts == 0 else "LARGER_ASSIGNMENT_REVERSAL_EXPORT_HAS_ERRORS",
        "created_utc": now(),
        "scientific_purpose": "Provide a larger literal-value assignment-reversal pool using exact BabyLM source sentences and the research A/B answer-slot contract, with more relation diversity than the 120-map export.",
        "triple_sources": {"narrow": rel(NARROW_TRIPLES), "broad": rel(BROAD_TRIPLES)},
        "triple_summary_after_filter": triple_summary,
        "relation_caps": caps,
        "n_base_maps": len(maps),
        "base_split_counts": dict(Counter(m["split"] for m in maps)),
        "base_relation_counts": dict(Counter(m["relation"] for m in maps)),
        "base_split_relation_counts": {split: dict(Counter(m["relation"] for m in maps if m["split"] == split)) for split in ["train", "heldout"]},
        "frames": FRAMES,
        "outputs": output_files,
        "operation_maps": rel(op_path),
        "sample": rel(args.out_dir / "operation_map_sample.json"),
        "static_checks": checks,
        "contract": {
            "row_schema": "A/B answer-slot: exactly two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "assignment_reversal": "Each base map produces update_a and update_b packets with the same source_context and shared_new_value but opposite updated entity.",
            "scope": "Mechanism-facing controlled counterfactual maps from extracted relation literals; source sentences are exact BabyLM text, update sentences and replacement values are controlled augmentations.",
            "needed_controls": "Use no-update, wrong-recipient, candidate-type, held-relation/frame and broad BabyLM controls before interpreting model behavior as transferable state tracking.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if fatal_counts != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
