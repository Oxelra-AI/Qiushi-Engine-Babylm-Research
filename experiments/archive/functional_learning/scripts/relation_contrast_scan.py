#!/usr/bin/env python3
"""research: scan for same-entity compatible-relation contrast candidates.

Purpose: test whether the acquired state-update operation indexes only an entity or an
entity--relation pair.  If a source passage states both birthplace and death place for
the same entity, a controlled update can change one relation while the other should be
retained.  Because both values are places, a syntactic type cue alone is less able to
solve the contrast.

This script only builds a candidate pool and static sample.  It does not establish that
all candidates are semantically valid; later use must inspect/admit examples.
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
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SOURCE_SENTENCES = [
    _public_path('experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl'),
    _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences.jsonl'),
]
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/relation_contrast_scan')

NAME = r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,4}"
# Includes multiword places and lower-case name connectors, but stops before comma,
# sentence punctuation, and ordinary clauses.
PLACE = r"[A-Z][A-Za-z'’.-]+(?:\s+(?:of|the|and|de|del|la|le|di|du|van|von|dos|das|[A-Z][A-Za-z'’.-]+)){0,5}"
BAD_ENTITY_START = {"He", "She", "It", "They", "This", "That", "The", "His", "Her", "Biography"}
BAD_PLACE_WORDS = {"August", "September", "October", "November", "December", "January", "February", "March", "April", "May", "June", "July"}
REPLACEMENT_CITIES = [
    "Amsterdam", "Barcelona", "Dublin", "Milan", "Tokyo", "Sydney", "Vienna",
    "Copenhagen", "Stockholm", "Athens", "Lisbon", "Prague", "Budapest",
    "Helsinki", "Cairo", "Montreal", "Zurich", "Brisbane", "Edinburgh",
    "Florence", "Geneva", "Hamburg", "Kyoto", "Munich", "Osaka", "Salzburg",
    "Venice", "Brussels", "Oslo", "Marseille", "Ankara", "Lima", "Bogota",
]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def read_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def trim_place(raw: str) -> str:
    s = clean(raw).strip(' "“”')
    # Cut common post-value clause/prepositional continuations that are not part of the
    # place name in this bounded contrast construction.
    s = re.split(r"\s+(?:from|after|before|at|aged|while|when|where|who|which|that)\b", s)[0]
    s = s.strip(" ,.;:()[]{}\"“”")
    # Remove trailing lowercase connectors if a regex stopped awkwardly.
    words = s.split()
    while words and words[-1].lower() in {"of", "the", "and", "de", "del", "la", "le", "di", "du", "van", "von", "dos", "das"}:
        words.pop()
    return " ".join(words).strip(" ,.;:")


def valid_name(name: str) -> bool:
    name = clean(name)
    if not name or len(name) < 3 or len(name) > 70:
        return False
    if name.split()[0] in BAD_ENTITY_START:
        return False
    if re.search(r"[=\[\]{}<>/\\]", name):
        return False
    return True


def valid_place(place: str) -> bool:
    place = trim_place(place)
    if not place or len(place) < 3 or len(place) > 60:
        return False
    if place.split()[0] in BAD_PLACE_WORDS:
        return False
    if re.search(r"[=\[\]{}<>/\\]", place):
        return False
    if any(w.lower() in {"heart", "failure", "cancer", "pneumonia", "diabetes"} for w in place.split()):
        return False
    return True


def no_prefix_continuation(text: str, value: str) -> bool:
    i = text.find(value)
    if i < 0:
        return False
    after = text[i + len(value): i + len(value) + 35]
    # If an immediate lower-case connector follows, the stored value is likely incomplete
    # (Rio de Janeiro, Santiago del Estero, King and Pierce).  Allow comma and sentence end.
    if re.match(r"^\s+(?:of|and|de|del|la|le|di|du|van|von|dos|das)\b", after):
        return False
    return True


def candidate_from_match(row: Dict[str, Any], m: re.Match, order: str) -> Dict[str, Any] | None:
    ent = clean(m.group("entity"))
    b = trim_place(m.group("birth"))
    d = trim_place(m.group("death"))
    text = clean(row["text"])
    if not valid_name(ent) or not valid_place(b) or not valid_place(d):
        return None
    if b.lower() == d.lower():
        return None
    if not no_prefix_continuation(text, b) or not no_prefix_continuation(text, d):
        return None
    bi = text.find(b)
    di = text.find(d)
    ei = text.lower().find(ent.lower())
    if bi < 0 or di < 0 or ei < 0:
        return None
    # Keep compact enough passages where both facts are plausibly about the same entity.
    if abs(bi - di) > 260:
        return None
    return {
        "source_id": row.get("id", f"ex_{row.get('example_id')}"),
        "example_id": row.get("example_id"),
        "source": row.get("source"),
        "entity": ent,
        "birthplace": b,
        "death_place": d,
        "source_text": text,
        "birth_value_span": [bi, bi + len(b)],
        "death_value_span": [di, di + len(d)],
        "entity_span": [ei, ei + len(ent)],
        "extraction_pattern": order,
        "audit_status": "candidate_needs_semantic_review",
    }


def scan_rows(max_rows: int | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    patterns = [
        (
            "born_then_died_same_entity_repeated",
            re.compile(
                rf"(?P<entity>{NAME})\s+was\s+born(?:\s+on\s+[^.;]{{0,70}}?)?\s+in\s+(?P<birth>{PLACE})[,.;]?[^.\n]{{0,220}}?\b(?:he|she|it|they|{NAME})?\s*(?:died|passed away)\b(?:\s+on\s+[^.;]{{0,70}}?)?(?:\s+at\s+[^.;]{{0,80}}?)?\s+(?:in|at)\s+(?P<death>{PLACE})",
                re.I,
            ),
        ),
        (
            "died_then_born_same_entity_repeated",
            re.compile(
                rf"(?P<entity>{NAME})\s+(?:died|passed away)\b(?:\s+on\s+[^.;]{{0,70}}?)?(?:\s+at\s+[^.;]{{0,80}}?)?\s+(?:in|at)\s+(?P<death>{PLACE})[,.;]?[^.\n]{{0,220}}?\b(?:was\s+)?born(?:\s+on\s+[^.;]{{0,70}}?)?\s+in\s+(?P<birth>{PLACE})",
                re.I,
            ),
        ),
    ]
    candidates: List[Dict[str, Any]] = []
    seen = set()
    n_rows = 0
    source_counts = Counter()
    for path in SOURCE_SENTENCES:
        for row in read_jsonl(path):
            n_rows += 1
            if max_rows and n_rows > max_rows:
                break
            if row.get("source") != "simple_wiki":
                continue
            text = clean(row.get("text", ""))
            if "born" not in text.lower() or "died" not in text.lower():
                continue
            for label, pat in patterns:
                for m in pat.finditer(text):
                    cand = candidate_from_match(row, m, label)
                    if cand is None:
                        continue
                    key = (cand["source_id"], cand["entity"].lower(), cand["birthplace"].lower(), cand["death_place"].lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(cand)
                    source_counts[str(row.get("source"))] += 1
        if max_rows and n_rows > max_rows:
            break
    summary = {"rows_seen": n_rows, "candidate_count": len(candidates), "source_counts": dict(source_counts)}
    return candidates, summary


def choose_new_value(c: Dict[str, Any], rng: random.Random) -> str:
    used = {c["birthplace"].lower(), c["death_place"].lower()}
    avail = [v for v in REPLACEMENT_CITIES if v.lower() not in used and v.lower() not in c["source_text"].lower()]
    return rng.choice(avail[:20] if len(avail) > 20 else avail)


def make_probe_rows(candidates: List[Dict[str, Any]], seed: int, max_maps: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    # Deduplicate by entity to avoid a single biography dominating.
    best = {}
    for c in candidates:
        k = c["entity"].lower()
        old = best.get(k)
        if old is None or len(c["source_text"]) < len(old["source_text"]):
            best[k] = c
    chosen = list(best.values())
    rng.shuffle(chosen)
    chosen = chosen[:max_maps]
    maps: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(chosen):
        new = choose_new_value(c, rng)
        split = "heldout" if i < max(1, round(len(chosen) * 0.25)) else "train"
        base_id = f"relcontrast_birth_death_{split}_{i:04d}"
        m = {**c, "base_pair_id": base_id, "shared_new_value": new, "split": split}
        maps.append(m)
        source = c["source_text"].strip()
        update_birth = f"However, recent records show that {c['entity']} was actually born in {new}."
        update_death = f"However, updated records confirm that {c['entity']} actually died in {new}."
        specs = [
            ("update_birthplace", update_birth, "birthplace", new, "death_place", c["death_place"]),
            ("update_death_place", update_death, "birthplace", c["birthplace"], "death_place", new),
        ]
        templates = {
            "birthplace": "According to this information, the birthplace of {entity} is {answer}.",
            "death_place": "According to this information, the place where {entity} died is {answer}.",
        }
        for assignment, update_sent, rel1, ans1, rel2, ans2 in specs:
            for query_rel, answer, role in [(rel1, ans1, "birthplace_query"), (rel2, ans2, "death_place_query")]:
                q = templates[query_rel].format(entity=c["entity"], answer=answer)
                context = source + " " + update_sent + " " + q
                a = context.find(answer, len(source) + len(update_sent))
                if a < 0:
                    raise RuntimeError(f"answer span missing {answer} in {context}")
                rows.append({
                    "row_id": f"{split}:{base_id}:{assignment}:{role}",
                    "base_pair_id": base_id,
                    "split": split,
                    "packet_type": "A01_RELATION_CONTRAST_BIRTH_DEATH_USE",
                    "entity": c["entity"],
                    "updated_relation": "birthplace" if assignment == "update_birthplace" else "death_place",
                    "query_relation": query_rel,
                    "answer_text": answer,
                    "answer_kind": "updated_new_value" if answer == new else "retained_source_value",
                    "source_text": source,
                    "update_sentence": update_sent,
                    "context_text": context,
                    "answer_char_start": a,
                    "answer_char_end": a + len(answer),
                    "source_birthplace": c["birthplace"],
                    "source_death_place": c["death_place"],
                    "shared_new_value": new,
                    "source_id": c["source_id"],
                    "extraction_pattern": c["extraction_pattern"],
                    "scientific_contract": "Same entity has two place-valued relations. Updating one relation should not change the other; distinguishes entity-only update gating from entity-relation-indexed state update.",
                    "audit_status": "candidate_needs_semantic_review_before_training",
                })
    return maps, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=66066)
    ap.add_argument("--max-maps", type=int, default=80)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cands, scan_summary = scan_rows(args.max_rows)
    maps, rows = make_probe_rows(cands, args.seed, args.max_maps)
    cand_path = args.out_dir / "candidate_birthplace_deathplace_same_entity.jsonl"
    maps_path = args.out_dir / "relation_contrast_operation_maps.jsonl"
    rows_path = args.out_dir / "relation_contrast_probe_rows.jsonl"
    sample_path = args.out_dir / "relation_contrast_sample.json"
    write_jsonl(cand_path, cands)
    write_jsonl(maps_path, maps)
    write_jsonl(rows_path, rows)
    sample_path.write_text(json.dumps(maps[:20], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    span_errors = []
    for r in rows:
        if r["context_text"][r["answer_char_start"]:r["answer_char_end"]] != r["answer_text"]:
            span_errors.append(r["row_id"])
    summary = {
        "status": "RELATION_CONTRAST_SCAN_DONE",
        "created_utc": now(),
        "scientific_purpose": "Find same-entity birthplace/death_place candidates for distinguishing entity-only update gates from entity-relation-indexed state updates.",
        "scan_summary": scan_summary,
        "candidate_path": rel(cand_path),
        "operation_maps_path": rel(maps_path),
        "probe_rows_path": rel(rows_path),
        "sample_path": rel(sample_path),
        "n_operation_maps": len(maps),
        "n_probe_rows": len(rows),
        "split_counts": dict(Counter(m["split"] for m in maps)),
        "answer_kind_counts": dict(Counter(r["answer_kind"] for r in rows)),
        "updated_relation_counts": dict(Counter(r["updated_relation"] for r in rows)),
        "query_relation_counts": dict(Counter(r["query_relation"] for r in rows)),
        "span_error_count": len(span_errors),
        "span_error_examples": span_errors[:20],
        "interpretation_boundary": "Candidate pool is automatically extracted and must be manually reviewed before any learner result is interpreted; absence or small count would indicate that BabyLM exact source may not support this contrast without new curated text.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
