#!/usr/bin/env python3
"""research: audit the larger assignment-reversal export before scientific use.

The export provides a large controlled map pool. This audit
quantifies what the pool does and does not provide: relation counts, source/entity
reuse, train-held leakage, source sentence multiplicity, answer-span integrity, and
examples from each relation.  It is an evidence check, not a learner experiment.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
DEFAULT_DIR = _public_path('experiments/archive/functional_learning/data/larger_assignment_reversal_export')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_counter(c: Counter, n: int = 20) -> List[Dict[str, Any]]:
    return [{"key": k, "count": v} for k, v in c.most_common(n)]


def value_type(relation: str) -> str:
    if relation in {"birth_year", "founded_year"}:
        return "year"
    if relation in {"birthplace", "death_place", "located_in"}:
        return "place_like_literal"
    if relation == "nationality":
        return "adjectival_nationality"
    if relation == "occupation":
        return "occupation_noun"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=pathlib.Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    export_dir = args.export_dir

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    maps = load_jsonl(export_dir / "a01_larger_assignment_reversal_operation_maps.jsonl")
    all_rows = []
    for split in ["train_frame_all", "heldout_frame_all"]:
        path = pathlib.Path(manifest["outputs"][split]["rows_path"])
        all_rows.extend(load_jsonl(ROOT / path if not path.is_absolute() else path))

    # Base-map-level reuse and split overlap.
    source_ids_by_split = defaultdict(set)
    entity_by_split = defaultdict(set)
    value_by_split = defaultdict(set)
    map_by_rel_split = defaultdict(Counter)
    source_pair_counter = Counter()
    source_id_counter = Counter()
    entity_counter = Counter()
    value_counter = Counter()
    shared_counter = Counter()
    source_sentence_counter = Counter()
    provenance_counter = Counter()
    map_span_source_errors = []
    update_assignment_errors = []
    for m in maps:
        split = m["split"]
        r = m["relation"]
        map_by_rel_split[split][r] += 1
        ids = [m.get("source_a_id"), m.get("source_b_id")]
        ents = [m.get("entity_a"), m.get("entity_b")]
        vals = [m.get("value_a"), m.get("value_b")]
        source_ids_by_split[split].update(x for x in ids if x)
        entity_by_split[split].update(str(x).lower() for x in ents if x)
        value_by_split[split].update(str(x).lower() for x in vals if x)
        source_pair_counter[tuple(sorted(str(x) for x in ids))] += 1
        for sid in ids:
            source_id_counter[sid] += 1
        for ent in ents:
            entity_counter[str(ent).lower()] += 1
        for val in vals:
            value_counter[str(val).lower()] += 1
        shared_counter[(r, m.get("shared_new_value"))] += 1
        source_sentence_counter[m.get("source_a_sentence", "")] += 1
        source_sentence_counter[m.get("source_b_sentence", "")] += 1
        provenance_counter[(m.get("triple_provenance_a"), r)] += 1
        provenance_counter[(m.get("triple_provenance_b"), r)] += 1
        # Check declared literal values occur in corresponding exact source sentences.
        if str(m.get("value_a")) not in str(m.get("source_a_sentence")):
            map_span_source_errors.append({"base_pair_id": m["base_pair_id"], "side": "a", "value": m.get("value_a"), "sentence": m.get("source_a_sentence")})
        if str(m.get("value_b")) not in str(m.get("source_b_sentence")):
            map_span_source_errors.append({"base_pair_id": m["base_pair_id"], "side": "b", "value": m.get("value_b"), "sentence": m.get("source_b_sentence")})
        if m.get("entity_a") not in m.get("update_a_sentence", "") or m.get("shared_new_value") not in m.get("update_a_sentence", ""):
            update_assignment_errors.append({"base_pair_id": m["base_pair_id"], "assignment": "update_a"})
        if m.get("entity_b") not in m.get("update_b_sentence", "") or m.get("shared_new_value") not in m.get("update_b_sentence", ""):
            update_assignment_errors.append({"base_pair_id": m["base_pair_id"], "assignment": "update_b"})

    row_span_errors = []
    pair_roles = defaultdict(list)
    pair_answers = defaultdict(list)
    for r in all_rows:
        pair_roles[r["pair_id"]].append(r["role"])
        pair_answers[r["pair_id"]].append((r["answer_kind"], r["answer_text"], r["query_entity"]))
        a, b = int(r["answer_char_start"]), int(r["answer_char_end"])
        if r["context_text"][a:b] != r["answer_text"]:
            row_span_errors.append(r["row_id"])
    malformed_pairs = {pid: roles for pid, roles in pair_roles.items() if sorted(roles) != ["unchanged_entity", "updated_entity"]}
    same_answer_pairs = {pid: ans for pid, ans in pair_answers.items() if len(ans) == 2 and ans[0][1] == ans[1][1]}

    train_sources = source_ids_by_split.get("train", set())
    held_sources = source_ids_by_split.get("heldout", set())
    train_entities = entity_by_split.get("train", set())
    held_entities = entity_by_split.get("heldout", set())
    train_values = value_by_split.get("train", set())
    held_values = value_by_split.get("heldout", set())

    examples_by_relation = {}
    for relation in sorted({m["relation"] for m in maps}):
        examples_by_relation[relation] = [m for m in maps if m["relation"] == relation][:2]

    relation_quality_notes = {
        "birth_year": "Literal years are clean values, but some extracted entities are surnames or abbreviated page titles; useful for numeric assignment reversal, not entity-disjoint natural biography evidence.",
        "birthplace": "Mostly exact born-in location literals from the narrower extractor; source sentences can contain additional locations that make provenance inspection important.",
        "death_place": "Largest high-confidence relation inherited from the repaired natural specialist, but source/entity reuse is heavy because many pair combinations are formed from finite triples.",
        "founded_year": "Small but type-consistent year-valued set from exact source strings.",
        "located_in": "Very small and place-like; values may be administrative subspans such as counties/provinces, so it is best used as controlled type variety rather than a clean geography fact benchmark.",
        "nationality": "Small broad-filtered adjective set; some values are demonyms/ethnic affiliations. Treat as additional literal type pressure, not robust nationality extraction.",
        "occupation": "Small broad-filtered noun set; useful as non-place/non-year literal-value type but not enough for relation-disjoint conclusions.",
    }

    summary = {
        "status": "LARGER_ASSIGNMENT_REVERSAL_EXPORT_AUDITED",
        "export_dir": rel(export_dir),
        "manifest_status": manifest.get("status"),
        "n_base_maps": len(maps),
        "base_split_counts": dict(Counter(m["split"] for m in maps)),
        "base_relation_counts": dict(Counter(m["relation"] for m in maps)),
        "base_split_relation_counts": {split: dict(map_by_rel_split[split]) for split in sorted(map_by_rel_split)},
        "row_outputs": {k: v for k, v in manifest.get("outputs", {}).items()},
        "static_check_fatal_counts": {
            name: chk.get("duplicate_row_ids", 0) + chk.get("duplicate_pair_ids", 0) + chk.get("n_span_errors", 0) + chk.get("n_bad_pair_groups", 0)
            for name, chk in manifest.get("static_checks", {}).items()
        },
        "map_literal_value_source_errors": len(map_span_source_errors),
        "map_literal_value_source_error_examples": map_span_source_errors[:10],
        "update_assignment_errors": len(update_assignment_errors),
        "update_assignment_error_examples": update_assignment_errors[:10],
        "row_span_errors": len(row_span_errors),
        "malformed_two_row_pairs": len(malformed_pairs),
        "same_answer_pairs": len(same_answer_pairs),
        "unique_source_ids": len(source_id_counter),
        "unique_entities_lower": len(entity_counter),
        "unique_source_values_lower": len(value_counter),
        "unique_shared_replacements_by_relation": len(shared_counter),
        "train_held_overlap": {
            "source_ids": len(train_sources & held_sources),
            "entities_lower": len(train_entities & held_entities),
            "source_values_lower": len(train_values & held_values),
            "source_id_examples": sorted(list(train_sources & held_sources))[:20],
            "entity_examples": sorted(list(train_entities & held_entities))[:20],
            "value_examples": sorted(list(train_values & held_values))[:20],
        },
        "reuse_profile": {
            "top_source_ids": top_counter(source_id_counter, 15),
            "top_entities_lower": top_counter(entity_counter, 15),
            "top_source_values_lower": top_counter(value_counter, 15),
            "top_source_pairs": top_counter(source_pair_counter, 15),
            "max_maps_per_single_source_id": max(source_id_counter.values(), default=0),
            "max_maps_per_entity_lower": max(entity_counter.values(), default=0),
        },
        "relation_value_type": {r: value_type(r) for r in sorted(Counter(m["relation"] for m in maps))},
        "triple_provenance_counts": {f"{prov}|{r}": c for (prov, r), c in sorted(provenance_counter.items())},
        "relation_quality_notes": relation_quality_notes,
        "examples_by_relation": examples_by_relation,
        "interpretation": (
            "The export satisfies the requested scale and static A/B answer-span contract, but it is not an entity/source-disjoint benchmark: "
            "516 base maps are generated as many pair combinations from 309 filtered triples, with train-held source/entity overlap and repeated high-degree triples. "
            "Use it to stress whether literal-value assignment reversal can be learned across relation/value types and frames; do not interpret heldout success as fully novel entity generalization unless a stricter source-disjoint split is built."
        ),
    }
    out = export_dir / "export_audit.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if any(summary["static_check_fatal_counts"].values()) or row_span_errors or malformed_pairs or same_answer_pairs or map_span_source_errors or update_assignment_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
