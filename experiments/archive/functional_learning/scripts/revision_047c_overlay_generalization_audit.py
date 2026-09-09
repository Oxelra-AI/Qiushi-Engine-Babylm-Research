#!/usr/bin/env python3
"""Step047c: audit bridge overlay against relation readout generalization boundaries.

This CPU/string audit complements the corrected trainer.  It does not score a
model; it checks whether bridge-training relation rows overlap the held readout
by pair id/entity/value/template, whether row keys are unique, and how often the
supervised final answer is also visible elsewhere in the same relation packet.
The latter is not necessarily a flaw—these tasks are meant to learn contextual
selection from visible evidence—but it bounds interpretation as context-copying
with relation/query selection rather than closed-book fact acquisition.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import saved_state_replicate_neutral_threeentity as repl  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

OVERLAY = _public_path('experiments/archive/functional_learning/data/bridge_schedule_plan/overlay_tail_relation_e080_interspersed_wordpaced.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/overlay_generalization_audit')


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def count_occurrences_excluding_span(text: str, needle: str, span: Tuple[int, int]) -> int:
    if not needle:
        return 0
    n = 0
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        j = i + len(needle)
        if not (i == span[0] and j == span[1]):
            n += 1
        start = i + 1
    return n


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(OVERLAY)
    rel_rows = [r for r in rows if r.get("bridge_kind") == "relation_answer_packet"]
    ord_rows = [r for r in rows if r.get("bridge_kind") != "relation_answer_packet"]

    pairs, train_pairs, held_pairs, _train_rows, _held_rows = repl.fixed_repaired_pairs(40040)
    train_pair_ids = {p["pair_id"] for p in train_pairs}
    held_pair_ids = {p["pair_id"] for p in held_pairs}
    overlay_pair_ids = {r.get("pair_id") for r in rel_rows}
    train_entities = {p["entity_a"] for p in train_pairs} | {p["entity_b"] for p in train_pairs}
    held_entities = {p["entity_a"] for p in held_pairs} | {p["entity_b"] for p in held_pairs}
    train_values = {p["value_a"] for p in train_pairs} | {p["value_b"] for p in train_pairs} | {p.get("shared_new_value") for p in train_pairs}
    held_values = {p["value_a"] for p in held_pairs} | {p["value_b"] for p in held_pairs} | {p.get("shared_new_value") for p in held_pairs}

    key_counts = collections.Counter(tuple(bridge.row_key(r)) for r in rows)
    duplicate_keys = [{"row_key": list(k), "count": c} for k, c in key_counts.items() if c > 1]
    relation_key_counts = collections.Counter(tuple(bridge.row_key(r)) for r in rel_rows)
    relation_duplicate_keys = [{"row_key": list(k), "count": c} for k, c in relation_key_counts.items() if c > 1]
    ordinary_key_counts = collections.Counter(tuple(bridge.row_key(r)) for r in ord_rows)
    ordinary_duplicate_keys = [{"row_key": list(k), "count": c} for k, c in ordinary_key_counts.items() if c > 1]

    copy_records = []
    role_counts = collections.Counter()
    visible_counts_by_role = collections.Counter()
    for r in rel_rows:
        text = str(r.get("text", ""))
        ans = str(r.get("answer_text", ""))
        span = tuple(int(x) for x in r.get("answer_char_span", [-1, -1]))
        exact_ok = span[0] >= 0 and span[1] <= len(text) and text[span[0]:span[1]] == ans
        visible_occ = count_occurrences_excluding_span(text, ans, span) if exact_ok else None
        role = str(r.get("role"))
        role_counts[role] += 1
        if visible_occ and visible_occ > 0:
            visible_counts_by_role[role] += 1
        if len(copy_records) < 200 or (visible_occ is not None and visible_occ == 0):
            copy_records.append({
                "pair_id": r.get("pair_id"),
                "row_type": r.get("row_type"),
                "role": role,
                "relation_epoch": r.get("relation_epoch"),
                "answer_text": ans,
                "span_exact": exact_ok,
                "visible_occurrences_excluding_final_span": visible_occ,
                "text_prefix": text[:240],
            })

    held_fam = []
    for p in held_pairs:
        ents = [p["entity_a"], p["entity_b"]]
        vals = [p["value_a"], p["value_b"], p.get("shared_new_value")]
        held_fam.append({
            "pair_id": p["pair_id"],
            "relation": p["relation"],
            "entity_a": p["entity_a"],
            "entity_b": p["entity_b"],
            "entity_overlap_count": sum(1 for e in ents if e in train_entities),
            "entities_both_unseen": all(e not in train_entities for e in ents),
            "value_overlap_count": sum(1 for v in vals if v in train_values),
            "source_value_overlap_count": sum(1 for v in vals[:2] if v in train_values),
        })

    summary = {
        "status": "OVERLAY_GENERALIZATION_AUDIT_READY",
        "overlay": rel(OVERLAY),
        "total_rows": len(rows),
        "relation_rows": len(rel_rows),
        "ordinary_rows": len(ord_rows),
        "train_pairs": len(train_pairs),
        "held_pairs": len(held_pairs),
        "relation_pair_ids_in_overlay": len(overlay_pair_ids),
        "overlay_pair_id_intersection_with_held": sorted(list(overlay_pair_ids & held_pair_ids)),
        "overlay_pair_id_subset_of_train": bool(overlay_pair_ids <= train_pair_ids),
        "held_pair_entity_overlap_counts": dict(collections.Counter(r["entity_overlap_count"] for r in held_fam)),
        "held_pairs_both_entities_unseen": sum(1 for r in held_fam if r["entities_both_unseen"]),
        "held_source_value_overlap_counts": dict(collections.Counter(r["source_value_overlap_count"] for r in held_fam)),
        "held_shared_new_value_overlap_with_train": sum(1 for p in held_pairs if p.get("shared_new_value") in train_values),
        "train_entity_count": len(train_entities),
        "held_entity_count": len(held_entities),
        "entity_intersection_count": len(train_entities & held_entities),
        "train_value_count": len(train_values),
        "held_value_count": len(held_values),
        "value_intersection_count": len(train_values & held_values),
        "row_key_duplicate_count_all": len(duplicate_keys),
        "row_key_duplicate_count_relation": len(relation_duplicate_keys),
        "row_key_duplicate_count_ordinary": len(ordinary_duplicate_keys),
        "row_key_duplicate_examples": duplicate_keys[:20],
        "relation_role_counts": dict(role_counts),
        "relation_answer_visible_elsewhere_by_role": dict(visible_counts_by_role),
        "relation_answer_visible_elsewhere_total": sum(visible_counts_by_role.values()),
        "relation_span_exact_count": sum(1 for r in copy_records if r["span_exact"]),
        "interpretation": "Relation held readout is pair-disjoint from overlay training if intersection is empty, but entity/value overlap remains a boundary; answer labels are usually visible elsewhere in the packet by design, so success means contextual selection/copying from evidence, not closed-book fact acquisition.",
    }
    (_public_path('experiments/archive/functional_learning/data/overlay_generalization_audit/overlay_generalization_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(_public_path('experiments/archive/functional_learning/data/overlay_generalization_audit/held_pair_overlap_records.jsonl'), held_fam)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/overlay_generalization_audit/answer_copyability_sample.jsonl'), copy_records)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
