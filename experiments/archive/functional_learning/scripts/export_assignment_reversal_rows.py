#!/usr/bin/env python3
"""Export operation-preserving assignment-reversal rows.

The export uses train+held rows in the same A/B answer-slot contract as the
binding machinery. This export takes the semantically repaired
relation-first pairs and renders two assignment-specific two-row packets per base pair:

  update_a assignment: entity_a receives the shared new value, entity_b must retain
                       its source value.
  update_b assignment: entity_b receives the shared new value, entity_a must retain
                       its source value.

Each assignment packet has exactly two rows in the schema: an unchanged_entity row
(answer_kind=source_state) and an updated_entity row (answer_kind=new_state).  The two
assignments for the same base pair form an explicit assignment reversal using the same
source_context and the same shared_new_value.  Extra fields preserve the typed
relation and original entity/value map so stricter three-candidate scoring remains
possible outside the A/B scorer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_INPUT = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/fixed_repaired_pairs.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/assignment_reversal_export')

FRAMES = [
    {"frame_id": "f00_original_a01", "frame_split": "train_seen", "style": "original_a01_query"},
    {"frame_id": "f01_after_records", "frame_split": "train_seen", "style": "after_records"},
    {"frame_id": "f02_passage_indicates", "frame_split": "train_seen", "style": "evidence_report"},
    {"frame_id": "f03_using_details", "frame_split": "eval_unseen", "style": "using_details"},
]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def wc(text: str) -> int:
    return len((text or "").strip().split())


def template_for(relation: str, frame_id: str) -> str:
    if relation == "birthplace":
        if frame_id == "f00_original_a01":
            return "According to this information, the birthplace of {entity} is {answer}."
        if frame_id == "f01_after_records":
            return "After these records, {entity} was born in {answer}."
        if frame_id == "f02_passage_indicates":
            return "The passage indicates that {entity}'s birthplace is {answer}."
        if frame_id == "f03_using_details":
            return "Using the details above, {entity} was born in {answer}."
    if relation == "death_place":
        if frame_id == "f00_original_a01":
            return "According to this information, the place where {entity} died is {answer}."
        if frame_id == "f01_after_records":
            return "After these records, {entity} died in {answer}."
        if frame_id == "f02_passage_indicates":
            return "The passage indicates that the place where {entity} died is {answer}."
        if frame_id == "f03_using_details":
            return "Using the details above, {entity}'s death place is {answer}."
    # Fallback keeps the slot type intentionally generic but still answer-slot compatible.
    return "According to this information, the relevant {relation} value for {entity} is {answer}."


def render_query(relation: str, frame_id: str, entity: str, answer: str) -> Tuple[str, int, int]:
    tmpl = template_for(relation, frame_id)
    if "{relation}" in tmpl:
        text = tmpl.format(relation=relation.replace("_", " "), entity=entity, answer=answer)
    else:
        text = tmpl.format(entity=entity, answer=answer)
    a = text.find(answer)
    if a < 0:
        raise RuntimeError(f"answer {answer!r} not found in rendered query {text!r}")
    return text, a, a + len(answer)


def assignment_specs(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    # In the A/B contract, entity_a means the unchanged row's queried entity and
    # entity_b means the updated row's queried entity.  These are assignment-local.
    return [
        {
            "assignment": "update_a",
            "updated_original_side": "entity_a",
            "unchanged_original_side": "entity_b",
            "updated_entity": pair["entity_a"],
            "unchanged_entity": pair["entity_b"],
            "update_sentence": pair["update_a_sentence"],
            "source_state": pair["value_b"],
            "new_state": pair["shared_new_value"],
            "wrong_other_source": pair["value_a"],
        },
        {
            "assignment": "update_b",
            "updated_original_side": "entity_b",
            "unchanged_original_side": "entity_a",
            "updated_entity": pair["entity_b"],
            "unchanged_entity": pair["entity_a"],
            "update_sentence": pair["update_b_sentence"],
            "source_state": pair["value_a"],
            "new_state": pair["shared_new_value"],
            "wrong_other_source": pair["value_b"],
        },
    ]


def make_two_rows(pair: Dict[str, Any], spec: Dict[str, Any], frame: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split_in = pair.get("split", "")
    split = "heldout" if split_in == "held" else str(split_in or "train")
    base_id = pair["pair_id"]
    pair_id = f"{base_id}::{spec['assignment']}::{frame['frame_id']}"
    source_context = pair["source_context"].strip()
    update_sentence = spec["update_sentence"].strip()
    context_prefix = source_context + " " + update_sentence

    rows = []
    for half, role, query_entity, answer_kind, answer in [
        ("A", "unchanged_entity", spec["unchanged_entity"], "source_state", spec["source_state"]),
        ("B", "updated_entity", spec["updated_entity"], "new_state", spec["new_state"]),
    ]:
        query, qa0, qb0 = render_query(pair["relation"], frame["frame_id"], query_entity, answer)
        context_text = context_prefix + " " + query
        answer_char_start = len(context_prefix) + 1 + qa0
        answer_char_end = len(context_prefix) + 1 + qb0
        if context_text[answer_char_start:answer_char_end] != answer:
            raise RuntimeError("answer span mismatch")
        rows.append({
            "row_id": f"{split}:{pair_id}:{role}",
            "pair_id": pair_id,
            "split": split,
            "packet_type": "A01_ASSIGNMENT_REVERSAL_USE",
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
            "training_contract": "assignment-reversal A/B answer-slot: same source_context and shared_new value; two rows per assignment; mask/supervise answer_text tokens only.",
            "base_pair_id": base_id,
            "assignment": spec["assignment"],
            "relation": pair["relation"],
            "frame_id": frame["frame_id"],
            "frame_split": frame["frame_split"],
            "frame_style": frame["style"],
            "query_template": template_for(pair["relation"], frame["frame_id"]),
            "original_entity_a": pair["entity_a"],
            "original_entity_b": pair["entity_b"],
            "original_value_a": pair["value_a"],
            "original_value_b": pair["value_b"],
            "shared_new_value": pair["shared_new_value"],
            "wrong_other_source": spec["wrong_other_source"],
            "original_split": split_in,
        })
    binding_pair = {
        "pair_id": pair_id,
        "row_a_id": rows[0]["row_id"],
        "row_b_id": rows[1]["row_id"],
        "entity_a": rows[0]["query_entity"],
        "entity_b": rows[1]["query_entity"],
        "answer_a": rows[0]["answer_text"],
        "answer_b": rows[1]["answer_text"],
        "base_pair_id": base_id,
        "assignment": spec["assignment"],
        "relation": pair["relation"],
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
    for r in rows:
        group[r["pair_id"]].append(r)
        if r["context_text"][int(r["answer_char_start"]):int(r["answer_char_end"])] != r["answer_text"]:
            span_errors.append(r["row_id"])
    bad_groups = {pid: [r.get("role") for r in rs] for pid, rs in group.items() if len(rs) != 2 or sorted(r.get("role") for r in rs) != ["unchanged_entity", "updated_entity"]}
    return {
        "rows": len(rows),
        "pairs": len(pairs),
        "duplicate_row_ids": len(row_ids) - len(set(row_ids)),
        "duplicate_pair_ids": len(pids) - len(set(pids)),
        "span_errors": span_errors[:20],
        "n_span_errors": len(span_errors),
        "bad_pair_groups": dict(list(bad_groups.items())[:20]),
        "n_bad_pair_groups": len(bad_groups),
        "row_words": sum(wc(r["context_text"]) for r in rows),
        "by_relation": dict(Counter(r.get("relation") for r in rows)),
        "by_frame": dict(Counter(r.get("frame_id") for r in rows)),
        "by_assignment": dict(Counter(r.get("assignment") for r in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_pairs = load_jsonl(args.input)
    all_rows: List[Dict[str, Any]] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}
    for pair in base_pairs:
        for spec in assignment_specs(pair):
            for frame in FRAMES:
                rows, bp = make_two_rows(pair, spec, frame)
                all_rows.extend(rows)
                pair_meta[bp["pair_id"]] = bp

    # Main files: original single-frame and frame-varied variants.
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
        row_path = args.out_dir / f"a01_assignment_reversal_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        write_jsonl(row_path, rows)
        write_jsonl(pair_path, pairs)
        chk = static_check(rows, pairs)
        checks[name] = chk
        output_files[name] = {
            "rows_path": rel(row_path),
            "binding_pairs_path": rel(pair_path),
            "rows": len(rows),
            "pairs": len(pairs),
        }

    # Also export a compact operation map without frame expansion.
    operation_maps = []
    for p in base_pairs:
        operation_maps.append({
            "base_pair_id": p["pair_id"],
            "split": "heldout" if p.get("split") == "held" else p.get("split"),
            "relation": p["relation"],
            "source_context": p["source_context"],
            "entity_a": p["entity_a"],
            "entity_b": p["entity_b"],
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "shared_new_value": p["shared_new_value"],
            "update_a_sentence": p["update_a_sentence"],
            "update_b_sentence": p["update_b_sentence"],
            "interpretation": "update_a and update_b are assignment reversals with the same source_context and shared new value; correct retained source depends on queried entity.",
        })
    operation_map_path = args.out_dir / "a01_assignment_reversal_operation_maps.jsonl"
    write_jsonl(operation_map_path, operation_maps)

    summary = {
        "status": "A01_ASSIGNMENT_REVERSAL_EXPORT_READY",
        "created_utc": now(),
        "scientific_purpose": "Provide semantically repaired assignment-reversal rows in the A/B answer-slot schema, while preserving typed relation metadata for stricter controls.",
        "source_pairs": rel(args.input),
        "n_base_pairs": len(base_pairs),
        "base_split_counts": dict(Counter("heldout" if p.get("split") == "held" else p.get("split") for p in base_pairs)),
        "frames": FRAMES,
        "outputs": output_files,
        "operation_maps": rel(operation_map_path),
        "static_checks": checks,
        "contract": {
            "row_schema": "A/B answer-slot: exactly two rows per assignment-specific pair_id; pair_half A is unchanged/source_state, pair_half B is updated/new_state.",
            "assignment_reversal": "Each base pair produces update_a and update_b packets with the same source_context and shared_new_value but opposite updated entity.",
            "not_by_itself": "This export enables matched training/evaluation; broad BabyLM improvement still requires paired training, no-context/assignment controls, and broad official-compatible evaluation.",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
