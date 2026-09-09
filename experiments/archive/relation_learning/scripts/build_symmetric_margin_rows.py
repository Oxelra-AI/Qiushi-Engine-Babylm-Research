#!/usr/bin/env python3
"""research: build a shortcut-symmetric local row family for margin-based binding pilots.

Purpose
-------
The single-frame and frame-varied binding rows failed in official Entity because the
training family let cheaper policies absorb answer credit: every row contained an
operation, B/update answers were easy local copies from the update sentence, and entity
position was not fully neutralized.  This script builds a local pilot row family whose
feature table is explicit before any GPU pilot:

  * operation count 1--4 has exactly half source-retention rows and half updated rows;
  * entity_a and entity_b are equally represented as the queried entity in both roles;
  * relevant-update position is balanced across first/middle/last for updated rows;
  * retention rows contain 0--4 distractor operations while the queried entity is untouched;
  * paired margin records compare the same source-vs-new candidate margin on a retention
    row and an updated row, so an identity-blind global source/new shift moves both
    margins together.

The data source is the semantically repaired assignment-reversal export.  It is small
(90 train, 30 held base maps) but has the exact two-entity source values and mirrored
update_a/update_b assignments needed to test the objective without spending official eval.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
DEFAULT_MAPS = _public_path('experiments/archive/functional_learning/data/assignment_reversal_export/a01_assignment_reversal_operation_maps.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/symmetric_margin_rows')

FRAMES = [
    {"frame_id": "f00_original_a01", "frame_split": "train_seen", "style": "original_a01_query"},
    {"frame_id": "f01_after_records", "frame_split": "train_seen", "style": "after_records"},
    {"frame_id": "f02_passage_indicates", "frame_split": "train_seen", "style": "evidence_report"},
    {"frame_id": "f03_using_details", "frame_split": "eval_unseen", "style": "using_details"},
]
SIDES = ["a", "b"]
OP_COUNTS_FOR_PAIRS = [1, 2, 3, 4]
RETENTION_OP_COUNTS = [0, 1, 2, 3, 4]

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def wc(text: str) -> int:
    return len((text or "").strip().split())


def norm_lite(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def toks(s: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(s or "") if len(m.group(0)) > 2}


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
    return "According to this information, the relevant {relation} value for {entity} is {answer}."


def render_query(relation: str, frame_id: str, entity: str, answer: str) -> tuple[str, int, int, str]:
    tmpl = template_for(relation, frame_id)
    if "{relation}" in tmpl:
        text = tmpl.format(relation=relation.replace("_", " "), entity=entity, answer=answer)
    else:
        text = tmpl.format(entity=entity, answer=answer)
    a = text.find(answer)
    if a < 0:
        raise RuntimeError(f"answer {answer!r} not found in {text!r}")
    return text, a, a + len(answer), tmpl


def side_fields(m: dict[str, Any], side: str) -> dict[str, str]:
    assert side in ("a", "b")
    return {
        "entity": str(m[f"entity_{side}"]),
        "source_value": str(m[f"value_{side}"]),
        "relevant_update": str(m[f"update_{side}_sentence"]),
        "other_side": "b" if side == "a" else "a",
        "other_entity": str(m["entity_b" if side == "a" else "entity_a"]),
        "other_source_value": str(m["value_b" if side == "a" else "value_a"]),
    }


def candidate_distractors(maps: list[dict[str, Any]], idx: int, side: str, needed: int) -> list[dict[str, Any]]:
    """Choose deterministic unrelated update sentences that avoid the queried entity/value tokens."""
    base = maps[idx]
    sf = side_fields(base, side)
    avoid = toks(sf["entity"]) | toks(sf["source_value"]) | toks(str(base["shared_new_value"]))
    candidates = []
    n = len(maps)
    # Use a stride co-prime-ish with typical n to avoid repeatedly sampling neighbors.
    stride = 17 if n % 17 else 19
    for step in range(1, n * 2 + 1):
        j = (idx + step * stride + (0 if side == "a" else 7)) % n
        if j == idx:
            continue
        other = maps[j]
        for upd_side in ("a", "b"):
            sent = str(other[f"update_{upd_side}_sentence"])
            # Avoid accidental direct mention of the queried entity/source answer/new candidate.
            if avoid and (avoid & toks(sent)):
                continue
            candidates.append({
                "base_pair_id": other["base_pair_id"],
                "side": upd_side,
                "sentence": sent,
                "updated_entity": str(other[f"entity_{upd_side}"]),
                "new_value": str(other["shared_new_value"]),
            })
            if len(candidates) >= needed:
                return candidates
    # If strict avoidance cannot supply enough, fall back deterministically but mark it.
    for step in range(1, n * 2 + 1):
        j = (idx + step * 13 + (3 if side == "b" else 0)) % n
        if j == idx:
            continue
        other = maps[j]
        for upd_side in ("a", "b"):
            candidates.append({
                "base_pair_id": other["base_pair_id"],
                "side": upd_side,
                "sentence": str(other[f"update_{upd_side}_sentence"]),
                "updated_entity": str(other[f"entity_{upd_side}"]),
                "new_value": str(other["shared_new_value"]),
                "fallback_possible_overlap": True,
            })
            if len(candidates) >= needed:
                return candidates
    raise RuntimeError("not enough distractors")


def make_context(source_context: str, ops: list[str], query: str) -> tuple[str, int]:
    prefix_parts = [source_context.strip()] + [o.strip() for o in ops if o.strip()]
    prefix = " ".join(prefix_parts).strip()
    context = prefix + " " + query
    query_start = len(prefix) + 1
    return context, query_start


def make_row(m: dict[str, Any], idx: int, split: str, side: str, frame: dict[str, str], role: str,
             op_count: int, ops: list[dict[str, Any]], relevant_position: int | None) -> dict[str, Any]:
    sf = side_fields(m, side)
    source_value = sf["source_value"]
    new_value = str(m["shared_new_value"])
    answer = source_value if role == "retention_source" else new_value
    answer_kind = "source_state" if role == "retention_source" else "new_state"
    query, qa, qb, tmpl = render_query(str(m["relation"]), frame["frame_id"], sf["entity"], answer)
    op_sents = [o["sentence"] for o in ops]
    context, qstart = make_context(str(m["source_context"]), op_sents, query)
    a0 = qstart + qa
    b0 = qstart + qb
    if context[a0:b0] != answer:
        raise RuntimeError(f"span mismatch {context[a0:b0]!r} != {answer!r}")
    row_id = f"{split}:{m['base_pair_id']}::{side}::k{op_count}::{role}::{frame['frame_id']}"
    return {
        "row_id": row_id,
        "pair_id": f"{m['base_pair_id']}::{side}::k{op_count}::{frame['frame_id']}",
        "split": split,
        "packet_type": "SYMMETRIC_MARGIN_BINDING",
        "query_entity": sf["entity"],
        "query_side": side,
        "answer_text": answer,
        "answer_kind": answer_kind,
        "role": "unchanged_entity" if role == "retention_source" else "updated_entity",
        "pair_half": "A" if role == "retention_source" else "B",
        "source_sentence": str(m["source_context"]),
        "update_sentence": " ".join(op_sents),
        "source_state": source_value,
        "new_state": new_value,
        "target_entity": sf["entity"],
        "updated_entity": sf["entity"] if role == "updated_target" else "",
        "context_text": context,
        "answer_char_start": a0,
        "answer_char_end": b0,
        "base_pair_id": m["base_pair_id"],
        "relation": m["relation"],
        "frame_id": frame["frame_id"],
        "frame_split": frame["frame_split"],
        "frame_style": frame["style"],
        "query_template": tmpl,
        "original_entity_a": m["entity_a"],
        "original_entity_b": m["entity_b"],
        "original_value_a": m["value_a"],
        "original_value_b": m["value_b"],
        "shared_new_value": new_value,
        "operation_count": op_count,
        "operation_role": "retention_distractors_only" if role == "retention_source" else "target_update_plus_distractors",
        "relevant_update_position_0based": relevant_position,
        "distractor_update_count": sum(1 for o in ops if not o.get("is_relevant")),
        "operation_specs": ops,
        "context_words": wc(context),
        "training_contract": "paired source-vs-new margin: A/retention must prefer source_state, B/updated must prefer new_state; op_count and query side balanced.",
    }


def build_for_split(maps: list[dict[str, Any]], split: str, use_frames: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for idx, m in enumerate(maps):
        for frame_idx, frame in enumerate(use_frames):
            for side in SIDES:
                sf = side_fields(m, side)
                # Standalone retention rows with zero operations are useful for measuring the desired flatness at op_count=0.
                if 0 in RETENTION_OP_COUNTS:
                    r0 = make_row(m, idx, split, side, frame, "retention_source", 0, [], None)
                    rows.append(r0)
                for k in OP_COUNTS_FOR_PAIRS:
                    # Retention row: queried entity untouched; all operations are distractors.
                    ret_ops = candidate_distractors(maps, idx, side, k)
                    for o in ret_ops:
                        o["is_relevant"] = False
                    row_a = make_row(m, idx, split, side, frame, "retention_source", k, ret_ops, None)
                    # Updated row: queried entity updated, plus k-1 distractors; relevant position balanced.
                    dist = candidate_distractors(maps, idx, side, max(0, k - 1))
                    for o in dist:
                        o["is_relevant"] = False
                    rel_op = {
                        "base_pair_id": m["base_pair_id"],
                        "side": side,
                        "sentence": sf["relevant_update"],
                        "updated_entity": sf["entity"],
                        "new_value": str(m["shared_new_value"]),
                        "is_relevant": True,
                    }
                    pos = (idx + (0 if side == "a" else 1) + frame_idx) % k
                    upd_ops = list(dist)
                    upd_ops.insert(pos, rel_op)
                    row_b = make_row(m, idx, split, side, frame, "updated_target", k, upd_ops, pos)
                    rows.extend([row_a, row_b])
                    pairs.append({
                        "pair_id": row_a["pair_id"],
                        "row_a_id": row_a["row_id"],
                        "row_b_id": row_b["row_id"],
                        "entity_a": row_a["query_entity"],
                        "entity_b": row_b["query_entity"],
                        "answer_a": row_a["answer_text"],
                        "answer_b": row_b["answer_text"],
                        "source_state": row_a["source_state"],
                        "new_state": row_a["new_state"],
                        "base_pair_id": m["base_pair_id"],
                        "query_side": side,
                        "relation": m["relation"],
                        "frame_id": frame["frame_id"],
                        "frame_split": frame["frame_split"],
                        "operation_count": k,
                        "relevant_update_position_0based": pos,
                    })
    return rows, pairs


def first_mention_side(row: dict[str, Any]) -> str | None:
    src = str(row["source_sentence"])
    ia = src.find(str(row["original_entity_a"]))
    ib = src.find(str(row["original_entity_b"]))
    if ia < 0 or ib < 0:
        return None
    return "a" if ia < ib else "b"


def feature_audit(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    def p_source(rs: list[dict[str, Any]]) -> float | None:
        if not rs:
            return None
        return sum(1 for r in rs if r["answer_kind"] == "source_state") / len(rs)
    by_op = {}
    for k in sorted({int(r["operation_count"]) for r in rows}):
        rs = [r for r in rows if int(r["operation_count"]) == k]
        by_op[str(k)] = {"n": len(rs), "p_source": p_source(rs), "roles": dict(Counter(r["answer_kind"] for r in rs))}
    by_op_side = {}
    for k in sorted({int(r["operation_count"]) for r in rows}):
        for side in SIDES:
            rs = [r for r in rows if int(r["operation_count"]) == k and r["query_side"] == side]
            by_op_side[f"k{k}_{side}"] = {"n": len(rs), "p_source": p_source(rs), "roles": dict(Counter(r["answer_kind"] for r in rs))}
    op_present = [r for r in rows if int(r["operation_count"]) > 0]
    by_query_first = {}
    for qfirst in [True, False]:
        rs = []
        for r in op_present:
            fm = first_mention_side(r)
            if fm is None:
                continue
            rs.append(r) if ((r["query_side"] == fm) == qfirst) else None
        by_query_first[str(qfirst)] = {"n": len(rs), "p_source": p_source(rs), "roles": dict(Counter(r["answer_kind"] for r in rs))}
    rel_pos = Counter()
    for p in pairs:
        k = int(p["operation_count"])
        pos = int(p["relevant_update_position_0based"])
        rel_pos[f"k{k}_pos{pos}"] += 1
    span_errors = [r["row_id"] for r in rows if r["context_text"][int(r["answer_char_start"]):int(r["answer_char_end"])] != r["answer_text"]]
    pair_ids = [p["pair_id"] for p in pairs]
    row_ids = [r["row_id"] for r in rows]
    rows_by_id = {r["row_id"]: r for r in rows}
    bad_pairs = []
    for p in pairs:
        a = rows_by_id.get(p["row_a_id"]); b = rows_by_id.get(p["row_b_id"])
        if not a or not b or a["answer_kind"] != "source_state" or b["answer_kind"] != "new_state":
            bad_pairs.append(p["pair_id"])
    answer_overlap_counts = {"source_answer_in_ops": 0, "new_answer_in_ops": 0, "source_rows": 0, "new_rows": 0}
    for r in rows:
        ops = str(r.get("update_sentence", "")).lower()
        ans = str(r["answer_text"]).lower()
        if r["answer_kind"] == "source_state":
            answer_overlap_counts["source_rows"] += 1
            if ans and ans in ops:
                answer_overlap_counts["source_answer_in_ops"] += 1
        else:
            answer_overlap_counts["new_rows"] += 1
            if ans and ans in ops:
                answer_overlap_counts["new_answer_in_ops"] += 1
    return {
        "rows": len(rows),
        "pairs": len(pairs),
        "row_words": sum(int(r["context_words"]) for r in rows),
        "duplicate_row_ids": len(row_ids) - len(set(row_ids)),
        "duplicate_pair_ids": len(pair_ids) - len(set(pair_ids)),
        "span_errors_count": len(span_errors),
        "span_errors_first": span_errors[:10],
        "bad_pairs_count": len(bad_pairs),
        "bad_pairs_first": bad_pairs[:10],
        "by_answer_kind": dict(Counter(r["answer_kind"] for r in rows)),
        "by_operation_count": by_op,
        "by_operation_count_and_query_side": by_op_side,
        "op_present_by_query_is_first_mentioned": by_query_first,
        "relevant_update_position_counts": dict(rel_pos),
        "by_relation": dict(Counter(r["relation"] for r in rows)),
        "by_frame": dict(Counter(r["frame_id"] for r in rows)),
        "answer_string_literal_overlap_with_operation_sentence": answer_overlap_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", type=pathlib.Path, default=DEFAULT_MAPS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_maps_raw = load_jsonl(args.maps)
    # The operation map uses split "heldout" for held rows; keep train/held separated.
    train_maps = [m for m in all_maps_raw if str(m.get("split")) == "train"]
    held_maps = [m for m in all_maps_raw if str(m.get("split")) == "heldout"]
    if not train_maps or not held_maps:
        raise RuntimeError(f"unexpected split counts {Counter(m.get('split') for m in all_maps_raw)}")
    # Preserve the field name expected by this script.
    for m in all_maps_raw:
        if "base_pair_id" not in m and "pair_id" in m:
            m["base_pair_id"] = m["pair_id"]

    frame_seen = [f for f in FRAMES if f["frame_split"] == "train_seen"]
    frame_all = FRAMES
    train_rows_seen, train_pairs_seen = build_for_split(train_maps, "train", frame_seen)
    train_rows_all, train_pairs_all = build_for_split(train_maps, "train", frame_all)
    held_rows_seen, held_pairs_seen = build_for_split(held_maps, "heldout", frame_seen)
    held_rows_all, held_pairs_all = build_for_split(held_maps, "heldout", frame_all)

    outputs = {}
    for name, rows, pairs in [
        ("train_frame_seen", train_rows_seen, train_pairs_seen),
        ("train_frame_all", train_rows_all, train_pairs_all),
        ("heldout_frame_seen", held_rows_seen, held_pairs_seen),
        ("heldout_frame_all", held_rows_all, held_pairs_all),
    ]:
        row_path = args.out_dir / f"symmetric_margin_{name}.jsonl"
        pair_path = args.out_dir / f"binding_pairs_{name}.jsonl"
        row_sha = write_jsonl(row_path, rows)
        pair_sha = write_jsonl(pair_path, pairs)
        outputs[name] = {
            "rows_path": rel(row_path),
            "pairs_path": rel(pair_path),
            "rows": len(rows),
            "pairs": len(pairs),
            "row_words": sum(int(r["context_words"]) for r in rows),
            "rows_sha256": row_sha,
            "pairs_sha256": pair_sha,
            "audit": feature_audit(rows, pairs),
        }

    manifest = {
        "status": "SYMMETRIC_MARGIN_ROWS_BUILT",
        "created_utc": now(),
        "source_operation_maps": rel(args.maps),
        "source_map_counts": dict(Counter(m.get("split") for m in all_maps_raw)),
        "scientific_reading": {
            "primary_failure_being_repaired": "A training family with one operation in every row plus answer-only credit let an anti-source/update-prior movement absorb credit when identity gating failed.",
            "objective_link": "The paired files support m_A - m_B, where m is logP(source_state)-logP(new_state); identity-blind global source/new shifts move both rows together and should not be rewarded by the pair term.",
            "not_a_broad_endpoint": "This is a local pilot asset. Passing the local readouts is required before any official Entity/cheap7 spend.",
        },
        "frames": FRAMES,
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable summary with the most important balance numbers.
    lines = [
        "# research symmetric margin row family",
        "",
        "This file family repairs the binding-row design for local pilots: operation-count and entity-position features are balanced for op_count 1--4, while retention rows with 0--4 distractor operations make source retention explicit under operations.",
        "",
    ]
    for name in ["train_frame_seen", "heldout_frame_all"]:
        audit = outputs[name]["audit"]
        lines.append(f"## {name}")
        lines.append(f"Rows {audit['rows']}; pairs {audit['pairs']}; row words {audit['row_words']}; span errors {audit['span_errors_count']}; bad pairs {audit['bad_pairs_count']}.")
        lines.append("Operation-count source fractions:")
        for k, rec in audit["by_operation_count"].items():
            lines.append(f"- k={k}: n={rec['n']}, P(source)={rec['p_source']:.3f}, roles={rec['roles']}")
        lines.append("Query-first source fractions for operation-present rows:")
        for k, rec in audit["op_present_by_query_is_first_mentioned"].items():
            p = rec["p_source"]
            lines.append(f"- query_is_first={k}: n={rec['n']}, P(source)={p:.3f}, roles={rec['roles']}")
        lines.append("")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "out_dir": rel(args.out_dir),
        "source_map_counts": manifest["source_map_counts"],
        "train_frame_seen": {k: outputs["train_frame_seen"][k] for k in ["rows", "pairs", "row_words"]},
        "heldout_frame_all": {k: outputs["heldout_frame_all"][k] for k in ["rows", "pairs", "row_words"]},
        "train_balance_by_op": outputs["train_frame_seen"]["audit"]["by_operation_count"],
        "train_query_first": outputs["train_frame_seen"]["audit"]["op_present_by_query_is_first_mentioned"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
