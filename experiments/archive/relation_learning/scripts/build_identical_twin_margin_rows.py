#!/usr/bin/env python3
"""research: build and audit identical-context twin rows for shortcut-resistant binding pilots.

The research family balanced latent answer roles by operation count, but the row pairs
were not identical contexts: the retention half lacked the queried update sentence and
had one fewer update than the update half.  This script rebuilds the local binding
pilot from assignment-reversal maps under the current scientific
row contract:

  A/retention row: source_context + update(other co-introduced entity -> shared_new)
                   + the same distractor updates + query(target entity -> source value)
  B/updated row:   source_context + update(target entity -> shared_new)
                   + the same distractor updates + query(target entity -> shared_new)

Within each pair A and B have the same source, same query entity, same source values,
same shared new value, same distractor updates, same update count, and the assignment
update in the same operation position.  They differ only in which co-introduced entity
is named in the assignment update.  The answer source/new distinction is therefore
aligned with identity-conditioned state matching rather than with operation count,
length, absent alternatives, or unrelated-update entities.

The script also audits the previous research rows on observable features so the failed
family is preserved as a negative construction rather than silently replaced.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
OLD_STEP088 = _public_path('experiments/archive/relation_learning/data/symmetric_margin_rows/symmetric_margin_heldout_frame_seen.jsonl')
DEFAULT_MAPS = _public_path('experiments/archive/functional_learning/data/assignment_reversal_export/a01_assignment_reversal_operation_maps.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/identical_twin_margin_rows')
DEFAULT_TOKENIZER = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')

FRAMES = [
    {"frame_id": "f00_original_a01", "frame_split": "train_seen", "style": "original_a01_query"},
    {"frame_id": "f01_after_records", "frame_split": "train_seen", "style": "after_records"},
    {"frame_id": "f02_passage_indicates", "frame_split": "train_seen", "style": "evidence_report"},
    {"frame_id": "f03_using_details", "frame_split": "eval_unseen", "style": "using_details"},
]
SIDES = ["a", "b"]
DISTRACTOR_COUNTS = [0, 1, 2, 3, 4]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def wc(s: str) -> int:
    return len(norm(s).split()) if norm(s) else 0


def toks(s: str) -> set[str]:
    return {m.group(0).lower() for m in WORD_RE.finditer(s or "") if len(m.group(0)) > 2}


def maybe_tokenizer(path: pathlib.Path | None):
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    except Exception as e:
        print(json.dumps({"event": "tokenizer_unavailable", "path": rel(path), "error": repr(e)}), flush=True)
        return None


def tok_len(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])


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
    if relation == "founded_year":
        if frame_id == "f01_after_records":
            return "After these records, {entity} was founded in {answer}."
        if frame_id == "f02_passage_indicates":
            return "The passage indicates that {entity} was founded in {answer}."
        if frame_id == "f03_using_details":
            return "Using the details above, {entity}'s founding year is {answer}."
    return "According to this information, the relevant {relation} value for {entity} is {answer}."


def render_query(relation: str, frame_id: str, entity: str, answer: str) -> tuple[str, int, int, str]:
    tmpl = template_for(relation, frame_id)
    if "{relation}" in tmpl:
        text = tmpl.format(relation=relation.replace("_", " "), entity=entity, answer=answer)
    else:
        text = tmpl.format(entity=entity, answer=answer)
    a = text.find(answer)
    if a < 0:
        raise RuntimeError(f"answer {answer!r} not found in query {text!r}")
    return text, a, a + len(answer), tmpl


def side_fields(m: dict[str, Any], side: str) -> dict[str, str]:
    other = "b" if side == "a" else "a"
    return {
        "side": side,
        "other_side": other,
        "entity": norm(m[f"entity_{side}"]),
        "other_entity": norm(m[f"entity_{other}"]),
        "source_value": norm(m[f"value_{side}"]),
        "other_source_value": norm(m[f"value_{other}"]),
        "target_update": norm(m[f"update_{side}_sentence"]),
        "other_update": norm(m[f"update_{other}_sentence"]),
    }


def valid_map(m: dict[str, Any]) -> bool:
    required = ["base_pair_id", "split", "relation", "source_context", "entity_a", "entity_b", "value_a", "value_b", "shared_new_value", "update_a_sentence", "update_b_sentence"]
    return all(norm(m.get(k)) for k in required)


def assignment_update_token_delta(m: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    ua = norm(m["update_a_sentence"])
    ub = norm(m["update_b_sentence"])
    return {
        "update_a_words": wc(ua),
        "update_b_words": wc(ub),
        "update_word_delta_a_minus_b": wc(ua) - wc(ub),
        "update_a_tokens": tok_len(tokenizer, ua),
        "update_b_tokens": tok_len(tokenizer, ub),
        "update_token_delta_a_minus_b": (None if tokenizer is None else tok_len(tokenizer, ua) - tok_len(tokenizer, ub)),
    }


def filter_maps_for_length(maps: list[dict[str, Any]], tokenizer: Any, require_equal_update_tokens: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for m in maps:
        if not valid_map(m):
            bad.append({"base_pair_id": m.get("base_pair_id"), "reason": "missing_required"})
            continue
        d = assignment_update_token_delta(m, tokenizer)
        if require_equal_update_tokens and tokenizer is not None and d["update_token_delta_a_minus_b"] != 0:
            bad.append({"base_pair_id": m.get("base_pair_id"), "reason": "unequal_update_token_len", **d})
            continue
        if require_equal_update_tokens and tokenizer is None and d["update_word_delta_a_minus_b"] != 0:
            bad.append({"base_pair_id": m.get("base_pair_id"), "reason": "unequal_update_word_len_no_tokenizer", **d})
            continue
        good.append({**m, "length_audit": d})
    return good, {"input_maps": len(maps), "kept_maps": len(good), "excluded_maps": len(bad), "excluded_first": bad[:12]}


def choose_distractors(maps: list[dict[str, Any]], idx: int, side: str, n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    base = maps[idx]
    sf = side_fields(base, side)
    avoid = toks(sf["entity"]) | toks(sf["other_entity"]) | toks(sf["source_value"]) | toks(sf["other_source_value"]) | toks(norm(base["shared_new_value"]))
    out: list[dict[str, Any]] = []
    stride = 17 if len(maps) % 17 else 19
    for step in range(1, len(maps) * 3 + 1):
        j = (idx + step * stride + (5 if side == "b" else 0)) % len(maps)
        if j == idx:
            continue
        other = maps[j]
        for uside in ("a", "b"):
            sent = norm(other[f"update_{uside}_sentence"])
            val = norm(other["shared_new_value"])
            if not sent or val == norm(base["shared_new_value"]):
                continue
            if avoid & toks(sent):
                continue
            out.append({
                "kind": "unrelated_distractor_update",
                "base_pair_id": other["base_pair_id"],
                "side": uside,
                "sentence": sent,
                "updated_entity": norm(other[f"entity_{uside}"]),
                "new_value": val,
                "is_assignment_update": False,
                "is_target_entity_update": False,
                "is_other_cointroduced_update": False,
            })
            if len(out) >= n:
                return out
    # Deterministic fallback, still mark it for later exclusion if needed.
    for step in range(1, len(maps) * 3 + 1):
        j = (idx + step * 13 + (3 if side == "b" else 0)) % len(maps)
        if j == idx:
            continue
        other = maps[j]
        for uside in ("a", "b"):
            val = norm(other["shared_new_value"])
            if val == norm(base["shared_new_value"]):
                continue
            out.append({
                "kind": "unrelated_distractor_update",
                "base_pair_id": other["base_pair_id"],
                "side": uside,
                "sentence": norm(other[f"update_{uside}_sentence"]),
                "updated_entity": norm(other[f"entity_{uside}"]),
                "new_value": val,
                "is_assignment_update": False,
                "is_target_entity_update": False,
                "is_other_cointroduced_update": False,
                "fallback_possible_overlap": True,
            })
            if len(out) >= n:
                return out
    raise RuntimeError("not enough distractors")


def candidate_values(m: dict[str, Any], sf: dict[str, str], ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vals: list[dict[str, Any]] = []
    def add(cid: str, value: str, origin: str, entity: str = "") -> None:
        value = norm(value)
        if not value:
            return
        if any(v["value"] == value for v in vals):
            return
        vals.append({"candidate_id": cid, "value": value, "origin": origin, "entity": entity})
    add("query_source", sf["source_value"], "source_context_query_entity", sf["entity"])
    add("other_source", sf["other_source_value"], "source_context_other_entity", sf["other_entity"])
    add("shared_new", norm(m["shared_new_value"]), "assignment_update_shared_new", "assignment_entity")
    for i, op in enumerate(ops):
        if op.get("kind") == "unrelated_distractor_update":
            add(f"distractor_new_{i}", op.get("new_value", ""), "unrelated_update", op.get("updated_entity", ""))
    return vals


def make_context(source: str, ops: list[dict[str, Any]], query: str) -> tuple[str, int]:
    prefix = " ".join([norm(source)] + [norm(o["sentence"]) for o in ops if norm(o.get("sentence", ""))]).strip()
    context = prefix + " " + query
    return context, len(prefix) + 1


def make_row(m: dict[str, Any], idx: int, split: str, side: str, frame: dict[str, str], pair_half: str,
             distractor_count: int, ops: list[dict[str, Any]], assignment_position: int, tokenizer: Any) -> dict[str, Any]:
    sf = side_fields(m, side)
    answer = sf["source_value"] if pair_half == "A" else norm(m["shared_new_value"])
    answer_kind = "source_state" if pair_half == "A" else "new_state"
    query, qa, qb, tmpl = render_query(norm(m["relation"]), frame["frame_id"], sf["entity"], answer)
    context, qstart = make_context(norm(m["source_context"]), ops, query)
    a0, b0 = qstart + qa, qstart + qb
    if context[a0:b0] != answer:
        raise RuntimeError(f"answer span mismatch {context[a0:b0]!r} != {answer!r}")
    cands = candidate_values(m, sf, ops)
    cand_values = {c["value"] for c in cands}
    if sf["source_value"] not in cand_values or norm(m["shared_new_value"]) not in cand_values:
        raise RuntimeError("required candidate missing")
    prefix = context[:a0]
    assignment_update = [o for o in ops if o.get("is_assignment_update")]
    row_id = f"{split}:{m['base_pair_id']}::{side}::d{distractor_count}::{pair_half}::{frame['frame_id']}"
    return {
        "row_id": row_id,
        "pair_id": f"{m['base_pair_id']}::{side}::d{distractor_count}::{frame['frame_id']}",
        "split": split,
        "packet_type": "IDENTICAL_CONTEXT_TWIN_BINDING",
        "pair_half": pair_half,
        "role": "unchanged_entity" if pair_half == "A" else "updated_entity",
        "answer_kind": answer_kind,
        "answer_text": answer,
        "correct_candidate_id": "query_source" if pair_half == "A" else "shared_new",
        "query_entity": sf["entity"],
        "query_side": side,
        "target_entity": sf["entity"],
        "other_cointroduced_entity": sf["other_entity"],
        "source_state": sf["source_value"],
        "other_source_state": sf["other_source_value"],
        "new_state": norm(m["shared_new_value"]),
        "source_sentence": norm(m["source_context"]),
        "update_sentence": " ".join(norm(o["sentence"]) for o in ops),
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
        "shared_new_value": norm(m["shared_new_value"]),
        "operation_count": len(ops),
        "distractor_update_count": distractor_count,
        "assignment_update_position_0based": assignment_position,
        "operation_specs": ops,
        "state_candidates": cands,
        "context_words": wc(context),
        "context_tokens_chck82": tok_len(tokenizer, context),
        "prefix_contains_query_source": sf["source_value"].lower() in prefix.lower(),
        "prefix_contains_shared_new": norm(m["shared_new_value"]).lower() in prefix.lower(),
        "query_entity_in_update_sentence": sf["entity"].lower() in (" ".join(norm(o["sentence"]) for o in ops)).lower(),
        "other_entity_in_assignment_update": bool(assignment_update and sf["other_entity"].lower() in norm(assignment_update[0].get("sentence", "")).lower()),
        "target_entity_in_assignment_update": bool(assignment_update and sf["entity"].lower() in norm(assignment_update[0].get("sentence", "")).lower()),
        "construction_reading": "A/B twins share source, query, values, distractors, update count, and assignment-update position; the intended differing cue is whether the assignment update names the queried entity or its co-introduced partner.",
    }


def build_split(maps: list[dict[str, Any]], split: str, frames: list[dict[str, str]], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for idx, m in enumerate(maps):
        for frame_idx, frame in enumerate(frames):
            for side_idx, side in enumerate(SIDES):
                sf = side_fields(m, side)
                for d in DISTRACTOR_COUNTS:
                    distractors = choose_distractors(maps, idx, side, d)
                    total_ops = d + 1
                    pos = (idx + side_idx + frame_idx + d) % total_ops
                    assign_a = {
                        "kind": "other_cointroduced_assignment_update",
                        "base_pair_id": m["base_pair_id"],
                        "side": sf["other_side"],
                        "sentence": sf["other_update"],
                        "updated_entity": sf["other_entity"],
                        "new_value": norm(m["shared_new_value"]),
                        "is_assignment_update": True,
                        "is_target_entity_update": False,
                        "is_other_cointroduced_update": True,
                    }
                    assign_b = {
                        "kind": "target_assignment_update",
                        "base_pair_id": m["base_pair_id"],
                        "side": side,
                        "sentence": sf["target_update"],
                        "updated_entity": sf["entity"],
                        "new_value": norm(m["shared_new_value"]),
                        "is_assignment_update": True,
                        "is_target_entity_update": True,
                        "is_other_cointroduced_update": False,
                    }
                    ops_a = list(distractors); ops_b = list(distractors)
                    ops_a.insert(pos, assign_a); ops_b.insert(pos, assign_b)
                    row_a = make_row(m, idx, split, side, frame, "A", d, ops_a, pos, tokenizer)
                    row_b = make_row(m, idx, split, side, frame, "B", d, ops_b, pos, tokenizer)
                    rows.extend([row_a, row_b])
                    pairs.append({
                        "pair_id": row_a["pair_id"],
                        "row_a_id": row_a["row_id"],
                        "row_b_id": row_b["row_id"],
                        "base_pair_id": m["base_pair_id"],
                        "split": split,
                        "query_side": side,
                        "query_entity": sf["entity"],
                        "other_cointroduced_entity": sf["other_entity"],
                        "source_state": sf["source_value"],
                        "other_source_state": sf["other_source_value"],
                        "new_state": norm(m["shared_new_value"]),
                        "frame_id": frame["frame_id"],
                        "frame_split": frame["frame_split"],
                        "operation_count": total_ops,
                        "distractor_update_count": d,
                        "assignment_update_position_0based": pos,
                        "a_context_words": row_a["context_words"],
                        "b_context_words": row_b["context_words"],
                        "a_context_tokens_chck82": row_a["context_tokens_chck82"],
                        "b_context_tokens_chck82": row_b["context_tokens_chck82"],
                        "context_word_delta_a_minus_b": row_a["context_words"] - row_b["context_words"],
                        "context_token_delta_a_minus_b": None if row_a["context_tokens_chck82"] is None or row_b["context_tokens_chck82"] is None else row_a["context_tokens_chck82"] - row_b["context_tokens_chck82"],
                    })
    return rows, pairs


def first_mention_side(row: dict[str, Any]) -> str | None:
    src = str(row.get("source_sentence", ""))
    ia = src.find(str(row.get("original_entity_a", "")))
    ib = src.find(str(row.get("original_entity_b", "")))
    if ia < 0 or ib < 0:
        return None
    return "a" if ia < ib else "b"


def source_frac(rs: list[dict[str, Any]]) -> float | None:
    return None if not rs else sum(1 for r in rs if r.get("answer_kind") == "source_state") / len(rs)


def group_table(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(r)
    out: dict[str, Any] = {}
    for k, rs in sorted(buckets.items(), key=lambda kv: kv[0]):
        out[k] = {"n": len(rs), "p_source": source_frac(rs), "roles": dict(Counter(r.get("answer_kind") for r in rs))}
    return out


def majority_rule_accuracy(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(r)
    correct = 0
    table = {}
    for k, rs in buckets.items():
        nsrc = sum(1 for r in rs if r.get("answer_kind") == "source_state")
        pred_source = nsrc >= (len(rs) - nsrc)
        cc = sum(1 for r in rs if (r.get("answer_kind") == "source_state") == pred_source)
        correct += cc
        table[str(k)] = {"n": len(rs), "p_source": nsrc / len(rs), "predict_source": pred_source, "correct": cc}
    return {"accuracy": correct / len(rows) if rows else None, "table": dict(sorted(table.items(), key=lambda kv: kv[0]))}


def audit_new_rows(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["row_id"]: r for r in rows}
    bad_pairs = []
    for p in pairs:
        a = by_id.get(p["row_a_id"]); b = by_id.get(p["row_b_id"])
        if not a or not b:
            bad_pairs.append({"pair_id": p["pair_id"], "reason": "missing_row"})
            continue
        checks = {
            "same_query_entity": a["query_entity"] == b["query_entity"],
            "same_source": a["source_sentence"] == b["source_sentence"],
            "same_operation_count": a["operation_count"] == b["operation_count"],
            "same_distractor_count": a["distractor_update_count"] == b["distractor_update_count"],
            "same_assignment_position": a["assignment_update_position_0based"] == b["assignment_update_position_0based"],
            "a_source_b_new": a["answer_kind"] == "source_state" and b["answer_kind"] == "new_state",
            "shared_new_present_a": a["prefix_contains_shared_new"],
            "shared_new_present_b": b["prefix_contains_shared_new"],
            "source_present_a": a["prefix_contains_query_source"],
            "source_present_b": b["prefix_contains_query_source"],
            "word_delta_zero": p["context_word_delta_a_minus_b"] == 0,
            "token_delta_zero_or_unknown": p["context_token_delta_a_minus_b"] in (0, None),
        }
        if not all(checks.values()):
            bad_pairs.append({"pair_id": p["pair_id"], "checks": checks, "word_delta": p["context_word_delta_a_minus_b"], "token_delta": p["context_token_delta_a_minus_b"]})
    span_errors = [r["row_id"] for r in rows if r["context_text"][int(r["answer_char_start"]):int(r["answer_char_end"])] != r["answer_text"]]
    qfirst_rows = []
    for r in rows:
        fm = first_mention_side(r)
        if fm is not None:
            rr = dict(r)
            rr["query_is_first_mentioned"] = (r["query_side"] == fm)
            qfirst_rows.append(rr)
    length_values = [int(r["context_words"]) for r in rows]
    token_values = [r.get("context_tokens_chck82") for r in rows if r.get("context_tokens_chck82") is not None]
    table = {
        "rows": len(rows),
        "pairs": len(pairs),
        "row_words": sum(int(r["context_words"]) for r in rows),
        "duplicate_row_ids": len(rows) - len({r["row_id"] for r in rows}),
        "duplicate_pair_ids": len(pairs) - len({p["pair_id"] for p in pairs}),
        "span_errors_count": len(span_errors),
        "span_errors_first": span_errors[:8],
        "bad_pairs_count": len(bad_pairs),
        "bad_pairs_first": bad_pairs[:8],
        "by_answer_kind": dict(Counter(r["answer_kind"] for r in rows)),
        "by_relation": group_table(rows, lambda r: r["relation"]),
        "by_frame": group_table(rows, lambda r: r["frame_id"]),
        "by_operation_count": group_table(rows, lambda r: r["operation_count"]),
        "by_distractor_count": group_table(rows, lambda r: r["distractor_update_count"]),
        "by_query_side": group_table(rows, lambda r: r["query_side"]),
        "by_query_is_first_mentioned": group_table(qfirst_rows, lambda r: r["query_is_first_mentioned"]),
        "by_query_entity_in_update_sentence": group_table(rows, lambda r: r["query_entity_in_update_sentence"]),
        "by_assignment_position": group_table(rows, lambda r: r["assignment_update_position_0based"]),
        "candidate_presence": {
            "rows_with_query_source_present_before_answer": sum(1 for r in rows if r["prefix_contains_query_source"]),
            "rows_with_shared_new_present_before_answer": sum(1 for r in rows if r["prefix_contains_shared_new"]),
            "rows_with_all_candidate_values": sum(1 for r in rows if len(r.get("state_candidates", [])) >= 3),
        },
        "simple_rule_accuracy": {
            "always_source": sum(1 for r in rows if r["answer_kind"] == "source_state") / len(rows) if rows else None,
            "operation_count_majority": majority_rule_accuracy(rows, lambda r: r["operation_count"]),
            "distractor_count_majority": majority_rule_accuracy(rows, lambda r: r["distractor_update_count"]),
            "context_words_majority": majority_rule_accuracy(rows, lambda r: r["context_words"]),
            "context_tokens_majority": majority_rule_accuracy([r for r in rows if r.get("context_tokens_chck82") is not None], lambda r: r["context_tokens_chck82"]),
            "query_entity_in_update_sentence": majority_rule_accuracy(rows, lambda r: r["query_entity_in_update_sentence"]),
        },
        "context_words_range": [min(length_values), max(length_values)] if length_values else None,
        "context_tokens_range": [min(token_values), max(token_values)] if token_values else None,
        "relevant_update_position_counts": dict(Counter(str(p["assignment_update_position_0based"]) for p in pairs)),
        "context_word_delta_counts": dict(Counter(str(p["context_word_delta_a_minus_b"]) for p in pairs)),
        "context_token_delta_counts": dict(Counter(str(p["context_token_delta_a_minus_b"]) for p in pairs)),
    }
    return table


def audit_old_step088(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pair[str(r.get("pair_id"))].append(r)
    pair_records = []
    for pid, rs in by_pair.items():
        if len(rs) < 2:
            continue
        a = next((r for r in rs if r.get("pair_half") == "A"), None)
        b = next((r for r in rs if r.get("pair_half") == "B"), None)
        if not a or not b:
            continue
        av = norm(a.get("new_state"))
        a_prefix = str(a.get("context_text", ""))[:int(a.get("answer_char_start", 0))].lower()
        b_prefix = str(b.get("context_text", ""))[:int(b.get("answer_char_start", 0))].lower()
        q = norm(a.get("query_entity"))
        pair_records.append({
            "pair_id": pid,
            "a_update_count": len(a.get("operation_specs", [])),
            "b_update_count": len(b.get("operation_specs", [])),
            "a_has_new_value_before_answer": av.lower() in a_prefix if av else False,
            "b_has_new_value_before_answer": av.lower() in b_prefix if av else False,
            "a_query_entity_in_update": q.lower() in norm(a.get("update_sentence", "")).lower(),
            "b_query_entity_in_update": q.lower() in norm(b.get("update_sentence", "")).lower(),
            "a_update_entities_in_source": [norm(o.get("updated_entity")) for o in a.get("operation_specs", []) if norm(o.get("updated_entity")) and norm(o.get("updated_entity")).lower() in norm(a.get("source_sentence", "")).lower()],
            "b_update_entities_in_source": [norm(o.get("updated_entity")) for o in b.get("operation_specs", []) if norm(o.get("updated_entity")) and norm(o.get("updated_entity")).lower() in norm(b.get("source_sentence", "")).lower()],
        })
    def frac(pred) -> float:
        return sum(1 for x in pair_records if pred(x)) / len(pair_records) if pair_records else float("nan")
    sent_rule = majority_rule_accuracy(rows, lambda r: len(r.get("operation_specs", [])))
    return {
        "status": "OBSERVABLE_FAILURE_AUDIT",
        "old_rows": len(rows),
        "old_pairs": len(pair_records),
        "fraction_A_has_one_fewer_update_than_B": frac(lambda x: x["a_update_count"] + 1 == x["b_update_count"]),
        "fraction_A_new_value_absent_before_answer": frac(lambda x: not x["a_has_new_value_before_answer"]),
        "fraction_B_new_value_present_before_answer": frac(lambda x: x["b_has_new_value_before_answer"]),
        "fraction_A_query_entity_absent_from_updates": frac(lambda x: not x["a_query_entity_in_update"]),
        "fraction_B_query_entity_present_in_updates": frac(lambda x: x["b_query_entity_in_update"]),
        "fraction_A_no_update_entity_introduced_in_source": frac(lambda x: len(x["a_update_entities_in_source"]) == 0),
        "observable_update_sentence_count_table": group_table(rows, lambda r: len(r.get("operation_specs", []))),
        "sentence_count_majority_rule": sent_rule,
        "always_source_accuracy": sum(1 for r in rows if r.get("answer_kind") == "source_state") / len(rows) if rows else None,
        "first_pairs": pair_records[:5],
        "scientific_reading": "The research pair files are not an invariant margin family: A/retention often lacks the shared new candidate and has one fewer update than B; unrelated distractors name entities absent from the source; sentence-count and candidate-presence rules can absorb credit.",
    }


def build_outputs(maps_path: pathlib.Path, out_dir: pathlib.Path, tokenizer: Any, require_equal_update_tokens: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_maps = [m for m in read_jsonl(maps_path) if valid_map(m)]
    split_raw = defaultdict(list)
    for m in raw_maps:
        split_raw[str(m.get("split"))].append(m)
    train_maps, train_filter = filter_maps_for_length(split_raw["train"], tokenizer, require_equal_update_tokens)
    held_maps, held_filter = filter_maps_for_length(split_raw["heldout"], tokenizer, require_equal_update_tokens)
    if len(train_maps) < 8 or len(held_maps) < 4:
        raise RuntimeError(f"too few maps after filter: train {len(train_maps)}, held {len(held_maps)}")
    frames_seen = [f for f in FRAMES if f["frame_split"] == "train_seen"]
    outputs = {}
    for name, maps, split, frames in [
        ("train_frame_seen", train_maps, "train", frames_seen),
        ("train_frame_all", train_maps, "train", FRAMES),
        ("heldout_frame_seen", held_maps, "heldout", frames_seen),
        ("heldout_frame_all", held_maps, "heldout", FRAMES),
    ]:
        rows, pairs = build_split(maps, split, frames, tokenizer)
        rp = out_dir / f"identical_twin_{name}.jsonl"
        pp = out_dir / f"binding_pairs_{name}.jsonl"
        outputs[name] = {
            "rows_path": rel(rp),
            "pairs_path": rel(pp),
            "rows_sha256": write_jsonl(rp, rows),
            "pairs_sha256": write_jsonl(pp, pairs),
            "rows": len(rows),
            "pairs": len(pairs),
            "row_words": sum(int(r["context_words"]) for r in rows),
            "audit": audit_new_rows(rows, pairs),
        }
    old_audit = None
    if OLD_STEP088.exists():
        old_audit = audit_old_step088(read_jsonl(OLD_STEP088))
        (out_dir / "observable_failure_audit.json").write_text(json.dumps(old_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "IDENTICAL_TWIN_MARGIN_ROWS_BUILT",
        "created_utc": now(),
        "source_maps": rel(maps_path),
        "tokenizer": None if tokenizer is None else rel(DEFAULT_TOKENIZER),
        "require_equal_update_token_lengths": require_equal_update_tokens,
        "source_map_counts_raw": {k: len(v) for k, v in split_raw.items()},
        "filter_train": train_filter,
        "filter_heldout": held_filter,
        "frames": FRAMES,
        "distractor_counts": DISTRACTOR_COUNTS,
        "construction_reading": {
            "status": "replaced by an observable-feature audit; do not train it",
            "new_pair_invariant": "A and B twins have same query, source context, source values, shared new value, distractor updates, operation count, and assignment-update position; they differ by the entity named in the assignment update.",
            "candidate_set": "Each row records all state values present in context: query source value, other source value, shared new update value, and unrelated distractor update values.",
            "pilot_decision": "Before private-phase or research, score untrained reference and require A/retention not at ceiling, retention flat across distractor count, and joint held-pair learning above both-wrong floor after a small local pilot.",
        },
        "outputs": outputs,
        "old_step088_observable_audit_path": rel(out_dir / "observable_failure_audit.json") if old_audit is not None else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research identical-context twin row family",
        "",
        "research was rejected because its A/B contexts differed in observable features. This family uses the assignment-reversal maps so A and B twins differ only in whether the assignment update names the queried entity or its co-introduced partner.",
        "",
        f"Raw maps: train={len(split_raw['train'])}, heldout={len(split_raw['heldout'])}; kept after equal-update-token filter: train={len(train_maps)}, heldout={len(held_maps)}.",
        "",
    ]
    if old_audit:
        lines += [
            "## research observable failure",
            f"Old heldout_frame_seen pairs: {old_audit['old_pairs']}",
            f"A has one fewer update than B: {old_audit['fraction_A_has_one_fewer_update_than_B']:.3f}",
            f"A shared-new alternative absent before answer: {old_audit['fraction_A_new_value_absent_before_answer']:.3f}",
            f"Sentence-count majority rule accuracy on rows: {old_audit['sentence_count_majority_rule']['accuracy']:.3f} vs always-source {old_audit['always_source_accuracy']:.3f}",
            "",
        ]
    for name in ["train_frame_seen", "heldout_frame_all"]:
        aud = outputs[name]["audit"]
        lines += [
            f"## {name}",
            f"Rows {aud['rows']}; pairs {aud['pairs']}; row words {aud['row_words']}; bad pairs {aud['bad_pairs_count']}; span errors {aud['span_errors_count']}.",
            f"Candidate presence: {aud['candidate_presence']}",
            f"Context token delta counts A-B: {aud['context_token_delta_counts']}",
            "By distractor count:",
        ]
        for k, rec in aud["by_distractor_count"].items():
            lines.append(f"- d={k}: n={rec['n']}, P(source)={rec['p_source']:.3f}, roles={rec['roles']}")
        lines.append("Observable rule accuracies:")
        sr = aud["simple_rule_accuracy"]
        lines.append(f"- always_source: {sr['always_source']:.3f}")
        lines.append(f"- operation_count_majority: {sr['operation_count_majority']['accuracy']:.3f}")
        lines.append(f"- context_words_majority: {sr['context_words_majority']['accuracy']:.3f}")
        lines.append(f"- context_tokens_majority: {sr['context_tokens_majority']['accuracy']:.3f}")
        lines.append(f"- query_entity_in_update_sentence: {sr['query_entity_in_update_sentence']['accuracy']:.3f} (desired identity cue, not a nuisance balance target)")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", type=pathlib.Path, default=DEFAULT_MAPS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--no-equal-update-token-filter", action="store_true")
    args = ap.parse_args()
    tokenizer = maybe_tokenizer(args.tokenizer)
    manifest = build_outputs(args.maps, args.out_dir, tokenizer, require_equal_update_tokens=not args.no_equal_update_token_filter)
    print(json.dumps({
        "status": manifest["status"],
        "out_dir": rel(args.out_dir),
        "raw_counts": manifest["source_map_counts_raw"],
        "kept_train": manifest["filter_train"]["kept_maps"],
        "kept_heldout": manifest["filter_heldout"]["kept_maps"],
        "train_frame_seen": {k: manifest["outputs"]["train_frame_seen"][k] for k in ["rows", "pairs", "row_words"]},
        "heldout_frame_all": {k: manifest["outputs"]["heldout_frame_all"][k] for k in ["rows", "pairs", "row_words"]},
        "heldout_rule_accuracy": manifest["outputs"]["heldout_frame_all"]["audit"]["simple_rule_accuracy"],
        "old_step088_audit_path": manifest["old_step088_observable_audit_path"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
