#!/usr/bin/env python3
"""research: presentation-schedule planner for the relation-tail BabyLM bridge.

The research overlay stores all relation packets before the ordinary tail for
transparent construction.  This planner makes that storage choice explicit and
constructs the two scientific schedules that should not be conflated:

* frontloaded: relation acquisition block followed by ordinary tail;
* interspersed_wordpaced: relation rows distributed across the same substituted
  corpus so relation learning and ordinary learning coexist.

It also records row-count versus word-paced update counts.  This matters because
short relation packets inflate row count: matching legal words alone does not
match optimizer updates if the old research row-batched trainer is used literally.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_OVERLAY_DIR = _public_path('experiments/archive/functional_learning/data/relation_tail_overlay')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/bridge_schedule_plan')
DEFAULT_REFERENCE_UPDATES = 354  # research standard continuation over 13,994,705 tail words.
DEFAULT_MACRO_ROWS = 256
DEFAULT_INITIAL_WORDS = 86_005_295


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def word_sum(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows)


def interleave_word_fraction(relation: List[Dict[str, Any]], ordinary: List[Dict[str, Any]], target_fraction: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ri = oi = 0
    rel_w = ord_w = 0
    while ri < len(relation) or oi < len(ordinary):
        total = rel_w + ord_w
        current_frac = (rel_w / total) if total > 0 else 0.0
        if ri < len(relation) and (oi >= len(ordinary) or current_frac <= target_fraction):
            r = dict(relation[ri])
            r["presentation_schedule"] = "interspersed_wordpaced"
            r["presentation_source_index"] = ri
            out.append(r)
            rel_w += int(r["words"])
            ri += 1
        else:
            r = dict(ordinary[oi])
            r["presentation_schedule"] = "interspersed_wordpaced"
            r["presentation_source_index"] = oi
            out.append(r)
            ord_w += int(r["words"])
            oi += 1
    return out


def checkpoint_stats(rows: List[Dict[str, Any]], checkpoints: List[int]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    cp_i = 0
    total = rel_words = rel_rows = rows_seen = 0
    for r in rows:
        if cp_i >= len(checkpoints):
            break
        w = int(r["words"])
        total += w
        rows_seen += 1
        if r.get("bridge_kind") == "relation_answer_packet":
            rel_words += w
            rel_rows += 1
        while cp_i < len(checkpoints) and total >= checkpoints[cp_i]:
            stats.append({
                "tail_word_checkpoint": int(checkpoints[cp_i]),
                "total_words_seen_at_crossing": total,
                "total_rows_seen_at_crossing": rows_seen,
                "relation_rows_seen": rel_rows,
                "relation_words_seen": rel_words,
                "relation_word_fraction_seen": rel_words / max(1, total),
            })
            cp_i += 1
    return stats


def row_macro_updates(n_rows: int, macro_rows: int) -> int:
    return int(math.ceil(n_rows / int(macro_rows)))


def word_paced_batches(rows: List[Dict[str, Any]], n_updates: int) -> Dict[str, Any]:
    total_words = word_sum(rows)
    target = total_words / int(n_updates)
    batches: List[Dict[str, Any]] = []
    cur_rows: List[Dict[str, Any]] = []
    cur_w = 0
    cur_rel_w = 0
    cur_rel_rows = 0
    boundary = target
    cum = 0
    for r in rows:
        w = int(r["words"])
        cur_rows.append(r)
        cur_w += w
        cum += w
        if r.get("bridge_kind") == "relation_answer_packet":
            cur_rel_w += w
            cur_rel_rows += 1
        if len(batches) < n_updates - 1 and cum >= boundary:
            batches.append({"rows": len(cur_rows), "words": cur_w, "relation_rows": cur_rel_rows, "relation_words": cur_rel_w})
            cur_rows = []
            cur_w = cur_rel_w = cur_rel_rows = 0
            boundary = target * (len(batches) + 1)
    if cur_rows:
        batches.append({"rows": len(cur_rows), "words": cur_w, "relation_rows": cur_rel_rows, "relation_words": cur_rel_w})
    rows_per = [b["rows"] for b in batches]
    words_per = [b["words"] for b in batches]
    rel_words_per = [b["relation_words"] for b in batches]
    rel_rows_per = [b["relation_rows"] for b in batches]
    return {
        "n_updates": len(batches),
        "target_words_per_update": target,
        "rows_per_update_mean": statistics.mean(rows_per),
        "rows_per_update_min": min(rows_per),
        "rows_per_update_max": max(rows_per),
        "words_per_update_mean": statistics.mean(words_per),
        "words_per_update_min": min(words_per),
        "words_per_update_max": max(words_per),
        "relation_rows_per_update_mean": statistics.mean(rel_rows_per),
        "relation_words_per_update_mean": statistics.mean(rel_words_per),
        "updates_with_relation": sum(1 for x in rel_rows_per if x > 0),
        "first10": batches[:10],
        "last10": batches[-10:],
    }


def estimate_relation_answer_tokens(relation: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Use the known repeated rows.  If the tokenizer is available, count actual
    # subword positions in the recorded answer_char_span; otherwise fall back to
    # answer word count.  This is a planning quantity, not a training score.
    try:
        from transformers import AutoTokenizer
        import relation_first_repaired_harness as h
        tok = AutoTokenizer.from_pretrained(str(h.MODEL_PATH), local_files_only=True, use_fast=True)
        counts: List[int] = []
        for r in relation:
            text = str(r["text"])
            a0, a1 = [int(x) for x in r["answer_char_span"]]
            enc = tok(text, add_special_tokens=True, return_offsets_mapping=True, max_length=512, truncation=True)
            pos = []
            for i, (a, b) in enumerate(enc["offset_mapping"]):
                if b <= a0 or a >= a1 or a == b:
                    continue
                pos.append(i)
            counts.append(len(pos))
        return {
            "method": "tokenizer_offset_count",
            "relation_rows": len(relation),
            "total_answer_label_token_positions": sum(counts),
            "mean_answer_tokens_per_row": statistics.mean(counts),
            "max_answer_tokens_per_row": max(counts),
        }
    except Exception as exc:
        counts = [len(str(r.get("answer_text", "")).split()) for r in relation]
        return {
            "method": "answer_word_fallback",
            "error": repr(exc),
            "relation_rows": len(relation),
            "total_answer_word_positions": sum(counts),
            "mean_answer_words_per_row": statistics.mean(counts) if counts else 0,
            "max_answer_words_per_row": max(counts) if counts else 0,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlay-dir", default=str(DEFAULT_OVERLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--reference-updates", type=int, default=DEFAULT_REFERENCE_UPDATES)
    ap.add_argument("--macro-rows", type=int, default=DEFAULT_MACRO_ROWS)
    ap.add_argument("--initial-words", type=int, default=DEFAULT_INITIAL_WORDS)
    ap.add_argument("--write-interspersed", action="store_true")
    args = ap.parse_args()

    overlay_dir = pathlib.Path(args.overlay_dir)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((overlay_dir / "overlay_manifest.json").read_text(encoding="utf-8"))
    overlay_path = pathlib.Path(ROOT / manifest["outputs"]["overlay_tail"])
    relation_path = pathlib.Path(ROOT / manifest["outputs"]["relation_rows"])
    overlay = read_jsonl(overlay_path)
    relation = read_jsonl(relation_path)
    ordinary = [r for r in overlay if r.get("bridge_kind") != "relation_answer_packet"]
    if len(ordinary) + len(relation) != len(overlay):
        raise RuntimeError("row partition mismatch")
    total_words = word_sum(overlay)
    relation_words = word_sum(relation)
    ordinary_words = word_sum(ordinary)
    frac = relation_words / total_words
    interspersed = interleave_word_fraction(relation, ordinary, frac)
    if word_sum(interspersed) != total_words:
        raise RuntimeError("interspersed word mismatch")

    checkpoints = [x for x in range(1_000_000, total_words + 1, 1_000_000)]
    if checkpoints[-1] != total_words:
        checkpoints.append(total_words)

    front_transition = {
        "tail_words_at_relation_to_ordinary_transition": relation_words,
        "total_consumed_words_at_transition": int(args.initial_words) + relation_words,
        "relation_rows_before_transition": len(relation),
        "row_batched_updates_before_transition_macro256": row_macro_updates(len(relation), int(args.macro_rows)),
    }
    answer_tokens = estimate_relation_answer_tokens(relation)
    summary = {
        "status": "BRIDGE_SCHEDULE_PLAN_READY",
        "scientific_purpose": "Separate storage order from experimental curriculum and make legal-word, update-count, relation-dose, answer-label, and schedule-transition quantities explicit for the first BabyLM bridge.",
        "inputs": {"manifest": rel(overlay_dir / "overlay_manifest.json"), "overlay_tail": rel(overlay_path), "relation_rows": rel(relation_path)},
        "word_accounting": {
            "tail_words": total_words,
            "relation_words": relation_words,
            "ordinary_words": ordinary_words,
            "relation_fraction_tail_words": frac,
            "initial_words": int(args.initial_words),
            "final_total_words": int(args.initial_words) + total_words,
        },
        "row_accounting": {
            "reference_tail_rows_from_manifest": manifest["accounting"]["reference_tail_rows"],
            "overlay_rows": len(overlay),
            "relation_rows": len(relation),
            "ordinary_rows": len(ordinary),
            "row_macro_batch_size": int(args.macro_rows),
            "reference_row_batched_updates": row_macro_updates(int(manifest["accounting"]["reference_tail_rows"]), int(args.macro_rows)),
            "overlay_row_batched_updates": row_macro_updates(len(overlay), int(args.macro_rows)),
            "row_batched_update_inflation_vs_reference": row_macro_updates(len(overlay), int(args.macro_rows)) / max(1, row_macro_updates(int(manifest["accounting"]["reference_tail_rows"]), int(args.macro_rows))),
            "reference_updates": int(args.reference_updates),
        },
        "answer_label_accounting_relation_rows": answer_tokens,
        "frontloaded_transition": front_transition,
        "word_paced_354_update_batches": {
            "frontloaded": word_paced_batches(overlay, int(args.reference_updates)),
            "interspersed_wordpaced": word_paced_batches(interspersed, int(args.reference_updates)),
        },
        "checkpoint_relation_dose": {
            "frontloaded": checkpoint_stats(overlay, checkpoints),
            "interspersed_wordpaced": checkpoint_stats(interspersed, checkpoints),
        },
        "experimental_arms": {
            "ordinary_tail_reference": "Reuse research standard continuation as the no-substitution, ordinary-WWM reference: same parent, same tail words, 354 updates, final equal_valid_mean 44.307857 and Entity 27.24 on the fast screen.",
            "overlay_wwm_same_schedule": "Train on the relation-substituted corpus with ordinary WWM on every row under the chosen presentation schedule. This tests whether data substitution alone, without answer allocation, helps or hurts.",
            "overlay_answer_allocation_frontloaded": "Relation rows first with answer-span labels on relation rows and ordinary WWM on ordinary rows; score relation readout and Cheap7/Entity immediately after the relation block and along the ordinary-tail transition.",
            "overlay_answer_allocation_interspersed": "Same rows and objectives, but relation rows distributed across the ordinary tail; this is the coexistence test and should be the main bridge arm if the goal is cumulative learning rather than phase-specialization survival.",
        },
        "interpretation_note": "Do not interpret the frontloaded overlay endpoint alone: a negative endpoint can mean acquisition followed by loss, while a positive early relation-block state can be transient specialization. Read out relation behavior and broader competence on the same checkpoints across the transition.",
    }
    (out / "bridge_schedule_plan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # A small schedule-index file is enough for later trainer implementation without
    # duplicating the full text stream unless explicitly requested.
    schedule_index = []
    cum = rel_w = ord_w = 0
    for i, r in enumerate(interspersed):
        w = int(r["words"])
        cum += w
        if r.get("bridge_kind") == "relation_answer_packet":
            rel_w += w
            source_idx = r.get("presentation_source_index")
        else:
            ord_w += w
            source_idx = r.get("presentation_source_index")
        schedule_index.append({
            "presentation_index": i,
            "bridge_kind": r.get("bridge_kind"),
            "source": r.get("source"),
            "source_index_within_kind": source_idx,
            "words": w,
            "cum_words": cum,
            "cum_relation_words": rel_w,
            "cum_ordinary_words": ord_w,
        })
    write_jsonl(out / "interspersed_wordpaced_schedule_index.jsonl", schedule_index)
    if args.write_interspersed:
        write_jsonl(out / "overlay_tail_relation_e080_interspersed_wordpaced.jsonl", interspersed)
    print(json.dumps({
        "status": summary["status"],
        "out": rel(out / "bridge_schedule_plan_summary.json"),
        "tail_words": total_words,
        "relation_words": relation_words,
        "relation_fraction": frac,
        "reference_row_updates": summary["row_accounting"]["reference_row_batched_updates"],
        "overlay_row_updates": summary["row_accounting"]["overlay_row_batched_updates"],
        "front_transition_total_words": front_transition["total_consumed_words_at_transition"],
        "answer_token_positions": answer_tokens,
        "schedule_index": rel(out / "interspersed_wordpaced_schedule_index.jsonl"),
        "interspersed_written": bool(args.write_interspersed),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
