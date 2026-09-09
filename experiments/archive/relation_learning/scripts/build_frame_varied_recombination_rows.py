#!/usr/bin/env python3
"""research: deterministic frame-varied recombination rows.

The existing research binding rows fix the source sentence, update sentence,
unchanged/updated query entities, and source/new state candidates.  This script
changes only the answer-bearing query sentence across several natural phrasings.
It creates pair IDs that include the frame ID so pair-batch trainers can treat each
frame-pair independently while preserving the identical-context A/B structure within
that frame.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path(".").resolve()
IN_DIR = ROOT / "experiments/archive/relation_learning/data/recombination_rows"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/frame_varied_recombination_rows"

# Original frame plus deterministic natural rephrasings.  Train frames are meant to
# practice the same relation under form variation.  Eval-only frames test whether the
# readout remains tied to seen query wording.
FRAMES = [
    {"frame_id": "f00_original_relevant_state", "frame_split": "train_seen", "template": "The relevant state of {entity} is {answer}.", "style": "original_step072"},
    {"frame_id": "f01_at_this_point", "frame_split": "train_seen", "template": "At this point, {entity} is {answer}.", "style": "short_state_predication"},
    {"frame_id": "f02_for_current_state", "frame_split": "train_seen", "template": "For {entity}, the current state is {answer}.", "style": "fronted_entity"},
    {"frame_id": "f03_currently_associated", "frame_split": "train_seen", "template": "The state currently associated with {entity} is {answer}.", "style": "nominal_association"},
    {"frame_id": "f04_given_context", "frame_split": "train_seen", "template": "Given this context, {entity} should be described as {answer}.", "style": "description_frame"},
    {"frame_id": "f05_after_descriptions", "frame_split": "eval_unseen", "template": "After these descriptions, {entity} is {answer}.", "style": "temporal_summary"},
    {"frame_id": "f06_by_the_end", "frame_split": "eval_unseen", "template": "By the end of the passage, {entity} is {answer}.", "style": "end_state"},
    {"frame_id": "f07_context_indicates", "frame_split": "eval_unseen", "template": "The context indicates that {entity} is {answer}.", "style": "evidence_report"},
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_sentence(s: str) -> str:
    t = re.sub(r"\s+", " ", str(s or "")).strip()
    if t and t[-1] not in ".!?]":
        t += "."
    return t


def clean_answer(s: str) -> str:
    t = re.sub(r"\s+", " ", str(s or "")).strip()
    return t.rstrip(" .")


def materialize_row(row: dict[str, Any], frame: dict[str, str], split: str) -> dict[str, Any]:
    entity = re.sub(r"\s+", " ", str(row["query_entity"])).strip()
    answer = clean_answer(str(row["answer_text"]))
    query = frame["template"].format(entity=entity, answer=answer)
    prefix = f"{clean_sentence(row['source_sentence'])} {clean_sentence(row['update_sentence'])} "
    context = prefix + query
    answer_start = len(prefix) + query.index(answer)
    answer_end = answer_start + len(answer)
    old_pair = str(row["pair_id"])
    new_pair = f"{old_pair}::{frame['frame_id']}"
    new = dict(row)
    new.update({
        "row_id": f"{split}:{old_pair}:{frame['frame_id']}:{row['role']}",
        "pair_id": new_pair,
        "base_pair_id": old_pair,
        "split": split,
        "frame_id": frame["frame_id"],
        "frame_split": frame["frame_split"],
        "frame_style": frame["style"],
        "query_template": frame["template"],
        "answer_text": answer,
        "context_text": context,
        "answer_char_start": answer_start,
        "answer_char_end": answer_end,
        "training_contract": "frame-varied answer-only: source_sentence and update_sentence unchanged; mask/supervise answer_text tokens; query frame varied deterministically",
    })
    assert context[answer_start:answer_end] == answer
    return new


def build(rows: list[dict[str, Any]], split: str, frames: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("pair_half") not in ("A", "B"):
            continue
        for frame in frames:
            out.append(materialize_row(row, frame, split))
    return out


def pair_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[r["pair_id"]].append(r)
    out = []
    for pid, rs in sorted(by.items()):
        if len(rs) != 2:
            continue
        a = [r for r in rs if r.get("pair_half") == "A"]
        b = [r for r in rs if r.get("pair_half") == "B"]
        if len(a) != 1 or len(b) != 1:
            continue
        ar, br = a[0], b[0]
        out.append({
            "pair_id": pid,
            "base_pair_id": ar.get("base_pair_id"),
            "frame_id": ar.get("frame_id"),
            "frame_split": ar.get("frame_split"),
            "row_a_id": ar["row_id"],
            "row_b_id": br["row_id"],
            "source_sentence": ar.get("source_sentence"),
            "update_sentence": ar.get("update_sentence"),
            "source_state": ar.get("source_state"),
            "new_state": ar.get("new_state"),
            "unchanged_query_entity": ar.get("query_entity"),
            "updated_query_entity": br.get("query_entity"),
        })
    return out


def count_words(rows: list[dict[str, Any]]) -> int:
    return sum(len(str(r.get("context_text", "")).split()) for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train_in = read_jsonl(IN_DIR / "recombination_train.jsonl")
    held_in = read_jsonl(IN_DIR / "recombination_heldout.jsonl")
    train_frames = [f for f in FRAMES if f["frame_split"] == "train_seen"]
    all_frames = FRAMES
    train_seen_rows = build(train_in, "train", train_frames)
    train_all_rows = build(train_in, "train", all_frames)
    held_all_rows = build(held_in, "heldout", all_frames)
    held_seen_rows = [r for r in held_all_rows if r["frame_split"] == "train_seen"]
    held_unseen_rows = [r for r in held_all_rows if r["frame_split"] == "eval_unseen"]
    files = {
        "train_seen_frames": (out / "recombination_train_frame_seen.jsonl", train_seen_rows),
        "train_all_frames": (out / "recombination_train_frame_all.jsonl", train_all_rows),
        "heldout_all_frames": (out / "recombination_heldout_frame_all.jsonl", held_all_rows),
        "heldout_seen_frames": (out / "recombination_heldout_frame_seen.jsonl", held_seen_rows),
        "heldout_unseen_frames": (out / "recombination_heldout_frame_unseen.jsonl", held_unseen_rows),
    }
    for _name, (path, rows) in files.items():
        write_jsonl(path, rows)
    pair_files = {
        "binding_pairs_heldout_all_frames": (out / "binding_pairs_heldout_frame_all.jsonl", pair_records(held_all_rows)),
        "binding_pairs_heldout_seen_frames": (out / "binding_pairs_heldout_frame_seen.jsonl", pair_records(held_seen_rows)),
        "binding_pairs_heldout_unseen_frames": (out / "binding_pairs_heldout_frame_unseen.jsonl", pair_records(held_unseen_rows)),
    }
    for _name, (path, rows) in pair_files.items():
        write_jsonl(path, rows)
    by_frame = Counter(r["frame_id"] for r in train_all_rows)
    manifest = {
        "status": "FRAME_VARIED_RECOMBINATION_ROWS",
        "created_utc": now(),
        "input_dir": rel(IN_DIR),
        "out_dir": rel(out),
        "frames": FRAMES,
        "counts": {name: {"path": rel(path), "rows": len(rows), "pairs": len(pair_records(rows)) if "heldout" in name else len(set(r["pair_id"] for r in rows)), "words": count_words(rows)} for name, (path, rows) in files.items()},
        "pair_files": {name: {"path": rel(path), "pairs": len(rows)} for name, (path, rows) in pair_files.items()},
        "train_all_rows_by_frame": dict(by_frame),
        "science_note": "Only the final query sentence is rephrased; source/update evidence, entities, source_state and new_state are preserved. Train-seen frames allow multi-form practice; eval-unseen frames test whether the readout escapes a single query wording.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research frame-varied recombination rows", "", manifest["science_note"], "", "| file | rows | pairs | words |", "|---|---:|---:|---:|"]
    for name, rec in manifest["counts"].items():
        lines.append(f"| {name} | {rec['rows']} | {rec['pairs']} | {rec['words']} |")
    lines += ["", "## Frames", "", "| frame | split | template |", "|---|---|---|"]
    for f in FRAMES:
        lines.append(f"| {f['frame_id']} | {f['frame_split']} | {f['template']} |")
    (out / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": rel(out / "manifest.json"), "counts": manifest["counts"], "pair_files": manifest["pair_files"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
