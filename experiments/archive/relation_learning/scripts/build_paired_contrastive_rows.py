#!/usr/bin/env python3
"""research: Build paired A+B frame-varied training rows.

The research frame-varied rows only contain UNCHANGED_DISTRACTOR_USE (row A).
For the paired contrastive objective, we need BOTH halves of each pair:
- Row A: query the unchanged entity → correct answer is source state
- Row B: query the updated entity → correct answer is new state

This script reads the original research train rows and the research frame templates,
then builds matched A+B pairs with frame variation on both halves.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import re
import hashlib
import collections
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')

TRAIN = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl')
HELD = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
PAIRS_HELD = _public_path('experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl')

# Frame templates from research
TRAIN_FRAMES = [
    ("f00_original_relevant_state", "train_seen", "The relevant state of {entity} is {answer}."),
    ("f01_at_this_point", "train_seen", "At this point, {entity} is {answer}."),
    ("f02_for_current_state", "train_seen", "For {entity}, the current state is {answer}."),
    ("f03_currently_associated", "train_seen", "The state currently associated with {entity} is {answer}."),
    ("f04_given_context", "train_seen", "Given the context, {entity} is {answer}."),
]

HELD_EXTRA_FRAMES = [
    ("f05_based_on_passage", "eval_unseen", "Based on the passage, {entity} is {answer}."),
    ("f06_by_end_of_passage", "eval_unseen", "By the end of the passage, {entity} is {answer}."),
    ("f07_context_indicates", "eval_unseen", "The context indicates that {entity} is {answer}."),
]

ALL_HELD_FRAMES = TRAIN_FRAMES + HELD_EXTRA_FRAMES


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def strip_query_from_context(context_text: str) -> tuple[str, str]:
    """Split a row's context_text into base_context and query_sentence.
    
    The query sentence is the last sentence containing the answer slot or
    entity-state phrasing.
    """
    # The original rows end with a query sentence like:
    # "The relevant state of {entity} is {answer}."
    # Try to find it by looking for common patterns
    patterns = [
        r"The relevant state of .+? is .+?\.$",
        r"At this point, .+? is .+?\.$",
        r"For .+?, the current state is .+?\.$",
        r"The state currently associated with .+? is .+?\.$",
        r"Given the context, .+? is .+?\.$",
        r"Based on the passage, .+? is .+?\.$",
        r"By the end of the passage, .+? is .+?\.$",
        r"The context indicates that .+? is .+?\.$",
    ]
    for pat in patterns:
        m = re.search(pat, context_text)
        if m:
            query = m.group(0)
            base = context_text[:m.start()].rstrip()
            return base, query
    
    # Fallback: split at last sentence
    sentences = re.split(r'(?<=\.)\s+', context_text)
    if len(sentences) >= 2:
        return ' '.join(sentences[:-1]), sentences[-1]
    return context_text, ""


def build_query_sentence(frame_template: str, entity: str, answer: str) -> str:
    return frame_template.format(entity=entity, answer=answer)


def build_paired_rows(base_rows: list[dict], split: str, frames: list[tuple[str, str, str]]) -> tuple[list[dict], list[dict]]:
    """Build paired A+B rows with frame variation.
    
    Returns (paired_rows, binding_pairs).
    """
    # Group by pair_id
    by_pair: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in base_rows:
        pid = r["pair_id"]
        ptype = r["packet_type"]
        by_pair[pid][ptype] = r
    
    paired_rows = []
    binding_pairs = []
    
    for pid, type_map in sorted(by_pair.items()):
        if "UNCHANGED_DISTRACTOR_USE" not in type_map or "UPDATED_USE" not in type_map:
            continue
        
        row_a_base = type_map["UNCHANGED_DISTRACTOR_USE"]
        row_b_base = type_map["UPDATED_USE"]
        
        # Extract base context (without query sentence) from row A
        base_ctx_a, _ = strip_query_from_context(row_a_base["context_text"])
        base_ctx_b, _ = strip_query_from_context(row_b_base["context_text"])
        
        # The base context should be the same for both rows of a pair
        # (same source + update, just different query)
        base_ctx = base_ctx_a  # Use row A's base context
        
        entity_a = row_a_base["query_entity"]
        entity_b = row_b_base["query_entity"]
        answer_a = row_a_base["answer_text"]
        answer_b = row_b_base["answer_text"]
        
        for frame_id, frame_split, template in frames:
            # Build row A: query unchanged entity
            query_a = build_query_sentence(template, entity_a, answer_a)
            full_ctx_a = base_ctx + " " + query_a
            
            # Find answer span in full context
            ans_start_a = full_ctx_a.rfind(answer_a)
            
            row_a = {
                "row_id": f"{split}:{pid}:{frame_id}:unchanged_entity",
                "pair_id": f"{pid}::{frame_id}",
                "split": split,
                "packet_type": "UNCHANGED_DISTRACTOR_USE",
                "query_entity": entity_a,
                "answer_text": answer_a,
                "context_text": full_ctx_a,
                "answer_char_start": ans_start_a,
                "answer_char_end": ans_start_a + len(answer_a) if ans_start_a >= 0 else -1,
                "frame_id": frame_id,
                "frame_split": frame_split,
                "pair_role": "A",
                "context_words": len(full_ctx_a.split()),
            }
            
            # Build row B: query updated entity
            query_b = build_query_sentence(template, entity_b, answer_b)
            full_ctx_b = base_ctx + " " + query_b
            
            ans_start_b = full_ctx_b.rfind(answer_b)
            
            row_b = {
                "row_id": f"{split}:{pid}:{frame_id}:updated_entity",
                "pair_id": f"{pid}::{frame_id}",
                "split": split,
                "packet_type": "UPDATED_USE",
                "query_entity": entity_b,
                "answer_text": answer_b,
                "context_text": full_ctx_b,
                "answer_char_start": ans_start_b,
                "answer_char_end": ans_start_b + len(answer_b) if ans_start_b >= 0 else -1,
                "frame_id": frame_id,
                "frame_split": frame_split,
                "pair_role": "B",
                "context_words": len(full_ctx_b.split()),
            }
            
            paired_rows.extend([row_a, row_b])
            
            binding_pairs.append({
                "pair_id": f"{pid}::{frame_id}",
                "base_pair_id": pid,
                "frame_id": frame_id,
                "frame_split": frame_split,
                "row_a_id": row_a["row_id"],
                "row_b_id": row_b["row_id"],
                "entity_a": entity_a,
                "entity_b": entity_b,
                "answer_a": answer_a,
                "answer_b": answer_b,
            })
    
    return paired_rows, binding_pairs


def main():
    out = _public_path('experiments/archive/relation_learning/data/paired_contrastive_rows')
    out.mkdir(parents=True, exist_ok=True)
    
    # Load original research rows
    train_rows = load_rows(TRAIN)
    held_rows = load_rows(HELD)
    
    print(f"research train: {len(train_rows)} rows", flush=True)
    print(f"research held: {len(held_rows)} rows", flush=True)
    
    # Count types
    train_types = collections.Counter(r["packet_type"] for r in train_rows)
    held_types = collections.Counter(r["packet_type"] for r in held_rows)
    print(f"Train types: {dict(train_types)}")
    print(f"Held types: {dict(held_types)}")
    
    # Build paired frame-varied rows
    train_paired, train_pairs = build_paired_rows(train_rows, "train", TRAIN_FRAMES)
    held_paired, held_pairs = build_paired_rows(held_rows, "heldout", ALL_HELD_FRAMES)
    
    # Verify: check answer spans
    bad_spans = 0
    for r in train_paired + held_paired:
        if r["answer_char_start"] < 0:
            bad_spans += 1
    
    print(f"\nPaired train rows: {len(train_paired)} ({len(train_pairs)} pairs)")
    print(f"Paired held rows: {len(held_paired)} ({len(held_pairs)} pairs)")
    print(f"Bad answer spans: {bad_spans}")
    
    # Split train into frame-seen only
    train_seen_rows = [r for r in train_paired if r["frame_split"] == "train_seen"]
    train_seen_pairs = [p for p in train_pairs if p["frame_split"] == "train_seen"]
    
    # Held all, seen, unseen
    held_seen_rows = [r for r in held_paired if r["frame_split"] == "train_seen"]
    held_unseen_rows = [r for r in held_paired if r["frame_split"] == "eval_unseen"]
    held_seen_pairs = [p for p in held_pairs if p["frame_split"] == "train_seen"]
    held_unseen_pairs = [p for p in held_pairs if p["frame_split"] == "eval_unseen"]
    
    # Write files
    def write_jsonl(path, items):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(items)
    
    files = {
        "train_frame_seen": (train_seen_rows, train_seen_pairs),
        "train_frame_all": (train_paired, train_pairs),
        "heldout_frame_all": (held_paired, held_pairs),
        "heldout_frame_seen": (held_seen_rows, held_seen_pairs),
        "heldout_frame_unseen": (held_unseen_rows, held_unseen_pairs),
    }
    
    manifest = {
        "status": "PAIRED_CONTRASTIVE_ROWS",
        "source": "research recombination rows with research frame templates",
        "frames_train_seen": [f[0] for f in TRAIN_FRAMES],
        "frames_eval_unseen": [f[0] for f in HELD_EXTRA_FRAMES],
    }
    
    for name, (rows, pairs) in files.items():
        row_path = out / f"paired_{name}.jsonl"
        pair_path = out / f"binding_pairs_{name}.jsonl"
        n_rows = write_jsonl(row_path, rows)
        n_pairs = write_jsonl(pair_path, pairs)
        manifest[name] = {
            "rows_path": str(row_path.relative_to(ROOT)),
            "pairs_path": str(pair_path.relative_to(ROOT)),
            "n_rows": n_rows,
            "n_pairs": n_pairs,
            "types": dict(collections.Counter(r["packet_type"] for r in rows)),
        }
    
    # Compute hash of train file
    h = hashlib.sha256()
    with open(out / "paired_train_frame_seen.jsonl", "rb") as f:
        h.update(f.read())
    manifest["train_frame_seen_sha256"] = h.hexdigest()
    manifest["bad_answer_spans"] = bad_spans
    
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
