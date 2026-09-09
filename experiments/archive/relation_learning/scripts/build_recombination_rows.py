#!/usr/bin/env python3
"""research: Build deterministic recombination rows for entity-conditioned training.

From each DISTRACTOR packet (two-entity source), create TWO training rows:
  Row A: "The relevant state of {target_entity} is ___" → answer = source_state
         (target_entity was NOT updated; its state is unchanged)
  Row B: "The relevant state of {updated_entity} is ___" → answer = new_state
         (updated_entity WAS updated; its state changed)

These share the same source+update prefix and differ ONLY in the queried entity name.
This is the minimal pair that tests entity-conditioned retrieval: the same local context
yields opposite answers depending on which entity is queried.

From each UPDATED packet (single-entity source), create ONE training row:
  Row: "The relevant state of {entity} is ___" → answer = new_state
  (The entity was updated; use the new state.)

The resulting dataset supports:
  - Answer-only adapter training (freeze base, mask only answer tokens)
  - Joint correctness evaluation on DISTRACTOR pairs
  - T/U/N comparison against the research trusted base rescore

Usage:
    python build_recombination_rows.py [--max-items 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time
from typing import Any

ROOT = pathlib.Path.cwd()
TRAIN_PACKETS = ROOT / "experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_train.jsonl"
HELDOUT_PACKETS = ROOT / "experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_heldout.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/recombination_rows"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except: return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_space(s: str | None) -> str:
    if not s: return ""
    return re.sub(r'\s+', ' ', str(s)).strip()


def build_template_context(source: str, update: str | None, entity: str, answer: str) -> str:
    """Build the template-mode context: source [+ update] + query + answer."""
    parts = [norm_space(source)]
    if update:
        parts.append(norm_space(update))
    query = f"The relevant state of {norm_space(entity)} is "
    full_context = " ".join(parts) + " " + query + norm_space(answer) + "."
    # Record where the answer starts and ends in the full context
    answer_start = len(" ".join(parts) + " " + query)
    answer_end = answer_start + len(norm_space(answer))
    return full_context, answer_start, answer_end


def make_recombination_row(
    pkt: dict[str, Any],
    query_entity: str,
    answer_text: str,
    answer_kind: str,  # "source_state" or "new_state"
    role: str,  # "unchanged_entity" or "updated_entity"
    pair_half: str,  # "A" or "B" or "single"
    split: str,
) -> dict[str, Any]:
    """Create one recombination training/eval row."""
    source = norm_space(pkt.get("source_sentence"))
    update = norm_space(pkt.get("update_sentence"))
    
    context, ans_start, ans_end = build_template_context(
        source, update, query_entity, answer_text
    )
    
    return {
        "row_id": f"{split}:{pkt['pair_id']}:{role}",
        "pair_id": pkt["pair_id"],
        "split": split,
        "packet_type": pkt["packet_type"],
        "query_entity": norm_space(query_entity),
        "answer_text": norm_space(answer_text),
        "answer_kind": answer_kind,
        "role": role,
        "pair_half": pair_half,
        "source_sentence": source,
        "update_sentence": update,
        "source_state": norm_space(pkt.get("source_state")),
        "new_state": norm_space(pkt.get("new_state")),
        "target_entity": norm_space(pkt.get("target_entity")),
        "updated_entity": norm_space(pkt.get("updated_entity")),
        "context_text": context,
        "answer_char_start": ans_start,
        "answer_char_end": ans_end,
        "training_contract": "answer-only: mask/supervise only answer_text tokens; source+update+query visible",
    }


def process_packets(packets: list[dict[str, Any]], split: str, max_items: int = 0) -> tuple[list[dict], dict]:
    """Convert packets into recombination rows."""
    if max_items > 0:
        packets = packets[:max_items]
    
    rows = []
    stats = {"total_packets": 0, "distractor": 0, "updated": 0,
             "distractor_rows": 0, "updated_rows": 0, "skipped": 0}
    
    for pkt in packets:
        stats["total_packets"] += 1
        pid = pkt.get("pair_id", "")
        ptype = pkt.get("packet_type", "")
        
        target_ent = norm_space(pkt.get("target_entity"))
        updated_ent = norm_space(pkt.get("updated_entity"))
        source_state = norm_space(pkt.get("source_state"))
        new_state = norm_space(pkt.get("new_state"))
        
        if not all([target_ent, source_state, new_state, pkt.get("source_sentence"), pkt.get("update_sentence")]):
            stats["skipped"] += 1
            continue
        
        if ptype == "UNCHANGED_DISTRACTOR_USE":
            stats["distractor"] += 1
            # Row A: query target_entity → source_state (unchanged)
            row_a = make_recombination_row(
                pkt, query_entity=target_ent, answer_text=source_state,
                answer_kind="source_state", role="unchanged_entity",
                pair_half="A", split=split,
            )
            # Row B: query updated_entity → new_state (changed)
            row_b = make_recombination_row(
                pkt, query_entity=updated_ent, answer_text=new_state,
                answer_kind="new_state", role="updated_entity",
                pair_half="B", split=split,
            )
            rows.extend([row_a, row_b])
            stats["distractor_rows"] += 2
            
        elif ptype == "UPDATED_USE":
            stats["updated"] += 1
            # Single row: query updated_entity → new_state
            row = make_recombination_row(
                pkt, query_entity=updated_ent, answer_text=new_state,
                answer_kind="new_state", role="updated_entity",
                pair_half="single", split=split,
            )
            rows.append(row)
            stats["updated_rows"] += 1
        else:
            stats["skipped"] += 1
    
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read packets
    train_pkts = read_jsonl(TRAIN_PACKETS)
    heldout_pkts = read_jsonl(HELDOUT_PACKETS)
    
    # Process
    train_rows, train_stats = process_packets(train_pkts, "train", args.max_items)
    heldout_rows, heldout_stats = process_packets(heldout_pkts, "heldout", args.max_items)
    
    # Write rows
    train_path = OUT_DIR / "recombination_train.jsonl"
    heldout_path = OUT_DIR / "recombination_heldout.jsonl"
    all_path = OUT_DIR / "recombination_all.jsonl"
    
    for path, rows in [(train_path, train_rows), (heldout_path, heldout_rows), (all_path, train_rows + heldout_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Held-out binding pairs for joint evaluation
    # Only DISTRACTOR packets produce true pairs (A and B halves)
    binding_pairs = []
    heldout_by_pid = {}
    for r in heldout_rows:
        pid = r["pair_id"]
        if pid not in heldout_by_pid:
            heldout_by_pid[pid] = {}
        heldout_by_pid[pid][r["pair_half"]] = r
    
    for pid, halves in heldout_by_pid.items():
        if "A" in halves and "B" in halves:
            binding_pairs.append({
                "pair_id": pid,
                "row_a_id": halves["A"]["row_id"],
                "row_b_id": halves["B"]["row_id"],
                "entity_a": halves["A"]["query_entity"],
                "entity_b": halves["B"]["query_entity"],
                "answer_a": halves["A"]["answer_text"],
                "answer_b": halves["B"]["answer_text"],
            })
    
    binding_path = OUT_DIR / "binding_pairs_heldout.jsonl"
    with open(binding_path, "w", encoding="utf-8") as f:
        for bp in binding_pairs:
            f.write(json.dumps(bp, ensure_ascii=False) + "\n")
    
    # Context length audit
    from collections import Counter
    length_bins = Counter()
    for r in train_rows + heldout_rows:
        wc = len(r["context_text"].split())
        if wc <= 64: length_bins["0-64"] += 1
        elif wc <= 128: length_bins["65-128"] += 1
        elif wc <= 192: length_bins["129-192"] += 1
        elif wc <= 256: length_bins["193-256"] += 1
        else: length_bins["257+"] += 1
    
    # Answer token length audit
    ans_lengths = Counter()
    for r in train_rows + heldout_rows:
        nw = len(r["answer_text"].split())
        ans_lengths[nw] += 1
    
    summary = {
        "status": "RECOMBINATION_ROWS_DONE",
        "created_utc": now_utc(),
        "source_shas": {
            "train": sha256_file(TRAIN_PACKETS),
            "heldout": sha256_file(HELDOUT_PACKETS),
        },
        "train_stats": train_stats,
        "heldout_stats": heldout_stats,
        "row_counts": {
            "train": len(train_rows),
            "heldout": len(heldout_rows),
            "total": len(train_rows) + len(heldout_rows),
        },
        "binding_pairs_heldout": len(binding_pairs),
        "context_word_length_distribution": dict(sorted(length_bins.items())),
        "answer_word_length_distribution": dict(sorted(ans_lengths.items())),
        "paths": {
            "train": rel(train_path),
            "heldout": rel(heldout_path),
            "all": rel(all_path),
            "binding_pairs": rel(binding_path),
        },
        "design": {
            "distractor_pairs": "Each DISTRACTOR packet → 2 rows: query unchanged entity (source_state) + query updated entity (new_state). Same context, different entity name.",
            "updated_singles": "Each UPDATED packet → 1 row: query entity (new_state). No second entity for pairing.",
            "joint_criterion": "For DISTRACTOR pairs, model must get BOTH halves correct. This is the binding test.",
            "training_contract": "Answer-only MLM: mask only answer tokens; source+update+query visible; freeze base; train adapter only.",
        },
    }
    
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
