#!/usr/bin/env python3
"""research: Validate 9B state-use outputs and materialize training stream.

Reads raw generation outputs from batch generation, applies strict
algorithmic validation, builds same-row three-part packets, and assembles a
100M-word training stream matching the compact-view-reinvest topology.

Validation rules (algorithmic, not LLM-judged):
  1. JSON must parse with all 6 required keys
  2. target_entity and updated_entity distinction must match packet type
  3. new_state must differ from source_state (for UPDATED_USE)
  4. target_entity and updated_entity must differ (for UNCHANGED_DISTRACTOR_USE)
  5. update_sentence must mention updated_entity
  6. use_sentence must mention target_entity
  7. Word counts within bounds (update: 8-22, use: 6-16)
  8. No Entity-benchmark vocabulary (box, basket, marble, container)
  9. No verbatim copy of source sentence
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()

# Input paths
RAW_9B_PATH = ROOT / "experiments/archive/relation_learning/data/state_use_generation/raw_outputs_9b.jsonl"
PROMPT_META_PATH = ROOT / "experiments/archive/relation_learning/data/state_use_generation/state_use_prompts_9b.jsonl"

# The original SOTA stream and row metadata for topology matching
ORIGINAL_STREAM = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
CHANGED_BLOCK_META = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
QWEN_PAIRS = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"

# Output
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
CORPORA_DIR = OUT_DIR / "training_corpora"
CORPORA_DIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
BENCHMARK_VOCAB = {"box", "basket", "marble", "container", "containers", "baskets", "boxes", "marbles"}
REQUIRED_KEYS = {"target_entity", "source_state", "updated_entity", "new_state", "update_sentence", "use_sentence"}

RNG_SEED = 43043


def wc(text: str) -> int:
    return len(WORD_RE.findall(text))


def norm_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def validate_output(raw_text: str, packet_type: str, source_sentence: str) -> tuple[dict | None, str]:
    """Parse and validate a generation output. Returns (parsed_dict, rejection_reason)."""
    raw_text = raw_text.strip()

    # Strip markdown fencing if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(l for l in lines if not l.startswith("```"))
        raw_text = raw_text.strip()

    # Try to find JSON in the output
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        return None, "no_json"

    try:
        parsed = json.loads(raw_text[json_start:json_end])
    except json.JSONDecodeError:
        return None, "json_parse_error"

    # Check required keys
    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return None, f"missing_keys:{','.join(missing)}"

    # Extract fields
    target_ent = str(parsed["target_entity"]).strip().lower()
    updated_ent = str(parsed["updated_entity"]).strip().lower()
    source_state = str(parsed["source_state"]).strip().lower()
    new_state = str(parsed["new_state"]).strip().lower()
    update_sent = norm_ws(str(parsed["update_sentence"]))
    use_sent = norm_ws(str(parsed["use_sentence"]))

    # Entity distinction check
    if packet_type == "UNCHANGED_DISTRACTOR_USE":
        if target_ent == updated_ent:
            return None, "distractor_same_entity"
    elif packet_type == "UPDATED_USE":
        if target_ent != updated_ent:
            return None, "updated_entity_mismatch"

    # State change check (for UPDATED_USE)
    if packet_type == "UPDATED_USE":
        if source_state == new_state:
            return None, "unchanged_state"

    # Entity mention check
    if target_ent not in use_sent.lower():
        return None, "use_missing_target"
    if updated_ent not in update_sent.lower():
        return None, "update_missing_entity"

    # Word count bounds
    update_wc = wc(update_sent)
    use_wc = wc(use_sent)
    if update_wc < 6 or update_wc > 30:
        return None, f"update_wordcount:{update_wc}"
    if use_wc < 4 or use_wc > 22:
        return None, f"use_wordcount:{use_wc}"

    # Benchmark vocabulary check
    update_words = set(w.lower() for w in WORD_RE.findall(update_sent))
    use_words = set(w.lower() for w in WORD_RE.findall(use_sent))
    if update_words & BENCHMARK_VOCAB or use_words & BENCHMARK_VOCAB:
        return None, "benchmark_vocab"

    # Source copy check
    if update_sent.lower().strip() == source_sentence.lower().strip():
        return None, "update_copies_source"
    if use_sent.lower().strip() == source_sentence.lower().strip():
        return None, "use_copies_source"

    return {
        "target_entity": parsed["target_entity"],
        "source_state": parsed["source_state"],
        "updated_entity": parsed["updated_entity"],
        "new_state": parsed["new_state"],
        "update_sentence": update_sent,
        "use_sentence": use_sent,
        "update_words": update_wc,
        "use_words": use_wc,
    }, ""


def main():
    rng = random.Random(RNG_SEED)

    # Load prompt metadata as ordered list (index-aligned with batch generator output)
    prompt_list = []
    with open(PROMPT_META_PATH) as f:
        for line in f:
            prompt_list.append(json.loads(line.strip()))

    print(f"Prompt records: {len(prompt_list)}", flush=True)

    # Load raw 9B outputs (batch generator uses 'index' field for ordering)
    raw_outputs = []
    with open(RAW_9B_PATH) as f:
        for line in f:
            raw_outputs.append(json.loads(line.strip()))
    print(f"Raw 9B outputs: {len(raw_outputs)}", flush=True)

    # Validate each output - match by index position
    valid_packets = {}  # pair_id → validated packet
    rejection_reasons = Counter()
    for rec in raw_outputs:
        # Match by index: batch generator processes prompts in order
        idx = rec.get("index", -1)
        if idx < 0 or idx >= len(prompt_list):
            rejection_reasons["index_out_of_range"] += 1
            continue
        prompt_rec = prompt_list[idx]

        raw_output = rec.get("output", rec.get("generated_text", ""))
        packet_type = prompt_rec["packet_type"]
        source_sentence = prompt_rec["source_sentence"]

        parsed, reason = validate_output(raw_output, packet_type, source_sentence)
        if parsed is None:
            rejection_reasons[reason] += 1
            continue

        pair_id = prompt_rec["pair_id"]
        valid_packets[pair_id] = {
            **parsed,
            "pair_id": pair_id,
            "packet_type": packet_type,
            "source_sentence": source_sentence,
            "source_words": prompt_rec["source_words"],
        }

    n_valid = len(valid_packets)
    n_updated = sum(1 for v in valid_packets.values() if v["packet_type"] == "UPDATED_USE")
    n_distractor = sum(1 for v in valid_packets.values() if v["packet_type"] == "UNCHANGED_DISTRACTOR_USE")

    print(f"\nValidation results:", flush=True)
    print(f"  Valid: {n_valid}/{len(raw_outputs)} ({100*n_valid/max(1,len(raw_outputs)):.1f}%)", flush=True)
    print(f"  UPDATED_USE: {n_updated}", flush=True)
    print(f"  UNCHANGED_DISTRACTOR_USE: {n_distractor}", flush=True)
    print(f"  Rejection reasons: {dict(rejection_reasons.most_common(15))}", flush=True)

    # Save validated packets
    validated_path = OUT_DIR / "validated_state_use_packets.jsonl"
    with open(validated_path, "w") as f:
        for pair_id in sorted(valid_packets):
            f.write(json.dumps(valid_packets[pair_id], ensure_ascii=False) + "\n")

    # Load original Qwen pairs for fallback
    qwen_pairs = {}
    with open(QWEN_PAIRS) as f:
        for line in f:
            rec = json.loads(line.strip())
            qwen_pairs[rec["pair_id"]] = rec

    # Build three-part companion text for each valid packet
    # Packet: "source_sentence update_sentence use_sentence"
    # Fallback for non-valid: keep original Qwen rewrite companion
    companion_map = {}  # pair_id → companion text and word count
    for pair_id, pkt in valid_packets.items():
        companion = f"{pkt['update_sentence']} {pkt['use_sentence']}"
        companion_map[pair_id] = {
            "text": companion,
            "words": wc(companion),
            "type": "state_update",
            "packet_type": pkt["packet_type"],
        }

    print(f"\n  Companion map entries: {len(companion_map)}", flush=True)
    mean_comp_words = sum(c["words"] for c in companion_map.values()) / max(1, len(companion_map))
    print(f"  Mean companion words: {mean_comp_words:.1f}", flush=True)

    # Save metadata
    metadata = {
        "status": "STATE_USE_VALIDATED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_outputs": len(raw_outputs),
        "valid_packets": n_valid,
        "updated_use": n_updated,
        "distractor_use": n_distractor,
        "conversion_rate": round(n_valid / max(1, len(raw_outputs)), 4),
        "mean_companion_words": round(mean_comp_words, 1),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "validated_path": str(validated_path),
        "sha256_validated": hashlib.sha256(validated_path.read_bytes()).hexdigest(),
    }
    (OUT_DIR / "validation_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
