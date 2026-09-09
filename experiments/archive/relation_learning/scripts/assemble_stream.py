#!/usr/bin/env python3
"""research: Assemble 100M training stream with state-update companions.

Reads the validated state-use packets, the original compact-view-reinvest stream,
and the changed-block metadata, then:
1. For pair-packed rows with validated state-update companions: rebuild the row
   with original + update_sentence + use_sentence instead of original + rewrite.
2. For pair-packed rows without validated companions: keep Qwen rewrite (fallback).
3. For non-pair rows: keep unchanged.
4. Verify total = 100M words; trim/pad filler as needed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import re
import time
from typing import Any

ROOT = pathlib.Path.cwd()

# Inputs
VALIDATED_PACKETS = ROOT / "experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets.jsonl"
ORIGINAL_STREAM = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
CHANGED_BLOCK_META = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
QWEN_PAIRS = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"

# Output
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/state_use_generation/training_corpora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
TARGET_WORDS = 100_000_000
RNG_SEED = 43043


def wc(text: str) -> int:
    return len(WORD_RE.findall(text))


def norm_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()

    # 1. Load validated state-update packets
    print("Loading validated packets...", flush=True)
    state_updates = {}  # pair_id → packet
    with open(VALIDATED_PACKETS) as f:
        for line in f:
            pkt = json.loads(line.strip())
            state_updates[pkt["pair_id"]] = pkt
    print(f"  {len(state_updates)} validated packets", flush=True)

    # 2. Load Qwen selected pairs (for fallback)
    print("Loading Qwen pairs...", flush=True)
    qwen_pairs = {}  # pair_id → {original, rewrite}
    with open(QWEN_PAIRS) as f:
        for line in f:
            p = json.loads(line.strip())
            qwen_pairs[p["pair_id"]] = p
    print(f"  {len(qwen_pairs)} Qwen pairs", flush=True)

    # 3. Load changed-block row metadata → map row_index to ordered pair_ids
    print("Loading changed-block metadata...", flush=True)
    changed_rows = {}  # row_index → meta dict
    with open(CHANGED_BLOCK_META) as f:
        for line in f:
            meta = json.loads(line.strip())
            changed_rows[meta["row_index"]] = meta
    print(f"  {len(changed_rows)} changed-block rows", flush=True)

    # 4. Build companion map for each pair_id
    # Priority: state-update > Qwen rewrite
    def get_companion(pair_id: str) -> tuple[str, str, str]:
        """Returns (original_text, companion_text, relation_type)."""
        qp = qwen_pairs.get(pair_id, {})
        original = norm_ws(qp.get("original", ""))

        if pair_id in state_updates:
            pkt = state_updates[pair_id]
            companion = f"{pkt['update_sentence']} {pkt['use_sentence']}"
            return original, companion, pkt["packet_type"]
        else:
            return original, norm_ws(qp.get("rewrite", "")), "restatement_fallback"

    # 5. Read original stream and rebuild
    print("Rebuilding stream...", flush=True)
    new_rows = []
    total_words = 0
    pair_row_count = 0
    converted_pairs = 0
    fallback_pairs = 0
    filler_row_indices = []
    relation_counts = collections.Counter()

    with open(ORIGINAL_STREAM) as f:
        for row_idx, line in enumerate(f):
            row = json.loads(line.strip())

            if row_idx in changed_rows:
                # This is a pair-packed row → rebuild from pair data
                meta = changed_rows[row_idx]
                pair_ids = meta["pair_ids"]
                segments = []
                row_word_sum = 0

                for pid in pair_ids:
                    orig, comp, rel_type = get_companion(pid)
                    relation_counts[rel_type] += 1
                    if rel_type != "restatement_fallback":
                        converted_pairs += 1
                    else:
                        fallback_pairs += 1
                    segments.append(orig)
                    segments.append(comp)
                    row_word_sum += wc(orig) + wc(comp)

                new_text = " ".join(s for s in segments if s)
                actual_wc = wc(new_text)
                new_rows.append({
                    "text": new_text,
                    "words": actual_wc,
                    "row_index": row_idx,
                    "row_type": "pair_packed",
                    "pair_ids": pair_ids,
                })
                total_words += actual_wc
                pair_row_count += 1
            else:
                # Non-pair row → keep unchanged
                new_rows.append({
                    "text": row["text"],
                    "words": row["words"],
                    "row_index": row_idx,
                    "row_type": "unchanged",
                })
                total_words += row["words"]
                filler_row_indices.append(len(new_rows) - 1)

    print(f"  Total rows: {len(new_rows)} ({pair_row_count} pair, {len(new_rows)-pair_row_count} unchanged)", flush=True)
    print(f"  Converted pairs: {converted_pairs}, Fallback pairs: {fallback_pairs}", flush=True)
    print(f"  Relation types: {dict(relation_counts)}", flush=True)
    print(f"  Total words BEFORE adjustment: {total_words:,} (target: {TARGET_WORDS:,})", flush=True)

    # 6. Adjust to exactly 100M words
    excess = total_words - TARGET_WORDS
    rng = random.Random(RNG_SEED)

    if excess > 0:
        # Need to remove words. Drop smallest filler rows.
        filler_by_size = sorted(filler_row_indices, key=lambda i: new_rows[i]["words"])
        dropped_indices = set()
        removed_words = 0
        for fi in filler_by_size:
            if removed_words >= excess:
                break
            removed_words += new_rows[fi]["words"]
            dropped_indices.add(fi)
        print(f"  Dropping {len(dropped_indices)} smallest filler rows ({removed_words} words) to compensate +{excess} excess", flush=True)
        new_rows = [r for i, r in enumerate(new_rows) if i not in dropped_indices]
        total_words -= removed_words
    elif excess < 0:
        print(f"  WARNING: {-excess} words below target. This is unusual. Stream will be slightly under 100M.", flush=True)

    # Recount for verification
    final_words = sum(r["words"] for r in new_rows)
    print(f"  Final total words: {final_words:,} (target: {TARGET_WORDS:,}, diff: {final_words - TARGET_WORDS:+,})", flush=True)

    # 7. Write output stream
    stream_path = OUT_DIR / "state_update_100M.jsonl"
    with open(stream_path, "w") as f:
        for r in new_rows:
            out_row = {"text": r["text"], "words": r["words"]}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    # 8. Save metadata
    metadata = {
        "status": "STATE_UPDATE_STREAM_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream_path": str(stream_path),
        "sha256": sha256_file(stream_path),
        "total_rows": len(new_rows),
        "pair_rows": pair_row_count,
        "filler_rows": len(new_rows) - pair_row_count,
        "converted_pairs": converted_pairs,
        "fallback_pairs": fallback_pairs,
        "relation_counts": dict(relation_counts),
        "total_words": final_words,
        "target_words": TARGET_WORDS,
        "word_difference": final_words - TARGET_WORDS,
        "original_stream": str(ORIGINAL_STREAM),
        "validated_packets": str(VALIDATED_PACKETS),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "stream_materialization_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
