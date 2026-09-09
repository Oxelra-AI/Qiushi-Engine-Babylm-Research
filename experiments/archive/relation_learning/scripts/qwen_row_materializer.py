#!/usr/bin/env python3
"""research: Qwen-row-aware stream materializer for state-use intervention.

Replaces inherited COMPACT_EXPERIENCE Qwen pair-packed companions in the compact-view-reinvest
100M training stream with validated plain-use state-update packets.

Design invariants:
  1. Only rows with source='qwen_pair_packed' are candidates
  2. Each Qwen row matched by example_id to COMPACT_EXPERIENCE qwen_pair_packed_rows_meta
  3. Each pair_id with a validated packet: original+rewrite -> original+update+use
  4. Converted rows tokenized; if >256 tokens, revert pairs from last
  5. Net word changes absorbed in common filler (last filler rows per pass)
  6. Non-target rows preserved byte-for-byte
  7. Pass order (10 passes × 64740 rows) and row positions preserved
  8. Full provenance: input/output SHAs, per-row conversion, accounting

Usage:
  python qwen_row_materializer.py \\
    --validated-packets .../validated_..._train.jsonl \\
    --output-dir .../intervention_stream/
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter

# ── Defaults (relative to repository root) ──
BASE_STREAM = "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
QWEN_META   = "experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl"
SELECTED_PAIRS = "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
TOKENIZER_DIR  = "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
EXPECTED_BASE_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"

ROWS_PER_PASS = 64_740
PASSES = 10
MAX_SEQ_TOKENS = 256
FILLER_SOURCES = frozenset({"childes", "gutenberg", "open_subtitles", "simple_wiki",
                             "bnc_spoken", "switchboard"})
MIN_FILLER_WORDS_AFTER_TRIM = 10

# ── Utilities ──
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def wc(text: str) -> int:
    return len(text.split())

def load_jsonl_dict(path: str, key: str) -> dict:
    d: dict = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            d[item[key]] = item
    return d

# ── Core text logic ──
def reconstruct_row_text(pair_ids: list[str], pairs: dict) -> str:
    """Reconstruct the original row text from its pairs (space-joined segments)."""
    return " ".join(pairs[pid]["original"] + " " + pairs[pid]["rewrite"]
                    for pid in pair_ids)

def build_replaced_text(pair_ids: list[str], pairs: dict,
                         packets: dict, replace_set: set) -> str:
    """Build row text with some pairs replaced by state-use packets."""
    segs = []
    for pid in pair_ids:
        if pid in replace_set:
            pkt = packets[pid]
            segs.append(pairs[pid]["original"] + " " +
                        pkt["update_sentence"] + " " + pkt["use_sentence"])
        else:
            segs.append(pairs[pid]["original"] + " " + pairs[pid]["rewrite"])
    return " ".join(segs)

def attempt_replacement(pair_ids: list[str], pairs: dict,
                         packets: dict, tokenizer) -> tuple:
    """Try replacing as many pairs as possible, handling token overflow.

    Returns (new_text | None, replaced_pids, overflow_pids, token_count).
    """
    replaceable = [pid for pid in pair_ids if pid in packets]
    if not replaceable:
        return None, [], [], 0

    # Try all replaceable, then remove from the end one at a time
    for n in range(len(replaceable), 0, -1):
        replace_set = set(replaceable[:n])
        text = build_replaced_text(pair_ids, pairs, packets, replace_set)
        tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        if tok_len <= MAX_SEQ_TOKENS:
            return text, replaceable[:n], replaceable[n:], tok_len

    # Even one replacement overflows
    return None, [], replaceable, 0

# ── Filler absorption ──
def absorb_filler(rows: list[dict], delta_words: int) -> tuple[int, list]:
    """Trim common filler rows near end of pass to absorb positive word delta."""
    if delta_words <= 0:
        return 0, []
    remaining = delta_words
    trims: list[dict] = []
    for i in range(len(rows) - 1, -1, -1):
        if remaining <= 0:
            break
        src = rows[i].get("source", "")
        if src not in FILLER_SOURCES:
            continue
        avail = rows[i]["words"] - MIN_FILLER_WORDS_AFTER_TRIM
        if avail <= 0:
            continue
        trim = min(remaining, avail)
        words_list = rows[i]["text"].split()
        rows[i]["text"] = " ".join(words_list[:-trim])
        rows[i]["words"] = len(words_list) - trim
        remaining -= trim
        trims.append({"row_pos_in_pass": i, "example_id": rows[i].get("example_id"),
                       "source": src, "trimmed_words": trim})
    return delta_words - remaining, trims

# ── Pass processing ──
def process_one_pass(rows: list[dict], pass_num: int, pairs: dict,
                      qwen_meta: dict, packets: dict, tokenizer,
                      global_stats: dict) -> dict:
    original_total = sum(r["words"] for r in rows)
    ps = {"pass": pass_num, "qwen_rows": 0, "rows_with_packets": 0,
          "rows_replaced": 0, "pairs_replaced": 0, "pairs_overflow": 0,
          "meta_miss": 0, "text_mismatch": 0,
          "original_words": original_total, "modified_words": 0,
          "word_delta": 0, "absorbed": 0, "filler_trims": []}

    for row in rows:
        if row.get("source") != "qwen_pair_packed":
            continue
        ps["qwen_rows"] += 1
        global_stats["total_qwen_rows"] += 1

        eid = row.get("example_id")
        if eid not in qwen_meta:
            ps["meta_miss"] += 1; global_stats["total_meta_miss"] += 1
            continue

        meta_entry = qwen_meta[eid]
        pair_ids = meta_entry["pair_ids"]

        if any(pid not in pairs for pid in pair_ids):
            ps["meta_miss"] += 1; global_stats["total_meta_miss"] += 1
            continue

        # Verify text reconstruction (detects silent upstream changes)
        expected = reconstruct_row_text(pair_ids, pairs)
        if expected != row["text"]:
            ps["text_mismatch"] += 1; global_stats["total_text_mismatch"] += 1
            continue

        available = [pid for pid in pair_ids if pid in packets]
        if not available:
            continue

        ps["rows_with_packets"] += 1
        global_stats["total_rows_with_packets"] += 1

        new_text, replaced, overflow, tok_len = attempt_replacement(
            pair_ids, pairs, packets, tokenizer)

        if new_text is not None and replaced:
            old_w = row["words"]
            new_w = wc(new_text)
            row["text"] = new_text
            row["words"] = new_w
            ps["rows_replaced"] += 1
            ps["pairs_replaced"] += len(replaced)
            global_stats["total_pairs_replaced"] += len(replaced)
            global_stats["token_counts"].append(tok_len)
            global_stats["word_deltas"].append(new_w - old_w)
            for pid in replaced:
                global_stats["packet_types"][packets[pid]["packet_type"]] += 1
                global_stats["replaced_pair_ids"].add(pid)

        if overflow:
            ps["pairs_overflow"] += len(overflow)
            global_stats["total_pairs_overflow"] += len(overflow)

    modified_total = sum(r["words"] for r in rows)
    ps["modified_words"] = modified_total
    ps["word_delta"] = modified_total - original_total

    # Absorb positive word delta in filler
    if ps["word_delta"] > 0:
        absorbed, trims = absorb_filler(rows, ps["word_delta"])
        ps["absorbed"] = absorbed
        ps["filler_trims"] = trims
        ps["modified_words"] = sum(r["words"] for r in rows)  # recompute
        ps["word_delta"] = ps["modified_words"] - original_total

    return ps

# ── Main ──
def main():
    ap = argparse.ArgumentParser(description="Qwen-row-aware stream materializer")
    ap.add_argument("--validated-packets", required=True,
                    help="Validated train-split packets JSONL")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for stream and metadata")
    ap.add_argument("--base-stream", default=BASE_STREAM)
    ap.add_argument("--qwen-meta", default=QWEN_META)
    ap.add_argument("--selected-pairs", default=SELECTED_PAIRS)
    ap.add_argument("--tokenizer", default=TOKENIZER_DIR)
    ap.add_argument("--skip-sha", action="store_true",
                    help="Skip base stream SHA verification")
    ap.add_argument("--dry-run", action="store_true",
                    help="Process but do not write output stream")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stream = out_dir / "state_use_intervention_100M.jsonl"
    out_meta = out_dir / "materializer_metadata.json"

    # ── Load references ──
    print("Loading references...", flush=True)
    pairs = load_jsonl_dict(args.selected_pairs, "pair_id")
    qwen_meta = load_jsonl_dict(args.qwen_meta, "example_id")
    packets = load_jsonl_dict(args.validated_packets, "pair_id")
    print(f"  {len(pairs)} selected pairs, {len(qwen_meta)} Qwen row entries, "
          f"{len(packets)} validated packets", flush=True)

    # Packet-pair cross-check
    missing = [pid for pid in packets if pid not in pairs]
    if missing:
        print(f"  WARNING: {len(missing)} packet pair_ids not in selected_pairs", flush=True)

    # Tokenizer
    from transformers import AutoTokenizer
    tok_path = str(pathlib.Path(args.tokenizer).resolve())
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    print(f"  Tokenizer loaded (vocab={tokenizer.vocab_size})", flush=True)

    # Base SHA
    if not args.skip_sha:
        print("Verifying base stream SHA...", flush=True)
        base_sha = sha256_file(args.base_stream)
        sha_ok = base_sha == EXPECTED_BASE_SHA
        if not sha_ok:
            print(f"  MISMATCH: expected {EXPECTED_BASE_SHA}, got {base_sha}",
                  file=sys.stderr, flush=True)
        else:
            print(f"  Base SHA OK: {base_sha[:16]}...", flush=True)
    else:
        base_sha = "skipped"
        sha_ok = True

    # ── Process stream ──
    gs = {"total_qwen_rows": 0, "total_rows_with_packets": 0,
          "total_pairs_replaced": 0, "total_pairs_overflow": 0,
          "total_meta_miss": 0, "total_text_mismatch": 0,
          "token_counts": [], "word_deltas": [],
          "packet_types": Counter(), "replaced_pair_ids": set()}
    pass_stats: list[dict] = []

    t0 = time.time()
    row_global = 0
    pass_buf: list[dict] = []
    pass_num = 0

    fout = None if args.dry_run else open(out_stream, "w")
    try:
        with open(args.base_stream) as fin:
            for line in fin:
                pass_buf.append(json.loads(line))
                if len(pass_buf) == ROWS_PER_PASS:
                    ps = process_one_pass(pass_buf, pass_num, pairs,
                                          qwen_meta, packets, tokenizer, gs)
                    pass_stats.append(ps)
                    if fout:
                        for r in pass_buf:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    row_global += len(pass_buf)
                    pass_num += 1
                    pass_buf = []
                    print(f"  Pass {pass_num}/{PASSES}: "
                          f"replaced={ps['rows_replaced']} rows / {ps['pairs_replaced']} pairs, "
                          f"delta={ps['word_delta']:+d} words", flush=True)

            # Tail (shouldn't happen if stream is correct multiple of ROWS_PER_PASS)
            if pass_buf:
                ps = process_one_pass(pass_buf, pass_num, pairs,
                                      qwen_meta, packets, tokenizer, gs)
                pass_stats.append(ps)
                if fout:
                    for r in pass_buf:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                row_global += len(pass_buf)
                pass_num += 1
    finally:
        if fout:
            fout.close()

    elapsed = time.time() - t0

    # Output SHA
    out_sha = sha256_file(str(out_stream)) if not args.dry_run else "dry_run"

    # ── Word-change and token-count summaries ──
    wds = gs["word_deltas"]
    tcs = gs["token_counts"]
    wc_summary = {}
    if wds:
        wds_s = sorted(wds)
        wc_summary = {"mean": sum(wds) / len(wds), "min": wds_s[0], "max": wds_s[-1],
                       "median": wds_s[len(wds_s)//2], "total": sum(wds)}
    tc_summary = {}
    if tcs:
        tcs_s = sorted(tcs)
        tc_summary = {"mean": sum(tcs) / len(tcs), "min": tcs_s[0], "max": tcs_s[-1],
                       "median": tcs_s[len(tcs_s)//2],
                       "over_240": sum(1 for t in tcs if t > 240),
                       "over_250": sum(1 for t in tcs if t > 250)}

    # ── Metadata ──
    metadata = {
        "status": "MATERIALIZER_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_stream": str(args.base_stream),
        "base_sha": base_sha, "base_sha_ok": sha_ok,
        "output_stream": str(out_stream),
        "output_sha": out_sha,
        "validated_packets": str(args.validated_packets),
        "packets_loaded": len(packets),
        "unique_pair_ids_replaced": len(gs["replaced_pair_ids"]),
        "total_rows": row_global,
        "total_passes": pass_num,
        "rows_per_pass": ROWS_PER_PASS,
        "total_qwen_rows": gs["total_qwen_rows"],
        "total_rows_with_packets": gs["total_rows_with_packets"],
        "total_pairs_replaced_instances": gs["total_pairs_replaced"],
        "total_pairs_overflow": gs["total_pairs_overflow"],
        "total_meta_miss": gs["total_meta_miss"],
        "total_text_mismatch": gs["total_text_mismatch"],
        "packet_types_used": dict(gs["packet_types"]),
        "word_change": wc_summary,
        "token_counts": tc_summary,
        "per_pass": pass_stats,
        "elapsed_sec": round(elapsed, 1),
        "dry_run": args.dry_run,
    }
    out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
