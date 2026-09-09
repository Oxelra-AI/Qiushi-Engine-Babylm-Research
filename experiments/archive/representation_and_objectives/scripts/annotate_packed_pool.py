#!/usr/bin/env python3
"""research: Annotate the compact-view-reinvest 10M pool at word level.

For each row in the 10M pool (64,740 rows), produce a per-word category:
  filler, source, rw_copied, rw_abs_content, rw_abs_other

Changed rows (3,006 rows containing compact pair content) have source and
rewrite text embedded as exact substrings. Rewrite words are classified as
"copied" (present in the paired source) or "source-absent" using research
logic. Source text words are "source". Filler words are "filler".

Output: annotation JSONL file, one line per 10M pool row, with:
  {"row_index": i, "example_id": ..., "word_categories": [...], "n_words": ...}
  where word_categories[j] is the category for the j-th whitespace word.

Also outputs a selection of whole-word copied-content groups matched to
source-absent-content groups (like research) for the target-selective trainer.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import pathlib
import re
import time

PAIR_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
META_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl")
POOL_PATH = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
TOK_PATH = pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
FUNCTION = set("a an and are as at be been being but by can could did do does for from had has have he her hers him his i if in into is it its me my no not of on or our ours she so than that the their theirs them there they this to was we were what when where which who whom whose will with would you your yours".split())

CAT_FILLER = "filler"
CAT_SOURCE = "source"
CAT_RW_COPIED = "rw_copied"
CAT_RW_ABS_CONTENT = "rw_abs_content"
CAT_RW_ABS_OTHER = "rw_abs_other"


def norm_word(w: str) -> str:
    m = WORD_RE.search(w)
    return m.group(0).lower() if m else w.lower().strip()


def word_class(w: str) -> str:
    nw = norm_word(w)
    if any(c.isdigit() for c in nw):
        return "number"
    if nw in FUNCTION:
        return "function"
    if len(nw) <= 2:
        return "short_other"
    return "content"


def find_span_in_text(text_words: list[str], span_text: str) -> tuple[int, int] | None:
    """Find the word-index span [start, end) of span_text in text_words."""
    span_words = span_text.split()
    n = len(span_words)
    if n == 0:
        return None
    # Try exact word-sequence matching
    for i in range(len(text_words) - n + 1):
        if text_words[i:i + n] == span_words:
            return (i, i + n)
    return None


def classify_rewrite_words(source_text: str, rewrite_text: str) -> list[str]:
    """Classify each rewrite word as copied, abs_content, or abs_other."""
    src_words_norm = set(norm_word(w) for w in source_text.split())
    rw_words = rewrite_text.split()
    cats = []
    for w in rw_words:
        nw = norm_word(w)
        if nw in src_words_norm:
            cats.append(CAT_RW_COPIED)
        else:
            wc = word_class(w)
            if wc in ("content", "number"):
                cats.append(CAT_RW_ABS_CONTENT)
            else:
                cats.append(CAT_RW_ABS_OTHER)
    return cats


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pairs
    pairs = {}
    with open(PAIR_PATH) as f:
        for line in f:
            p = json.loads(line)
            pairs[p["pair_id"]] = p
    print(f"Loaded {len(pairs)} pairs", flush=True)

    # Load meta
    meta_by_index = {}
    with open(META_PATH) as f:
        for line in f:
            m = json.loads(line)
            meta_by_index[m["row_index"]] = m
    print(f"Loaded {len(meta_by_index)} changed row metadata", flush=True)

    # Load tokenizer (for BPE word group computation)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH), use_fast=True)
    vocab_size = tok.vocab_size
    print(f"Tokenizer vocab: {vocab_size}", flush=True)

    # Process pool rows
    annotations = []
    stats = collections.Counter()
    pair_match_failures = []

    with open(POOL_PATH) as f:
        for row_idx, line in enumerate(f):
            row = json.loads(line)
            text = row["text"]
            words = text.split()
            n_words = len(words)
            example_id = row.get("example_id", row_idx)

            if row_idx not in meta_by_index:
                # Pure filler row
                cats = [CAT_FILLER] * n_words
                stats["filler_rows"] += 1
                stats["filler_words"] += n_words
            else:
                # Changed row - contains source + rewrite content
                meta = meta_by_index[row_idx]
                cats = [CAT_FILLER] * n_words  # default to filler
                claimed = [False] * n_words

                for pid in meta["pair_ids"]:
                    p = pairs.get(pid)
                    if not p:
                        pair_match_failures.append({"row": row_idx, "pid": pid, "reason": "not_found"})
                        continue

                    # Find and mark source text span
                    src_span = find_span_in_text(words, p["source_text"])
                    if src_span:
                        for i in range(src_span[0], src_span[1]):
                            if not claimed[i]:
                                cats[i] = CAT_SOURCE
                                claimed[i] = True
                                stats["source_words"] += 1

                    # Find and mark rewrite text span
                    rw_span = find_span_in_text(words, p["rewrite_text"])
                    if rw_span:
                        rw_cats = classify_rewrite_words(p["source_text"], p["rewrite_text"])
                        rw_words = p["rewrite_text"].split()
                        for j, (i, cat) in enumerate(zip(range(rw_span[0], rw_span[1]), rw_cats)):
                            if not claimed[i]:
                                cats[i] = cat
                                claimed[i] = True
                                stats[f"{cat}_words"] += 1

                    if not src_span:
                        pair_match_failures.append({"row": row_idx, "pid": pid, "reason": "src_not_found"})
                    if not rw_span:
                        pair_match_failures.append({"row": row_idx, "pid": pid, "reason": "rw_not_found"})

                # Count unclaimed filler in changed rows
                unclaimed_filler = sum(1 for c in cats if c == CAT_FILLER)
                stats["changed_row_filler_words"] += unclaimed_filler
                stats["changed_rows"] += 1

            annotations.append({
                "row_index": row_idx,
                "example_id": example_id,
                "word_categories": cats,
                "n_words": n_words,
            })

            if (row_idx + 1) % 10000 == 0:
                print(json.dumps({"event": "progress", "rows": row_idx + 1,
                                  "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # Save annotations
    ann_path = OUT_DIR / "packed_pool_annotations.jsonl"
    with open(ann_path, "w") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    ann_sha = hashlib.sha256(ann_path.read_bytes()).hexdigest()[:16]

    # ── Build whole-word copied-content matched selection ──
    # Collect all source-absent-content word positions and copied word positions
    # across the pool, then match by BPE piece count and position
    print("Building whole-word matched selection...", flush=True)

    # For each rewrite word group in the pool, record:
    # (row_index, word_index, bpe_pieces, category, norm_word)
    abs_content_groups = []
    copied_groups = []

    for a in annotations:
        row_idx = a["row_index"]
        cats = a["word_categories"]
        words = None  # lazy load

        for wi, cat in enumerate(cats):
            if cat in (CAT_RW_ABS_CONTENT, CAT_RW_COPIED):
                if words is None:
                    # Reload the row text for BPE analysis
                    pass  # Will do batch below

    # Batch BPE analysis: load rows that have rewrite content
    changed_rows_data = {}
    with open(POOL_PATH) as f:
        for idx, line in enumerate(f):
            if idx in meta_by_index:
                changed_rows_data[idx] = json.loads(line)

    for a in annotations:
        row_idx = a["row_index"]
        if row_idx not in changed_rows_data:
            continue
        row = changed_rows_data[row_idx]
        text_words = row["text"].split()
        cats = a["word_categories"]

        # Tokenize the full text to get word→BPE mapping
        encoded = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        # Map each character position to word index
        char_to_word = {}
        pos = 0
        for wi, w in enumerate(text_words):
            start = row["text"].index(w, pos)
            for ci in range(start, start + len(w)):
                char_to_word[ci] = wi
            pos = start + len(w)

        # Map each token to its word index
        token_to_word = {}
        for ti, (cs, ce) in enumerate(offsets):
            if cs == ce:
                continue
            for ci in range(cs, ce):
                if ci in char_to_word:
                    token_to_word[ti] = char_to_word[ci]
                    break

        # Count BPE pieces per word
        bpe_per_word = collections.Counter()
        for ti, wi in token_to_word.items():
            bpe_per_word[wi] += 1

        # Collect groups
        for wi, cat in enumerate(cats):
            if wi < len(text_words):
                nw = norm_word(text_words[wi])
                bpe = bpe_per_word.get(wi, 0)
                if bpe == 0:
                    continue
                group = {
                    "row_index": row_idx,
                    "word_index": wi,
                    "bpe_pieces": bpe,
                    "norm_word": nw,
                    "word_class": word_class(text_words[wi]),
                }
                if cat == CAT_RW_ABS_CONTENT:
                    abs_content_groups.append(group)
                elif cat == CAT_RW_COPIED:
                    # Only copied content words (not function)
                    wc = word_class(text_words[wi])
                    if wc in ("content", "number"):
                        copied_groups.append(group)

    print(f"Source-absent content groups: {len(abs_content_groups)}", flush=True)
    print(f"Copied content groups: {len(copied_groups)}", flush=True)

    # Match: for each abs_content group, find a copied group with same BPE piece count
    # Using greedy matching by BPE length, similar to research
    import random
    rng = random.Random(22543023)

    # Build bucket by bpe_pieces
    copied_by_bpe = collections.defaultdict(list)
    for g in copied_groups:
        copied_by_bpe[g["bpe_pieces"]].append(g)
    for k in copied_by_bpe:
        rng.shuffle(copied_by_bpe[k])

    selected_copied = []
    unmatched_abs = 0
    total_abs_pieces = sum(g["bpe_pieces"] for g in abs_content_groups)

    for ag in abs_content_groups:
        bpe = ag["bpe_pieces"]
        bucket = copied_by_bpe.get(bpe, [])
        if bucket:
            cg = bucket.pop()
            selected_copied.append(cg)
        else:
            unmatched_abs += 1

    total_selected_pieces = sum(g["bpe_pieces"] for g in selected_copied)
    print(f"Matched {len(selected_copied)}/{len(abs_content_groups)} abs groups, "
          f"unmatched={unmatched_abs}, abs_pieces={total_abs_pieces}, "
          f"selected_pieces={total_selected_pieces}", flush=True)

    # Save selection
    sel_path = OUT_DIR / "packed_wholeword_copied_selection.jsonl"
    with open(sel_path, "w") as f:
        for g in selected_copied:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "status": "PACKED_POOL_ANNOTATION",
        "out": str(OUT_DIR),
        "pool_path": str(POOL_PATH),
        "pool_rows": len(annotations),
        "changed_rows": int(stats["changed_rows"]),
        "filler_rows": int(stats["filler_rows"]),
        "annotation_sha": ann_sha,
        "word_stats": {k: int(v) for k, v in sorted(stats.items())},
        "pair_match_failures": len(pair_match_failures),
        "pair_match_failure_types": dict(collections.Counter(
            f["reason"] for f in pair_match_failures)),
        "abs_content_groups": len(abs_content_groups),
        "copied_content_groups": len(copied_groups),
        "matched_copied_groups": len(selected_copied),
        "unmatched_abs_groups": unmatched_abs,
        "total_abs_pieces": total_abs_pieces,
        "total_selected_pieces": total_selected_pieces,
        "fraction_exact_bpe_match": 1.0 if unmatched_abs == 0 else
            len(selected_copied) / max(1, len(abs_content_groups)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    (OUT_DIR / "packed_pool_annotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
