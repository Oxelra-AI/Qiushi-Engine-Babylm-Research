#!/usr/bin/env python3
"""research: materialize nested aligned-restatement dose streams.

Scientific purpose
------------------
The restatement-dose arm should be a quantitative curve on one substrate, not a
single go/no-go comparison.  This script makes the two added-dose streams nested
both in selected pair content and in row positions:

  * dose21 adds exactly 443,200 aligned source->rewrite pair words per 10M pass,
    selected only from validated shard1.  Together with the inherited COMPACT_EXPERIENCE ALN
    1,656,800 pair words/pass, this is exactly 21.0% pair words.
  * dose25 is a strict superset: it keeps the dose21 selected pairs in the same
    packed rows and the same filler slots, then adds exactly 400,000 pair words
    per pass selected from validated shard0.  Together this gives exactly
    843,200 added pair words/pass and 25.0% total pair words.

The base stream is the compact-view-reinvest SOTA substrate.  Inherited
qwen_pair_packed and compact-view-reinvest rows are untouched; only ordinary
160-word filler rows are replaced.  Each replacement row begins with intact
new source+rewrite pairs and is topped up with words from the displaced filler
row so that row count, pass order and the 100M-word accounting remain fixed.

research repair after dry-run
----------------------------
The first dry-run used word-only packing.  It kept word accounting exact but
created a small number of packed pair segments above 256 tokenizer tokens and
hundreds of full rows above 256 tokens once filler top-up was appended.  Since
the trainer truncates to 256 tokens, this would make some declared pair/top-up
exposure partly invisible.  The repaired script is token-aware:

  * pair-only segments are greedily packed under both 160 words and 256 tokens;
  * pair-only segments are always placed at the beginning of a row and must
    tokenize to <=256 tokens, so the added relation practice itself is visible;
  * filler top-up is appended after the pair segment to preserve official word
    accounting. Some top-up tokens can be truncated, as in ordinary 160-word
    rows; this is recorded rather than allowed to truncate the pair segment;
  * dose25 reuses dose21 core row positions and content exactly, then adds
    extra rows from unused filler slots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()
SCRIPT_DIR = ROOT / "experiments/archive/relation_learning/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import pack_materialize_dose_arm as base  # type: ignore  # noqa: E402

SHARD1_ACCEPTED = ROOT / "experiments/archive/relation_learning/data/dose_arm_validated/accepted_dose_arm_pairs_shard1.jsonl"
SHARD0_ACCEPTED = ROOT / "experiments/archive/relation_learning/data/dose_arm_validated/accepted_dose_arm_pairs_shard0.jsonl"
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/nested_restatement_dose_streams"

DOSE21_ADDED = 443_200
DOSE25_ADDED = 843_200
DOSE25_TOPUP = DOSE25_ADDED - DOSE21_ADDED
MAX_TOKENS = 256


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_json(path: pathlib.Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def token_len(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def pack_pairs_token_aware(selected: list[dict[str, Any]], tokenizer: Any | None) -> list[dict[str, Any]]:
    """Greedy COMPACT_EXPERIENCE-style packing, with an added tokenizer-visibility limit.

    If tokenizer is unavailable this reduces to the word-only packer.  In the
    normal repaired path the tokenizer is available, and every pair-only segment
    in every replacement row is <=256 tokens before the filler top-up is added.
    """
    if tokenizer is None:
        return base.pack_pairs(selected)
    rows: list[dict[str, Any]] = []
    cur_segments: list[str] = []
    cur_pair_ids: list[str] = []
    cur_original_ids: list[str] = []
    cur_words = 0
    cur_sources: Counter[str] = Counter()

    def emit() -> None:
        nonlocal cur_segments, cur_pair_ids, cur_original_ids, cur_words, cur_sources
        if not cur_segments:
            return
        text = " ".join(cur_segments)
        tlen = token_len(tokenizer, text)
        if base.wc(text) != cur_words:
            raise RuntimeError("internal packed word mismatch")
        if tlen is not None and tlen > MAX_TOKENS:
            raise RuntimeError(f"internal packed token overflow {tlen}")
        rows.append({
            "text_pairs_only": text,
            "pair_words": cur_words,
            "pair_ids": list(cur_pair_ids),
            "original_ids": list(cur_original_ids),
            "n_pairs": len(cur_pair_ids),
            "component_sources": dict(cur_sources),
            "pair_segment_tokens_no_special": tlen,
        })
        cur_segments, cur_pair_ids, cur_original_ids, cur_words, cur_sources = [], [], [], 0, Counter()

    for p in selected:
        w = int(p["pair_words"])
        seg = f"{p['original']} {p['rewrite']}"
        if base.wc(seg) != w:
            raise RuntimeError(f"segment word mismatch {p['pair_id']}")
        seg_tokens = token_len(tokenizer, seg)
        if seg_tokens is not None and seg_tokens > MAX_TOKENS:
            raise RuntimeError(f"single pair {p['pair_id']} tokenizes to {seg_tokens} tokens > {MAX_TOKENS}")
        tentative_segments = cur_segments + [seg]
        tentative_text = " ".join(tentative_segments)
        tentative_words = cur_words + w
        tentative_tokens = token_len(tokenizer, tentative_text)
        if cur_segments and (tentative_words > base.MAX_PACK_PAIR_WORDS or (tentative_tokens is not None and tentative_tokens > MAX_TOKENS)):
            emit()
        cur_segments.append(seg)
        cur_pair_ids.append(str(p["pair_id"]))
        cur_original_ids.append(str(p["original_id"]))
        cur_words += w
        cur_sources[str(p.get("source", ""))] += w
        if cur_words > base.MAX_PACK_PAIR_WORDS:
            raise RuntimeError("single packed row exceeded word limit after emit")
    emit()
    return rows


def candidate_filler_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [
        i for i, r in enumerate(rows)
        if str(r.get("source", "")) in base.FILLER_SOURCES and int(r.get("words", 0)) == base.ROW_WORDS
    ]


def replacement_text_and_token_ok(old: dict[str, Any], pack: dict[str, Any], tokenizer: Any | None) -> tuple[str, int, int | None, int | None, bool]:
    old_words = str(old["text"]).split()
    pair_words = int(pack["pair_words"])
    topup_n = base.ROW_WORDS - pair_words
    if topup_n < 0:
        raise RuntimeError(f"packed row has {pair_words} pair words > {base.ROW_WORDS}")
    if len(old_words) < topup_n:
        raise RuntimeError("displaced filler row shorter than requested topup")
    topup = " ".join(old_words[:topup_n])
    text = pack["text_pairs_only"] if not topup else pack["text_pairs_only"] + " " + topup
    if base.wc(text) != base.ROW_WORDS:
        raise RuntimeError(f"new row word mismatch: {base.wc(text)}")
    pair_toks = token_len(tokenizer, pack["text_pairs_only"])
    full_toks = token_len(tokenizer, text)
    # The scientific relation material is the pair segment at the beginning of
    # the row.  It must be visible to the 256-token trainer.  Filler top-up comes
    # after the pair segment and may be truncated in a small number of rows; this
    # preserves the official word substitution budget while keeping relation
    # exposure intact.  Full-row token counts are recorded for later accounting.
    ok = True
    if pair_toks is not None and pair_toks > MAX_TOKENS:
        ok = False
    return text, topup_n, pair_toks, full_toks, ok


def assign_slots_token_aware(rows: list[dict[str, Any]], candidates: list[int], packs: list[dict[str, Any]], tokenizer: Any | None, used_rows: set[int] | None = None) -> list[int]:
    """Assign packs to valid filler rows near evenly spaced positions.

    The resulting list has one row index per pack in pack order.  Because the
    pair segment is placed before top-up and pack_pairs_token_aware already
    enforces pair-segment visibility, slot assignment can stay close to the
    original evenly spaced construction.  We still call the row builder during
    selection so malformed rows fail early and full-row token counts are later
    recorded.  If tokenizer is missing, fall back to exact spaced slots.
    """
    if used_rows is None:
        used_rows = set()
    available_positions = [pos for pos, idx in enumerate(candidates) if idx not in used_rows]
    if len(packs) > len(available_positions):
        raise RuntimeError(f"need {len(packs)} slots but only {len(available_positions)} available")
    if tokenizer is None:
        chosen = base.spaced_indices([candidates[pos] for pos in available_positions], len(packs))
        return chosen

    chosen: list[int] = []
    used = set(used_rows)
    n = len(candidates)
    for j, pack in enumerate(packs):
        preferred = int((j + 0.5) * n / max(1, len(packs)))
        preferred = min(n - 1, max(0, preferred))
        found: int | None = None
        best_dist = None
        # Expand deterministically by candidate-list distance.  The full scan is
        # rarely needed but protects exact materialization if local rows contain
        # tokenizer-heavy text.
        for radius in range(n):
            for pos in (preferred - radius, preferred + radius):
                if pos < 0 or pos >= n:
                    continue
                idx = candidates[pos]
                if idx in used:
                    continue
                _, _, _, _, ok = replacement_text_and_token_ok(rows[idx], pack, tokenizer)
                if ok:
                    found = idx
                    best_dist = radius
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"could not find pair-visible filler slot for pack {j}/{len(packs)}")
        used.add(found)
        chosen.append(found)
    return chosen


def build_replacement_row(old: dict[str, Any], pack: dict[str, Any], source_label: str, example_id: int, tokenizer: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    new_text, topup_n, pair_token_count, full_token_count, ok = replacement_text_and_token_ok(old, pack, tokenizer)
    if not ok:
        raise RuntimeError("attempted to build a replacement row that would truncate")
    pair_words = int(pack["pair_words"])
    row = {
        "text": new_text,
        "words": base.ROW_WORDS,
        "example_id": example_id,
        "source": source_label,
    }
    meta = {
        "old_example_id": old.get("example_id"),
        "old_source": old.get("source"),
        "pair_words": pair_words,
        "topup_words": topup_n,
        "row_words": base.ROW_WORDS,
        "n_pairs": pack["n_pairs"],
        "pair_ids": pack["pair_ids"],
        "original_ids": pack["original_ids"],
        "component_sources": pack.get("component_sources", {}),
        "token_count_no_special_full_row": full_token_count,
        "token_count_no_special_pair_segment": pair_token_count,
    }
    return row, meta


def process_one_pass(rows: list[dict[str, Any]], pass_i: int, packed21: list[dict[str, Any]], packed_extra: list[dict[str, Any]], tokenizer: Any | None, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return modified rows and pass metadata.

    mode='dose21' applies only the dose21 rows.  mode='dose25' applies dose21
    rows at exactly the same indices plus the extra top-up rows selected from
    the remaining candidate filler slots.
    """
    if mode not in {"dose21", "dose25"}:
        raise ValueError(mode)
    before_words = sum(int(r["words"]) for r in rows)
    if before_words != base.TOTAL_WORDS_PER_PASS:
        raise RuntimeError(f"pass {pass_i} has {before_words} words before replacement")
    candidates = candidate_filler_indices(rows)
    dose21_idxs = assign_slots_token_aware(rows, candidates, packed21, tokenizer)
    used = set(dose21_idxs)
    extra_idxs: list[int] = []
    if mode == "dose25":
        extra_idxs = assign_slots_token_aware(rows, candidates, packed_extra, tokenizer, used_rows=used)
        if set(extra_idxs) & used:
            raise RuntimeError("dose25 extra indices overlap dose21 indices")

    out = list(rows)
    row_meta: list[dict[str, Any]] = []
    source_words = Counter()
    topup_words_by_old_source = Counter()
    pair_tokens: list[float] = []
    full_tokens: list[float] = []
    pair_segment_over_256 = 0
    full_row_over_256 = 0

    for local_j, (idx, pack) in enumerate(zip(dose21_idxs, packed21)):
        old = out[idx]
        new_row, meta = build_replacement_row(
            old, pack, "dose21_qwen_pair_packed", 881_000_000 + pass_i * 100_000 + local_j, tokenizer
        )
        out[idx] = new_row
        meta.update({
            "pass": pass_i,
            "dose_component": "dose21_core",
            "row_index_in_pass": idx,
            "global_row_index": pass_i * base.ROWS_PER_PASS + idx,
            "new_example_id": new_row["example_id"],
        })
        row_meta.append(meta)
    if mode == "dose25":
        for local_j, (idx, pack) in enumerate(zip(extra_idxs, packed_extra)):
            old = out[idx]
            new_row, meta = build_replacement_row(
                old, pack, "dose25_extra_qwen_pair_packed", 882_000_000 + pass_i * 100_000 + local_j, tokenizer
            )
            out[idx] = new_row
            meta.update({
                "pass": pass_i,
                "dose_component": "dose25_topup",
                "row_index_in_pass": idx,
                "global_row_index": pass_i * base.ROWS_PER_PASS + idx,
                "new_example_id": new_row["example_id"],
            })
            row_meta.append(meta)

    for m in row_meta:
        old_source = str(m.get("old_source", ""))
        source_words[old_source] += int(m["row_words"])
        topup_words_by_old_source[old_source] += int(m["topup_words"])
        if m.get("token_count_no_special_pair_segment") is not None:
            pt = float(m["token_count_no_special_pair_segment"])
            ft = float(m["token_count_no_special_full_row"])
            pair_tokens.append(pt)
            full_tokens.append(ft)
            pair_segment_over_256 += int(pt > MAX_TOKENS)
            full_row_over_256 += int(ft > MAX_TOKENS)

    after_words = sum(int(r["words"]) for r in out)
    if after_words != base.TOTAL_WORDS_PER_PASS:
        raise RuntimeError(f"pass {pass_i} has {after_words} words after replacement")

    added_pair_words = sum(int(r["pair_words"]) for r in packed21)
    if mode == "dose25":
        added_pair_words += sum(int(r["pair_words"]) for r in packed_extra)
    return out, {
        "pass": pass_i,
        "mode": mode,
        "before_words": before_words,
        "after_words": after_words,
        "candidate_filler_rows_160w": len(candidates),
        "dose21_replaced_rows": len(packed21),
        "dose25_extra_replaced_rows": len(packed_extra) if mode == "dose25" else 0,
        "replaced_rows": len(row_meta),
        "added_pair_words": added_pair_words,
        "topup_words": sum(int(m["topup_words"]) for m in row_meta),
        "displaced_source_words": dict(source_words),
        "topup_source_words": dict(topup_words_by_old_source),
        "pair_segment_token_stats": base.stat(pair_tokens),
        "full_row_token_stats": base.stat(full_tokens),
        "pair_segment_rows_over_256": pair_segment_over_256,
        "full_rows_over_256": full_row_over_256,
        "row_meta": row_meta,
    }


def materialize_mode(mode: str, packed21: list[dict[str, Any]], packed_extra: list[dict[str, Any]], out_dir: pathlib.Path, tokenizer: Any | None, write_stream: bool) -> dict[str, Any]:
    if mode not in {"dose21", "dose25"}:
        raise ValueError(mode)
    out_mode = out_dir / mode
    out_mode.mkdir(parents=True, exist_ok=True)
    stream_path = out_mode / f"{mode}_compact_view_reinvest_100M.jsonl"
    row_meta_path = out_mode / f"{mode}_dose_rows_meta.jsonl"
    pass_stats_path = out_mode / f"{mode}_pass_stats.jsonl"

    source_words = Counter()
    all_row_meta: list[dict[str, Any]] = []
    pass_stats_public: list[dict[str, Any]] = []
    fout = stream_path.open("w", encoding="utf-8") if write_stream else None
    pass_i = 0
    buf: list[dict[str, Any]] = []
    try:
        with base.BASE_STREAM.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                buf.append(json.loads(line))
                if len(buf) == base.ROWS_PER_PASS:
                    new_rows, ps = process_one_pass(buf, pass_i, packed21, packed_extra, tokenizer, mode)
                    for r in new_rows:
                        source_words[str(r.get("source", ""))] += int(r["words"])
                        if fout is not None:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    all_row_meta.extend(ps.pop("row_meta"))
                    pass_stats_public.append(ps)
                    pass_i += 1
                    buf = []
        if buf:
            raise RuntimeError(f"base stream ended with partial pass of {len(buf)} rows")
    finally:
        if fout is not None:
            fout.close()
    if pass_i != base.PASSES:
        raise RuntimeError(f"expected {base.PASSES} passes, found {pass_i}")

    base.write_jsonl(row_meta_path, all_row_meta)
    base.write_jsonl(pass_stats_path, pass_stats_public)
    total_words = sum(source_words.values())
    target_added = DOSE21_ADDED if mode == "dose21" else DOSE25_ADDED
    if any(int(ps["added_pair_words"]) != target_added for ps in pass_stats_public):
        raise RuntimeError(f"{mode} per-pass added words mismatch")
    if total_words != base.TOTAL_WORDS_PER_PASS * base.PASSES:
        raise RuntimeError(f"{mode} total words {total_words}")
    if sum(int(ps["pair_segment_rows_over_256"]) for ps in pass_stats_public) != 0:
        raise RuntimeError(f"{mode} still has pair-segment token overflow")
    # Full-row >256 counts indicate that some appended filler top-up is beyond
    # the trainer window.  They are a recorded cost/accounting feature, not a
    # fatal error, because the pair segment is first and remains visible.
    return {
        "mode": mode,
        "write_stream": write_stream,
        "stream_path": str(stream_path) if write_stream else None,
        "stream_sha256": base.sha256_file(stream_path) if write_stream else None,
        "row_meta_path": str(row_meta_path),
        "row_meta_sha256": base.sha256_file(row_meta_path),
        "pass_stats_path": str(pass_stats_path),
        "pass_stats_sha256": base.sha256_file(pass_stats_path),
        "passes": pass_i,
        "rows_per_pass": base.ROWS_PER_PASS,
        "total_words": total_words,
        "output_source_words": dict(source_words),
        "replacement_rows_per_pass": pass_stats_public[0]["replaced_rows"],
        "added_pair_words_per_pass": target_added,
        "topup_words_per_pass": pass_stats_public[0]["topup_words"],
        "pair_segment_rows_over_256_total": sum(int(ps["pair_segment_rows_over_256"]) for ps in pass_stats_public),
        "full_rows_over_256_total": sum(int(ps["full_rows_over_256"]) for ps in pass_stats_public),
        "pair_segment_token_stats_first_pass": pass_stats_public[0]["pair_segment_token_stats"],
        "full_row_token_stats_first_pass": pass_stats_public[0]["full_row_token_stats"],
        "pass_stats_preview": pass_stats_public[:2],
    }


def summarize_by_source(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    words = Counter()
    for p in pairs:
        src = str(p.get("source", ""))
        counts[src] += 1
        words[src] += int(p["pair_words"])
    return {"counts": dict(counts), "words": dict(words)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard1-accepted", default=str(SHARD1_ACCEPTED))
    ap.add_argument("--shard0-accepted", default=str(SHARD0_ACCEPTED))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--selection-seed", type=int, default=59000)
    ap.add_argument("--write-stream", action="store_true")
    ap.add_argument("--skip-base-sha", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sha = "skipped"
    if not args.skip_base_sha:
        base_sha = base.sha256_file(base.BASE_STREAM)
        if base_sha != base.EXPECTED_BASE_SHA:
            raise RuntimeError(f"base SHA mismatch {base_sha} != {base.EXPECTED_BASE_SHA}")

    tokenizer = base.load_tokenizer(base.TOKENIZER)
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for research repaired token-aware materialization")

    shard1 = base.load_accepted([pathlib.Path(args.shard1_accepted)])
    shard0 = base.load_accepted([pathlib.Path(args.shard0_accepted)])
    shard1_ids = {str(p["original_id"]) for p in shard1}
    shard0_ids = {str(p["original_id"]) for p in shard0}
    overlap = sorted(shard1_ids & shard0_ids)
    if overlap:
        raise RuntimeError(f"shard accepted original_id overlap, first={overlap[:5]}")

    dose21_selected = base.exact_select(shard1, DOSE21_ADDED, args.selection_seed + 21)
    dose21_words = sum(int(p["pair_words"]) for p in dose21_selected)
    if dose21_words != DOSE21_ADDED:
        raise RuntimeError("dose21 exact selection failed")
    dose25_topup_selected = base.exact_select(shard0, DOSE25_TOPUP, args.selection_seed + 25)
    topup_words = sum(int(p["pair_words"]) for p in dose25_topup_selected)
    if topup_words != DOSE25_TOPUP:
        raise RuntimeError("dose25 topup exact selection failed")
    dose25_selected = dose21_selected + dose25_topup_selected
    if sum(int(p["pair_words"]) for p in dose25_selected) != DOSE25_ADDED:
        raise RuntimeError("dose25 total exact selection failed")

    packed21 = pack_pairs_token_aware(dose21_selected, tokenizer)
    packed_topup = pack_pairs_token_aware(dose25_topup_selected, tokenizer)

    selected21_path = out_dir / "dose21_selected_pairs.jsonl"
    selected25_topup_path = out_dir / "dose25_topup_selected_pairs_from_shard0.jsonl"
    selected25_path = out_dir / "dose25_selected_pairs_superset.jsonl"
    packed21_path = out_dir / "dose21_packed_pair_rows.jsonl"
    packed_topup_path = out_dir / "dose25_topup_packed_pair_rows.jsonl"
    base.write_jsonl(selected21_path, dose21_selected)
    base.write_jsonl(selected25_topup_path, dose25_topup_selected)
    base.write_jsonl(selected25_path, dose25_selected)
    base.write_jsonl(packed21_path, packed21)
    base.write_jsonl(packed_topup_path, packed_topup)

    dose21_meta = materialize_mode("dose21", packed21, [], out_dir, tokenizer, args.write_stream)
    dose25_meta = materialize_mode("dose25", packed21, packed_topup, out_dir, tokenizer, args.write_stream)

    # Strict nesting checks over row metadata.  The dose21 core rows in dose25
    # must occupy the same pass/index and carry the same selected pair IDs.
    dose21_rows = base.read_jsonl(pathlib.Path(dose21_meta["row_meta_path"]))
    dose25_rows = base.read_jsonl(pathlib.Path(dose25_meta["row_meta_path"]))
    d21_core = {(r["pass"], r["row_index_in_pass"]): r for r in dose21_rows}
    d25_core = {(r["pass"], r["row_index_in_pass"]): r for r in dose25_rows if r.get("dose_component") == "dose21_core"}
    if set(d21_core) != set(d25_core):
        raise RuntimeError("dose25 does not preserve dose21 row positions")
    for k, r in d21_core.items():
        rr = d25_core[k]
        for field in ["pair_ids", "original_ids", "pair_words", "topup_words", "old_example_id", "old_source", "token_count_no_special_full_row", "token_count_no_special_pair_segment"]:
            if r.get(field) != rr.get(field):
                raise RuntimeError(f"dose25 core row mismatch at {k} field {field}")

    meta = {
        "status": "NESTED_RESTATEMENT_DOSE_MATERIALIZED" if args.write_stream else "NESTED_RESTATEMENT_DOSE_DRYRUN",
        "created_utc": now_utc(),
        "base_stream": str(base.BASE_STREAM),
        "base_sha256": base_sha,
        "write_stream": bool(args.write_stream),
        "tokenizer": str(base.TOKENIZER),
        "max_tokens_no_special_for_replacement_rows": MAX_TOKENS,
        "shard1_accepted": str(args.shard1_accepted),
        "shard0_accepted": str(args.shard0_accepted),
        "shard1_pairs": len(shard1),
        "shard1_pair_words": sum(int(p["pair_words"]) for p in shard1),
        "shard0_pairs": len(shard0),
        "shard0_pair_words": sum(int(p["pair_words"]) for p in shard0),
        "selection_seed": args.selection_seed,
        "targets": {
            "dose21_added_pair_words_per_10M": DOSE21_ADDED,
            "dose25_added_pair_words_per_10M": DOSE25_ADDED,
            "dose25_topup_pair_words_from_shard0_per_10M": DOSE25_TOPUP,
            "inherited_pair_words_per_10M": base.INHERITED_PAIR_WORDS,
            "dose21_total_pair_fraction": (base.INHERITED_PAIR_WORDS + DOSE21_ADDED) / base.TOTAL_WORDS_PER_PASS,
            "dose25_total_pair_fraction": (base.INHERITED_PAIR_WORDS + DOSE25_ADDED) / base.TOTAL_WORDS_PER_PASS,
        },
        "dose21_selection": {
            "source": "validated shard1 only",
            "selected_pairs": len(dose21_selected),
            "selected_pair_words": dose21_words,
            "selected_sha256_virtual": sha256_jsonl_rows(dose21_selected),
            "packed_rows_per_pass": len(packed21),
            "pair_word_stats": base.stat([float(p["pair_words"]) for p in dose21_selected]),
            "pair_source": summarize_by_source(dose21_selected),
            "packed_pair_word_stats": base.stat([float(r["pair_words"]) for r in packed21]),
            "packed_pair_token_stats": base.stat([float(r["pair_segment_tokens_no_special"]) for r in packed21]),
            "outputs": {
                "selected_pairs": str(selected21_path),
                "packed_pair_rows": str(packed21_path),
            },
        },
        "dose25_selection": {
            "source": "strict superset: dose21 selected pairs plus exact topup from validated shard0",
            "selected_pairs": len(dose25_selected),
            "selected_pair_words": sum(int(p["pair_words"]) for p in dose25_selected),
            "topup_pairs_from_shard0": len(dose25_topup_selected),
            "topup_pair_words_from_shard0": topup_words,
            "selected_sha256_virtual": sha256_jsonl_rows(dose25_selected),
            "packed_core_rows_per_pass": len(packed21),
            "packed_topup_rows_per_pass": len(packed_topup),
            "topup_pair_word_stats": base.stat([float(p["pair_words"]) for p in dose25_topup_selected]),
            "topup_pair_source": summarize_by_source(dose25_topup_selected),
            "packed_topup_pair_word_stats": base.stat([float(r["pair_words"]) for r in packed_topup]),
            "packed_topup_pair_token_stats": base.stat([float(r["pair_segment_tokens_no_special"]) for r in packed_topup]),
            "outputs": {
                "topup_selected_pairs": str(selected25_topup_path),
                "selected_pairs_superset": str(selected25_path),
                "topup_packed_pair_rows": str(packed_topup_path),
            },
        },
        "nesting_checks": {
            "content_superset": set(str(p["original_id"]) for p in dose21_selected).issubset(set(str(p["original_id"]) for p in dose25_selected)),
            "row_position_superset": True,
            "dose21_core_rows_identical_in_dose25_metadata": True,
        },
        "dose21_materialization": dose21_meta,
        "dose25_materialization": dose25_meta,
        "artifact_sha256": {
            "dose21_selected_pairs": base.sha256_file(selected21_path),
            "dose25_topup_selected_pairs_from_shard0": base.sha256_file(selected25_topup_path),
            "dose25_selected_pairs_superset": base.sha256_file(selected25_path),
            "dose21_packed_pair_rows": base.sha256_file(packed21_path),
            "dose25_topup_packed_pair_rows": base.sha256_file(packed_topup_path),
        },
        "scientific_note": "The three-point compact-view-reinvest substrate is nested and pair-visible: 0 added pair words is the inherited research/research base, dose21 adds a shard1-only subset, and dose25 preserves those rows and adds shard0 topup rows. Pair-only segments are at the beginning of replacement rows and tokenize to <=256, so the added relation practice is not truncated; full-row >256 counts record possible truncation of appended filler top-up.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "nested_dose_materialization_metadata.json"
    write_json(meta_path, meta)
    print(json.dumps({
        "status": meta["status"],
        "write_stream": meta["write_stream"],
        "dose21_selected_pairs": len(dose21_selected),
        "dose21_words": dose21_words,
        "dose21_packed_rows": len(packed21),
        "dose25_topup_pairs": len(dose25_topup_selected),
        "dose25_topup_words": topup_words,
        "dose25_topup_packed_rows": len(packed_topup),
        "dose21_pair_segment_rows_over_256_total": dose21_meta["pair_segment_rows_over_256_total"],
        "dose25_pair_segment_rows_over_256_total": dose25_meta["pair_segment_rows_over_256_total"],
        "dose21_full_rows_over_256_total": dose21_meta["full_rows_over_256_total"],
        "dose25_full_rows_over_256_total": dose25_meta["full_rows_over_256_total"],
        "dose21_stream": dose21_meta["stream_path"],
        "dose25_stream": dose25_meta["stream_path"],
        "metadata": str(meta_path),
        "elapsed_sec": meta["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
