#!/usr/bin/env python3
"""research: materialize pair-boundary-aware stagewise clean-Qwen 10M data.

The research row-chunked data fixed hidden-word exposure but split many Qwen
original/rewrite pairs across separate training examples. This script builds a
mechanism-preserving 10M stagewise arm:

  - Every Qwen original+rewrite pair remains one atomic example.
  - A pair is assigned to the shortest stage whose sequence window contains it
    under the exact 40k tokenizer used by the research stagewise run:
       tokens <= 64  -> stage1, seq64,  batch512
       tokens <= 128 -> stage2, seq128, batch256
       tokens <= 256 -> stage3, seq256, batch128
  - Non-Qwen official/filler words from the inherited qwen_aligned_10M corpus are
    streamed in original row order and split into chunks that fill the remaining
    stage word budgets exactly.
  - Stage files are written in one pass over the inherited corpus so pairs and
    same-stage filler preserve as much original ordering as possible, rather than
    putting all pairs at the front of a stage.

The output preserves the 10M-word budget and all 1,656,800 Qwen pair words while
making pair-joint visibility a property of the training examples rather than an
accidental casualty of row chunking.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience")
ROOT = Path("experiments/archive/frontier_consolidation")
COMPACT_EXPERIENCE_SCRIPT_DIR = (COMPACT_EXPERIENCE / "scripts").resolve()
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPT_DIR))
import phase2_sota_trainer as tr  # type: ignore  # noqa: E402

QWEN10 = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
SELECTED_PAIRS = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/selected_pairs.jsonl"
PAIR_ROW_META = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl"
TOKENIZER40 = COMPACT_EXPERIENCE / "data/shared_tokenizer/hf_tokenizer_40k_shared"
OUT = ROOT / "data/qwen10_stagewise_pairfit"
SUMMARY = OUT / "stagewise_pairfit_summary.json"

STAGES = [
    {"stage": 1, "target_words": 3_000_000, "seq_len": 64, "batch_size": 512, "max_chunk_words": 32, "max_pair_tokens": 64},
    {"stage": 2, "target_words": 3_000_000, "seq_len": 128, "batch_size": 256, "max_chunk_words": 64, "max_pair_tokens": 128},
    {"stage": 3, "target_words": 4_000_000, "seq_len": 256, "batch_size": 128, "max_chunk_words": 160, "max_pair_tokens": 256},
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_chunks(words: list[str], max_n: int):
    for i in range(0, len(words), max_n):
        ww = words[i:i + max_n]
        if ww:
            yield i // max_n, ww


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    with QWEN10.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            text = str(r["text"])
            words = int(r.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"row {i} word mismatch field={words} actual={actual}")
            rows.append({
                "row_index": i,
                "text": text,
                "words": words,
                "source": str(r.get("source", "")),
                "example_id": r.get("example_id"),
            })
            total += words
    if total != 10_000_000:
        raise RuntimeError(f"expected 10M words, got {total}")
    return rows


def load_pairs_and_meta() -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    pairs: dict[str, dict[str, Any]] = {}
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs[str(r["pair_id"])] = r
    meta: dict[int, dict[str, Any]] = {}
    with PAIR_ROW_META.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[int(r["row_index"])] = r
    return pairs, meta


def choose_pair_stage(n_tokens: int) -> int:
    if n_tokens <= 64:
        return 1
    if n_tokens <= 128:
        return 2
    return 3


def empty_stats(stage: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "stage": stage["stage"],
        "target_words": stage["target_words"],
        "seq_len": stage["seq_len"],
        "batch_size": stage["batch_size"],
        "max_chunk_words": stage["max_chunk_words"],
        "max_pair_tokens": stage["max_pair_tokens"],
        "path": str(path),
        "rows": 0,
        "words": 0,
        "qwen_pair_rows": 0,
        "qwen_pair_words": 0,
        "filler_rows": 0,
        "filler_words": 0,
        "source_words": {},
        "source_rows": {},
        "qwen_pair_token_lengths": [],
        "qwen_pair_word_lengths": [],
    }


def write_rec(handle, stats: dict[str, Any], rec: dict[str, Any], kind: str, source_for_mix: str) -> None:
    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    w = int(rec["words"])
    stats["rows"] += 1
    stats["words"] += w
    stats["source_words"][source_for_mix] = stats["source_words"].get(source_for_mix, 0) + w
    stats["source_rows"][source_for_mix] = stats["source_rows"].get(source_for_mix, 0) + 1
    if kind == "qwen_pair":
        stats["qwen_pair_rows"] += 1
        stats["qwen_pair_words"] += w
    else:
        stats["filler_rows"] += 1
        stats["filler_words"] += w


def percentile(vals: list[int], q: float) -> int | None:
    if not vals:
        return None
    ss = sorted(vals)
    idx = min(len(ss) - 1, int(round(q * (len(ss) - 1))))
    return int(ss[idx])


def summarize_lengths(vals: list[int]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": min(vals),
        "p10": percentile(vals, 0.10),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": max(vals),
        "mean": round(sum(vals) / len(vals), 6),
    }


def precompute_pair_stage(pairs: dict[str, dict[str, Any]], tokenizer) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    counts = defaultdict(int)
    words_by_stage = defaultdict(int)
    token_lengths: list[int] = []
    word_lengths: list[int] = []
    long_pairs = []
    by_bucket_words = defaultdict(int)
    by_bucket_count = defaultdict(int)
    for pid, p in pairs.items():
        text = str(p["original"]) + " " + str(p["rewrite"])
        words = int(p["pair_words"])
        if words != len(text.split()):
            raise RuntimeError(f"pair word mismatch {pid}")
        n_tokens = len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        st = choose_pair_stage(n_tokens)
        bucket = "<=64" if n_tokens <= 64 else "<=128" if n_tokens <= 128 else "<=256" if n_tokens <= 256 else ">256"
        counts[st] += 1
        words_by_stage[st] += words
        by_bucket_count[bucket] += 1
        by_bucket_words[bucket] += words
        token_lengths.append(n_tokens)
        word_lengths.append(words)
        if n_tokens > 256:
            long_pairs.append({"pair_id": pid, "tokens": n_tokens, "words": words, "text": text[:240]})
        info[pid] = {
            "stage": st,
            "pair_tokens_40k": n_tokens,
            "bucket": bucket,
            "text": text,
        }
    summary = {
        "pair_stage_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "pair_stage_words": {str(k): int(v) for k, v in sorted(words_by_stage.items())},
        "token_bucket_counts": dict(sorted(by_bucket_count.items())),
        "token_bucket_words": dict(sorted(by_bucket_words.items())),
        "all_pair_token_lengths": summarize_lengths(token_lengths),
        "all_pair_word_lengths": summarize_lengths(word_lengths),
        "long_pairs_gt_256_tokens": long_pairs,
    }
    return info, summary


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    pairs, meta = load_pairs_and_meta()
    tokenizer = tr.make_portable_tokenizer(str(TOKENIZER40))
    pair_info, pair_summary = precompute_pair_stage(pairs, tokenizer)

    stage_paths = {
        s["stage"]: OUT / f"qwen10_pairfit_stage{s['stage']}_seq{s['seq_len']}_{s['target_words']//1_000_000}M.jsonl"
        for s in STAGES
    }
    handles = {k: p.open("w", encoding="utf-8") for k, p in stage_paths.items()}
    stats = {int(s["stage"]): empty_stats(s, stage_paths[int(s["stage"])]) for s in STAGES}

    # Remaining budgets after all pairs assigned by token length.
    pair_words_by_stage = {int(k): int(v) for k, v in pair_summary["pair_stage_words"].items()}
    filler_remaining = {int(s["stage"]): int(s["target_words"]) - pair_words_by_stage.get(int(s["stage"]), 0) for s in STAGES}
    if any(v < 0 for v in filler_remaining.values()):
        raise RuntimeError(f"pair words exceed a stage target: {filler_remaining}")
    total_filler_capacity = sum(filler_remaining.values())
    expected_non_qwen = 10_000_000 - sum(pair_words_by_stage.values())
    if total_filler_capacity != expected_non_qwen:
        raise RuntimeError(f"filler capacity {total_filler_capacity} != non-qwen words {expected_non_qwen}")

    written_pairs = set()
    qwen_rows = 0
    qwen_words = 0
    filler_stage_idx = 0
    filler_part_counter = 0
    try:
        for row in rows:
            if row["source"] == "qwen_pair_packed":
                qwen_rows += 1
                qwen_words += int(row["words"])
                m = meta.get(int(row["row_index"]))
                if m is None:
                    raise RuntimeError(f"missing qwen row meta for row {row['row_index']}")
                for pid in m["pair_ids"]:
                    pid = str(pid)
                    p = pairs[pid]
                    inf = pair_info[pid]
                    st = int(inf["stage"])
                    rec = {
                        "text": inf["text"],
                        "words": int(p["pair_words"]),
                        "example_id": stats[st]["rows"],
                        "source": f"qwen_pair_atomic::stage{st}::pair::{pid}",
                        "orig_source": "qwen_pair_packed",
                        "component_source": p.get("source"),
                        "orig_example_id": p.get("example_id"),
                        "orig_row_index": row["row_index"],
                        "pair_id": pid,
                        "original_words": int(p["original_words"]),
                        "rewrite_words": int(p["rewrite_words"]),
                        "pair_tokens_40k": int(inf["pair_tokens_40k"]),
                        "pairfit_stage_reason": inf["bucket"],
                    }
                    write_rec(handles[st], stats[st], rec, kind="qwen_pair", source_for_mix="qwen_pair_packed")
                    stats[st]["qwen_pair_token_lengths"].append(int(inf["pair_tokens_40k"]))
                    stats[st]["qwen_pair_word_lengths"].append(int(p["pair_words"]))
                    written_pairs.add(pid)
                continue

            # Stream non-Qwen words into the remaining stage capacities in order.
            words = str(row["text"]).split()
            cursor = 0
            while cursor < len(words):
                while filler_stage_idx < len(STAGES) and filler_remaining[int(STAGES[filler_stage_idx]["stage"])] <= 0:
                    filler_stage_idx += 1
                if filler_stage_idx >= len(STAGES):
                    raise RuntimeError("filler capacity exhausted before all non-Qwen rows were consumed")
                stage = STAGES[filler_stage_idx]
                st = int(stage["stage"])
                take = min(filler_remaining[st], len(words) - cursor)
                part = words[cursor:cursor + take]
                for chunk_id, ww in split_chunks(part, int(stage["max_chunk_words"])):
                    rec = {
                        "text": " ".join(ww),
                        "words": len(ww),
                        "example_id": stats[st]["rows"],
                        "source": f"{row['source']}::pairfit_stage{st}::row{row['row_index']}::part{filler_part_counter}::chunk{chunk_id}",
                        "orig_source": row["source"],
                        "orig_example_id": row.get("example_id"),
                        "orig_row_index": row["row_index"],
                        "row_part_id": filler_part_counter,
                        "chunk_id": chunk_id,
                    }
                    write_rec(handles[st], stats[st], rec, kind="filler", source_for_mix=str(row["source"]))
                filler_remaining[st] -= take
                cursor += take
                filler_part_counter += 1
    finally:
        for h in handles.values():
            h.close()

    if len(written_pairs) != len(pairs):
        raise RuntimeError(f"wrote {len(written_pairs)} pairs but selected_pairs has {len(pairs)}")
    if sum(pair_words_by_stage.values()) != 1_656_800:
        raise RuntimeError(f"qwen pair words changed: {sum(pair_words_by_stage.values())}")
    if any(v != 0 for v in filler_remaining.values()):
        raise RuntimeError(f"unused filler capacity remains: {filler_remaining}")
    for s in STAGES:
        st = int(s["stage"])
        if stats[st]["words"] != int(s["target_words"]):
            raise RuntimeError(f"stage {st} words {stats[st]['words']} != {s['target_words']}")

    out_stats: dict[str, Any] = {}
    for s in STAGES:
        st = int(s["stage"])
        path = stage_paths[st]
        rec = dict(stats[st])
        rec["qwen_pair_token_lengths"] = summarize_lengths(stats[st]["qwen_pair_token_lengths"])
        rec["qwen_pair_word_lengths"] = summarize_lengths(stats[st]["qwen_pair_word_lengths"])
        rec["sha256"] = sha256_file(path)
        rec["bytes"] = path.stat().st_size
        out_stats[str(st)] = rec

    payload = {
        "status": "QWEN10_STAGEWISE_PAIRFIT_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": "Qwen pairs are atomic and assigned by exact 40k token length to shortest containing seq window; non-Qwen text fills exact stage budgets in original order while preserving within-stage encounter order.",
        "input": str(QWEN10),
        "input_sha256": sha256_file(QWEN10),
        "tokenizer_path": str(TOKENIZER40),
        "tokenizer_vocab_size": len(tokenizer),
        "selected_pairs": str(SELECTED_PAIRS),
        "pair_row_meta": str(PAIR_ROW_META),
        "qwen_pair_rows_in_inherited_corpus": qwen_rows,
        "qwen_pair_words_in_inherited_corpus": qwen_words,
        "written_qwen_pairs": len(written_pairs),
        "written_qwen_pair_words": sum(pair_words_by_stage.values()),
        "filler_capacity_words_by_stage": {str(k): int(v) for k, v in sorted((int(s["stage"]), int(s["target_words"]) - pair_words_by_stage.get(int(s["stage"]), 0)) for s in STAGES)},
        **pair_summary,
        "stages": out_stats,
        "total_words": sum(s["target_words"] for s in STAGES),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "summary": str(SUMMARY),
        "elapsed_sec": payload["elapsed_sec"],
        "stage_rows": {k: v["rows"] for k, v in out_stats.items()},
        "stage_words": {k: v["words"] for k, v in out_stats.items()},
        "pair_stage_counts": payload["pair_stage_counts"],
        "pair_stage_words": payload["pair_stage_words"],
        "long_pairs_gt_256_tokens": len(payload["long_pairs_gt_256_tokens"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
