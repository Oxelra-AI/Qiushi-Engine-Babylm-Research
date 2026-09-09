#!/usr/bin/env python3
"""research: materialize repaired low-truncation stagewise clean-Qwen 10M data.

The research exposure audit showed that the unchanged research seq64->256
schedule hides roughly half of counted word groups because it crops 160-word rows
while still counting all words.  This script constructs a training-control corpus
that matches the leader-style inverse-batch sequence curriculum without hidden
word exposure:

  stage1: 3M words, chunks <=32 words, seq64, batch512
  stage2: 3M words, chunks <=64 words, seq128, batch256
  stage3: 4M words, chunks <=160 words, seq256, batch128

It deliberately uses row/chunk materialization, including Qwen pair-packed rows,
so nearly all counted Qwen words are visible under short sequence lengths.  This
is a training-dynamics control, not a claim that pair-atom semantics are the final
best data representation.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/frontier_consolidation")
COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience")
SRC = COMPACT_EXPERIENCE / "data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl"
OUT = ROOT / "data/qwen10_stagewise_rowchunk"
SUMMARY = OUT / "stagewise_rowchunk_summary.json"

STAGES = [
    {"stage": 1, "target_words": 3_000_000, "max_chunk_words": 32, "seq_len": 64, "batch_size": 512},
    {"stage": 2, "target_words": 3_000_000, "max_chunk_words": 64, "seq_len": 128, "batch_size": 256},
    {"stage": 3, "target_words": 4_000_000, "max_chunk_words": 160, "seq_len": 256, "batch_size": 128},
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


def write_stage_chunk(handle, stats: dict[str, Any], source_row: dict[str, Any], stage: dict[str, Any], row_idx: int, row_part_idx: int, part_words: list[str]) -> None:
    max_chunk = int(stage["max_chunk_words"])
    for chunk_id, ww in split_chunks(part_words, max_chunk):
        source = str(source_row.get("source", ""))
        rec = {
            "text": " ".join(ww),
            "words": len(ww),
            "example_id": stats["rows"],
            "source": f"{source}::stage{stage['stage']}::row{row_idx}::part{row_part_idx}::chunk{chunk_id}",
            "orig_source": source,
            "orig_example_id": source_row.get("example_id"),
            "orig_row_index": row_idx,
            "row_part_id": row_part_idx,
            "chunk_id": chunk_id,
        }
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["rows"] += 1
        stats["words"] += len(ww)
        stats["source_words"][source] = stats["source_words"].get(source, 0) + len(ww)
        stats["orig_rows_seen"].add(row_idx)


def materialize() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    stage_paths = {
        s["stage"]: OUT / f"qwen10_rowchunk_stage{s['stage']}_seq{s['seq_len']}_{s['target_words']//1_000_000}M.jsonl"
        for s in STAGES
    }
    handles = {k: p.open("w", encoding="utf-8") for k, p in stage_paths.items()}
    stats: dict[int, dict[str, Any]] = {
        s["stage"]: {
            "stage": s["stage"],
            "target_words": s["target_words"],
            "seq_len": s["seq_len"],
            "batch_size": s["batch_size"],
            "max_chunk_words": s["max_chunk_words"],
            "path": str(stage_paths[s["stage"]]),
            "rows": 0,
            "words": 0,
            "source_words": {},
            "orig_rows_seen": set(),
        }
        for s in STAGES
    }

    stage_idx = 0
    stage_words = 0
    input_rows = 0
    input_words = 0
    row_part_counter = 0
    try:
        with SRC.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = str(r["text"])
                words = text.split()
                field = int(r.get("words", len(words)))
                if field != len(words):
                    raise RuntimeError(f"word mismatch at input row {row_idx}: field={field} actual={len(words)}")
                input_rows += 1
                input_words += field
                cursor = 0
                while cursor < len(words):
                    if stage_idx >= len(STAGES):
                        raise RuntimeError(f"more than {sum(s['target_words'] for s in STAGES)} words in input")
                    stage = STAGES[stage_idx]
                    remaining = int(stage["target_words"]) - stage_words
                    if remaining <= 0:
                        stage_idx += 1
                        stage_words = 0
                        continue
                    take = min(remaining, len(words) - cursor)
                    part_words = words[cursor:cursor + take]
                    st = int(stage["stage"])
                    write_stage_chunk(handles[st], stats[st], r, stage, row_idx, row_part_counter, part_words)
                    row_part_counter += 1
                    cursor += take
                    stage_words += take
                    if stage_words == int(stage["target_words"]):
                        stage_idx += 1
                        stage_words = 0
    finally:
        for h in handles.values():
            h.close()

    target_total = sum(int(s["target_words"]) for s in STAGES)
    if input_words != target_total:
        raise RuntimeError(f"input words {input_words} != expected {target_total}")
    if sum(int(v["words"]) for v in stats.values()) != target_total:
        raise RuntimeError("materialized total mismatch")

    out_stats: dict[str, Any] = {}
    for s in STAGES:
        st = int(s["stage"])
        if stats[st]["words"] != s["target_words"]:
            raise RuntimeError(f"stage {st} words {stats[st]['words']} != {s['target_words']}")
        path = stage_paths[st]
        rec = dict(stats[st])
        rec["orig_rows_seen"] = len(stats[st]["orig_rows_seen"])
        rec["sha256"] = sha256_file(path)
        rec["bytes"] = path.stat().st_size
        out_stats[str(st)] = rec

    payload = {
        "status": "QWEN10_STAGEWISE_ROWCHUNK_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": "low-truncation row chunks for repaired leader-style sequence curriculum; Qwen pair-packed rows may be split deliberately to expose counted words during seq64/128 stages",
        "input": str(SRC),
        "input_sha256": sha256_file(SRC),
        "input_rows": input_rows,
        "input_words": input_words,
        "stages": out_stats,
        "total_words": target_total,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    t0 = time.time()
    payload = materialize()
    payload["elapsed_sec"] = round(time.time() - t0, 3)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "summary": str(SUMMARY),
        "elapsed_sec": payload["elapsed_sec"],
        "stage_rows": {k: v["rows"] for k, v in payload["stages"].items()},
        "stage_words": {k: v["words"] for k, v in payload["stages"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
