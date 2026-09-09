#!/usr/bin/env python3
"""research: prepare the FW source-repeat attribution arm without launching it.

The current expensive comparison is compact_view vs whole-sentence source_breadth.
If the result is close, or compact wins and we must distinguish compact restatement
from literal same-proposition reinforcement, the research source_repeat arm is the
next attribution comparator. This script verifies that the existing 10M
source_repeat pool can be expanded to a matched 100M stream using the same pass
orders as the already-trained compact/breadth streams. It writes only a small
dry-run manifest unless --write-stream is explicitly provided.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import time
from typing import Any, Iterable

USER_ROOT = pathlib.Path(".").resolve()
A01_WS = pathlib.Path("experiments/archive/representation_and_objectives")
A02_WS = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DIR = A02_WS / "data/fw_source_repeat_contingency"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/fw_source_repeat_contingency.md')

COMPACT_10M = A01_WS / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
SOURCE_REPEAT_10M = A01_WS / "data/fw_full_arms/fw_preserved_source_repeat_10M.jsonl"
MANIFEST = A01_WS / "data/fw_full_arms/fw_preservation_aware_arms_manifest.json"
CHECK = A01_WS / "data/fw_compact_vs_wholesentence_breadth_check/fw_compact_vs_wholesentence_breadth_check.json"
TOKENIZER_DIR = A01_WS / "data/shared_tokenizer/shared_16k_tokenizer"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
EXPECTED_TOKENIZER_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"

STREAM_SEED = 10289931
PASSES = 10
TOTAL_WORDS = 10_000_000
TOTAL_EXPOSURE = 100_000_000
SOURCE_REPEAT_LABEL = "fw_preserved_source_repeat"
COMPACT_LABEL = "fw_preserved_compact_view"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_words(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pool(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            text = str(r.get("text") or "")
            words = int(r.get("words") or norm_words(text))
            if words != norm_words(text):
                raise RuntimeError(f"word-count mismatch in {path} line {line_no}: field={words} actual={norm_words(text)}")
            rows.append({"text": text, "words": words, "example_id": int(r.get("example_id") or 0), "source": str(r.get("source") or "")})
    return rows


def source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[str(r.get("source") or "")] += int(r["words"])
    return dict(c)


def pass_order_hashes(n: int, seed: int) -> list[str]:
    hashes: list[str] = []
    for pass_i in range(PASSES):
        order = list(range(n))
        random.Random(seed + 1000 + pass_i).shuffle(order)
        hashes.append(sha256_text(",".join(map(str, order))))
    return hashes


def write_training_stream(path: pathlib.Path, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    order_hashes: list[str] = []
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(len(rows)))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            order_hashes.append(sha256_text(",".join(map(str, order))))
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += int(r["words"])
    if total != TOTAL_EXPOSURE:
        raise RuntimeError(f"stream exposure {total} != {TOTAL_EXPOSURE}")
    return {"path": str(path), "rows": len(rows) * PASSES, "words": total, "sha256": sha256_file(path), "pass_order_hashes": order_hashes, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-stream", action="store_true", help="Actually write the 100M source_repeat JSONL stream. Do not use unless the attribution run is authorized.")
    ap.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)

    compact_rows = load_pool(COMPACT_10M)
    repeat_rows = load_pool(SOURCE_REPEAT_10M)
    pool_manifest = read_json(MANIFEST)
    breadth_check = read_json(CHECK)

    errors: list[str] = []
    if len(compact_rows) != len(repeat_rows):
        errors.append(f"row_count mismatch compact={len(compact_rows)} repeat={len(repeat_rows)}")
    compact_words = sum(r["words"] for r in compact_rows)
    repeat_words = sum(r["words"] for r in repeat_rows)
    if compact_words != TOTAL_WORDS or repeat_words != TOTAL_WORDS:
        errors.append(f"word totals compact={compact_words} repeat={repeat_words}")
    if [r["words"] for r in compact_rows] != [r["words"] for r in repeat_rows]:
        errors.append("row word sequence differs between compact and source_repeat")
    if [r["example_id"] for r in compact_rows] != [r["example_id"] for r in repeat_rows]:
        errors.append("example_id sequence differs between compact and source_repeat")
    compact_fw = sum(1 for r in compact_rows if r["source"] == COMPACT_LABEL)
    repeat_fw = sum(1 for r in repeat_rows if r["source"] == SOURCE_REPEAT_LABEL)
    if compact_fw != repeat_fw:
        errors.append(f"FW row count mismatch compact={compact_fw} repeat={repeat_fw}")
    tok_sha = sha256_file(TOKENIZER_JSON) if TOKENIZER_JSON.exists() else None
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        errors.append(f"tokenizer SHA mismatch: {tok_sha}")

    generated_order_hashes = pass_order_hashes(len(repeat_rows), args.stream_seed)
    compact_stream_record = breadth_check.get("training_streams", {}).get("compact_stream_manifest_record", {})
    expected_order_hashes = compact_stream_record.get("pass_order_hashes") or []
    pass_orders_match_step104 = generated_order_hashes == expected_order_hashes
    if expected_order_hashes and not pass_orders_match_step104:
        errors.append("generated source_repeat pass orders do not match research compact stream pass orders")

    stream_record: dict[str, Any] | None = None
    if args.write_stream:
        if errors:
            raise SystemExit(f"Refusing to write stream with errors: {errors}")
        stream_record = write_training_stream(OUT_DIR / "fw_preserved_source_repeat_100M.jsonl", repeat_rows, args.stream_seed)

    manifest = {
        "status": "SOURCE_REPEAT_CONTINGENCY_READY" if not errors else "SOURCE_REPEAT_CONTINGENCY_FAILED",
        "created_utc": now_utc(),
        "scientific_purpose": "Prepare the literal same-source reinforcement attribution arm for the FW mechanism family; no training is launched here.",
        "write_stream": args.write_stream,
        "inputs": {
            "compact_10m": str(COMPACT_10M),
            "compact_10m_sha256": sha256_file(COMPACT_10M),
            "source_repeat_10m": str(SOURCE_REPEAT_10M),
            "source_repeat_10m_sha256": sha256_file(SOURCE_REPEAT_10M),
            "manifest": str(MANIFEST),
            "check": str(CHECK),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "tokenizer_json_sha256": tok_sha,
        },
        "checks": {
            "errors": errors,
            "compact_rows": len(compact_rows),
            "source_repeat_rows": len(repeat_rows),
            "compact_words": compact_words,
            "source_repeat_words": repeat_words,
            "row_word_sequence_matches_compact": [r["words"] for r in compact_rows] == [r["words"] for r in repeat_rows],
            "example_id_sequence_matches_compact": [r["example_id"] for r in compact_rows] == [r["example_id"] for r in repeat_rows],
            "compact_fw_rows": compact_fw,
            "source_repeat_fw_rows": repeat_fw,
            "tokenizer_sha_matches_expected": tok_sha == EXPECTED_TOKENIZER_SHA,
            "generated_pass_order_hashes": generated_order_hashes,
            "compact_pass_order_hashes": expected_order_hashes,
            "pass_orders_match_step104_compact_and_breadth": pass_orders_match_step104,
            "projected_100m_rows": len(repeat_rows) * PASSES,
            "projected_100m_words": repeat_words * PASSES,
        },
        "source_word_counts_10m": {
            "compact_view": source_counts(compact_rows),
            "source_repeat": source_counts(repeat_rows),
        },
        "budget_reference": pool_manifest.get("budgets", {}),
        "training_stream": stream_record,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = OUT_DIR / ("source_repeat_stream_manifest.json" if args.write_stream else "source_repeat_stream_dryrun.json")
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    NOTE.write_text(
        "# research FW source-repeat contingency\n\n"
        "This note records a CPU-only preparation for the literal source-repeat arm. It does not authorize or launch training.\n\n"
        f"- Status: {manifest['status']}\n"
        f"- Compact/source-repeat rows: {len(compact_rows):,} / {len(repeat_rows):,}\n"
        f"- Compact/source-repeat words: {compact_words:,} / {repeat_words:,}\n"
        f"- Row word sequence matches: {manifest['checks']['row_word_sequence_matches_compact']}\n"
        f"- Example-id sequence matches: {manifest['checks']['example_id_sequence_matches_compact']}\n"
        f"- Generated pass orders match research streams: {pass_orders_match_step104}\n"
        f"- Source-repeat 10M SHA: {manifest['inputs']['source_repeat_10m_sha256']}\n"
        f"- Manifest: `{out_json}`\n\n"
        "Use only if the compact-vs-breadth result requires an attribution arm. The first authorized action would be `--write-stream`, then the same research training/eval recipe with this source-repeat 100M stream.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": manifest["status"],
        "out_json": str(out_json),
        "errors": errors,
        "rows": len(repeat_rows),
        "words": repeat_words,
        "pass_orders_match_step104": pass_orders_match_step104,
        "write_stream": args.write_stream,
        "elapsed_sec": manifest["elapsed_sec"],
    }, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
