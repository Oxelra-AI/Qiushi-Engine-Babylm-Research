#!/usr/bin/env python3
"""research: CPU smoke test for a future source-view consistency collate path.

If the mature legal-tokenizer trajectory later selects source-view consistency,
The comparison requires a trainer whose dataset/collate carries source/rewrite spans in
addition to the inherited MLM tensors.  This script prototypes that interface on
bounded samples of the frozen 100M stream.  It joins by example_id, uses the
research legal tokenizer, and emits batch-level auxiliary payload statistics.  It
performs no model training/evaluation and changes no corpus/tokenizer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TRAIN_ROWS = 647_400
SEQ_LEN = 256
BATCH_SIZE = 256
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
TOKENIZER_DIR = WORKSPACE / "data/compliant_tokenizer"
OUT_DIR = WORKSPACE / "data/consistency_collate_smoke"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


@dataclass
class StreamExample:
    text: str
    words: int
    example_id: int
    source: str
    global_row_1based: int


def load_span_map(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[int(obj["example_id"])] = obj
    return out


def read_stream_sample(path: pathlib.Path, mode: str, rows: int) -> list[StreamExample]:
    if rows <= 0:
        raise ValueError("rows must be positive for smoke test")
    examples: list[StreamExample] = []
    if mode == "front":
        wanted = set(range(1, min(rows, TRAIN_ROWS) + 1))
    elif mode == "stride":
        if rows == 1:
            wanted = {1}
        else:
            wanted = {1 + round(i * (TRAIN_ROWS - 1) / (rows - 1)) for i in range(rows)}
    else:
        raise ValueError(f"unknown sample mode: {mode}")
    with path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            if row_1 not in wanted:
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            examples.append(StreamExample(
                text=text,
                words=int(obj.get("words", len(text.split()))),
                example_id=int(obj.get("example_id", row_1 - 1)),
                source=str(obj.get("source", "")),
                global_row_1based=row_1,
            ))
            if len(examples) >= len(wanted):
                break
    examples.sort(key=lambda x: x.global_row_1based)
    return examples


class ConsistencySpanDataset(Dataset):
    def __init__(self, examples: list[StreamExample], tokenizer, span_by_ex: dict[int, dict[str, Any]], seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.span_by_ex = span_by_ex
        self.seq_len = seq_len
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start_cache: dict[int, bool] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _word_start_flag(self, tid: int) -> bool:
        v = self._word_start_cache.get(tid)
        if v is None:
            tok = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(tok is not None and is_word_start(str(tok)))
            self._word_start_cache[tid] = v
        return v

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if int(attention_mask[i]) == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid
        span = self.span_by_ex.get(ex.example_id)
        aux_pairs: list[dict[str, Any]] = []
        aux_error: str | None = None
        if span is not None:
            if ex.source != CHANGED_SOURCE:
                aux_error = "span_joined_to_nonchanged_source"
            elif int(span.get("token_len_truncated", 0)) != int(attention_mask.sum().item()):
                aux_error = "token_length_mismatch"
            else:
                for p in span.get("pairs") or []:
                    if p.get("visibility") != "both_visible":
                        continue
                    s_ranges = [[int(a), int(b)] for a, b in p.get("source_token_ranges") or []]
                    r_ranges = [[int(a), int(b)] for a, b in p.get("rewrite_token_ranges") or []]
                    s_count = sum(b - a for a, b in s_ranges)
                    r_count = sum(b - a for a, b in r_ranges)
                    if s_count <= 0 or r_count <= 0:
                        continue
                    if any(a < 0 or b > self.seq_len or b <= a for a, b in s_ranges + r_ranges):
                        aux_error = "invalid_range_bounds"
                        continue
                    if any(attention_mask[a:b].sum().item() != (b - a) for a, b in s_ranges + r_ranges):
                        aux_error = "range_crosses_padding"
                        continue
                    aux_pairs.append({
                        "pair_id": str(p.get("pair_id", "")),
                        "source_ranges": s_ranges,
                        "rewrite_ranges": r_ranges,
                        "source_token_count": s_count,
                        "rewrite_token_count": r_count,
                    })
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "words": ex.words,
            "example_id": ex.example_id,
            "source": ex.source,
            "global_row_1based": ex.global_row_1based,
            "aux_pairs": aux_pairs,
            "aux_error": aux_error,
        }


def consistency_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    aux_records: list[dict[str, Any]] = []
    errors = collections.Counter()
    for bi, item in enumerate(batch):
        if item.get("aux_error"):
            errors[str(item["aux_error"])] += 1
        for p in item.get("aux_pairs") or []:
            aux_records.append({
                "batch_row": bi,
                "global_row_1based": int(item["global_row_1based"]),
                "example_id": int(item["example_id"]),
                "pair_id": p["pair_id"],
                "source_ranges": p["source_ranges"],
                "rewrite_ranges": p["rewrite_ranges"],
                "source_token_count": int(p["source_token_count"]),
                "rewrite_token_count": int(p["rewrite_token_count"]),
            })
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "global_row_1based": torch.tensor([x["global_row_1based"] for x in batch], dtype=torch.long),
        "aux_records": aux_records,
        "aux_errors": dict(errors),
    }


def basic_stats(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "std": float(statistics.pstdev(vals)),
    }


def summarize_sample(name: str, examples: list[StreamExample], tokenizer, span_by_ex: dict[int, dict[str, Any]], seq_len: int, batch_size: int) -> dict[str, Any]:
    ds = ConsistencySpanDataset(examples, tokenizer, span_by_ex, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=consistency_collate)
    batches = 0
    rows = 0
    words = 0
    aux_batches = 0
    aux_records_total = 0
    aux_rows_set: set[tuple[int, int]] = set()
    aux_source_tokens = 0
    aux_rewrite_tokens = 0
    batch_aux_counts: list[float] = []
    batch_aux_rows: list[float] = []
    pair_source_lengths: list[float] = []
    pair_rewrite_lengths: list[float] = []
    error_counts = collections.Counter()
    first_aux_records: list[dict[str, Any]] = []
    first_batch_shapes: dict[str, Any] | None = None
    for batch in loader:
        batches += 1
        rows += int(batch["input_ids"].shape[0])
        words += int(batch["words"].sum().item())
        if first_batch_shapes is None:
            first_batch_shapes = {
                "input_ids": list(batch["input_ids"].shape),
                "attention_mask": list(batch["attention_mask"].shape),
                "word_group": list(batch["word_group"].shape),
            }
        aux = batch["aux_records"]
        aux_records_total += len(aux)
        batch_aux_counts.append(float(len(aux)))
        rows_this = {(int(r["global_row_1based"]), int(r["example_id"])) for r in aux}
        batch_aux_rows.append(float(len(rows_this)))
        if aux:
            aux_batches += 1
        aux_rows_set.update(rows_this)
        for k, v in (batch.get("aux_errors") or {}).items():
            error_counts[k] += int(v)
        for r in aux:
            aux_source_tokens += int(r["source_token_count"])
            aux_rewrite_tokens += int(r["rewrite_token_count"])
            pair_source_lengths.append(float(r["source_token_count"]))
            pair_rewrite_lengths.append(float(r["rewrite_token_count"]))
            if len(first_aux_records) < 6:
                first_aux_records.append(r)
    return {
        "name": name,
        "rows": rows,
        "words": words,
        "batches": batches,
        "first_batch_shapes": first_batch_shapes,
        "aux_batches": aux_batches,
        "fraction_batches_with_aux": aux_batches / batches if batches else 0.0,
        "aux_rows": len(aux_rows_set),
        "aux_pair_records": aux_records_total,
        "aux_source_tokens": aux_source_tokens,
        "aux_rewrite_tokens": aux_rewrite_tokens,
        "aux_rows_per_batch": basic_stats(batch_aux_rows),
        "aux_pairs_per_batch": basic_stats(batch_aux_counts),
        "pair_source_token_counts": basic_stats(pair_source_lengths),
        "pair_rewrite_token_counts": basic_stats(pair_rewrite_lengths),
        "aux_errors": dict(error_counts),
        "first_aux_records": first_aux_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--front-rows", type=int, default=4096)
    ap.add_argument("--stride-rows", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--skip-sha", action="store_true", help="development only")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sha = None if args.skip_sha else sha256_file(TRAIN_100M)
    tok_sha = None if args.skip_sha else sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if train_sha is not None and train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha is not None and tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    span_by_ex = load_span_map(SPAN_JSONL)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    samples = [
        ("front_4096_rows", read_stream_sample(TRAIN_100M, "front", args.front_rows)),
        ("stride_4096_rows", read_stream_sample(TRAIN_100M, "stride", args.stride_rows)),
    ]
    sample_summaries = [
        summarize_sample(name, examples, tokenizer, span_by_ex, args.seq_len, args.batch_size)
        for name, examples in samples
    ]
    any_errors = collections.Counter()
    for s in sample_summaries:
        for k, v in (s.get("aux_errors") or {}).items():
            any_errors[k] += int(v)

    summary = {
        "status": "CONSISTENCY_COLLATE_SMOKE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Prototype the dataset/collate interface needed for a future source-view consistency objective, using frozen training-side spans and tokenizer but no training/evaluation.",
        "inputs": {
            "train_100m": str(TRAIN_100M),
            "train_sha256": train_sha,
            "span_jsonl": str(SPAN_JSONL),
            "span_examples": len(span_by_ex),
            "tokenizer": str(TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        "sample_summaries": sample_summaries,
        "combined_aux_errors": dict(any_errors),
        "interface_contract_for_future_trainer": [
            "Dataset returns the inherited MLM tensors plus example_id, global_row_1based, and aux_pairs looked up by example_id.",
            "Collate keeps tensor stacking unchanged and adds an aux_records list with batch_row, example_id, pair_id, source_ranges, rewrite_ranges, and per-side token counts for both-visible pairs only.",
            "A future auxiliary loss should compute pooled source/rewrite representations from model hidden states after the forward pass, average within pair records, then normalize per eligible row or per eligible pair so batches with many changed rows do not dominate.",
            "No in-batch negatives or static-prior masking are part of this first source-view consistency interface; it must stay a single-variable auxiliary-loss screen if selected.",
        ],
        "scientific_interpretation": [
            "Successful smoke testing means source-view consistency is implementable without changing corpus order or tokenizer, but it is not evidence that it improves BabyLM scores.",
            "The mature legal-tokenizer treatment pattern remains necessary before spending GPU on this objective; this script only removes one engineering uncertainty from the broad-disappearance branch.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "consistency_collate_smoke.json"
    out_md = out_dir / "consistency_collate_smoke.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research consistency-collate smoke test",
        "",
        summary["purpose"],
        "",
        "This is CPU-only. It does not train, evaluate, alter the corpus/tokenizer, or choose the route.",
        "",
        "## Samples",
    ]
    for s in sample_summaries:
        lines.append(
            f"- {s['name']}: rows={s['rows']}, batches={s['batches']}, "
            f"aux_rows={s['aux_rows']}, aux_pair_records={s['aux_pair_records']}, "
            f"aux_batches={s['aux_batches']} ({s['fraction_batches_with_aux']:.4f}), "
            f"aux_pairs_per_batch mean={s['aux_pairs_per_batch']['mean']:.2f}, "
            f"max={s['aux_pairs_per_batch']['max']}, aux_errors={s['aux_errors']}"
        )
    lines += ["", "## Future trainer interface"]
    for item in summary["interface_contract_for_future_trainer"]:
        lines.append(f"- {item}")
    lines += ["", "## Interpretation"]
    for item in summary["scientific_interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "combined_aux_errors": dict(any_errors),
        "samples": [{"name": s["name"], "rows": s["rows"], "aux_pair_records": s["aux_pair_records"]} for s in sample_summaries],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
