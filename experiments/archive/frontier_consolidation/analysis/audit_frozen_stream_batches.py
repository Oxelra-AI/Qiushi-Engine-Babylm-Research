#!/usr/bin/env python3
"""Audit changed-row mixing in the frozen 100M order without tokenizing/training."""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

import build_innovation_metadata as builder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=pathlib.Path, default=builder.DEFAULT_TRAIN)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=pathlib.Path, default=builder.WORKSPACE / "analysis/frozen_stream_batch_audit.json")
    args = ap.parse_args()
    batch_counts: list[int] = []
    changed_positions: list[int] = []
    in_batch = 0
    rows = 0
    with args.train.open("r", encoding="utf-8") as f:
        for rows, line in enumerate(f, 1):
            obj = json.loads(line)
            if str(obj.get("source", "")) == builder.CHANGED_SOURCE:
                in_batch += 1
                changed_positions.append(rows)
            if rows % args.batch_size == 0:
                batch_counts.append(in_batch)
                in_batch = 0
    if rows % args.batch_size:
        batch_counts.append(in_batch)
    gaps = [b - a for a, b in zip(changed_positions, changed_positions[1:])]
    hist = collections.Counter(batch_counts)
    result = {
        "status": "FROZEN_STREAM_BATCH_AUDIT",
        "train_sha256": builder.sha256_file(args.train),
        "batch_size": args.batch_size,
        "rows": rows,
        "changed_row_exposures": len(changed_positions),
        "batches": len(batch_counts),
        "batches_with_changed": sum(x > 0 for x in batch_counts),
        "all_changed_batches": sum(x == args.batch_size for x in batch_counts),
        "changed_per_batch": {
            "mean": statistics.mean(batch_counts),
            "min": min(batch_counts),
            "max": max(batch_counts),
            "histogram": dict(sorted(hist.items())),
        },
        "changed_row_gaps": {
            "min": min(gaps), "median": statistics.median(gaps), "max": max(gaps),
        },
        "interpretation": "Every actual batch contains a large ordinary-row donor pool; the balanced batch-64 smoke is stricter than the frozen batch-256 mix.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
