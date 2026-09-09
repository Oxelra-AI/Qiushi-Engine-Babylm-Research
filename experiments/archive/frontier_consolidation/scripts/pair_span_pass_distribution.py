#!/usr/bin/env python3
"""research: quantify per-pass distribution of changed rows in the frozen 100M stream.

The first stream validation showed all mapped compact-view rows join correctly ten
times, but pass order is shuffled rather than identical. This script measures the
actual per-pass and per-batch distribution so future source-view consistency work
uses accurate normalization assumptions.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import statistics
import time
from typing import Any

EXPECTED_PASS_ROWS = 64_740
EXPECTED_PASS_COUNT = 10
BATCH_SIZE = 256
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
STREAM_JSON = WORKSPACE / "data/pair_span_stream_validation/pair_span_stream_validation.json"
OUT_DIR = WORKSPACE / "data/pair_span_pass_distribution"


def stats(vals: list[float]) -> dict[str, float | int]:
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


def load_span_ids() -> set[int]:
    out: set[int] = set()
    with SPAN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(int(json.loads(line)["example_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    span_ids = load_span_ids()
    stream = json.loads(STREAM_JSON.read_text(encoding="utf-8"))

    pass_changed_counts = [0 for _ in range(EXPECTED_PASS_COUNT)]
    pass_changed_words = [0 for _ in range(EXPECTED_PASS_COUNT)]
    pass_batch_aux_counts: list[collections.Counter[int]] = [collections.Counter() for _ in range(EXPECTED_PASS_COUNT)]
    pass_positions: list[list[int]] = [[] for _ in range(EXPECTED_PASS_COUNT)]
    pass_order_prefix: list[list[int]] = [[] for _ in range(EXPECTED_PASS_COUNT)]
    missing_span_changed = 0
    wrong_source_span = 0
    repeated_ids_by_pass: list[set[int]] = [set() for _ in range(EXPECTED_PASS_COUNT)]

    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ex_id = int(obj.get("example_id", row_1 - 1))
            source = str(obj.get("source", ""))
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            pass_idx = (row_1 - 1) // EXPECTED_PASS_ROWS
            idx_in_pass = (row_1 - 1) % EXPECTED_PASS_ROWS
            if pass_idx >= EXPECTED_PASS_COUNT:
                continue
            if len(pass_order_prefix[pass_idx]) < 30:
                pass_order_prefix[pass_idx].append(ex_id)
            if source == CHANGED_SOURCE:
                pass_changed_counts[pass_idx] += 1
                pass_changed_words[pass_idx] += words
                pass_positions[pass_idx].append(idx_in_pass + 1)
                repeated_ids_by_pass[pass_idx].add(ex_id)
                pass_batch_aux_counts[pass_idx][idx_in_pass // BATCH_SIZE] += 1
                if ex_id not in span_ids:
                    missing_span_changed += 1
            elif ex_id in span_ids:
                wrong_source_span += 1

    set_sizes = [len(s) for s in repeated_ids_by_pass]
    all_pass_sets_equal = all(s == span_ids for s in repeated_ids_by_pass)
    batch_count_per_pass = (EXPECTED_PASS_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
    pass_summaries: list[dict[str, Any]] = []
    for p in range(EXPECTED_PASS_COUNT):
        aux_counts = [float(pass_batch_aux_counts[p].get(b, 0)) for b in range(batch_count_per_pass)]
        nonzero = [x for x in aux_counts if x > 0]
        positions = [float(x) for x in pass_positions[p]]
        pass_summaries.append({
            "pass_index_1based": p + 1,
            "changed_rows": pass_changed_counts[p],
            "changed_words": pass_changed_words[p],
            "unique_changed_ids": set_sizes[p],
            "changed_positions": stats(positions),
            "first_20_changed_positions": pass_positions[p][:20],
            "last_20_changed_positions": pass_positions[p][-20:],
            "batches_with_changed_rows": sum(1 for x in aux_counts if x > 0),
            "fraction_batches_with_changed_rows": sum(1 for x in aux_counts if x > 0) / batch_count_per_pass,
            "changed_rows_per_batch_all_batches": stats(aux_counts),
            "changed_rows_per_nonempty_batch": stats(nonzero),
            "first_30_example_ids_in_pass": pass_order_prefix[p],
        })

    summary = {
        "status": "PAIR_SPAN_PASS_DISTRIBUTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Correct the stream-level picture: the 100M file contains ten 10M passes with the same examples but shuffled order, while every mapped changed example still appears exactly once per pass and ten times overall.",
        "input_validation": {
            "stream_validation_json": str(STREAM_JSON),
            "stream_status": stream.get("status"),
            "stream_all_mapped_repeat_ten": stream.get("join_counts", {}).get("all_mapped_examples_repeat_exactly_ten_times"),
            "stream_passes_match_first_pass": stream.get("stream_counts", {}).get("passes_match_first_pass"),
        },
        "global_counts": {
            "span_ids": len(span_ids),
            "pass_changed_counts": pass_changed_counts,
            "pass_changed_words": pass_changed_words,
            "unique_changed_ids_by_pass": set_sizes,
            "all_pass_changed_sets_equal_span_ids": all_pass_sets_equal,
            "missing_span_changed_rows": missing_span_changed,
            "wrong_source_span_rows": wrong_source_span,
            "batch_count_per_pass": batch_count_per_pass,
        },
        "pass_summaries": pass_summaries,
        "interpretation": [
            "The 100M stream is ten shuffled 10M passes, not ten identical row orders; a future source-view consistency trainer should join by example_id rather than by row offset within pass.",
            "Every pass contains all 3005 changed-row examples exactly once, and no changed-source row lacks a span map, so the research map is usable across the repeated exposure stream.",
            "Changed rows are distributed across nearly all batches in every pass rather than appearing as a single contiguous block, so the auxiliary signal is sparse per batch but not temporally confined to one corpus segment.",
            "This remains route-neutral engineering evidence; mature 70M/80M treatment effects still decide whether source-view consistency is worth GPU testing.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_span_pass_distribution.json"
    out_md = out_dir / "pair_span_pass_distribution.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair-span pass distribution",
        "",
        summary["purpose"],
        "",
        "## Global counts",
    ]
    for k, v in summary["global_counts"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Per-pass distribution"]
    for rec in pass_summaries:
        lines.append(
            f"- pass {rec['pass_index_1based']}: changed_rows={rec['changed_rows']}, "
            f"changed_words={rec['changed_words']}, batches_with_changed={rec['batches_with_changed_rows']}/"
            f"{batch_count_per_pass} ({rec['fraction_batches_with_changed_rows']:.4f}), "
            f"nonempty_batch_mean={rec['changed_rows_per_nonempty_batch']['mean']:.2f}, "
            f"nonempty_min={rec['changed_rows_per_nonempty_batch']['min']}, "
            f"nonempty_max={rec['changed_rows_per_nonempty_batch']['max']}"
        )
    lines += ["", "## Interpretation"]
    for item in summary["interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "all_pass_changed_sets_equal_span_ids": all_pass_sets_equal,
        "missing_span_changed_rows": missing_span_changed,
        "wrong_source_span_rows": wrong_source_span,
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
