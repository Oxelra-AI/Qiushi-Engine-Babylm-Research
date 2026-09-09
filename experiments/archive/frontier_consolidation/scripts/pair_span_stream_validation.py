#!/usr/bin/env python3
"""research: validate pair-span map against the frozen 100M trainer stream.

This is CPU-only support for a possible future source-view consistency trainer. It
checks whether the research example_id -> source/rewrite span map joins cleanly to
all ten repeated passes of the frozen legal compact-view-reinvest training file,
and quantifies where a consistency loss would occur under the inherited batch
order. It does not train, evaluate, alter data, or choose an intervention route.
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
from typing import Any


EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TRAIN_WORDS = 100_000_000
EXPECTED_TRAIN_ROWS = 647_400
EXPECTED_PASS_ROWS = 64_740
EXPECTED_PASS_WORDS = 10_000_000
EXPECTED_PASS_COUNT = 10
CHANGED_SOURCE = "cleanqwen_fineweb_compact_view_reinvest"
EXPOSURES = [20_000_000, 70_000_000, 80_000_000, 100_000_000]
BATCH_SIZE = 256


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
SPAN_JSONL = WORKSPACE / "data/pair_span_map/pair_span_map.jsonl"
SPAN_SUMMARY = WORKSPACE / "data/pair_span_map/pair_span_map_summary.json"
OUT_DIR = WORKSPACE / "data/pair_span_stream_validation"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_fingerprint(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=12).hexdigest()


def load_span_map(path: pathlib.Path) -> dict[int, dict[str, Any]]:
    by_ex: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ex_id = int(obj["example_id"])
            if ex_id in by_ex:
                raise RuntimeError(f"duplicate example_id in span map: {ex_id} at line {line_no}")
            by_ex[ex_id] = obj
    return by_ex


def basic_stats(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def burst_lengths(aux_by_batch: dict[int, int]) -> list[int]:
    if not aux_by_batch:
        return []
    keys = sorted(aux_by_batch)
    bursts: list[int] = []
    cur = 1
    for prev, x in zip(keys, keys[1:]):
        if x == prev + 1:
            cur += 1
        else:
            bursts.append(cur)
            cur = 1
    bursts.append(cur)
    return bursts


class ExposureAccumulator:
    def __init__(self, exposure_words: int, batch_size: int):
        self.exposure_words = exposure_words
        self.batch_size = batch_size
        self.rows = 0
        self.words = 0
        self.aux_rows = 0
        self.aux_pair_records = 0
        self.aux_both_visible_pairs = 0
        self.aux_source_tokens = 0
        self.aux_rewrite_tokens = 0
        self.aux_source_plus_rewrite_tokens = 0
        self.aux_batches: dict[int, dict[str, int]] = {}
        self.changed_row_positions: list[int] = []
        self.last_included_cum_words = 0
        self.boundary_crossing_rows = 0

    def maybe_add(self, row_idx: int, row_words: int, cum_words_after: int, span: dict[str, Any] | None) -> None:
        # The frozen exposure targets are pass boundaries. Refuse partial rows if a
        # future use gives a non-aligned exposure.
        if self.words >= self.exposure_words:
            return
        if cum_words_after > self.exposure_words:
            self.boundary_crossing_rows += 1
            return
        self.rows += 1
        self.words += row_words
        self.last_included_cum_words = cum_words_after
        if span is None:
            return
        self.aux_rows += 1
        self.changed_row_positions.append(row_idx)
        batch = (self.rows - 1) // self.batch_size
        rec = self.aux_batches.setdefault(batch, {"aux_rows": 0, "pair_records": 0, "both_visible_pairs": 0, "tokens": 0})
        pairs = span.get("pairs") or []
        pair_records = len(pairs)
        both_pairs = sum(1 for p in pairs if p.get("visibility") == "both_visible")
        row_counts = span.get("row_token_counts") or {}
        source_tokens = int(row_counts.get("source", 0))
        rewrite_tokens = int(row_counts.get("rewrite", 0))
        both_tokens = int(row_counts.get("source_plus_rewrite", source_tokens + rewrite_tokens))
        self.aux_pair_records += pair_records
        self.aux_both_visible_pairs += both_pairs
        self.aux_source_tokens += source_tokens
        self.aux_rewrite_tokens += rewrite_tokens
        self.aux_source_plus_rewrite_tokens += both_tokens
        rec["aux_rows"] += 1
        rec["pair_records"] += pair_records
        rec["both_visible_pairs"] += both_pairs
        rec["tokens"] += both_tokens

    def finalize(self) -> dict[str, Any]:
        batches = int(math.ceil(self.rows / self.batch_size)) if self.rows else 0
        aux_batch_rows = [float(x["aux_rows"]) for x in self.aux_batches.values()]
        aux_batch_pairs = [float(x["both_visible_pairs"]) for x in self.aux_batches.values()]
        bursts = burst_lengths({k: v["aux_rows"] for k, v in self.aux_batches.items()})
        return {
            "exposure_words": self.exposure_words,
            "included_rows": self.rows,
            "included_words": self.words,
            "last_included_cumulative_words": self.last_included_cum_words,
            "boundary_crossing_rows": self.boundary_crossing_rows,
            "batches": batches,
            "aux_rows": self.aux_rows,
            "aux_pair_records": self.aux_pair_records,
            "aux_both_visible_pairs": self.aux_both_visible_pairs,
            "aux_source_tokens": self.aux_source_tokens,
            "aux_rewrite_tokens": self.aux_rewrite_tokens,
            "aux_source_plus_rewrite_tokens": self.aux_source_plus_rewrite_tokens,
            "batches_with_aux": len(self.aux_batches),
            "fraction_batches_with_aux": (len(self.aux_batches) / batches) if batches else 0.0,
            "aux_rows_per_aux_batch": {
                **basic_stats(aux_batch_rows),
                "p05": percentile(aux_batch_rows, 0.05),
                "p95": percentile(aux_batch_rows, 0.95),
            },
            "both_visible_pairs_per_aux_batch": {
                **basic_stats(aux_batch_pairs),
                "p05": percentile(aux_batch_pairs, 0.05),
                "p95": percentile(aux_batch_pairs, 0.95),
            },
            "aux_batch_burst_lengths": basic_stats([float(x) for x in bursts]),
            "first_aux_batches": sorted(self.aux_batches)[:20],
            "last_aux_batches": sorted(self.aux_batches)[-20:],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--skip-sha", action="store_true", help="development only")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    span_by_ex = load_span_map(SPAN_JSONL)
    span_summary = json.loads(SPAN_SUMMARY.read_text(encoding="utf-8"))
    if len(span_by_ex) != int(span_summary["counts"]["rows_written"]):
        raise RuntimeError("span map row count disagrees with summary")

    train_sha = None if args.skip_sha else sha256_file(TRAIN_100M)
    if train_sha is not None and train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"100M train SHA mismatch: {train_sha}")

    first_pass_fps: list[tuple[int, int, str, str]] = []
    pass_words: list[int] = [0 for _ in range(EXPECTED_PASS_COUNT)]
    pass_rows: list[int] = [0 for _ in range(EXPECTED_PASS_COUNT)]
    pass_mismatches: list[dict[str, Any]] = []
    row_count = 0
    field_words = 0
    changed_source_rows = 0
    changed_joined_rows = 0
    changed_source_unmapped = 0
    mapped_wrong_source = 0
    mapped_occurrences = collections.Counter()
    pair_visibility_stream = collections.Counter()
    row_visibility_stream = collections.Counter()
    token_truncation_occurrences = 0
    exposure_accs = {e: ExposureAccumulator(e, BATCH_SIZE) for e in EXPOSURES}

    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            obj = json.loads(line)
            ex_id = int(obj.get("example_id", row_count - 1))
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            source = str(obj.get("source", ""))
            text = str(obj.get("text", ""))
            field_words += words
            pass_idx = (row_count - 1) // EXPECTED_PASS_ROWS
            idx_in_pass = (row_count - 1) % EXPECTED_PASS_ROWS
            if pass_idx >= EXPECTED_PASS_COUNT:
                pass_mismatches.append({"row": row_count, "kind": "extra_row_beyond_expected_passes"})
                continue
            pass_words[pass_idx] += words
            pass_rows[pass_idx] += 1
            fp = (ex_id, words, source, text_fingerprint(text))
            if pass_idx == 0:
                first_pass_fps.append(fp)
            else:
                ref = first_pass_fps[idx_in_pass]
                if fp != ref and len(pass_mismatches) < 20:
                    pass_mismatches.append({
                        "row": row_count,
                        "pass_index_1based": pass_idx + 1,
                        "idx_in_pass_1based": idx_in_pass + 1,
                        "seen": fp[:3],
                        "ref": ref[:3],
                    })

            span = span_by_ex.get(ex_id)
            if source == CHANGED_SOURCE:
                changed_source_rows += 1
                if span is None:
                    changed_source_unmapped += 1
            if span is not None:
                changed_joined_rows += 1
                mapped_occurrences[ex_id] += 1
                if source != CHANGED_SOURCE:
                    mapped_wrong_source += 1
                rv = span.get("row_pair_visibility") or {}
                if rv.get("both_visible", 0) == span.get("pair_count", len(span.get("pairs") or [])):
                    row_visibility_stream["all_pairs_both_visible"] += 1
                elif rv.get("both_visible", 0) > 0:
                    row_visibility_stream["some_pairs_both_visible"] += 1
                else:
                    row_visibility_stream["no_pairs_both_visible"] += 1
                for p in span.get("pairs") or []:
                    pair_visibility_stream[str(p.get("visibility", "unknown"))] += 1
                if bool(span.get("truncated_by_seq256")):
                    token_truncation_occurrences += 1
            cum_after = field_words
            for acc in exposure_accs.values():
                acc.maybe_add(row_count, words, cum_after, span)

    if row_count != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"row count mismatch {row_count} != {EXPECTED_TRAIN_ROWS}")
    if field_words != EXPECTED_TRAIN_WORDS:
        raise RuntimeError(f"word count mismatch {field_words} != {EXPECTED_TRAIN_WORDS}")

    occurrence_vals = list(mapped_occurrences.values())
    missing_map_examples = sorted(set(span_by_ex) - set(mapped_occurrences))
    repeated_all_ten = (
        len(missing_map_examples) == 0
        and occurrence_vals
        and min(occurrence_vals) == EXPECTED_PASS_COUNT
        and max(occurrence_vals) == EXPECTED_PASS_COUNT
    )

    summary = {
        "status": "PAIR_SPAN_STREAM_VALIDATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Validate research source/rewrite span map against the exact frozen 100M training stream and quantify auxiliary-signal placement under inherited batch order.",
        "inputs": {
            "train_100m": str(TRAIN_100M),
            "train_sha256": train_sha,
            "span_jsonl": str(SPAN_JSONL),
            "span_summary": str(SPAN_SUMMARY),
            "batch_size": BATCH_SIZE,
            "exposures": EXPOSURES,
        },
        "stream_counts": {
            "rows": row_count,
            "field_words": field_words,
            "pass_rows": pass_rows,
            "pass_words": pass_words,
            "passes_match_first_pass": len(pass_mismatches) == 0,
            "pass_mismatch_samples": pass_mismatches,
        },
        "join_counts": {
            "span_map_examples": len(span_by_ex),
            "changed_source_rows_in_100m": changed_source_rows,
            "span_joined_rows_in_100m": changed_joined_rows,
            "changed_source_unmapped_rows": changed_source_unmapped,
            "mapped_wrong_source_rows": mapped_wrong_source,
            "missing_map_examples_in_100m": len(missing_map_examples),
            "missing_map_example_samples": missing_map_examples[:20],
            "mapped_occurrences_per_example": {
                "n": len(occurrence_vals),
                "min": min(occurrence_vals) if occurrence_vals else 0,
                "max": max(occurrence_vals) if occurrence_vals else 0,
                "mean": float(statistics.mean(occurrence_vals)) if occurrence_vals else 0.0,
            },
            "all_mapped_examples_repeat_exactly_ten_times": repeated_all_ten,
            "pair_visibility_stream": dict(pair_visibility_stream),
            "row_visibility_stream": dict(row_visibility_stream),
            "token_truncation_occurrences": token_truncation_occurrences,
        },
        "exposure_summaries": {str(e): acc.finalize() for e, acc in exposure_accs.items()},
        "scientific_interpretation": [
            "If source-view consistency is later selected, the research span map can join by example_id across the repeated 100M stream only if changed_source_unmapped_rows and mapped_wrong_source_rows stay zero and every mapped example appears ten times.",
            "The auxiliary signal is sparse per batch but, as quantified by pair_span_pass_distribution, changed rows are shuffled across nearly all batches in each 10M pass rather than kept as one contiguous block; a future source-view consistency trainer should join by example_id and normalize the auxiliary loss inside rows/pairs that actually have both-visible source and rewrite spans.",
            "This validation is not score evidence and does not favor source-view consistency over static-prior masking; mature 70M/80M clean-vs-reinvest results still choose which intervention family, if any, is scientifically justified.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_span_stream_validation.json"
    out_md = out_dir / "pair_span_stream_validation.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair-span stream validation",
        "",
        summary["purpose"],
        "",
        "This is CPU-only stream validation. It does not train, evaluate, alter the corpus, or choose an intervention route.",
        "",
        "## 100M stream",
        f"- rows: `{row_count}`",
        f"- field_words: `{field_words}`",
        f"- train_sha256: `{train_sha}`",
        f"- pass_rows: `{pass_rows}`",
        f"- pass_words: `{pass_words}`",
        f"- passes_match_first_pass: `{summary['stream_counts']['passes_match_first_pass']}`",
        "",
        "## Span-map join",
    ]
    for k, v in summary["join_counts"].items():
        if k.endswith("samples"):
            continue
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Exposure-level auxiliary placement"]
    for e in EXPOSURES:
        rec = summary["exposure_summaries"][str(e)]
        lines.append(
            f"- {e//1_000_000}M: rows={rec['included_rows']}, batches={rec['batches']}, "
            f"aux_rows={rec['aux_rows']}, aux_both_visible_pairs={rec['aux_both_visible_pairs']}, "
            f"batches_with_aux={rec['batches_with_aux']} ({rec['fraction_batches_with_aux']:.4f}); "
            f"aux_rows_per_aux_batch mean={rec['aux_rows_per_aux_batch']['mean']:.2f}, "
            f"min={rec['aux_rows_per_aux_batch']['min']}, max={rec['aux_rows_per_aux_batch']['max']}"
        )
    lines += ["", "## Interpretation"]
    for item in summary["scientific_interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "rows": row_count,
        "words": field_words,
        "span_joined_rows": changed_joined_rows,
        "all_mapped_examples_repeat_exactly_ten_times": repeated_all_ten,
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
