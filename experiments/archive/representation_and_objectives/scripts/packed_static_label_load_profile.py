#!/usr/bin/env python3
"""research: static label-load profile for the packed target-selective intervention.

The two research 100M arms delete matched total realized WWM label mass, but the
training loss is a mean over the labels left in each batch.  This script asks
whether the *static* candidate words for source-absent deletion and matched copied
whole-word deletion have materially different timing/batch load before the WWM
random draw.  It uses the same token-to-word grouping logic as the packed trainer,
then summarizes expected per-batch deleted/kept label load under p=0.15.

It does not inspect the running 100M jobs and does not infer their endpoints.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import packed_target_selective_trainer as base  # noqa: E402

STREAM = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
ANNOTATION = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_pool_annotations.jsonl")
SELECTION = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_wholeword_copied_selection.jsonl")
TOKENIZER = pathlib.Path("experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_static_label_load_profile")

CAT_RW_COPIED = base.CAT_TO_IDX[base.CAT_RW_COPIED]
CAT_RW_ABS_CONTENT = base.CAT_TO_IDX[base.CAT_RW_ABS_CONTENT]


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qstats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "mean": round(statistics.mean(s), 8),
        "median": round(statistics.median(s), 8),
        "p01": round(q(0.01), 8),
        "p05": round(q(0.05), 8),
        "p10": round(q(0.10), 8),
        "p25": round(q(0.25), 8),
        "p75": round(q(0.75), 8),
        "p90": round(q(0.90), 8),
        "p95": round(q(0.95), 8),
        "p99": round(q(0.99), 8),
        "min": round(s[0], 8),
        "max": round(s[-1], 8),
    }


def load_annotations(path: pathlib.Path) -> tuple[dict[int, list[int]], dict[int, int]]:
    annotations: dict[int, list[int]] = {}
    row_to_eid: dict[int, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            eid = int(a["example_id"])
            annotations[eid] = [base.CAT_TO_IDX.get(c, 0) for c in a["word_categories"]]
            row_to_eid[int(a["row_index"])] = eid
    return annotations, row_to_eid


def load_selection(path: pathlib.Path, row_to_eid: dict[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            eid = row_to_eid.get(int(g["row_index"]))
            if eid is not None:
                out.add((eid, int(g["word_index"])))
    return out


class StaticTokenizerProfiler:
    def __init__(self, tok, annotations: dict[int, list[int]], selected: set[tuple[int, int]], seq_len: int):
        self.tok = tok
        self.annotations = annotations
        self.selected = selected
        self.seq_len = seq_len
        self.special_ids = set(int(x) for x in tok.all_special_ids)
        self._word_start: dict[int, bool] = {}
        self.cache: dict[int, dict[str, int]] = {}

    def is_word_start(self, tid: int) -> bool:
        if tid not in self._word_start:
            s = self.tok.convert_ids_to_tokens(int(tid))
            self._word_start[tid] = bool(s is not None and base.is_word_start(str(s)))
        return self._word_start[tid]

    def profile_example(self, eid: int, text: str) -> dict[str, int]:
        if eid in self.cache:
            return self.cache[eid]
        enc = self.tok(text, add_special_tokens=False, truncation=True, max_length=self.seq_len, padding="max_length")
        ids = list(enc["input_ids"])
        att = list(enc["attention_mask"])
        word_cats = self.annotations.get(eid, [])
        gid = -1
        seen_groups: set[int] = set()
        abs_groups: set[int] = set()
        sel_groups: set[int] = set()
        copied_groups: set[int] = set()
        counts = collections.Counter()
        for i, (tid, am) in enumerate(zip(ids, att)):
            if int(am) == 0:
                continue
            tid = int(tid)
            if tid in self.special_ids:
                continue
            if gid < 0 or self.is_word_start(tid) or i == 0:
                gid += 1
            seen_groups.add(gid)
            cat = word_cats[gid] if 0 <= gid < len(word_cats) else base.CAT_TO_IDX[base.CAT_FILLER]
            counts["total_candidate_pieces"] += 1
            if cat == CAT_RW_ABS_CONTENT:
                counts["abs_content_pieces"] += 1
                abs_groups.add(gid)
            if cat == CAT_RW_COPIED:
                counts["rw_copied_pieces"] += 1
                copied_groups.add(gid)
            if (eid, gid) in self.selected:
                counts["selected_copied_pieces"] += 1
                sel_groups.add(gid)
        counts["total_groups"] = len(seen_groups)
        counts["abs_content_groups"] = len(abs_groups)
        counts["rw_copied_groups"] = len(copied_groups)
        counts["selected_copied_groups"] = len(sel_groups)
        rec = {k: int(v) for k, v in counts.items()}
        self.cache[eid] = rec
        return rec


def add_to_counter(c: collections.Counter, rec: dict[str, int], prefix: str = "") -> None:
    for k, v in rec.items():
        c[prefix + k] += int(v)


def summarize_batches(batches: list[dict[str, Any]], mask_prob: float) -> dict[str, Any]:
    diffs = [float(b["selected_minus_abs_candidate_pieces"]) for b in batches]
    abs_load = [float(b["abs_content_pieces"]) for b in batches]
    sel_load = [float(b["selected_copied_pieces"]) for b in batches]
    total = [float(b["total_candidate_pieces"]) for b in batches]
    expected_denom_abs = [mask_prob * max(0.0, b["total_candidate_pieces"] - b["abs_content_pieces"]) for b in batches]
    expected_denom_sel = [mask_prob * max(0.0, b["total_candidate_pieces"] - b["selected_copied_pieces"]) for b in batches]
    # Relative per-kept-token loss weight in drop_abs versus drop_copied if the
    # only difference were mean-loss denominators: (1/denom_abs)/(1/denom_sel).
    weight_ratio_abs_over_copied = [
        (dc / da) if da > 0 and dc > 0 else 1.0
        for da, dc in zip(expected_denom_abs, expected_denom_sel)
    ]
    frac_abs = [(a / t) if t else 0.0 for a, t in zip(abs_load, total)]
    frac_sel = [(s / t) if t else 0.0 for s, t in zip(sel_load, total)]
    abs_gt_sel = sum(1 for a, s in zip(abs_load, sel_load) if a > s)
    sel_gt_abs = sum(1 for a, s in zip(abs_load, sel_load) if s > a)
    equal = len(batches) - abs_gt_sel - sel_gt_abs
    mean_abs = statistics.mean(abs_load) if abs_load else 0.0
    mean_sel = statistics.mean(sel_load) if sel_load else 0.0
    cov = 0.0
    corr = None
    if len(batches) >= 2:
        ma, ms = mean_abs, mean_sel
        cov = sum((a - ma) * (s - ms) for a, s in zip(abs_load, sel_load)) / (len(batches) - 1)
        va = sum((a - ma) ** 2 for a in abs_load) / (len(batches) - 1)
        vs = sum((s - ms) ** 2 for s in sel_load) / (len(batches) - 1)
        if va > 0 and vs > 0:
            corr = cov / math.sqrt(va * vs)
    top_abs = sorted(batches, key=lambda b: b["abs_content_pieces"], reverse=True)[:8]
    top_sel = sorted(batches, key=lambda b: b["selected_copied_pieces"], reverse=True)[:8]
    top_diff = sorted(batches, key=lambda b: abs(b["selected_minus_abs_candidate_pieces"]), reverse=True)[:12]
    keep_keys = ["batch_index", "row_start", "row_end", "epoch_guess", "total_candidate_pieces", "abs_content_pieces", "selected_copied_pieces", "selected_minus_abs_candidate_pieces", "denom_weight_ratio_abs_over_copied"]
    def slim(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: r.get(k) for k in keep_keys} for r in rows]
    return {
        "n_batches": len(batches),
        "candidate_piece_totals": {
            "total": int(sum(total)),
            "abs_content": int(sum(abs_load)),
            "selected_copied": int(sum(sel_load)),
            "selected_minus_abs": int(sum(diffs)),
        },
        "batch_abs_content_pieces": qstats(abs_load),
        "batch_selected_copied_pieces": qstats(sel_load),
        "batch_selected_minus_abs_candidate_pieces": qstats(diffs),
        "batch_abs_fraction_of_all_candidate_pieces": qstats(frac_abs),
        "batch_selected_fraction_of_all_candidate_pieces": qstats(frac_sel),
        "expected_kept_label_denominator_abs_deleted": qstats(expected_denom_abs),
        "expected_kept_label_denominator_copied_deleted": qstats(expected_denom_sel),
        "denom_weight_ratio_abs_over_copied": qstats(weight_ratio_abs_over_copied),
        "batch_relation_counts": {
            "abs_load_greater_than_selected": abs_gt_sel,
            "selected_load_greater_than_abs": sel_gt_abs,
            "equal_load": equal,
        },
        "batch_abs_selected_pearson": None if corr is None else round(corr, 8),
        "top_abs_batches": slim(top_abs),
        "top_selected_batches": slim(top_sel),
        "top_absolute_difference_batches": slim(top_diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--annotation", default=str(ANNOTATION))
    ap.add_argument("--selection", default=str(SELECTION))
    ap.add_argument("--tokenizer_path", default=str(TOKENIZER))
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--max_word_exposure", type=int, default=100_000_000)
    ap.add_argument("--progress_every", type=int, default=100_000)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations, row_to_eid = load_annotations(pathlib.Path(args.annotation))
    selected = load_selection(pathlib.Path(args.selection), row_to_eid)
    tok = base.make_portable_tokenizer(args.tokenizer_path)
    profiler = StaticTokenizerProfiler(tok, annotations, selected, args.max_seq_length)

    total_words = 0
    stream_rows = 0
    batch_rows: list[dict[str, int]] = []
    batches: list[dict[str, Any]] = []
    static_totals = collections.Counter()
    by_epoch: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    occurrence = collections.Counter()
    example_first_static: dict[int, dict[str, int]] = {}

    def flush_batch(row_start: int, row_end: int) -> None:
        if not batch_rows:
            return
        c = collections.Counter()
        for r in batch_rows:
            add_to_counter(c, r)
        bi = len(batches)
        # Since the 10M pool is replayed ten times in the preserved stream order,
        # the median occurrence of rows in the batch is the most stable epoch tag.
        epoch_vals = [r.get("epoch_seen", 0) for r in batch_rows]
        epoch_guess = int(statistics.median(epoch_vals)) if epoch_vals else 0
        total = int(c["total_candidate_pieces"])
        abs_p = int(c["abs_content_pieces"])
        sel_p = int(c["selected_copied_pieces"])
        denom_abs = args.mask_prob * max(0, total - abs_p)
        denom_sel = args.mask_prob * max(0, total - sel_p)
        batches.append({
            "batch_index": bi,
            "row_start": row_start,
            "row_end": row_end,
            "epoch_guess": epoch_guess,
            "total_candidate_pieces": total,
            "abs_content_pieces": abs_p,
            "selected_copied_pieces": sel_p,
            "selected_minus_abs_candidate_pieces": sel_p - abs_p,
            "abs_fraction_of_all_candidate_pieces": (abs_p / total) if total else 0.0,
            "selected_fraction_of_all_candidate_pieces": (sel_p / total) if total else 0.0,
            "expected_kept_labels_drop_abs": denom_abs,
            "expected_kept_labels_drop_copied": denom_sel,
            "denom_weight_ratio_abs_over_copied": (denom_sel / denom_abs) if denom_abs > 0 and denom_sel > 0 else 1.0,
        })

    row_start_for_batch = 0
    with pathlib.Path(args.stream).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if total_words + words > args.max_word_exposure:
                break
            eid = int(obj.get("example_id", stream_rows))
            epoch_seen = int(occurrence[eid])
            occurrence[eid] += 1
            rec = dict(profiler.profile_example(eid, text))
            rec["epoch_seen"] = epoch_seen
            add_to_counter(static_totals, rec)
            add_to_counter(by_epoch[epoch_seen], rec)
            example_first_static.setdefault(eid, rec)
            if not batch_rows:
                row_start_for_batch = stream_rows
            batch_rows.append(rec)
            stream_rows += 1
            total_words += words
            if len(batch_rows) == args.batch_size:
                flush_batch(row_start_for_batch, stream_rows)
                batch_rows = []
            if args.progress_every and stream_rows % args.progress_every == 0:
                print(json.dumps({"event": "progress", "stream_rows": stream_rows, "total_words": total_words, "unique_examples": len(profiler.cache), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    if batch_rows:
        flush_batch(row_start_for_batch, stream_rows)

    epoch_payload = {}
    for ep, c in sorted(by_epoch.items()):
        total = int(c["total_candidate_pieces"])
        abs_p = int(c["abs_content_pieces"])
        sel_p = int(c["selected_copied_pieces"])
        epoch_payload[str(ep)] = {
            "rows": int(c.get("rows", 0)),
            "total_candidate_pieces": total,
            "abs_content_pieces": abs_p,
            "selected_copied_pieces": sel_p,
            "selected_minus_abs_candidate_pieces": sel_p - abs_p,
            "abs_fraction": round(abs_p / total, 8) if total else None,
            "selected_fraction": round(sel_p / total, 8) if total else None,
        }

    # Add row counts by occurrence separately because rec has no row marker in the counters.
    epoch_row_counts = collections.Counter(str(v) for v in occurrence.values())
    occ_dist = collections.Counter(str(v) for v in occurrence.values())
    # Detailed per-epoch row counts from stream pass.
    by_epoch_rows = collections.Counter()
    for b in batches:
        by_epoch_rows[str(b["epoch_guess"])] += int(b["row_end"] - b["row_start"])
    for ep, n in by_epoch_rows.items():
        epoch_payload.setdefault(ep, {})["batch_row_count_median_epoch"] = int(n)

    totals = {k: int(v) for k, v in static_totals.items() if k != "epoch_seen"}
    profile = {
        "status": "PACKED_STATIC_LABEL_LOAD_PROFILE",
        "meaning": "Static packed-stream profile of the two deletion populations before WWM sampling. It tests whether matched total label deletion could still differ in training-time loss-denominator timing or batch concentration.",
        "inputs": {
            "stream": str(args.stream),
            "stream_sha256": sha256_file(pathlib.Path(args.stream)),
            "annotation": str(args.annotation),
            "annotation_sha256": sha256_file(pathlib.Path(args.annotation)),
            "selection": str(args.selection),
            "selection_sha256": sha256_file(pathlib.Path(args.selection)),
            "tokenizer_path": str(args.tokenizer_path),
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "mask_prob": args.mask_prob,
            "max_word_exposure": args.max_word_exposure,
        },
        "stream_rows_used": stream_rows,
        "word_exposure_used": total_words,
        "unique_examples_profiled": len(profiler.cache),
        "occurrence_count_distribution_by_example": dict(sorted(occ_dist.items(), key=lambda kv: int(kv[0]))),
        "static_totals_over_stream": totals,
        "expected_realized_bpe_at_mask_prob": {
            "abs_content": round(args.mask_prob * totals.get("abs_content_pieces", 0), 3),
            "selected_copied": round(args.mask_prob * totals.get("selected_copied_pieces", 0), 3),
            "selected_minus_abs": round(args.mask_prob * (totals.get("selected_copied_pieces", 0) - totals.get("abs_content_pieces", 0)), 3),
        },
        "by_epoch_static": epoch_payload,
        "batch_profile": summarize_batches(batches, args.mask_prob),
        "interpretation_fields": {
            "total_candidate_match": "selected_copied_pieces - abs_content_pieces should be near zero over the 100M stream if the pool-level whole-word control matches the deleted BPE mass before WWM.",
            "denominator_ratio": "denom_weight_ratio_abs_over_copied near 1.0 means mean-loss normalization changes are too small to explain a large endpoint contrast by themselves.",
            "batch_timing": "Large or structured batch load differences would require caution because the two arms would upweight different kept labels at different steps even with equal total deleted mass.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / "packed_static_label_load_profile.json"
    out_json.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research packed static label-load profile",
        "",
        f"JSON: `{out_json}`",
        "",
        "This profile was run while the research 100M arms were still managed asynchronously; it does not read their outputs.",
        "",
        "## Key static totals",
        "",
        f"- stream rows used: {stream_rows}",
        f"- word exposure used: {total_words}",
        f"- total candidate BPE pieces: {totals.get('total_candidate_pieces', 0)}",
        f"- source-absent-content candidate BPE pieces: {totals.get('abs_content_pieces', 0)}",
        f"- selected copied-content candidate BPE pieces: {totals.get('selected_copied_pieces', 0)}",
        f"- selected minus absent candidate pieces: {totals.get('selected_copied_pieces', 0) - totals.get('abs_content_pieces', 0)}",
        "",
        "## Batch-level denominator signal",
        "",
        "```json",
        json.dumps(profile["batch_profile"].get("denom_weight_ratio_abs_over_copied"), indent=2, ensure_ascii=False),
        "```",
        "",
        "A ratio near 1 says the per-batch mean-loss denominator difference is tiny. Endpoint interpretation still depends on the completed arm comparison and local denoising readout.",
        "",
    ]
    (out_dir / "packed_static_label_load_profile.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": profile["status"], "out_json": str(out_json), "stream_rows_used": stream_rows, "word_exposure_used": total_words, "static_totals_over_stream": profile["static_totals_over_stream"], "expected_realized_bpe_at_mask_prob": profile["expected_realized_bpe_at_mask_prob"], "denom_weight_ratio_abs_over_copied": profile["batch_profile"]["denom_weight_ratio_abs_over_copied"], "elapsed_sec": profile["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
