#!/usr/bin/env python3
"""research v2: LR-weighted static label-load profile for packed target selection.

This extends the research static profile by saving every optimizer-batch load and
weighting the two deletion populations by the historical warmup+cosine learning
rate schedule.  It tests a subtle confound before the 100M endpoints are read:
matched total deleted labels could still be concentrated at different learning
rates or create different per-batch mean-loss denominators.

No running training outputs are read.
"""
from __future__ import annotations

import argparse
import collections
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

import packed_static_label_load_profile as prof  # noqa: E402

DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2")


def lr_lambda(step_after: int, total_steps: int, warmup_steps: int) -> float:
    # Mirrors transformers.get_cosine_schedule_with_warmup lambda.  step_after is
    # the scheduler step count after an update.  `lr_used_approx` below uses
    # step_after-1 to approximate the LR applied to that update; `lr_after` uses
    # the value logged by the trainer after sched.step().
    if step_after < warmup_steps:
        return float(step_after) / float(max(1, warmup_steps))
    progress = float(step_after - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def qstats(xs: list[float]) -> dict[str, Any]:
    return prof.qstats(xs)


def weighted_mean(vals: list[float], weights: list[float]) -> float | None:
    den = sum(float(w) for w in weights)
    if den == 0:
        return None
    return sum(float(v) * float(w) for v, w in zip(vals, weights)) / den


def weighted_sum(vals: list[float], weights: list[float]) -> float:
    return sum(float(v) * float(w) for v, w in zip(vals, weights))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_batches": 0}
    abs_p = [float(r["abs_content_pieces"]) for r in rows]
    sel_p = [float(r["selected_copied_pieces"]) for r in rows]
    total_p = [float(r["total_candidate_pieces"]) for r in rows]
    diff = [s - a for a, s in zip(abs_p, sel_p)]
    lr_after = [float(r["lr_after_sched_step"]) for r in rows]
    lr_used = [float(r["lr_used_approx"]) for r in rows]
    denom_ratio = [float(r["denom_weight_ratio_abs_over_copied"]) for r in rows]
    return {
        "n_batches": len(rows),
        "step_range": [int(rows[0]["step"]), int(rows[-1]["step"])],
        "total_candidate_pieces": int(sum(total_p)),
        "abs_content_pieces": int(sum(abs_p)),
        "selected_copied_pieces": int(sum(sel_p)),
        "selected_minus_abs_candidate_pieces": int(sum(diff)),
        "expected_realized_selected_minus_abs_bpe": round(0.15 * sum(diff), 6),
        "mean_selected_minus_abs_per_batch": round(statistics.mean(diff), 8),
        "lr_after_weighted_selected_minus_abs_per_batch": None if weighted_mean(diff, lr_after) is None else round(weighted_mean(diff, lr_after), 8),
        "lr_used_weighted_selected_minus_abs_per_batch": None if weighted_mean(diff, lr_used) is None else round(weighted_mean(diff, lr_used), 8),
        "lr_after_weighted_denom_ratio_abs_over_copied": None if weighted_mean(denom_ratio, lr_after) is None else round(weighted_mean(denom_ratio, lr_after), 10),
        "lr_used_weighted_denom_ratio_abs_over_copied": None if weighted_mean(denom_ratio, lr_used) is None else round(weighted_mean(denom_ratio, lr_used), 10),
        "selected_minus_abs_distribution": qstats(diff),
        "denom_weight_ratio_distribution": qstats(denom_ratio),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(prof.STREAM))
    ap.add_argument("--annotation", default=str(prof.ANNOTATION))
    ap.add_argument("--selection", default=str(prof.SELECTION))
    ap.add_argument("--tokenizer_path", default=str(prof.TOKENIZER))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--max_word_exposure", type=int, default=100_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_total_steps", type=int, default=2529)
    ap.add_argument("--warmup_fraction", type=float, default=0.06)
    ap.add_argument("--progress_every", type=int, default=200000)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations, row_to_eid = prof.load_annotations(pathlib.Path(args.annotation))
    selected = prof.load_selection(pathlib.Path(args.selection), row_to_eid)
    tok = prof.base.make_portable_tokenizer(args.tokenizer_path)
    profiler = prof.StaticTokenizerProfiler(tok, annotations, selected, args.max_seq_length)

    warmup_steps = max(1, int(args.lr_total_steps * args.warmup_fraction))
    batches: list[dict[str, Any]] = []
    batch_recs: list[dict[str, int]] = []
    occurrence: collections.Counter[int] = collections.Counter()
    total_words = 0
    stream_rows = 0
    row_start = 0

    def flush(row_end: int) -> None:
        nonlocal batch_recs, row_start
        if not batch_recs:
            return
        c = collections.Counter()
        epoch_vals = []
        for r in batch_recs:
            for k, v in r.items():
                if k != "epoch_seen":
                    c[k] += int(v)
            epoch_vals.append(int(r.get("epoch_seen", 0)))
        step = len(batches) + 1
        lr_after = args.lr * lr_lambda(step, args.lr_total_steps, warmup_steps)
        lr_used = args.lr * lr_lambda(max(0, step - 1), args.lr_total_steps, warmup_steps)
        total = int(c["total_candidate_pieces"])
        abs_p = int(c["abs_content_pieces"])
        sel_p = int(c["selected_copied_pieces"])
        denom_abs = args.mask_prob * max(0, total - abs_p)
        denom_sel = args.mask_prob * max(0, total - sel_p)
        rec = {
            "step": step,
            "row_start": row_start,
            "row_end": row_end,
            "epoch_median": int(statistics.median(epoch_vals)) if epoch_vals else 0,
            "total_candidate_pieces": total,
            "abs_content_pieces": abs_p,
            "selected_copied_pieces": sel_p,
            "selected_minus_abs_candidate_pieces": sel_p - abs_p,
            "expected_abs_deleted_bpe": args.mask_prob * abs_p,
            "expected_selected_deleted_bpe": args.mask_prob * sel_p,
            "expected_selected_minus_abs_deleted_bpe": args.mask_prob * (sel_p - abs_p),
            "expected_kept_labels_drop_abs": denom_abs,
            "expected_kept_labels_drop_copied": denom_sel,
            "denom_weight_ratio_abs_over_copied": (denom_sel / denom_abs) if denom_abs > 0 and denom_sel > 0 else 1.0,
            "lr_after_sched_step": lr_after,
            "lr_used_approx": lr_used,
            "lr_after_x_expected_selected_minus_abs_deleted_bpe": lr_after * args.mask_prob * (sel_p - abs_p),
            "lr_used_x_expected_selected_minus_abs_deleted_bpe": lr_used * args.mask_prob * (sel_p - abs_p),
        }
        batches.append(rec)
        batch_recs = []

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
            ep = int(occurrence[eid])
            occurrence[eid] += 1
            rec = dict(profiler.profile_example(eid, text))
            rec["epoch_seen"] = ep
            if not batch_recs:
                row_start = stream_rows
            batch_recs.append(rec)
            stream_rows += 1
            total_words += words
            if len(batch_recs) == args.batch_size:
                flush(stream_rows)
            if args.progress_every and stream_rows % args.progress_every == 0:
                print(json.dumps({"event": "progress", "stream_rows": stream_rows, "words": total_words, "cached_examples": len(profiler.cache), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    if batch_recs:
        flush(stream_rows)

    groups = {
        "all": batches,
        "warmup_steps": [b for b in batches if int(b["step"]) <= warmup_steps],
        "post_warmup_top_lr_half": [b for b in batches if int(b["step"]) > warmup_steps and float(b["lr_used_approx"]) >= 0.0005],
        "post_warmup_low_lr_half": [b for b in batches if int(b["step"]) > warmup_steps and float(b["lr_used_approx"]) < 0.0005],
    }
    # Ten equal step bands make timing structure visible without saving huge text in the summary.
    band_size = math.ceil(len(batches) / 10)
    for band in range(10):
        lo = band * band_size
        hi = min(len(batches), (band + 1) * band_size)
        groups[f"step_decile_{band:02d}_{lo+1:04d}_{hi:04d}"] = batches[lo:hi]

    summary_groups = {k: summarize_group(v) for k, v in groups.items()}
    all_rows = groups["all"]
    lr_after = [float(r["lr_after_sched_step"]) for r in all_rows]
    lr_used = [float(r["lr_used_approx"]) for r in all_rows]
    diff_deleted = [float(r["expected_selected_minus_abs_deleted_bpe"]) for r in all_rows]
    denom_ratio = [float(r["denom_weight_ratio_abs_over_copied"]) for r in all_rows]
    max_abs_ratio_dev = max(abs(x - 1.0) for x in denom_ratio) if denom_ratio else 0.0

    # Save batch table for later exact inspection.
    batch_jsonl = out_dir / "packed_static_label_load_batches.jsonl"
    with batch_jsonl.open("w", encoding="utf-8") as f:
        for r in batches:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    payload = {
        "status": "PACKED_STATIC_LABEL_LOAD_PROFILE_V2",
        "meaning": "LR-weighted static profile of source-absent-content versus matched copied-content whole-word deletion populations in the packed historical stream. It does not inspect the running endpoints.",
        "inputs": {
            "stream": args.stream,
            "stream_sha256": prof.sha256_file(pathlib.Path(args.stream)),
            "annotation": args.annotation,
            "annotation_sha256": prof.sha256_file(pathlib.Path(args.annotation)),
            "selection": args.selection,
            "selection_sha256": prof.sha256_file(pathlib.Path(args.selection)),
            "tokenizer_path": args.tokenizer_path,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "mask_prob": args.mask_prob,
            "lr": args.lr,
            "lr_total_steps": args.lr_total_steps,
            "warmup_steps": warmup_steps,
        },
        "stream_rows_used": stream_rows,
        "word_exposure_used": total_words,
        "unique_examples_profiled": len(profiler.cache),
        "batch_jsonl": str(batch_jsonl),
        "groups": summary_groups,
        "global_lr_weighted_fields": {
            "unweighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": round(statistics.mean(diff_deleted), 10),
            "lr_after_weighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": None if weighted_mean(diff_deleted, lr_after) is None else round(weighted_mean(diff_deleted, lr_after), 10),
            "lr_used_weighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch": None if weighted_mean(diff_deleted, lr_used) is None else round(weighted_mean(diff_deleted, lr_used), 10),
            "sum_lr_after_x_expected_selected_minus_abs_deleted_bpe": round(weighted_sum(diff_deleted, lr_after), 12),
            "sum_lr_used_x_expected_selected_minus_abs_deleted_bpe": round(weighted_sum(diff_deleted, lr_used), 12),
            "lr_after_weighted_denom_ratio_abs_over_copied": None if weighted_mean(denom_ratio, lr_after) is None else round(weighted_mean(denom_ratio, lr_after), 12),
            "lr_used_weighted_denom_ratio_abs_over_copied": None if weighted_mean(denom_ratio, lr_used) is None else round(weighted_mean(denom_ratio, lr_used), 12),
            "max_abs_batch_denom_ratio_deviation_from_1": round(max_abs_ratio_dev, 12),
        },
        "scientific_reading": {
            "load_match": "Total static selected-copied minus source-absent candidate pieces and LR-weighted expected realized differences should be tiny relative to the ~8.5k kept labels per batch if loss-normalization timing is not a serious alternate explanation.",
            "endpoint_dependency": "This only protects the control interface; the mechanism still depends on the completed endpoint and local denoising comparison.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / "packed_static_label_load_profile_v2.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research v2 packed static label-load profile",
        "",
        f"JSON: `{out_json}`",
        f"Batch table: `{batch_jsonl}`",
        "",
        "This was computed without reading the running 100M arms.",
        "",
        "## Main fields",
        "",
        "```json",
        json.dumps(payload["global_lr_weighted_fields"], indent=2, ensure_ascii=False),
        "```",
        "",
        "The primary arm comparison still waits for the trained endpoints; this file only narrows a loss-normalization/timing confound.",
        "",
    ]
    (out_dir / "packed_static_label_load_profile_v2.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "batch_jsonl": str(batch_jsonl), "all": summary_groups["all"], "global_lr_weighted_fields": payload["global_lr_weighted_fields"], "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
