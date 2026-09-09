#!/usr/bin/env python3
"""research: packed-token geometry audit for the research causal topology 2x2 scaffold.

The GPT trainer concatenates tokenized JSONL rows plus EOS, truncates only the final
stream tail to a multiple of seq_len, and reshuffles documents each epoch with
random.Random(seed + epoch). This audit measures whether the four scaffold arms
really present the intended pair-unit contexts under that packing geometry.

It is CPU/file-only and produces no training result.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any

PAIR_PREFIX = "topology2x2_"
ARMS_DEFAULT = ["compact_oneway", "repeat_oneway", "compact_reciprocal", "repeat_reciprocal"]


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fstats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "n": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
    }


def load_rows(path: pathlib.Path, tokenizer, eos_id: int) -> tuple[list[dict[str, Any]], list[int]]:
    rows = []
    lengths = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            ids = tokenizer.encode(r["text"])
            n = len(ids) + 1  # trainer appends eos_id after every doc
            rr = {
                "idx": i,
                "source": r.get("source"),
                "words": int(r.get("words", 0)),
                "token_len": n,
                "text_token_len_without_eos": len(ids),
                "is_pair": isinstance(r.get("source"), str) and r["source"].startswith(PAIR_PREFIX),
                "pair_id": r.get("pair_id"),
                "topology_arm": r.get("topology_arm"),
                "semantic_arm": r.get("semantic_arm"),
                "base_order": r.get("base_order"),
                "realized_order": r.get("realized_order"),
                "occurrence_index": r.get("occurrence_index"),
                "selected_order_index": r.get("selected_order_index"),
                "view_kind": r.get("view_kind"),
            }
            rows.append(rr)
            lengths.append(n)
    return rows, lengths


def spans_for_order(rows: list[dict[str, Any]], lengths: list[int], order: list[int]) -> dict[int, tuple[int, int]]:
    spans = {}
    pos = 0
    for idx in order:
        n = lengths[idx]
        spans[idx] = (pos, pos + n)  # [start, end), includes appended EOS
        pos += n
    return spans


def span_chunk_metrics(start: int, end: int, seq_len: int, active_tokens: int) -> dict[str, Any]:
    # Tokens beyond active_tokens are the final dropped tail and never trained.
    active_end = min(end, active_tokens)
    if start >= active_tokens or active_end <= start:
        return {
            "active_len": 0,
            "first_chunk": None,
            "last_chunk": None,
            "n_chunks_touched": 0,
            "wholly_within_one_chunk": False,
            "crosses_chunk_boundary": False,
            "start_offset": start % seq_len,
            "end_offset_active": None,
            "remaining_context_after_row_start_in_first_chunk": seq_len - (start % seq_len),
        }
    first = start // seq_len
    last = (active_end - 1) // seq_len
    n_chunks = last - first + 1
    return {
        "active_len": active_end - start,
        "first_chunk": first,
        "last_chunk": last,
        "n_chunks_touched": n_chunks,
        "wholly_within_one_chunk": n_chunks == 1,
        "crosses_chunk_boundary": n_chunks > 1,
        "start_offset": start % seq_len,
        "end_offset_active": active_end % seq_len,
        "remaining_context_after_row_start_in_first_chunk": seq_len - (start % seq_len),
    }


def audit_epoch(rows: list[dict[str, Any]], lengths: list[int], seq_len: int, seed: int, epoch: int) -> dict[str, Any]:
    order = list(range(len(rows)))
    if epoch != 0:
        rng = random.Random(seed + epoch)
        rng.shuffle(order)
    spans = spans_for_order(rows, lengths, order)
    raw_tokens = sum(lengths[i] for i in order)
    active_tokens = (raw_tokens // seq_len) * seq_len
    chunks = active_tokens // seq_len
    order_rank = {idx: rank for rank, idx in enumerate(order)}

    pair_rows = [r for r in rows if r["is_pair"]]
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_metrics = []
    for r in pair_rows:
        s, e = spans[r["idx"]]
        m = span_chunk_metrics(s, e, seq_len, active_tokens)
        by_pair[str(r.get("pair_id"))].append({**r, "span_start": s, "span_end": e, "order_rank": order_rank[r["idx"]], **m})
        row_metrics.append(m)

    # Within each semantic pair, the two intended pair-unit rows can be adjacent in document order
    # before shuffling; after shuffling they are usually separated, which matters because the trainer
    # shuffles entire rows/documents per epoch. Pair-level metrics quantify this directly.
    pair_records = []
    for pair_id, recs in by_pair.items():
        recs = sorted(recs, key=lambda x: (int(x.get("occurrence_index") or 0), int(x.get("idx") or 0)))
        ranks = sorted(int(x["order_rank"]) for x in recs)
        chunks_touched = set()
        active_lens = []
        for x in recs:
            active_lens.append(int(x["active_len"]))
            if x.get("first_chunk") is not None:
                chunks_touched.update(range(int(x["first_chunk"]), int(x["last_chunk"]) + 1))
        same_chunk = False
        if len(recs) >= 2 and recs[0].get("first_chunk") is not None and recs[1].get("first_chunk") is not None:
            chunks0 = set(range(int(recs[0]["first_chunk"]), int(recs[0]["last_chunk"]) + 1))
            chunks1 = set(range(int(recs[1]["first_chunk"]), int(recs[1]["last_chunk"]) + 1))
            same_chunk = bool(chunks0 & chunks1)
        pair_records.append({
            "pair_id": pair_id,
            "n_units": len(recs),
            "doc_order_distance": (ranks[-1] - ranks[0]) if len(ranks) >= 2 else None,
            "adjacent_docs": (ranks[-1] - ranks[0] == 1) if len(ranks) >= 2 else None,
            "units_share_any_chunk": same_chunk,
            "n_chunks_touched_total": len(chunks_touched),
            "active_tokens_total": sum(active_lens),
            "unit_realized_orders": [x.get("realized_order") for x in recs],
            "unit_start_offsets": [x.get("start_offset") for x in recs],
            "unit_chunks_touched": [x.get("n_chunks_touched") for x in recs],
        })

    pair_row_lengths = [float(r["token_len"]) for r in pair_rows]
    pair_row_words = [float(r["words"]) for r in pair_rows]
    row_cross = [1.0 if m["crosses_chunk_boundary"] else 0.0 for m in row_metrics]
    row_within = [1.0 if m["wholly_within_one_chunk"] else 0.0 for m in row_metrics]
    row_dropped = [1.0 if m["active_len"] == 0 else 0.0 for m in row_metrics]
    pair_share = [1.0 if p["units_share_any_chunk"] else 0.0 for p in pair_records]
    pair_adj = [1.0 if p["adjacent_docs"] else 0.0 for p in pair_records if p["adjacent_docs"] is not None]
    distances = [float(p["doc_order_distance"]) for p in pair_records if p["doc_order_distance"] is not None]
    chunks_total = [float(p["n_chunks_touched_total"]) for p in pair_records]
    start_offsets = [float(m["start_offset"]) for m in row_metrics]
    chunk_touches = [float(m["n_chunks_touched"]) for m in row_metrics]
    return {
        "epoch": epoch,
        "seed_used_for_shuffle": None if epoch == 0 else seed + epoch,
        "raw_tokens": raw_tokens,
        "active_tokens": active_tokens,
        "dropped_tail_tokens": raw_tokens - active_tokens,
        "chunks": chunks,
        "n_rows": len(rows),
        "n_pair_rows": len(pair_rows),
        "n_pairs": len(pair_records),
        "pair_row_token_len_stats": fstats(pair_row_lengths),
        "pair_row_word_stats": fstats(pair_row_words),
        "pair_row_cross_chunk_fraction": float(mean(row_cross)) if row_cross else None,
        "pair_row_within_one_chunk_fraction": float(mean(row_within)) if row_within else None,
        "pair_row_dropped_tail_fraction": float(mean(row_dropped)) if row_dropped else None,
        "pair_row_start_offset_stats": fstats(start_offsets),
        "pair_row_chunks_touched_stats": fstats(chunk_touches),
        "pair_units_share_any_chunk_fraction": float(mean(pair_share)) if pair_share else None,
        "pair_units_adjacent_doc_fraction": float(mean(pair_adj)) if pair_adj else None,
        "pair_doc_order_distance_stats": fstats(distances),
        "pair_total_chunks_touched_stats": fstats(chunks_total),
        "pair_level_sample": pair_records[:8],
    }


def compare_epochs(arms: dict[str, dict[str, Any]], epochs: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metrics = [
        "raw_tokens", "active_tokens", "dropped_tail_tokens", "chunks", "pair_row_cross_chunk_fraction",
        "pair_row_within_one_chunk_fraction", "pair_units_share_any_chunk_fraction", "pair_units_adjacent_doc_fraction",
    ]
    for epoch in epochs:
        key = f"epoch_{epoch}"
        eblocks = {arm: arms[arm]["epochs"][key] for arm in arms}
        comp: dict[str, Any] = {}
        for metric in metrics:
            vals = {arm: eblocks[arm].get(metric) for arm in eblocks}
            comp[metric] = vals
            if "compact_oneway" in vals and "repeat_oneway" in vals:
                try:
                    comp[f"compact_minus_repeat_oneway_{metric}"] = float(vals["compact_oneway"] - vals["repeat_oneway"])
                except Exception:
                    pass
            if "compact_reciprocal" in vals and "repeat_reciprocal" in vals:
                try:
                    comp[f"compact_minus_repeat_reciprocal_{metric}"] = float(vals["compact_reciprocal"] - vals["repeat_reciprocal"])
                except Exception:
                    pass
            if "compact_reciprocal" in vals and "compact_oneway" in vals:
                try:
                    comp[f"reciprocal_minus_oneway_compact_{metric}"] = float(vals["compact_reciprocal"] - vals["compact_oneway"])
                except Exception:
                    pass
            if "repeat_reciprocal" in vals and "repeat_oneway" in vals:
                try:
                    comp[f"reciprocal_minus_oneway_repeat_{metric}"] = float(vals["repeat_reciprocal"] - vals["repeat_oneway"])
                except Exception:
                    pass
        out[key] = comp
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--epochs", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.manifest)
    tokenizer_path = args.tokenizer or pathlib.Path(manifest["neutral_tokenizer"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    arms_out: dict[str, Any] = {}
    for arm in ARMS_DEFAULT:
        rec = manifest["arms"][arm]
        rows, lengths = load_rows(pathlib.Path(rec["path"]), tokenizer, int(eos_id))
        epochs_out = {}
        for epoch in args.epochs:
            epochs_out[f"epoch_{epoch}"] = audit_epoch(rows, lengths, args.seq_len, args.seed, epoch)
        arms_out[arm] = {
            "path": rec["path"],
            "sha256": rec.get("sha256"),
            "n_rows": len(rows),
            "epochs": epochs_out,
        }

    result = {
        "status": "CAUSAL_TOPOLOGY_PACKING_AUDIT",
        "manifest": str(args.manifest),
        "tokenizer": str(tokenizer_path),
        "seq_len": args.seq_len,
        "trainer_shuffle_seed": args.seed,
        "epochs_audited": args.epochs,
        "arms": arms_out,
        "epoch_comparisons": compare_epochs(arms_out, args.epochs),
        "interpretation": [
            "The GPT trainer concatenates full tokenized rows plus EOS and chunks the stream; rows longer than 256 are legal for this trainer but can span multiple chunks.",
            "Epoch 0 preserves JSONL document order; later epochs shuffle whole rows with random.Random(seed+epoch), so two units of the same pair usually stop being adjacent even though each unit still contains source and view/repeat within one row.",
            "The intended causal conditioning is inside each pair-unit row. Pair-unit rows that cross chunk boundaries provide only partial within-row context for tokens after the boundary; the fractions here quantify that exposure geometry.",
            "A future topology interaction must be read with these packing quantities, copied-token overlap, and compact-vs-repeat token asymmetry; this audit does not authorize training.",
        ],
    }
    out_json = args.out_dir / "causal_topology_packing_audit.json"
    out_md = args.out_dir / "causal_topology_packing_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = ["# research causal topology packing audit\n\n"]
    lines.append(f"Manifest: `{args.manifest}`. Epochs audited: {args.epochs}. Seq length: {args.seq_len}.\n\n")
    lines.append("## Arm/epoch summary\n\n")
    lines.append("| arm | epoch | active tokens | chunks | dropped tail | pair rows | pair row cross-chunk frac | pair units share chunk frac | pair units adjacent-doc frac | pair row token mean | pair doc distance median |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm, block in arms_out.items():
        for ek, er in block["epochs"].items():
            lines.append(
                f"| {arm} | {er['epoch']} | {er['active_tokens']} | {er['chunks']} | {er['dropped_tail_tokens']} | {er['n_pair_rows']} | "
                f"{er['pair_row_cross_chunk_fraction']:.6f} | {er['pair_units_share_any_chunk_fraction']:.6f} | {er['pair_units_adjacent_doc_fraction']:.6f} | "
                f"{er['pair_row_token_len_stats']['mean']:.3f} | {er['pair_doc_order_distance_stats']['median']:.3f} |\n"
            )
    lines.append("\n## Key comparisons\n\n")
    lines.append("| epoch | C-R active tokens oneway | C-R active tokens reciprocal | recip-oneway active compact | recip-oneway active repeat | recip-oneway pair share compact | recip-oneway pair share repeat |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for epoch in args.epochs:
        comp = result["epoch_comparisons"][f"epoch_{epoch}"]
        lines.append(
            f"| {epoch} | {comp.get('compact_minus_repeat_oneway_active_tokens','')} | {comp.get('compact_minus_repeat_reciprocal_active_tokens','')} | "
            f"{comp.get('reciprocal_minus_oneway_compact_active_tokens','')} | {comp.get('reciprocal_minus_oneway_repeat_active_tokens','')} | "
            f"{comp.get('reciprocal_minus_oneway_compact_pair_units_share_any_chunk_fraction','')} | {comp.get('reciprocal_minus_oneway_repeat_pair_units_share_any_chunk_fraction','')} |\n"
        )
    lines.append("\n## Interpretation\n\n")
    for item in result["interpretation"]:
        lines.append(f"- {item}\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
