#!/usr/bin/env python3
"""research audit of the actual research/126 shuffled-control semantics.

The repaired trainer uses a per-batch derangement of AuxUnit source sequences rather
than the static precomputed decoy_source_ids in the pair-data file.  This preserves
the exact source multiset and total source-word exposure within each batch, but it
may assign a rewrite target a source from the same packed row/document or a very
different source length.  This CPU audit quantifies that control so later score
interpretation does not silently assume the unused decoy map.

It audits the first main examples loaded under the 20M main-word cap.  For all
constituent pairs in each batch (an approximation to the masked subset; in actual
training most rewrite pairs fire because WWM masks any selected rewrite word group),
it applies the same per-batch derangement rule as the trainer and records source
correspondence, document relation, row relation, and source-length mismatch.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import statistics
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DEFAULT_POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DEFAULT_PAIR_DATA = WORKSPACE / "data/aux_pair_data/aux_pair_data.json"
PAIR_JSONL = WORKSPACE / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
OUT_DIR = WORKSPACE / "data/dualview_shuffle_control_audit"


def norm_pid(pid: str) -> str:
    s = str(pid)
    return s.split(":", 1)[-1] if s.startswith("compact:") else s


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def load_pair_doc_map() -> dict[str, dict[str, Any]]:
    out = {}
    with PAIR_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = norm_pid(obj.get("pair_id"))
            out[pid] = {
                "doc_id": str(obj.get("doc_id", "")),
                "source_words": int(obj.get("source_words", 0)),
                "rewrite_words": int(obj.get("rewrite_words", 0)),
            }
    return out


def load_examples(pool: pathlib.Path, max_main_words: int) -> list[dict[str, Any]]:
    rows = []
    total = 0
    with pool.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            if total + words > max_main_words:
                break
            rows.append({"example_id": int(obj.get("example_id", -1)), "words": words, "source": str(obj.get("source", ""))})
            total += words
    return rows


def deranged_source_indices(n: int, seed: int, step: int) -> list[int] | None:
    if n < 2:
        return None
    rng = random.Random(seed + 1000003 * step)
    idxs = list(range(n))
    rng.shuffle(idxs)
    src_from = idxs[1:] + idxs[:1]
    assign = [-1] * n
    for target_i, src_i in zip(idxs, src_from):
        assign[target_i] = src_i
    return assign


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--pair-data", default=str(DEFAULT_PAIR_DATA))
    ap.add_argument("--max-main-words", type=int, default=20_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--shuffle-seed", type=int, default=43022)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    pair_raw = read_json(pathlib.Path(args.pair_data))
    pair_data = pair_raw["pair_data"]
    pair_docs = load_pair_doc_map()
    examples = load_examples(pathlib.Path(args.pool), args.max_main_words)

    row_pair_counts = []
    unit_records_all = []
    batch_records = []
    relation_counts = collections.Counter()
    len_diffs = []
    src_words_true = []
    src_words_assigned = []
    batches_with_pairs = 0
    skipped_single_unit_batches = 0

    for b_start in range(0, len(examples), args.batch_size):
        batch = examples[b_start:b_start + args.batch_size]
        step = b_start // args.batch_size + 1
        units = []
        for row_pos, ex in enumerate(batch):
            rec = pair_data.get(str(ex["example_id"]))
            if not rec:
                continue
            for pidx, pr in enumerate(rec.get("pairs", [])):
                pid_full = str(pr.get("pair_id"))
                pid = norm_pid(pid_full)
                doc = pair_docs.get(pid, {}).get("doc_id", "")
                units.append({
                    "row_example_id": ex["example_id"],
                    "batch_row_pos": row_pos,
                    "pair_index_in_row": pidx,
                    "pair_id": pid_full,
                    "pid_norm": pid,
                    "doc_id": doc,
                    "source_words": int(pr.get("source_words", pair_docs.get(pid, {}).get("source_words", 0))),
                    "rewrite_words": int(pr.get("rewrite_words", pair_docs.get(pid, {}).get("rewrite_words", 0))),
                })
        if units:
            batches_with_pairs += 1
            row_pair_counts.append({"loader_step": step, "n_units": len(units), "n_pair_rows": len(set(u["row_example_id"] for u in units)), "n_docs": len(set(u["doc_id"] for u in units))})
        assign = deranged_source_indices(len(units), args.shuffle_seed, step)
        if assign is None:
            if units:
                skipped_single_unit_batches += 1
            continue
        for i, src_i in enumerate(assign):
            u = units[i]
            s = units[src_i]
            same_pair = (i == src_i) or (u["pid_norm"] == s["pid_norm"])
            same_row = u["row_example_id"] == s["row_example_id"]
            same_doc = u["doc_id"] != "" and u["doc_id"] == s["doc_id"]
            relation_counts["total_assignments"] += 1
            relation_counts["same_pair"] += int(same_pair)
            relation_counts["same_row"] += int(same_row)
            relation_counts["same_doc"] += int(same_doc)
            relation_counts["different_doc"] += int(not same_doc)
            d = int(s["source_words"]) - int(u["source_words"])
            len_diffs.append(d)
            src_words_true.append(int(u["source_words"]))
            src_words_assigned.append(int(s["source_words"]))
            if len(unit_records_all) < 50:
                unit_records_all.append({
                    "loader_step": step,
                    "target_pair": u["pair_id"],
                    "assigned_source_pair": s["pair_id"],
                    "target_doc": u["doc_id"],
                    "assigned_doc": s["doc_id"],
                    "same_row": same_row,
                    "same_doc": same_doc,
                    "true_source_words": u["source_words"],
                    "assigned_source_words": s["source_words"],
                    "source_word_delta": d,
                })
        batch_records.append({
            "loader_step": step,
            "n_units": len(units),
            "n_pair_rows": len(set(u["row_example_id"] for u in units)),
            "n_docs": len(set(u["doc_id"] for u in units)),
            "same_doc_assignments": sum(1 for i, src_i in enumerate(assign) if units[i]["doc_id"] != "" and units[i]["doc_id"] == units[src_i]["doc_id"]),
            "same_row_assignments": sum(1 for i, src_i in enumerate(assign) if units[i]["row_example_id"] == units[src_i]["row_example_id"]),
        })

    total = relation_counts["total_assignments"]
    abs_len = [abs(x) for x in len_diffs]
    summary = {
        "status": "DUALVIEW_SHUFFLE_CONTROL_AUDIT",
        "pool": rel(args.pool),
        "pair_data": rel(args.pair_data),
        "max_main_words": args.max_main_words,
        "batch_size": args.batch_size,
        "shuffle_seed": args.shuffle_seed,
        "n_examples_loaded": len(examples),
        "main_words_loaded": sum(x["words"] for x in examples),
        "n_batches": (len(examples) + args.batch_size - 1) // args.batch_size,
        "batches_with_pairs": batches_with_pairs,
        "skipped_single_unit_batches": skipped_single_unit_batches,
        "potential_unit_assignments": total,
        "relation_counts": dict(relation_counts),
        "relation_pct": {k: pct(v, total) for k, v in relation_counts.items() if k != "total_assignments"},
        "source_word_delta_stats": {
            "mean_assigned_minus_true": statistics.mean(len_diffs) if len_diffs else None,
            "mean_abs": statistics.mean(abs_len) if abs_len else None,
            "median_abs": statistics.median(abs_len) if abs_len else None,
            "p90_abs": sorted(abs_len)[int(0.90 * (len(abs_len)-1))] if abs_len else None,
            "p95_abs": sorted(abs_len)[int(0.95 * (len(abs_len)-1))] if abs_len else None,
            "max_abs": max(abs_len) if abs_len else None,
        },
        "batch_unit_stats": {
            "mean_units_in_pair_batches": statistics.mean([r["n_units"] for r in row_pair_counts]) if row_pair_counts else None,
            "median_units_in_pair_batches": statistics.median([r["n_units"] for r in row_pair_counts]) if row_pair_counts else None,
            "max_units_in_pair_batches": max([r["n_units"] for r in row_pair_counts]) if row_pair_counts else None,
            "mean_pair_rows_in_pair_batches": statistics.mean([r["n_pair_rows"] for r in row_pair_counts]) if row_pair_counts else None,
            "mean_docs_in_pair_batches": statistics.mean([r["n_docs"] for r in row_pair_counts]) if row_pair_counts else None,
        },
        "first_assignment_examples": unit_records_all,
        "batch_records_sample": batch_records[:20],
        "interpretation": "Trainer shuffled mode uses batch derangement, not the static decoy_source_ids stored in pair_data. This preserves total source exposure exactly per batch; same_doc/same_row rates quantify how false the source-correspondence control is.",
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "shuffle_control_audit.json"
    out_md = out_dir / "shuffle_control_audit.md"
    summary["out_json"] = rel(out_json)
    summary["out_md"] = rel(out_md)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dual-view shuffled-control audit", ""]
    lines.append(f"Potential unit assignments: `{total}` over `{batches_with_pairs}` pair-containing batches.")
    lines.append(f"Same pair: `{relation_counts['same_pair']}` ({pct(relation_counts['same_pair'], total):.3f}%).")
    lines.append(f"Same row: `{relation_counts['same_row']}` ({pct(relation_counts['same_row'], total):.3f}%).")
    lines.append(f"Same doc: `{relation_counts['same_doc']}` ({pct(relation_counts['same_doc'], total):.3f}%).")
    lines.append(f"Mean absolute source-word mismatch: `{summary['source_word_delta_stats']['mean_abs']}`; p95 `{summary['source_word_delta_stats']['p95_abs']}`.")
    lines.append("")
    lines.append("The stored pair-data decoy map remains unused by the current trainer; any aligned-vs-shuffled score should be interpreted as a same-batch source-multiset derangement contrast.")
    lines.append(f"JSON: `{rel(out_json)}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "potential_unit_assignments": total,
                      "same_doc_pct": summary["relation_pct"].get("same_doc"),
                      "same_row_pct": summary["relation_pct"].get("same_row"),
                      "mean_abs_source_word_delta": summary["source_word_delta_stats"].get("mean_abs"),
                      "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
