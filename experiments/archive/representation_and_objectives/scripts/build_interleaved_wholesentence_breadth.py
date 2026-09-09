#!/usr/bin/env python3
"""research: build a second repaired source-breadth arm that keeps whole
independent FineWeb sentences but places them between the same common source
segments, approximating the compact arm's source/rewrite alternation.

Why this exists: the row-block whole-sentence repair removes cross-row sentence
splitting, but it places all common source segments first and all independent
breadth sentences afterward. The compact arm is source1+rewrite1, source2+rewrite2,
... within each packed FineWeb row. This script reuses exactly the same selected
whole independent breadth sentences from the research repair, assigns them to
per-source slots inside each row, and builds:
  source1 + independent whole sentence(s) + source2 + ...
with the exact same row word totals and no sentence splitting.

The assignment minimizes source-position displacement relative to the compact
source/rewrite layout while preserving whole independent sentences and exact row
word budgets.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SCRIPT = ROOT / "scripts/fw_source_breadth_arm.py"
COMPACT_ARM = ROOT / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
COMPACT_100M = ROOT / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
WHOLE_ROW_META = ROOT / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_row_meta.jsonl"
WHOLE_SOURCES = ROOT / "data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl"
WHOLE_MANIFEST = ROOT / "data/fw_source_breadth_wholesentence_arm/fw_source_breadth_wholesentence_manifest.json"
TOKENIZER_DIR = ROOT / "data/shared_tokenizer/shared_16k_tokenizer"
OUT_DIR = ROOT / "data/fw_source_breadth_interleaved_wholesentence_arm"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_source_breadth_interleaved_wholesentence.md')
TOTAL_WORDS = 10_000_000
PASSES = 10
STREAM_SEED = 10289931
COMPACT_LABEL = "fw_preserved_compact_view"
BREADTH_LABEL = "fw_preserved_source_breadth_interleaved_wholesentence"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def wc(text: str) -> int:
    return len(norm_text(text).split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_step102():
    spec = importlib.util.spec_from_file_location("fw_source_breadth_arm", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    arr = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        x = p * (len(arr) - 1)
        lo = int(x)
        hi = min(lo + 1, len(arr) - 1)
        a = x - lo
        return arr[lo] * (1 - a) + arr[hi] * a
    return {"n": len(arr), "sum": float(sum(arr)), "min": arr[0], "p05": q(0.05), "p10": q(0.10), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "max": arr[-1], "mean": float(statistics.mean(arr))}


def assign_sentences_to_slots(targets: list[int], lengths: list[int]) -> list[list[int]]:
    """Assign ordered sentence indices to ordered slots, allowing empty slots.

    Objective: minimize cumulative source-position displacement relative to the
    compact layout. Since total assigned length equals total target length, final
    displacement is zero; the internal cumulative errors determine where later
    sources move.
    """
    m = len(targets)
    n = len(lengths)
    prefix_L = [0]
    for x in lengths:
        prefix_L.append(prefix_L[-1] + x)
    prefix_T = [0]
    for x in targets:
        prefix_T.append(prefix_T[-1] + x)

    # Dynamic program over slots and used sentence count. Keep explicit
    # backpointers for each layer; the previous implementation overwrote earlier
    # layers and could not backtrack when m > 1.
    states: dict[int, float] = {0: 0.0}
    back: list[dict[int, int]] = []
    for slot in range(m):
        ndp: dict[int, float] = {}
        parent: dict[int, int] = {}
        for j, cost in states.items():
            # Remaining slots may be empty; use any next_j from j..n.
            for next_j in range(j, n + 1):
                group_sum = prefix_L[next_j] - prefix_L[j]
                slot_target = targets[slot]
                cum_err = abs(prefix_L[next_j] - prefix_T[slot + 1])
                slot_err = abs(group_sum - slot_target)
                empty_penalty = 0.05 if next_j == j and slot_target > 0 else 0.0
                # Cumulative alignment dominates; slot-level fit breaks ties.
                new_cost = cost + cum_err + 0.25 * slot_err + empty_penalty
                if next_j not in ndp or new_cost < ndp[next_j]:
                    ndp[next_j] = new_cost
                    parent[next_j] = j
        states = ndp
        back.append(parent)
    if n not in states:
        raise RuntimeError("no ordered assignment found")
    cuts = [n]
    cur = n
    for slot in range(m - 1, -1, -1):
        prev_j = back[slot][cur]
        cuts.append(prev_j)
        cur = prev_j
    cuts.reverse()
    out: list[list[int]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        out.append(list(range(a, b)))
    return out


def segment_starts(source_words: list[int], companion_words: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for sw, cw in zip(source_words, companion_words):
        starts.append(pos)
        pos += int(sw) + int(cw)
    return starts


def build_interleaved() -> dict[str, Any]:
    breadth_builder = load_step102()
    pairs = breadth_builder.load_pairs()
    pair_by_id = {str(p["norm_hash"]): p for p in pairs}
    compact_fw, filler_rows = breadth_builder.load_compact_arm()
    row_meta = read_jsonl(WHOLE_ROW_META)
    whole_sources = {str(r["norm_hash"]): r for r in read_jsonl(WHOLE_SOURCES)}
    if len(row_meta) != len(compact_fw):
        raise RuntimeError("row_meta / compact FW count mismatch")

    inter_rows: list[dict[str, Any]] = []
    inter_meta: list[dict[str, Any]] = []
    compact_vs_sourceblock_displacements: list[int] = []
    compact_vs_interleaved_displacements: list[int] = []
    abs_sourceblock_displacements: list[int] = []
    abs_interleaved_displacements: list[int] = []
    max_abs_interleaved_by_row: list[int] = []
    max_abs_sourceblock_by_row: list[int] = []
    slot_abs_errors: list[int] = []
    empty_slots = 0
    nonempty_slots = 0
    sources_per_row: list[int] = []
    pairs_per_row: list[int] = []

    for row_i, meta in enumerate(row_meta):
        pids = [str(x) for x in meta["pair_ids"]]
        ps = [pair_by_id[x] for x in pids]
        selected = [whole_sources[str(x)] for x in meta["breadth_source_ids"]]
        source_words = [int(p["source_words"]) for p in ps]
        rewrite_words = [int(p["rewrite_words"]) for p in ps]
        selected_words = [int(s["words"]) for s in selected]
        if sum(selected_words) != int(meta["breadth_companion_words"]):
            raise RuntimeError(f"row {row_i} selected words mismatch")
        assignments = assign_sentences_to_slots(rewrite_words, selected_words)
        assigned_words = [sum(selected_words[j] for j in group) for group in assignments]
        if sum(assigned_words) != sum(rewrite_words):
            raise RuntimeError(f"row {row_i} assigned sum mismatch")
        compact_starts = segment_starts(source_words, rewrite_words)
        inter_starts = segment_starts(source_words, assigned_words)
        sourceblock_starts = []
        pos = 0
        for sw in source_words:
            sourceblock_starts.append(pos)
            pos += sw
        d_inter = [a - b for a, b in zip(inter_starts, compact_starts)]
        d_block = [a - b for a, b in zip(sourceblock_starts, compact_starts)]
        compact_vs_interleaved_displacements.extend(d_inter)
        compact_vs_sourceblock_displacements.extend(d_block)
        abs_interleaved_displacements.extend(abs(x) for x in d_inter)
        abs_sourceblock_displacements.extend(abs(x) for x in d_block)
        max_abs_interleaved_by_row.append(max([abs(x) for x in d_inter] or [0]))
        max_abs_sourceblock_by_row.append(max([abs(x) for x in d_block] or [0]))
        slot_abs_errors.extend(abs(a - t) for a, t in zip(assigned_words, rewrite_words))
        empty_slots += sum(1 for x in assigned_words if x == 0)
        nonempty_slots += sum(1 for x in assigned_words if x > 0)
        sources_per_row.append(len(selected))
        pairs_per_row.append(len(ps))

        parts: list[str] = []
        slot_source_ids: list[list[str]] = []
        slot_source_words: list[list[int]] = []
        for p, group in zip(ps, assignments):
            parts.append(p["source_text"])
            slot_ids: list[str] = []
            slot_words: list[int] = []
            for j in group:
                s = selected[j]
                parts.append(str(s["text"]))
                slot_ids.append(str(s["norm_hash"]))
                slot_words.append(int(s["words"]))
            slot_source_ids.append(slot_ids)
            slot_source_words.append(slot_words)
        text = norm_text(" ".join(parts))
        if wc(text) != int(meta["total_words"]):
            raise RuntimeError(f"row {row_i} total words {wc(text)} != {meta['total_words']}")
        inter_rows.append({"text": text, "words": int(meta["total_words"]), "example_id": int(meta["example_id"]), "source": BREADTH_LABEL})
        inter_meta.append({
            "row_index": row_i,
            "example_id": int(meta["example_id"]),
            "pair_ids": pids,
            "pair_source_words": source_words,
            "compact_rewrite_word_targets": rewrite_words,
            "assigned_breadth_words": assigned_words,
            "assigned_breadth_source_ids_by_slot": slot_source_ids,
            "assigned_breadth_source_words_by_slot": slot_source_words,
            "compact_source_starts": compact_starts,
            "interleaved_source_starts": inter_starts,
            "sourceblock_source_starts": sourceblock_starts,
            "interleaved_minus_compact_source_start_words": d_inter,
            "sourceblock_minus_compact_source_start_words": d_block,
            "whole_sentences": True,
        })

    breadth_pool = inter_rows + filler_rows
    compact_pool = compact_fw + filler_rows
    if [int(r["words"]) for r in breadth_pool] != [int(r["words"]) for r in compact_pool]:
        raise RuntimeError("row word sequence mismatch")
    total = sum(int(r["words"]) for r in breadth_pool)
    if total != TOTAL_WORDS:
        raise RuntimeError(f"total {total} != {TOTAL_WORDS}")

    arm_path = OUT_DIR / "fw_preserved_source_breadth_interleaved_wholesentence_10M.jsonl"
    meta_path = OUT_DIR / "source_breadth_interleaved_wholesentence_row_meta.jsonl"
    write_jsonl(arm_path, breadth_pool)
    write_jsonl(meta_path, inter_meta)

    stream_path = OUT_DIR / "fw_preserved_source_breadth_interleaved_wholesentence_100M.jsonl"
    rows = breadth_pool
    order_hashes: list[str] = []
    words_stream = 0
    with stream_path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(len(rows)))
            random.Random(STREAM_SEED + 1000 + pass_i).shuffle(order)
            order_hashes.append(sha256_text(",".join(map(str, order))))
            for idx in order:
                r = rows[idx]
                rec = {"text": r["text"], "words": int(r["words"]), "example_id": int(r.get("example_id") or 0), "source": str(r.get("source") or "")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                words_stream += int(r["words"])
    if words_stream != TOTAL_WORDS * PASSES:
        raise RuntimeError("100M stream word count mismatch")
    whole_manifest = json.loads(WHOLE_MANIFEST.read_text(encoding="utf-8"))
    compact_stream_orders = ((whole_manifest.get("training_streams") or {}).get("compact_view_existing") or {}).get("pass_order_hashes")
    pass_orders_match = order_hashes == compact_stream_orders
    if not pass_orders_match:
        raise RuntimeError("interleaved stream order does not match compact stream")

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    def token_metrics(path: pathlib.Path) -> dict[str, Any]:
        rows_n = words_n = tok_un = tok_vis = over = groups = 0
        special = set(int(x) for x in tok.all_special_ids)
        by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for r in iter_jsonl(path):
            rows_n += 1
            text = str(r.get("text") or "")
            w = int(r.get("words") or wc(text))
            words_n += w
            src = str(r.get("source") or "")
            ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            vis = ids[:256]
            tok_un += len(ids)
            tok_vis += len(vis)
            over += int(len(ids) > 256)
            gc = 0
            prev = False
            for j, tid in enumerate(vis):
                if int(tid) in special:
                    prev = False
                    continue
                piece = tok.convert_ids_to_tokens(int(tid))
                start = j == 0 or str(piece).startswith("Ġ") or str(piece).startswith("▁") or not prev
                if start:
                    gc += 1
                prev = True
            groups += gc
            by_source[src]["rows"] += 1
            by_source[src]["words"] += w
            by_source[src]["tokens_visible"] += len(vis)
            by_source[src]["rows_over_256"] += int(len(ids) > 256)
            by_source[src]["wwm_groups_visible"] += gc
        return {"rows": rows_n, "words": words_n, "tokens_untruncated": tok_un, "tokens_visible": tok_vis, "tokens_per_word_visible": tok_vis/max(1, words_n), "rows_over_256": over, "wwm_groups_visible": groups, "by_source": {k: dict(v) for k, v in by_source.items()}}

    compact_tok = token_metrics(COMPACT_ARM)
    inter_tok = token_metrics(arm_path)

    result = {
        "status": "FW_SOURCE_BREADTH_INTERLEAVED_WHOLESENTENCE_READY",
        "created_utc": now_utc(),
        "scientific_purpose": "Reduce internal-layout asymmetry while keeping coherent independent FineWeb breadth: each common selected source segment is followed by assigned whole independent source sentence(s), with exact row word totals.",
        "inputs": {
            "rowblock_repaired_sources": str(WHOLE_SOURCES),
            "rowblock_repaired_row_meta": str(WHOLE_ROW_META),
            "compact_arm": str(COMPACT_ARM),
            "compact_stream": str(COMPACT_100M),
            "shared_tokenizer": str(TOKENIZER_DIR),
        },
        "budgets": {
            "total_words": total,
            "rows_total": len(breadth_pool),
            "fineweb_rows": len(inter_rows),
            "common_fineweb_source_words": sum(int(p["source_words"]) for p in pairs),
            "breadth_companion_words": sum(sum(row["assigned_breadth_words"]) for row in inter_meta),
            "row_word_sequence_matches_compact": True,
            "source_sentence_splits_across_rows": 0,
            "whole_sentence_companions": True,
        },
        "layout_alignment": {
            "pairs_per_row_stats": stats(pairs_per_row),
            "selected_breadth_sentences_per_row_stats": stats(sources_per_row),
            "slot_abs_error_stats": stats(slot_abs_errors),
            "empty_slots": empty_slots,
            "nonempty_slots": nonempty_slots,
            "source_start_abs_displacement_words_interleaved_vs_compact": stats(abs_interleaved_displacements),
            "source_start_abs_displacement_words_sourceblock_vs_compact": stats(abs_sourceblock_displacements),
            "max_abs_displacement_by_row_interleaved": stats(max_abs_interleaved_by_row),
            "max_abs_displacement_by_row_sourceblock": stats(max_abs_sourceblock_by_row),
            "displacement_improvement_mean_abs_words": statistics.mean(abs_sourceblock_displacements) - statistics.mean(abs_interleaved_displacements),
        },
        "token_metrics_10m": {
            "compact_view": compact_tok,
            "source_breadth_interleaved_wholesentence": inter_tok,
            "delta_interleaved_minus_compact": {
                "tokens_per_word_visible": inter_tok["tokens_per_word_visible"] - compact_tok["tokens_per_word_visible"],
                "rows_over_256": inter_tok["rows_over_256"] - compact_tok["rows_over_256"],
                "wwm_groups_visible": inter_tok["wwm_groups_visible"] - compact_tok["wwm_groups_visible"],
            },
        },
        "training_stream": {
            "path": str(stream_path),
            "rows": len(rows) * PASSES,
            "words": words_stream,
            "sha256": sha256_file(stream_path),
            "seed": STREAM_SEED,
            "pass_order_hashes": order_hashes,
            "pass_orders_match_compact": pass_orders_match,
        },
        "files": {
            "source_breadth_interleaved_wholesentence_10m": str(arm_path),
            "source_breadth_interleaved_wholesentence_100m": str(stream_path),
            "row_meta": str(meta_path),
            "manifest": str(OUT_DIR / "fw_source_breadth_interleaved_wholesentence_manifest.json"),
            "note": str(NOTE),
        },
        "hashes": {
            "source_breadth_interleaved_wholesentence_10m": sha256_file(arm_path),
            "source_breadth_interleaved_wholesentence_100m": sha256_file(stream_path),
            "row_meta": sha256_file(meta_path),
        },
        "scientific_reading": {
            "what_this_repairs": "Compared with the row-block repaired breadth arm, this arm makes common-source positions much closer to the compact source/rewrite alternation while preserving whole independent sentences.",
            "remaining_asymmetry": "Some rewrite-length slots cannot receive matching independent whole sentences, so empty slots and length mismatch remain; the semantic difference between same-content restatement and unrelated added facts is still the intended intervention.",
        },
        "elapsed_sec": round(time.time() - time.time() + 0, 2),
    }
    return result


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_interleaved()
    result["elapsed_sec"] = round(time.time() - t0, 2)
    manifest_path = OUT_DIR / "fw_source_breadth_interleaved_wholesentence_manifest.json"
    result["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    la = result["layout_alignment"]
    tok = result["token_metrics_10m"]
    inter = tok["source_breadth_interleaved_wholesentence"]
    comp = tok["compact_view"]
    NOTE.write_text(
        "# research — interleaved whole-sentence source-breadth arm\n\n"
        "This variant reuses the same repaired independent FineWeb sentences, but places them between the same common source segments to approximate the compact arm's source/rewrite alternation. It avoids cross-row sentence splitting and preserves exact row word totals.\n\n"
        "## Layout movement\n\n"
        f"- Mean absolute source-start displacement vs compact: interleaved {la['source_start_abs_displacement_words_interleaved_vs_compact']['mean']:.3f} words; source-block {la['source_start_abs_displacement_words_sourceblock_vs_compact']['mean']:.3f} words.\n"
        f"- Per-row max absolute displacement median: interleaved {la['max_abs_displacement_by_row_interleaved']['median']:.1f}; source-block {la['max_abs_displacement_by_row_sourceblock']['median']:.1f}.\n"
        f"- Slot length absolute error mean: {la['slot_abs_error_stats']['mean']:.3f}; empty slots {la['empty_slots']:,}, nonempty slots {la['nonempty_slots']:,}.\n\n"
        "## Token geometry\n\n"
        f"- Compact visible tokens/word: {comp['tokens_per_word_visible']:.4f}; interleaved breadth visible tokens/word: {inter['tokens_per_word_visible']:.4f}.\n"
        f"- Compact rows over 256: {comp['rows_over_256']:,}; interleaved breadth rows over 256: {inter['rows_over_256']:,}.\n"
        f"- Compact WWM groups/pass: {comp['wwm_groups_visible']:,}; interleaved breadth WWM groups/pass: {inter['wwm_groups_visible']:,}.\n\n"
        "## Use\n\n"
        "This arm is a layout-closer coherent-breadth comparator. It is useful if the first H100 pair should minimize common-source position movement relative to the compact arm. The row-block whole-sentence breadth arm remains a cleaner coherent-breadth block comparator but has larger source-position movement.\n\n"
        f"Manifest: `{manifest_path}`\n"
        f"10M arm: `{result['files']['source_breadth_interleaved_wholesentence_10m']}`\n"
        f"100M stream: `{result['files']['source_breadth_interleaved_wholesentence_100m']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "total_words": result["budgets"]["total_words"],
        "row_word_sequence_matches_compact": result["budgets"]["row_word_sequence_matches_compact"],
        "stream_words": result["training_stream"]["words"],
        "pass_orders_match_compact": result["training_stream"]["pass_orders_match_compact"],
        "mean_abs_displacement_interleaved": round(la["source_start_abs_displacement_words_interleaved_vs_compact"]["mean"], 3),
        "mean_abs_displacement_sourceblock": round(la["source_start_abs_displacement_words_sourceblock_vs_compact"]["mean"], 3),
        "interleaved_visible_tpw": round(inter["tokens_per_word_visible"], 4),
        "manifest": str(manifest_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
