#!/usr/bin/env python3
"""research: build a low-cost matched causal topology 2x2 scaffold.

Scientific purpose
------------------
research showed that compact semantic same-window views did not transfer broadly to
a one-direction GPT2 causal coordinate. research then showed that causal examples
only give paired-context lift to the second segment, but DeBERTa non-copy lift is
tiny. The prepared reciprocal scaffold is therefore not enough to justify another
large run. This script builds the *minimal interaction-isolation construction* that
would be needed if the DeBERTa triangle results revive the direction:

    semantic arm: compact view vs exact repeat
    topology arm: one-way duplicated exposure vs reciprocal both-way exposure

All four pools use the identical selected pair subset, identical pair word dose,
identical filler words, identical legal 10M-word total, identical row positions,
identical base direction seed, and identical recurrence count.  The only intended
topology difference is that one-way arms see the same direction twice for each
selected pair, while reciprocal arms see the base direction once and the opposite
direction once.

This script does not train or evaluate a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
PAIRS_FILE = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
SELECTED_FILE = ROOT / "data/reciprocal_causal_transfer_scaffold_dosematched_shuffle/selected_reciprocal_pairs.jsonl"
FILLER_FILE = ROOT / "data/causal_transfer_scaffold/filler_rows.jsonl"
TOKENIZER_DIR = ROOT / "data/causal_transfer_scaffold/neutral_tokenizer"
OUT_DEFAULT = ROOT / "data/causal_topology_2x2_scaffold"

TARGET_WORDS = 10_000_000
ORDER_SEED = 171043
SEQ_LEN = 256


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def repeat_text(source: str, n_words: int) -> str:
    words = source.split()
    if not words:
        return ""
    if n_words <= len(words):
        return " ".join(words[:n_words])
    # Preserve exact legal word count even in rare too-short cases.
    tiled = (words * ((n_words // len(words)) + 2))[:n_words]
    return " ".join(tiled)


def norm_word(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w.lower())


def bag_overlap_fraction(a_text: str, b_text: str) -> dict[str, float | int | None]:
    # Fraction of words in a_text that can be matched by words in b_text as a multiset.
    a = [norm_word(x) for x in a_text.split()]
    b = [norm_word(x) for x in b_text.split()]
    a = [x for x in a if x]
    b = [x for x in b if x]
    if not a:
        return {"n": 0, "matched": 0, "frac": None}
    cb = Counter(b)
    matched = 0
    for x in a:
        if cb[x] > 0:
            matched += 1
            cb[x] -= 1
    return {"n": len(a), "matched": matched, "frac": matched / len(a)}


def summary_stats(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    ys = sorted(xs)
    return {
        "n": len(ys),
        "mean": float(sum(ys) / len(ys)),
        "median": float(statistics.median(ys)),
        "min": float(ys[0]),
        "max": float(ys[-1]),
    }


def make_text(source: str, view: str, order: str) -> str:
    if order == "source_first":
        return source + " " + view
    if order == "view_first":
        return view + " " + source
    raise ValueError(order)


def opposite(order: str) -> str:
    if order == "source_first":
        return "view_first"
    if order == "view_first":
        return "source_first"
    raise ValueError(order)


def load_selected_pairs(pairs_file: pathlib.Path, selected_file: pathlib.Path) -> list[dict[str, Any]]:
    all_pairs = read_jsonl(pairs_file)
    by_id: dict[str, dict[str, Any]] = {}
    for original_idx, p in enumerate(all_pairs):
        pid = p.get("pair_id")
        if pid in by_id:
            raise RuntimeError(f"duplicate pair_id {pid}")
        q0 = dict(p)
        q0["original_pair_index"] = original_idx
        by_id[pid] = q0
    selected = read_jsonl(selected_file)
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for i, s in enumerate(selected):
        pid = s["pair_id"]
        p = by_id.get(pid)
        if p is None:
            missing.append(pid)
            continue
        q = dict(p)
        q["selected_order_index"] = i
        q["selected_record"] = s
        out.append(q)
    if missing:
        raise RuntimeError(f"missing selected pair ids: {missing[:5]} ... n={len(missing)}")
    if len(out) != len(selected):
        raise RuntimeError(f"selected length mismatch {len(out)} != {len(selected)}")
    return out


def truncate_filler(filler_file: pathlib.Path, filler_needed: int) -> list[dict[str, Any]]:
    filler = read_jsonl(filler_file)
    out: list[dict[str, Any]] = []
    total = 0
    for r in filler:
        text = r["text"]
        w = int(r.get("words") or wc(text))
        if total + w > filler_needed:
            remain = filler_needed - total
            if remain <= 0:
                break
            out.append({
                "text": " ".join(text.split()[:remain]),
                "words": remain,
                "source": f"shared_filler_truncated_for_topology2x2::{r.get('source', 'unknown')}",
            })
            total += remain
            break
        out.append({"text": text, "words": w, "source": r.get("source", "shared_filler")})
        total += w
    if total != filler_needed:
        raise RuntimeError(f"not enough filler: {total} != {filler_needed}")
    return out


def interleave(filler_rows: list[dict[str, Any]], pair_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_units = len(pair_units)
    stride = max(1, len(filler_rows) // max(1, n_units))
    unit_idx = 0
    for i, fr in enumerate(filler_rows):
        rows.append(dict(fr))
        while unit_idx < n_units and i == min(len(filler_rows) - 1, unit_idx * stride):
            rows.append(pair_units[unit_idx])
            unit_idx += 1
    while unit_idx < n_units:
        rows.append(pair_units[unit_idx])
        unit_idx += 1
    return rows


def arm_units(selected_pairs: list[dict[str, Any]], base_orders: list[str], semantic: str, topology: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for p, base in zip(selected_pairs, base_orders):
        source = p["source_text"]
        rewrite = p["rewrite_text"]
        rw = int(p.get("rewrite_words") or wc(rewrite))
        if wc(rewrite) != rw:
            rw = wc(rewrite)
        if semantic == "compact":
            view = rewrite
            view_kind = "compact_view"
        elif semantic == "repeat":
            view = repeat_text(source, rw)
            view_kind = "exact_repeat"
            if wc(view) != rw:
                raise RuntimeError(f"repeat word mismatch for {p.get('pair_id')}: {wc(view)} != {rw}")
        else:
            raise ValueError(semantic)
        orders = [base, base] if topology == "oneway" else [base, opposite(base)]
        for occ, order in enumerate(orders):
            text = make_text(source, view, order)
            expected_words = wc(source) + wc(view)
            units.append({
                "text": text,
                "words": expected_words,
                "source": f"topology2x2_{semantic}_{topology}_{order}",
                "pair_id": p.get("pair_id"),
                "selected_order_index": p.get("selected_order_index"),
                "semantic_arm": semantic,
                "topology_arm": topology,
                "view_kind": view_kind,
                "base_order": base,
                "realized_order": order,
                "occurrence_index": occ,
            })
    return units


def token_stats_for_rows(rows: list[dict[str, Any]], tokenizer: Any, seq_len: int, batch_sizes: list[int]) -> dict[str, Any]:
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.convert_tokens_to_ids("</s>")
    raw_tokens = 0
    words = 0
    row_max = 0
    rows_over = 0
    for r in rows:
        ids = tokenizer.encode(r["text"])
        n = len(ids) + 1  # EOS in trainer
        raw_tokens += n
        words += int(r.get("words") or wc(r["text"]))
        row_max = max(row_max, len(ids))
        if len(ids) > seq_len:
            rows_over += 1
    active = (raw_tokens // seq_len) * seq_len
    chunks = active // seq_len
    return {
        "rows": len(rows),
        "words": words,
        "raw_tokens_including_eos": raw_tokens,
        "active_tokens": active,
        "dropped_tail_tokens": raw_tokens - active,
        "chunks": chunks,
        "token_per_word_including_eos": raw_tokens / words if words else None,
        "max_row_tokens_without_eos": row_max,
        "rows_over_seq_len_without_eos": rows_over,
        "steps_per_epoch_by_batch": {str(bs): (chunks + bs - 1) // bs for bs in batch_sizes},
        "steps_10epochs_by_batch": {str(bs): 10 * ((chunks + bs - 1) // bs) for bs in batch_sizes},
    }


def is_pair_unit(row: dict[str, Any]) -> bool:
    return "semantic_arm" in row and "topology_arm" in row and str(row.get("source", "")).startswith("topology2x2_")


def pair_positions(rows: list[dict[str, Any]]) -> list[int]:
    return [i for i, r in enumerate(rows) if is_pair_unit(r)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=PAIRS_FILE)
    ap.add_argument("--selected", type=pathlib.Path, default=SELECTED_FILE)
    ap.add_argument("--filler", type=pathlib.Path, default=FILLER_FILE)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=TOKENIZER_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--target-words", type=int, default=TARGET_WORDS)
    ap.add_argument("--order-seed", type=int, default=ORDER_SEED)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--skip-token-stats", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_pairs = load_selected_pairs(args.pairs, args.selected)
    # Recover the exact per-pair order assignment that research would have used
    # over the full original pair list, then subset it by pair_id.  This keeps
    # the one-way cells aligned with the prior causal coordinate while the 2x2
    # itself remains internally matched.
    n_full_pairs = sum(1 for _ in args.pairs.open("r", encoding="utf-8") if _.strip())
    rng = random.Random(args.order_seed)
    full_orders = ["source_first" if rng.random() < 0.5 else "view_first" for _ in range(n_full_pairs)]
    base_orders = [full_orders[int(p["original_pair_index"])] for p in selected_pairs]
    order_counts = Counter(base_orders)

    selected_records: list[dict[str, Any]] = []
    pair_word_total = 0
    rewrite_copy_fracs: list[float] = []
    for p, order in zip(selected_pairs, base_orders):
        sw = wc(p["source_text"])
        rw = wc(p["rewrite_text"])
        dual_words = 2 * (sw + rw)
        pair_word_total += dual_words
        overlap = bag_overlap_fraction(p["rewrite_text"], p["source_text"])
        if overlap["frac"] is not None:
            rewrite_copy_fracs.append(float(overlap["frac"]))
        selected_records.append({
            "selected_order_index": p["selected_order_index"],
            "pair_id": p.get("pair_id"),
            "base_order": order,
            "original_pair_index": p.get("original_pair_index"),
            "source_words": sw,
            "rewrite_words": rw,
            "dual_pair_words": dual_words,
            "rewrite_word_multiset_overlap_with_source": overlap,
        })
    if pair_word_total >= args.target_words:
        raise RuntimeError(f"pair word total too large: {pair_word_total}")
    filler_needed = args.target_words - pair_word_total
    filler_rows = truncate_filler(args.filler, filler_needed)

    arms: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, pathlib.Path] = {}
    for semantic in ["compact", "repeat"]:
        for topology in ["oneway", "reciprocal"]:
            label = f"{semantic}_{topology}"
            units = arm_units(selected_pairs, base_orders, semantic, topology)
            rows = interleave(filler_rows, units)
            words = sum(int(r.get("words") or wc(r["text"])) for r in rows)
            if words != args.target_words:
                raise RuntimeError(f"{label} word mismatch {words} != {args.target_words}")
            arms[label] = rows
            paths[label] = args.out_dir / f"causal_topology2x2_{label}_10M.jsonl"
            write_jsonl(paths[label], rows)

    write_jsonl(args.out_dir / "selected_topology2x2_pairs.jsonl", selected_records)

    pos_by_arm = {label: pair_positions(rows) for label, rows in arms.items()}
    all_positions_match = len({tuple(v) for v in pos_by_arm.values()}) == 1
    row_words_by_arm = {label: [int(r.get("words") or wc(r["text"])) for r in rows] for label, rows in arms.items()}
    all_row_words_match = len({tuple(v) for v in row_words_by_arm.values()}) == 1

    manifest: dict[str, Any] = {
        "status": "CAUSAL_TOPOLOGY_2X2_SCAFFOLD_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": None,
        "scientific_intent": "prepare a matched compact-vs-repeat by one-way-vs-reciprocal causal interaction test without launching training",
        "training_launched": False,
        "target_words_per_arm": args.target_words,
        "order_seed": args.order_seed,
        "base_order_source": "research full original-pair order assignment, subset by selected pair_id",
        "selected_pairs_source": str(args.selected),
        "selected_pairs_source_sha256": sha256_file(args.selected),
        "pair_records_source": str(args.pairs),
        "pair_records_source_sha256": sha256_file(args.pairs),
        "filler_source": str(args.filler),
        "filler_source_sha256": sha256_file(args.filler),
        "neutral_tokenizer": str(args.tokenizer),
        "neutral_tokenizer_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
        "selected_pairs": len(selected_pairs),
        "base_order_counts": dict(order_counts),
        "pair_units_per_arm": 2 * len(selected_pairs),
        "pair_words_per_arm": pair_word_total,
        "filler_words_per_arm": filler_needed,
        "filler_rows_per_arm": len(filler_rows),
        "rewrite_copy_fraction_word_multiset_stats": summary_stats(rewrite_copy_fracs),
        "arms": {},
        "matched_structure": {
            "all_pair_positions_match": all_positions_match,
            "pair_positions_count": len(next(iter(pos_by_arm.values()))),
            "first_pair_positions": next(iter(pos_by_arm.values()))[:20],
            "last_pair_positions": next(iter(pos_by_arm.values()))[-20:],
            "all_row_word_sequences_match": all_row_words_match,
            "row_count_by_arm": {label: len(rows) for label, rows in arms.items()},
            "words_by_arm": {label: sum(row_words_by_arm[label]) for label in arms},
        },
        "topology_definition": {
            "oneway": "each selected pair appears twice with the same base direction determined by order_seed",
            "reciprocal": "each selected pair appears once in the same base direction and once in the opposite direction",
            "semantic_compact": "source plus generated compact rewrite from the legal compact-view pair file",
            "semantic_repeat": "source plus first-N-word exact source repeat where N equals compact rewrite word count",
        },
        "interpretation": [
            "This construction is not evidence that reciprocal causal training works; it only removes known design confounds if the direction is later revived.",
            "A future run should start as a short matched four-arm screen, not a pair of full 100M trainings.",
            "The key quantity would be a semantic-by-topology interaction: compact_reciprocal should improve over compact_oneway more than repeat_reciprocal improves over repeat_oneway, and the advantage must be broad across official families rather than copied-token or GlobalPIQA/Reading volatility.",
        ],
    }

    for label, path in paths.items():
        rows = arms[label]
        pair_src_counts = Counter(str(r.get("source", "")) for r in rows if is_pair_unit(r))
        manifest["arms"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "words": sum(int(r.get("words") or wc(r["text"])) for r in rows),
            "pair_source_counts": dict(pair_src_counts),
        }

    if not args.skip_token_stats:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for label, rows in arms.items():
            manifest["arms"][label]["token_stats"] = token_stats_for_rows(rows, tok, args.seq_len, [128, 256])
        # Difference summaries useful for choosing batch size if this ever trains.
        def delta(a: str, b: str, key: str) -> int | float:
            return manifest["arms"][a]["token_stats"][key] - manifest["arms"][b]["token_stats"][key]
        manifest["token_comparisons"] = {
            "compact_minus_repeat_within_oneway": {
                "raw_tokens": delta("compact_oneway", "repeat_oneway", "raw_tokens_including_eos"),
                "active_tokens": delta("compact_oneway", "repeat_oneway", "active_tokens"),
                "chunks": delta("compact_oneway", "repeat_oneway", "chunks"),
            },
            "compact_minus_repeat_within_reciprocal": {
                "raw_tokens": delta("compact_reciprocal", "repeat_reciprocal", "raw_tokens_including_eos"),
                "active_tokens": delta("compact_reciprocal", "repeat_reciprocal", "active_tokens"),
                "chunks": delta("compact_reciprocal", "repeat_reciprocal", "chunks"),
            },
            "reciprocal_minus_oneway_within_compact": {
                "raw_tokens": delta("compact_reciprocal", "compact_oneway", "raw_tokens_including_eos"),
                "active_tokens": delta("compact_reciprocal", "compact_oneway", "active_tokens"),
                "chunks": delta("compact_reciprocal", "compact_oneway", "chunks"),
            },
            "reciprocal_minus_oneway_within_repeat": {
                "raw_tokens": delta("repeat_reciprocal", "repeat_oneway", "raw_tokens_including_eos"),
                "active_tokens": delta("repeat_reciprocal", "repeat_oneway", "active_tokens"),
                "chunks": delta("repeat_reciprocal", "repeat_oneway", "chunks"),
            },
        }

    manifest["elapsed_sec"] = round(time.time() - t0, 3)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# research causal topology 2×2 scaffold\n\n")
    md.append("No training was launched. This is a matched construction for a possible later short four-arm screen.\n\n")
    md.append(f"Selected pairs: {len(selected_pairs):,}; pair units/arm: {2*len(selected_pairs):,}; pair words/arm: {pair_word_total:,}; filler words/arm: {filler_needed:,}; legal words/arm: {args.target_words:,}.\n\n")
    md.append(f"Base direction counts from seed {args.order_seed}: source_first={order_counts.get('source_first',0)}, view_first={order_counts.get('view_first',0)}.\n\n")
    md.append(f"Matched pair positions across four arms: {all_positions_match}; matched row word sequences: {all_row_words_match}.\n\n")
    md.append("| Arm | Rows | Words | SHA256 | Raw toks | Active toks | Chunks | steps/epoch b256 |\n")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|\n")
    for label in ["compact_oneway", "repeat_oneway", "compact_reciprocal", "repeat_reciprocal"]:
        arm = manifest["arms"][label]
        ts = arm.get("token_stats", {})
        md.append(
            f"| {label} | {arm['rows']} | {arm['words']} | `{arm['sha256']}` | "
            f"{ts.get('raw_tokens_including_eos','')} | {ts.get('active_tokens','')} | {ts.get('chunks','')} | "
            f"{(ts.get('steps_per_epoch_by_batch') or {}).get('256','')} |\n"
        )
    if "token_comparisons" in manifest:
        md.append("\n## Token comparisons\n\n")
        for name, vals in manifest["token_comparisons"].items():
            md.append(f"- {name}: raw {vals['raw_tokens']:+}, active {vals['active_tokens']:+}, chunks {vals['chunks']:+}.\n")
    md.append("\n## Scientific use\n\n")
    md.append("Use only if independent DeBERTa evidence revives the topology interaction. The first real test should be a short matched four-arm run, not a full 100M pair, and the readout should require broad official-family movement rather than copied-token lift or GlobalPIQA/Reading volatility.\n\n")
    md.append(f"JSON: `{manifest_path}`\n")
    (args.out_dir / "manifest.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "arms": {k: {"sha256": v["sha256"], "rows": v["rows"], "words": v["words"]} for k, v in manifest["arms"].items()},
        "all_pair_positions_match": all_positions_match,
        "all_row_word_sequences_match": all_row_words_match,
        "token_comparisons": manifest.get("token_comparisons"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
