#!/usr/bin/env python3
"""research: build dual-order reciprocal causal compact-vs-repeat pools.

Purpose: convert the causal-transfer negative into an objective-compatible test.
The research causal pools assigned each source/compact pair to one direction only.
A decoder-only model therefore never receives reciprocal conditioning for the
same semantic pair. This script constructs matched legal 10M pools in which each
selected compact pair appears twice: S->C and C->S. The matched repeat control
appears twice with source->repeat and repeat->source, where repeat is the first
N source words matching the compact view's word count.

The script selects the largest prefix of pair records that fits with a shared
filler prefix inside exactly 10M words. It keeps the neutral filler-only tokenizer
from research. This is a candidate construction; training should wait until the
pending DeBERTa trajectory comparison and Lead/Reviewer judgment.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random, statistics, time
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
FILLER = ROOT / "data/causal_transfer_scaffold/filler_rows.jsonl"
OUT_DEFAULT = ROOT / "data/reciprocal_causal_transfer_scaffold"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repeat_text(source: str, n_words: int) -> str:
    words = source.split()
    return " ".join(words[: min(n_words, len(words))])


def stats(xs: list[float]) -> dict[str, float | int | None]:
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


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--filler", default=str(FILLER))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--target-words", type=int, default=10_000_000)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 means as many as fit")
    ap.add_argument("--shuffle-seed", type=int, default=-1, help="shuffle pair order before selection when >=0")
    ap.add_argument("--min-dual-pair-words", type=int, default=0, help="stop after reaching at least this many dual-order pair words")
    args = ap.parse_args()
    t0 = time.time()
    pairs_path = pathlib.Path(args.pairs)
    filler_path = pathlib.Path(args.filler)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(pairs_path)
    if args.shuffle_seed >= 0:
        rng = random.Random(args.shuffle_seed)
        pairs = list(pairs)
        rng.shuffle(pairs)
    filler = read_jsonl(filler_path)
    compact_pair_units: list[dict[str, Any]] = []
    repeat_pair_units: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    total_pair_words = 0
    pair_limit = len(pairs) if args.max_pairs <= 0 else min(args.max_pairs, len(pairs))
    for i, r in enumerate(pairs[:pair_limit]):
        if args.min_dual_pair_words > 0 and total_pair_words >= args.min_dual_pair_words:
            break
        source = r["source_text"]
        rewrite = r["rewrite_text"]
        rw = int(r.get("rewrite_words") or wc(rewrite))
        rep = repeat_text(source, rw)
        sw = int(r.get("source_words") or wc(source))
        cw = wc(rewrite)
        repw = wc(rep)
        if cw != rw:
            # Trust actual text words for legal accounting; preserve discrepancy.
            rw = cw
        if repw != rw:
            # If source is too short (should be rare/none), skip to keep row-wise matching exact.
            continue
        unit_words = 2 * (sw + cw)
        if total_pair_words + unit_words >= args.target_words:
            break
        total_pair_words += unit_words
        pair_id = r.get("pair_id", f"pair_{i}")
        selected_pairs.append({
            "idx": i,
            "pair_id": pair_id,
            "source_words": sw,
            "rewrite_words": cw,
            "dual_order_words": unit_words,
        })
        compact_pair_units.append({
            "text": source + " " + rewrite,
            "words": sw + cw,
            "source": "reciprocal_compact_source_to_view",
            "pair_id": pair_id,
            "direction": "source_to_view",
        })
        compact_pair_units.append({
            "text": rewrite + " " + source,
            "words": cw + sw,
            "source": "reciprocal_compact_view_to_source",
            "pair_id": pair_id,
            "direction": "view_to_source",
        })
        repeat_pair_units.append({
            "text": source + " " + rep,
            "words": sw + repw,
            "source": "reciprocal_repeat_source_to_repeat",
            "pair_id": pair_id,
            "direction": "source_to_repeat",
        })
        repeat_pair_units.append({
            "text": rep + " " + source,
            "words": repw + sw,
            "source": "reciprocal_repeat_repeat_to_source",
            "pair_id": pair_id,
            "direction": "repeat_to_source",
        })

    if args.min_dual_pair_words > 0 and total_pair_words < args.min_dual_pair_words:
        raise RuntimeError(f"selected only {total_pair_words} dual pair words, below requested {args.min_dual_pair_words}")

    filler_needed = args.target_words - total_pair_words
    if filler_needed <= 0:
        raise RuntimeError("pair units exceed target words")
    filler_rows: list[dict[str, Any]] = []
    fwords = 0
    for r in filler:
        w = int(r.get("words") or wc(r["text"]))
        if fwords + w > filler_needed:
            remain = filler_needed - fwords
            if remain <= 0:
                break
            words = r["text"].split()[:remain]
            filler_rows.append({
                "text": " ".join(words),
                "words": remain,
                "source": f"reciprocal_shared_filler_truncated::{r.get('source','unknown')}",
            })
            fwords += remain
            break
        filler_rows.append({"text": r["text"], "words": w, "source": r.get("source", "shared_filler")})
        fwords += w
    if fwords != filler_needed:
        raise RuntimeError(f"not enough filler: {fwords} != {filler_needed}")

    # Interleave pair units with shared filler chunks deterministically, preserving paired positions identical across arms.
    # Use an anchor stride over filler rows so pair units are spread through the pool rather than all contiguous.
    compact_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    n_units = len(compact_pair_units)
    stride = max(1, len(filler_rows) // max(1, n_units))
    unit_idx = 0
    for i, fr in enumerate(filler_rows):
        compact_rows.append(fr)
        repeat_rows.append(dict(fr))
        while unit_idx < n_units and i == min(len(filler_rows)-1, unit_idx * stride):
            compact_rows.append(compact_pair_units[unit_idx])
            repeat_rows.append(repeat_pair_units[unit_idx])
            unit_idx += 1
    while unit_idx < n_units:
        compact_rows.append(compact_pair_units[unit_idx])
        repeat_rows.append(repeat_pair_units[unit_idx])
        unit_idx += 1

    compact_words = sum(int(r["words"]) for r in compact_rows)
    repeat_words = sum(int(r["words"]) for r in repeat_rows)
    if compact_words != args.target_words or repeat_words != args.target_words:
        raise RuntimeError(f"word mismatch compact={compact_words} repeat={repeat_words}")
    compact_path = out_dir / "causal_reciprocal_compact_10M.jsonl"
    repeat_path = out_dir / "causal_reciprocal_repeat_10M.jsonl"
    sel_path = out_dir / "selected_reciprocal_pairs.jsonl"
    write_jsonl(compact_path, compact_rows)
    write_jsonl(repeat_path, repeat_rows)
    write_jsonl(sel_path, selected_pairs)

    pair_pos_c = [i for i, r in enumerate(compact_rows) if str(r.get("source", "")).startswith("reciprocal_compact")]
    pair_pos_r = [i for i, r in enumerate(repeat_rows) if str(r.get("source", "")).startswith("reciprocal_repeat")]
    text_same = sum(1 for a, b in zip(compact_rows, repeat_rows) if a["text"] == b["text"])
    word_delta = [int(a["words"]) - int(b["words"]) for a, b in zip(compact_rows, repeat_rows)]
    result = {
        "status": "RECIPROCAL_CAUSAL_SCAFFOLD_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 3),
        "scientific_intent": "causal-compatible dual-order reciprocal S<->compact-view exposure for each selected pair, matched to exact dual-order repetition control",
        "target_words": args.target_words,
        "source_pairs": str(pairs_path),
        "source_pairs_sha256": sha256_file(pairs_path),
        "source_filler": str(filler_path),
        "source_filler_sha256": sha256_file(filler_path),
        "shuffle_seed": args.shuffle_seed,
        "min_dual_pair_words": args.min_dual_pair_words,
        "selected_pairs": len(selected_pairs),
        "pair_units_per_arm": n_units,
        "total_pair_words_per_arm": total_pair_words,
        "filler_words_per_arm": fwords,
        "filler_rows": len(filler_rows),
        "compact": {"path": str(compact_path), "rows": len(compact_rows), "words": compact_words, "sha256": sha256_file(compact_path)},
        "repeat": {"path": str(repeat_path), "rows": len(repeat_rows), "words": repeat_words, "sha256": sha256_file(repeat_path)},
        "selected_pairs_path": str(sel_path),
        "selected_pairs_sha256": sha256_file(sel_path),
        "pair_positions_match": pair_pos_c == pair_pos_r,
        "pair_positions_count": len(pair_pos_c),
        "first_pair_positions": pair_pos_c[:20],
        "last_pair_positions": pair_pos_c[-20:],
        "same_text_rows": text_same,
        "different_text_rows": len(compact_rows) - text_same,
        "row_word_delta_stats_compact_minus_repeat": stats([float(x) for x in word_delta]),
        "selected_pair_dual_word_stats": stats([float(r["dual_order_words"]) for r in selected_pairs]),
        "interpretation": [
            "This is a candidate scaffold only; it should not be trained before pending DeBERTa common-grid evidence is interpreted.",
            "Compared with research causal pools, this doubles each selected pair directionally and therefore uses more pair-budget but fewer distinct pairs/filler words.",
            "A valid training test would compare reciprocal_compact against reciprocal_repeat at matched legal words, tokenizer, architecture, seed, and selected checkpoints."
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research reciprocal causal transfer scaffold", ""]
    lines.append(f"Selected pairs: {len(selected_pairs)}; pair units/arm: {n_units}; pair words/arm: {total_pair_words:,}; filler words/arm: {fwords:,}")
    lines.append(f"Compact: `{compact_path}` SHA `{result['compact']['sha256']}`")
    lines.append(f"Repeat: `{repeat_path}` SHA `{result['repeat']['sha256']}`")
    lines.append(f"Rows compact/repeat: {len(compact_rows)} / {len(repeat_rows)}; row word delta mean {result['row_word_delta_stats_compact_minus_repeat']['mean']}")
    lines.append(f"Pair positions match: {result['pair_positions_match']} ({len(pair_pos_c)} positions)")
    lines.append("")
    lines.append("Scientific use: objective-compatible reciprocal causal test; not launched.")
    lines.append("")
    lines.append(f"JSON: `{out_dir / 'manifest.json'}`")
    (out_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(out_dir), "selected_pairs": len(selected_pairs), "compact_sha256": result["compact"]["sha256"], "repeat_sha256": result["repeat"]["sha256"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
