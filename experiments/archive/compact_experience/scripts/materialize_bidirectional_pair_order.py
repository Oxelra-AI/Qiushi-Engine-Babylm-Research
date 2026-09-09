#!/usr/bin/env python3
"""research: bidirectional same-window Qwen pair order materializer.

Builds a clean structural variant of the research clean-Qwen corpus:
  - exact same 10M row sequence and 100M pass/order sequence as research qwen_aligned;
  - exact same selected pair set, pair dose, filler rows, example IDs, source labels, word totals;
  - changes only within-pair segment order for half the packed pair rows/pairs, using a fixed
    prospective hash/parity rule: some pairs are original->rewrite, others rewrite->original.

Purpose: test whether the active same-window generated second-view correspondence is made
more robust by removing one-direction positional bias while preserving the mechanism.  This
uses only training corpus metadata and generated rewrites already counted in the clean-Qwen
pool.  It does not read official AoA/CDI words, child curves, AoA predictions, or AoA scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import hashlib
import json
import pathlib
from typing import Any, Dict, List

ROOT = _public_path('experiments/archive/compact_experience')
SRC_DIR = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
IN10 = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
IN100 = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
PAIR_META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl')
SELECTED = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
META_IN = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/bidirectional_pair_order')
OUT10 = _public_path('experiments/archive/compact_experience/data/bidirectional_pair_order/qwen_bidirectional_pair_order_10M.jsonl')
OUT100 = _public_path('experiments/archive/compact_experience/data/bidirectional_pair_order/qwen_bidirectional_pair_order_100M.jsonl')
OUT_META = _public_path('experiments/archive/compact_experience/data/bidirectional_pair_order/bidirectional_pair_order_metadata.json')


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_digest(obj: Dict[str, Any]) -> str:
    # Digest row identity ignoring text, so the structural edit can be audited separately.
    s = json.dumps({"words": obj.get("words"), "example_id": obj.get("example_id"), "source": obj.get("source")}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def choose_direction(pair_id: str) -> str:
    # Stable prospective 50/50-ish split, independent of evaluation outputs.
    b = hashlib.sha256(pair_id.encode("utf-8")).digest()[0]
    return "orig_rewrite" if (b % 2 == 0) else "rewrite_orig"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_in = json.loads(META_IN.read_text(encoding="utf-8"))
    if meta_in.get("status") != "CLEAN_QWEN_CORPORA_MATERIALIZED":
        raise SystemExit(f"unexpected input metadata: {meta_in.get('status')}")

    selected: Dict[str, Dict[str, Any]] = {}
    with SELECTED.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pid = str(p["pair_id"])
            selected[pid] = p
    if not selected:
        raise RuntimeError("no selected pairs")

    rows10 = read_jsonl(IN10)
    if sum(int(r["words"]) for r in rows10) != 10_000_000:
        raise RuntimeError("input 10M word total mismatch")
    metas = []
    with PAIR_META.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    meta_by_row = {int(m["row_index"]): m for m in metas}
    if len(meta_by_row) != len(metas):
        raise RuntimeError("duplicate pair meta row index")

    replacements_by_example: Dict[int, Dict[str, Any]] = {}
    direction_counts = {"orig_rewrite": 0, "rewrite_orig": 0}
    direction_words = {"orig_rewrite": 0, "rewrite_orig": 0}
    changed_rows = 0
    pair_words = 0

    rows10_new: List[Dict[str, Any]] = []
    for i, obj in enumerate(rows10):
        new = dict(obj)
        m = meta_by_row.get(i)
        if m is not None:
            segments: List[str] = []
            total = 0
            for pid in m["pair_ids"]:
                p = selected[str(pid)]
                ow = int(p["original_words"]); rw = int(p["rewrite_words"])
                direction = choose_direction(str(pid))
                direction_counts[direction] += 1
                direction_words[direction] += ow + rw
                if direction == "orig_rewrite":
                    segments.extend([p["original"], p["rewrite"]])
                else:
                    segments.extend([p["rewrite"], p["original"]])
                total += ow + rw
            text = " ".join(segments)
            if len(text.split()) != int(new["words"]) or total != int(new["words"]):
                raise RuntimeError(f"word mismatch replacing row {i}: meta {total} row {new['words']} text {len(text.split())}")
            if text != new["text"]:
                changed_rows += 1
            pair_words += total
            new["text"] = text
            replacements_by_example[int(new["example_id"])] = dict(new)
        rows10_new.append(new)

    if pair_words != int(meta_in.get("selected_pair_words", pair_words)):
        raise RuntimeError(f"pair word mismatch {pair_words} vs metadata {meta_in.get('selected_pair_words')}")
    if sum(int(r["words"]) for r in rows10_new) != 10_000_000:
        raise RuntimeError("output 10M word total mismatch")
    if [row_digest(r) for r in rows10_new] != [row_digest(r) for r in rows10]:
        raise RuntimeError("row identities changed")

    with OUT10.open("w", encoding="utf-8") as f:
        for r in rows10_new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Rewrite the original 100M stream in-place by pair-row example_id, preserving pass order.
    total100 = 0
    replaced100 = 0
    with IN100.open("r", encoding="utf-8") as fin, OUT100.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            exid = int(obj["example_id"])
            if exid in replacements_by_example:
                obj = replacements_by_example[exid]
                replaced100 += 1
            total100 += int(obj["words"])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if total100 != 100_000_000:
        raise RuntimeError(f"100M total mismatch: {total100}")
    if replaced100 != len(meta_by_row) * 10:
        raise RuntimeError(f"unexpected replacement count in 100M: {replaced100} vs {len(meta_by_row)*10}")

    payload = {
        "status": "BIDIRECTIONAL_PAIR_ORDER_MATERIALIZED",
        "non_leakage_statement": "Uses only research selected-pair training data and pair metadata. No official AoA/CDI words, child curves, AoA outputs, or evaluation results are read or used.",
        "input_10M": str(IN10),
        "input_100M": str(IN100),
        "input_metadata": str(META_IN),
        "output_10M": str(OUT10),
        "output_100M": str(OUT100),
        "word_total_10M": 10_000_000,
        "word_total_100M": 100_000_000,
        "row_count_10M": len(rows10_new),
        "pair_rows": len(meta_by_row),
        "pair_words": pair_words,
        "pair_word_fraction": pair_words / 10_000_000,
        "changed_pair_rows": changed_rows,
        "row_identity_sequence_preserved": True,
        "hundred_million_pass_order_preserved": True,
        "direction_rule": "sha256(pair_id)[0] parity: even original->rewrite, odd rewrite->original",
        "direction_counts": direction_counts,
        "direction_words": direction_words,
        "sha256_10M": sha_file(OUT10),
        "sha256_100M": sha_file(OUT100),
        "intended_mechanism": "preserve same-window generated second-view correspondence while removing one-direction positional bias, testing whether symmetry improves broad transfer/Reading/COMPS without losing Entity/SuperGLUE/Supplement",
        "baseline_runs": [
            "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022",
            "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122"
        ]
    }
    OUT_META.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(OUT_META), "pair_word_fraction": payload["pair_word_fraction"], "direction_counts": direction_counts, "direction_words": direction_words, "sha256_100M": payload["sha256_100M"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
