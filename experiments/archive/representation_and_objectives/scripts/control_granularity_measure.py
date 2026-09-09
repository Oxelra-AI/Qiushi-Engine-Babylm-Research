#!/usr/bin/env python3
"""Pair-only target granularity measure for research copied-label deletion control."""
from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(".").resolve()
SCRIPTS = ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossview_deberta_trainer_v2 as base  # noqa: E402
import target_selective_deberta_trainer as ts  # noqa: E402

OUT_DIR = ROOT / "experiments/archive/representation_and_objectives/data/control_granularity_measure"
DATA = "experiments/archive/representation_and_objectives/data/crossview_data_v2/crossview_consolidated_v2.jsonl"
TOKENIZER = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
DROP_K = 7649
SELECTION_SEED = 22543023


def contiguous_groups(pos: list[int]) -> list[list[int]]:
    if not pos:
        return []
    out = []
    cur = [pos[0]]
    for p in pos[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            out.append(cur)
            cur = [p]
    out.append(cur)
    return out


def q(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = base.make_portable_tokenizer(TOKENIZER)
    pool = base.load_pool(DATA)
    examples, aw, ep = base.build_stream(pool, 20_000_000, "own", 43)
    full = base.PairAwareDataset(examples, tok, 256, "own", False, 0.15, 430230221)

    # Select copied token positions exactly as research did, but skip tokenizing filler rows.
    scored = []
    copied_group_sizes = []
    copied_group_positions = []
    abs_group_sizes = []
    copied_piece_count = 0
    abs_piece_count = 0
    pair_examples = 0
    for i, ex in enumerate(examples):
        if ex.get("type") != "pair":
            continue
        pair_examples += 1
        item = full[i]
        labels = item["labels"]
        st = item["target_stratum"]
        m = labels != -100
        copied_positions = [int(x) for x in ((m & (st == base.S_RW_COPIED)).nonzero(as_tuple=False).view(-1).tolist())]
        abs_positions = [int(x) for x in ((m & (st == base.S_RW_ABS_CONTENT)).nonzero(as_tuple=False).view(-1).tolist())]
        copied_piece_count += len(copied_positions)
        abs_piece_count += len(abs_positions)
        for p in copied_positions:
            scored.append((ts.stable_u64(ts.target_key(i, p, "drop_copied_matched", SELECTION_SEED)), i, p))
        for g in contiguous_groups(copied_positions):
            gi = len(copied_group_sizes)
            copied_group_sizes.append(len(g))
            copied_group_positions.append((i, tuple(g)))
        for g in contiguous_groups(abs_positions):
            abs_group_sizes.append(len(g))
    scored.sort(key=lambda x: x[0])
    drop_positions = {(i, p) for _, i, p in scored[:DROP_K]}
    touched = []
    fully = []
    partial = []
    untouched = 0
    for size, (i, positions) in zip(copied_group_sizes, copied_group_positions):
        d = sum((i, p) in drop_positions for p in positions)
        if d == 0:
            untouched += 1
        elif d == size:
            fully.append((size, d))
            touched.append((size, d))
        else:
            partial.append((size, d))
            touched.append((size, d))
    payload = {
        "status": "CONTROL_GRANULARITY_MEASURE",
        "meaning": "Pair-only measure of whether the matched copied-label deletion removed whole copied target groups or partial groups. Groups are contiguous selected-token runs within copied/absent strata, so adjacent same-stratum selected words can be merged; partial copied deletion is therefore a conservative warning, not exact word_ids.",
        "inputs": {"data": DATA, "tokenizer": TOKENIZER, "word_exposure": aw, "epochs": ep, "pair_examples": pair_examples, "drop_k": DROP_K, "selection_seed": SELECTION_SEED},
        "pair_target_piece_counts": {"rw_copied": copied_piece_count, "rw_abs_content": abs_piece_count},
        "drop_positions": len(drop_positions),
        "copied_target_groups_contiguous": len(copied_group_sizes),
        "abs_target_groups_contiguous": len(abs_group_sizes),
        "copied_group_size_counts": dict(collections.Counter(copied_group_sizes)),
        "abs_group_size_counts": dict(collections.Counter(abs_group_sizes)),
        "copied_untouched_groups": untouched,
        "copied_touched_groups": len(touched),
        "copied_fully_dropped_groups": len(fully),
        "copied_partial_groups": len(partial),
        "fraction_touched_partial": len(partial) / max(1, len(touched)),
        "pieces_dropped_in_partial_groups": sum(d for s, d in partial),
        "pieces_dropped_in_fully_dropped_groups": sum(d for s, d in fully),
        "touched_group_size_stats": {"median": q([s for s, d in touched], 0.5), "p90": q([s for s, d in touched], 0.9), "max": max([s for s, d in touched]) if touched else None},
        "scientific_warning": "The research copied-drop arm matched deleted token-piece count, not whole-word target groups. If the source-absent channel remains important, rerun with a whole-word copied-drop control matched on piece count as closely as possible.",
    }
    out = OUT_DIR / "control_granularity_measure.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "pair_target_piece_counts", "drop_positions", "copied_touched_groups", "copied_fully_dropped_groups", "copied_partial_groups", "fraction_touched_partial", "pieces_dropped_in_partial_groups", "pieces_dropped_in_fully_dropped_groups", "scientific_warning"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
