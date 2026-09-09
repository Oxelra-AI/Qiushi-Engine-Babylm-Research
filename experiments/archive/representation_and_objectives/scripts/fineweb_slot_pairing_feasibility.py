#!/usr/bin/env python3
"""CPU-only feasibility study for a controlled FineWeb source/view slot family.

The corrected future family should not compare a view arm against an unrelated
source-breadth arm with different experience slots.  A cleaner unit is a source
slot S plus a second slot of the same length:

  A_natural : official BabyLM text of length len(S)+len(V)
  B_breadth : S + distinct live-FineWeb text D, len(D)=len(V)
  B_repeat  : S + literal repetition of S to len(V)
  C_view    : S + faithful generated view V

This script does not generate V and does not write training corpora.  It uses the
research 300k live-FineWeb seed to measure whether enough distinct D rows exist at
plausible view-length ratios, preserving different documents and optionally the
same relation focus.  The output is a contract/feasibility record for a later
materializer after real research/research results arrive.
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import Counter, defaultdict, deque
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SEED = ROOT / "data/stratified_live_fineweb_blueprint/live_fineweb_stratified_current_seed_300k.jsonl"
BLUEPRINT = ROOT / "data/stratified_live_fineweb_blueprint/stratified_live_fineweb_blueprint.json"
OUT_DIR = ROOT / "data/fineweb_slot_pairing_feasibility"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fineweb_slot_pairing_feasibility.md')
RATIOS = [0.75, 0.88, 0.95, 1.05]
BUDGETS = [100_000, 200_000, 300_000, 500_000, 1_000_000, 1_750_000]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_i"] = i
            r["words"] = int(r.get("words") or len(str(r.get("text", "")).split()))
            r["focus"] = str(r.get("focus") or "unknown")
            r["doc_id"] = str(r.get("doc_id") or "")
            rows.append(r)
    return rows


def target_len(words: int, ratio: float) -> int:
    # Views shorter than ten words become poor masked-LM/rewrite units; keep a
    # floor compatible with the research/020 source selectors.
    return max(10, int(round(words * ratio)))


def make_buckets(rows: list[dict[str, Any]], same_focus: bool) -> dict[Any, deque[int]]:
    buckets: dict[Any, deque[int]] = defaultdict(deque)
    for r in sorted(rows, key=lambda x: (int(x.get("blueprint_selection_rank", x["_i"])), x["_i"])):
        key = (r["focus"], r["words"]) if same_focus else r["words"]
        buckets[key].append(r["_i"])
    return buckets


def pop_match(
    rows_by_i: dict[int, dict[str, Any]],
    buckets: dict[Any, deque[int]],
    used: set[int],
    base: dict[str, Any],
    want_len: int,
    same_focus: bool,
    max_delta: int,
) -> tuple[dict[str, Any] | None, int | None]:
    candidate_keys = []
    for delta in range(0, max_delta + 1):
        for sign in ([0] if delta == 0 else [-1, 1]):
            length = want_len + sign * delta
            if length < 10:
                continue
            key = (base["focus"], length) if same_focus else length
            candidate_keys.append((key, abs(length - want_len)))
    for key, delta in candidate_keys:
        q = buckets.get(key)
        if not q:
            continue
        kept = deque()
        match = None
        while q:
            idx = q.popleft()
            cand = rows_by_i[idx]
            if idx in used or idx == base["_i"] or cand["doc_id"] == base["doc_id"]:
                kept.append(idx)
                continue
            match = cand
            break
        while kept:
            q.appendleft(kept.pop())
        if match is not None:
            return match, delta
    return None, None


def pair_rows(rows: list[dict[str, Any]], ratio: float, same_focus: bool, max_delta: int) -> dict[str, Any]:
    rows_by_i = {r["_i"]: r for r in rows}
    buckets = make_buckets(rows, same_focus=same_focus)
    used: set[int] = set()
    pairs = []
    skipped_no_match = 0
    for base in sorted(rows, key=lambda x: (int(x.get("blueprint_selection_rank", x["_i"])), x["_i"])):
        if base["_i"] in used:
            continue
        want = target_len(base["words"], ratio)
        match, delta = pop_match(rows_by_i, buckets, used, base, want, same_focus, max_delta)
        if match is None:
            skipped_no_match += 1
            continue
        used.add(base["_i"])
        used.add(match["_i"])
        pair = {
            "source_index": base["_i"],
            "distinct_index": match["_i"],
            "source_doc": base["doc_id"],
            "distinct_doc": match["doc_id"],
            "source_words": base["words"],
            "target_view_words": want,
            "distinct_words": match["words"],
            "length_delta_abs": delta,
            "packet_words": base["words"] + want,
            "source_focus": base["focus"],
            "distinct_focus": match["focus"],
            "source_pool": base.get("source_pool"),
            "source_class": base.get("blueprint_class"),
            "distinct_pool": match.get("source_pool"),
            "distinct_class": match.get("blueprint_class"),
        }
        pairs.append(pair)
    total_packet_words = sum(p["packet_words"] for p in pairs)
    source_words = sum(p["source_words"] for p in pairs)
    distinct_words = sum(p["distinct_words"] for p in pairs)
    target_words = sum(p["target_view_words"] for p in pairs)
    focus_packet_words = Counter()
    focus_rows = Counter()
    d_focus_words = Counter()
    class_words = Counter()
    for p in pairs:
        focus_packet_words[p["source_focus"]] += p["packet_words"]
        focus_rows[p["source_focus"]] += 1
        d_focus_words[p["distinct_focus"]] += p["distinct_words"]
        class_words[p["source_class"]] += p["source_words"]
    budget_capacity = {}
    running = 0
    count_at = {b: None for b in BUDGETS}
    for j, p in enumerate(pairs, start=1):
        running += p["packet_words"]
        for b in BUDGETS:
            if count_at[b] is None and running >= b:
                count_at[b] = j
    for b in BUDGETS:
        budget_capacity[str(b)] = {
            "reachable_from_current_seed": total_packet_words >= b,
            "pairs_needed_if_reachable": count_at[b],
            "coverage_fraction": round(min(1.0, total_packet_words / b), 6),
        }
    return {
        "ratio": ratio,
        "same_focus_distinct_slot": same_focus,
        "max_distinct_length_delta": max_delta,
        "pairs": len(pairs),
        "skipped_no_match": skipped_no_match,
        "used_rows": len(used),
        "unused_rows": len(rows) - len(used),
        "source_words": source_words,
        "target_view_words_simulated": target_words,
        "distinct_words": distinct_words,
        "c_view_or_b_repeat_packet_words": total_packet_words,
        "inventory_words_consumed_for_source_and_distinct_slots": source_words + distinct_words,
        "mean_packet_words": round(total_packet_words / len(pairs), 3) if pairs else None,
        "focus_packet_words": dict(focus_packet_words.most_common()),
        "focus_rows": dict(focus_rows.most_common()),
        "distinct_focus_words": dict(d_focus_words.most_common()),
        "source_class_words": dict(class_words.most_common()),
        "length_delta_counts": dict(Counter(p["length_delta_abs"] for p in pairs).most_common()),
        "budget_capacity": budget_capacity,
        "first_pairs": pairs[:20],
    }


def projection_from_seed(packet_words: int, inventory_words: int, seed_words: int) -> dict[str, Any]:
    if seed_words <= 0 or packet_words <= 0:
        return {}
    # Packet budget is the changed block in each arm. Inventory is the live-source
    # text needed to support S and D slots before generation.  This is not a
    # downstream-effect extrapolation.
    out = {}
    for b in BUDGETS:
        out[str(b)] = {
            "source_inventory_words_needed_at_observed_pairing_yield": math.ceil(seed_words * b / packet_words),
            "distinct_source_inventory_words_used_per_changed_word": round(inventory_words / packet_words, 6),
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SEED)
    blueprint = json.loads(BLUEPRINT.read_text(encoding="utf-8")) if BLUEPRINT.exists() else {}
    seed_words = sum(r["words"] for r in rows)
    results = []
    for ratio in RATIOS:
        for same_focus in [True, False]:
            for max_delta in [0, 2]:
                res = pair_rows(rows, ratio, same_focus=same_focus, max_delta=max_delta)
                res["projection_from_current_seed"] = projection_from_seed(
                    res["c_view_or_b_repeat_packet_words"],
                    res["inventory_words_consumed_for_source_and_distinct_slots"],
                    seed_words,
                )
                results.append(res)
    # Prefer exact same-focus matching if it can support at least a compact 200k arm;
    # otherwise fall back to exact any-focus.  This is only a materialization-readiness
    # suggestion, not a decision to train.
    ranked = sorted(
        results,
        key=lambda r: (
            r["max_distinct_length_delta"] != 0,
            not r["same_focus_distinct_slot"],
            abs(r["ratio"] - 0.88),
            -r["c_view_or_b_repeat_packet_words"],
        ),
    )
    preferred = ranked[0] if ranked else None
    payload = {
        "status": "FINEWEB_SLOT_PAIRING_FEASIBILITY",
        "purpose": "CPU-only readiness evidence for a future controlled source/view materializer; no generated views, training, or evaluation.",
        "source_seed": str(SEED),
        "blueprint_summary": str(BLUEPRINT),
        "seed_rows": len(rows),
        "seed_words": seed_words,
        "seed_docs": len({r["doc_id"] for r in rows}),
        "view_length_ratios_simulated": RATIOS,
        "future_four_arm_unit_contract": {
            "A_natural": "official BabyLM text chunked to each source-plus-view packet length",
            "B_breadth": "source S plus distinct live-FineWeb row D from a different document with len(D)=len(V) when possible",
            "B_repeat": "source S plus literal source-word repetition/truncation filling len(V)",
            "C_view": "source S plus a faithful generated view V that preserves the source proposition but changes wording",
            "required_real_inputs_before_training": [
                "accepted view text for every C_view source slot",
                "view source-faithfulness and substantive-change measurements",
                "distinct D slot ledger with no same-document reuse",
                "exact 10M/100M word ledgers and row-length match across arms",
                "trainer-token and WWM-unit exposure measurement for every arm",
            ],
        },
        "results": results,
        "preferred_contract_shape_from_current_seed": preferred,
        "interpretation": "The current 300k live seed is sufficient to test the slot ledger for a compact source/view pilot, but scaling to a million-word changed block requires more live-source streaming and real view acceptance measurements. This file should guide materialization only after the repaired cached source-breadth result arrives.",
    }
    out_json = OUT_DIR / "fineweb_slot_pairing_feasibility.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research FineWeb source/view slot-pairing feasibility\n\n"]
    lines.append("This CPU-only work formalizes a more controlled future FineWeb unit: source `S` plus a second slot. The second slot is official text in `A_natural`, a distinct FineWeb source `D` in `B_breadth`, literal source repetition in `B_repeat`, and a faithful generated view `V` in `C_view`. No view was generated and no training corpus was written.\n\n")
    lines.append(f"Input live seed: {len(rows):,} rows / {seed_words:,} words / {payload['seed_docs']:,} documents from `{SEED}`.\n\n")
    lines.append("| view length ratio | D focus match | max D length offset | pairs | C/B_repeat packet words | inventory words S+D | 300k arm reachable | 1M arm coverage |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---|---:|\n")
    for r in results:
        b300 = r["budget_capacity"]["300000"]
        b1m = r["budget_capacity"]["1000000"]
        lines.append(
            f"| {r['ratio']:.2f} | {r['same_focus_distinct_slot']} | {r['max_distinct_length_delta']} | {r['pairs']:,} | "
            f"{r['c_view_or_b_repeat_packet_words']:,} | {r['inventory_words_consumed_for_source_and_distinct_slots']:,} | "
            f"{b300['reachable_from_current_seed']} | {b1m['coverage_fraction']:.3f} |\n"
        )
    if preferred:
        lines.append("\nPreferred current-seed dry contract: ")
        lines.append(f"ratio {preferred['ratio']:.2f}, same_focus={preferred['same_focus_distinct_slot']}, max_length_offset={preferred['max_distinct_length_delta']}; it yields {preferred['pairs']:,} pairs and {preferred['c_view_or_b_repeat_packet_words']:,} changed-block words.\n\n")
        lines.append("Focus packet words for that contract:\n\n")
        lines.append("```json\n" + json.dumps(preferred["focus_packet_words"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("Scientific use: if cached source breadth is positive, this slot family lets the next experiment distinguish whether the second exposure budget is better spent on new facts, repeated wording, or faithful views while retaining a strong natural arm. If cached source breadth is weak, the same contract can still support a smaller source+view test without repeating the cached 96-word chunk route.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "seed_words": seed_words,
        "preferred_packet_words": preferred["c_view_or_b_repeat_packet_words"] if preferred else None,
        "preferred_pairs": preferred["pairs"] if preferred else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
