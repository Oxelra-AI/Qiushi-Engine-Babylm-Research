#!/usr/bin/env python3
"""research: select an intermediate compact-dose pair set without new generation.

The current MAX point (research) and inherited 1x point are not enough to locate
where fixed-budget restructuring stops paying.  This selector builds a nested
intermediate dose directly from the already accepted compact-view inventory.
It imports the research distribution functions so that the intermediate arm uses
exactly the same feature definitions (source length, rewrite/source ratio,
content density, novel rewrite fraction, content recall) as the repaired MAX
instrument.

Default target: the midpoint in restructured budget between the old 1x changed
block (423,520 words) and the research MAX changed block (1,118,720 words),
rounded to an intact 160-word row budget: 771,200 words, i.e. rho=0.07712 and
1.8210x the old dose.

No training, evaluation, SuperGLUE, AoA, upload, or leaderboard action is
performed here.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
SELECTOR = WORKSPACE / "scripts/dose_distribution_compare_and_select.py"
OUT_DIR_DEFAULT = WORKSPACE / "data/dose_intermediate_select"
BASE_CHANGED_BLOCK_WORDS = 423_520
MAX_CHANGED_BLOCK_WORDS = 1_118_720
TOTAL_WORDS = 10_000_000
MAX_PACKET_WORDS = 160
DEFAULT_TARGET_CHANGED_BUDGET = 771_200


def load_step256_module():
    spec = importlib.util.spec_from_file_location("dose_selector", SELECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SELECTOR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pairs(path: pathlib.Path, pairs: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(dataclasses.asdict(p), ensure_ascii=False) + "\n")


def feature_values(pairs: list[Any], name: str) -> list[float]:
    return [float(getattr(p, name)) for p in pairs]


def stat(vals: list[float], weights: list[float] | None = None) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    if weights is None:
        wmean = statistics.fmean(vals)
    else:
        s = sum(weights)
        wmean = sum(v * w for v, w in zip(vals, weights)) / s if s else None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "weighted_mean": wmean,
        "median": statistics.median(xs),
        "q10": xs[int(0.10 * (len(xs) - 1))],
        "q25": xs[int(0.25 * (len(xs) - 1))],
        "q75": xs[int(0.75 * (len(xs) - 1))],
        "q90": xs[int(0.90 * (len(xs) - 1))],
        "min": xs[0],
        "max": xs[-1],
        "sum": sum(vals),
    }


def summarize_group(mod: Any, name: str, pairs: list[Any]) -> dict[str, Any]:
    # Reuse research's own summary when possible so the feature definitions remain identical.
    return mod.summarize(name, pairs)


def divergence(mod: Any, old: list[Any], other: list[Any]) -> dict[str, Any]:
    return mod.divergence(old, other)


def select_target_increment(mod: Any, old: list[Any], candidates: list[Any], target_increment_words: int, bin_slack: float, loose_cap_mult: float, final_cap_mult: float) -> tuple[list[Any], dict[str, Any]]:
    """Select a near-old-distribution increment up to a requested word budget.

    The research MAX selector used the full candidate-total as its target.  Here
    the target is smaller: enough accepted rows exist to choose a midpoint.  The
    algorithm keeps old-bin proportional quotas, sorts candidates by standardized
    distance to the old selected block, and stops before exceeding the target.
    """
    edges = {
        "source_words": mod.quantile_edges(feature_values(old, "source_words"), [0.20, 0.40, 0.60, 0.80]),
        "length_ratio": mod.quantile_edges(feature_values(old, "length_ratio"), [0.25, 0.50, 0.75]),
        "pair_content_density": mod.quantile_edges(feature_values(old, "pair_content_density"), [0.33, 0.66]),
        "novel_content_fraction": mod.quantile_edges(feature_values(old, "novel_content_fraction"), [0.33, 0.66]),
    }
    old_bin_words: collections.Counter[tuple[int, int, int, int]] = collections.Counter()
    for p in old:
        old_bin_words[mod.make_bin(p, edges)] += p.pair_words
    old_total = sum(p.pair_words for p in old)

    means = {
        "source_words": statistics.fmean(feature_values(old, "source_words")),
        "length_ratio": statistics.fmean(feature_values(old, "length_ratio")),
        "pair_content_density": statistics.fmean(feature_values(old, "pair_content_density")),
        "novel_content_fraction": statistics.fmean(feature_values(old, "novel_content_fraction")),
        "content_recall": statistics.fmean([float(p.content_recall) for p in old if p.content_recall is not None]),
    }
    sds = {
        "source_words": mod.std_old(old, "source_words"),
        "length_ratio": mod.std_old(old, "length_ratio"),
        "pair_content_density": mod.std_old(old, "pair_content_density"),
        "novel_content_fraction": mod.std_old(old, "novel_content_fraction"),
        "content_recall": statistics.pstdev([float(p.content_recall) for p in old if p.content_recall is not None]) or 1.0,
    }

    cand_bins: dict[tuple[int, int, int, int], list[Any]] = collections.defaultdict(list)
    for p in candidates:
        cand_bins[mod.make_bin(p, edges)].append(p)
    for b in cand_bins:
        cand_bins[b].sort(key=lambda p: (mod.pair_distance(p, old, means, sds), p.pair_id))

    selected: list[Any] = []
    selected_bin_words: collections.Counter[tuple[int, int, int, int]] = collections.Counter()
    selected_keys: set[str] = set()
    total = 0

    def bin_target(b: tuple[int, int, int, int], mult: float) -> int:
        target = (old_bin_words.get(b, 0) / old_total) * target_increment_words if old_total else 0.0
        return max(0, int(round(target * mult)))

    # First pass: fill old-proportional bins with moderate slack.
    for b, rows in sorted(cand_bins.items(), key=lambda kv: (-old_bin_words.get(kv[0], 0), kv[0])):
        cap = bin_target(b, bin_slack)
        for p in rows:
            if total + p.pair_words > target_increment_words:
                continue
            if selected_bin_words[b] + p.pair_words <= cap:
                selected.append(p)
                selected_keys.add(p.key)
                selected_bin_words[b] += p.pair_words
                total += p.pair_words

    # Second pass: globally close rows, but with a looser per-bin cap.
    unused = [p for p in candidates if p.key not in selected_keys]
    unused.sort(key=lambda p: (mod.pair_distance(p, old, means, sds), p.pair_id))
    for p in unused:
        if total + p.pair_words > target_increment_words:
            continue
        b = mod.make_bin(p, edges)
        if selected_bin_words[b] + p.pair_words <= bin_target(b, loose_cap_mult):
            selected.append(p)
            selected_keys.add(p.key)
            selected_bin_words[b] += p.pair_words
            total += p.pair_words

    # Final small fill: stay below a wider cap so the target is not missed by a
    # few thousand words just because one coarse bin has exhausted its quota.
    unused2 = [p for p in candidates if p.key not in selected_keys]
    unused2.sort(key=lambda p: (mod.pair_distance(p, old, means, sds), p.pair_id))
    for p in unused2:
        if total + p.pair_words > target_increment_words:
            continue
        b = mod.make_bin(p, edges)
        if selected_bin_words[b] + p.pair_words <= bin_target(b, final_cap_mult):
            selected.append(p)
            selected_keys.add(p.key)
            selected_bin_words[b] += p.pair_words
            total += p.pair_words

    selected_bin_dist = {str(k): v for k, v in selected_bin_words.items()}
    candidate_bin_dist = {str(k): sum(p.pair_words for p in rows) for k, rows in cand_bins.items()}
    return selected, {
        "edges": edges,
        "bin_slack": bin_slack,
        "loose_cap_mult": loose_cap_mult,
        "final_cap_mult": final_cap_mult,
        "target_increment_pair_words": target_increment_words,
        "selected_increment_pair_words": total,
        "unfilled_increment_pair_words": target_increment_words - total,
        "candidate_increment_pair_words": sum(p.pair_words for p in candidates),
        "selected_increment_fraction_of_target": total / max(1, target_increment_words),
        "old_bin_words": {str(k): v for k, v in old_bin_words.items()},
        "candidate_bin_words": candidate_bin_dist,
        "selected_bin_words": selected_bin_dist,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-changed-budget", type=int, default=DEFAULT_TARGET_CHANGED_BUDGET)
    ap.add_argument("--label", default="dose1p82")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--bin-slack", type=float, default=1.12)
    ap.add_argument("--loose-cap-mult", type=float, default=1.50)
    ap.add_argument("--final-cap-mult", type=float, default=2.00)
    args = ap.parse_args()

    if args.target_changed_budget % MAX_PACKET_WORDS:
        raise SystemExit("target changed budget must be divisible by 160 for row-holdout compatibility")
    if args.target_changed_budget <= BASE_CHANGED_BLOCK_WORDS or args.target_changed_budget >= MAX_CHANGED_BLOCK_WORDS:
        raise SystemExit("default use is an intermediate target strictly between old 1x and research MAX")

    mod = load_step256_module()
    old = mod.read_pairs(mod.OLD_SELECTED_DEFAULT, "old_selected_1x_medium")
    medium_all = mod.read_pairs(mod.MEDIUM_ACCEPTED_DEFAULT, "medium_unused_accepted")
    expansion = mod.read_pairs(mod.EXPANSION_ACCEPTED_DEFAULT, "war_expansion_accepted")
    old_keys = {p.key for p in old}
    medium_remaining = [p for p in medium_all if p.key not in old_keys]
    candidates = medium_remaining + expansion

    old_pair_words = sum(p.pair_words for p in old)
    target_increment_words = args.target_changed_budget - old_pair_words
    if target_increment_words <= 0:
        raise RuntimeError("target increment is nonpositive")

    selected_increment, select_meta = select_target_increment(
        mod, old, candidates, target_increment_words, args.bin_slack, args.loose_cap_mult, args.final_cap_mult
    )
    selected = old + selected_increment
    selected_pair_words = sum(p.pair_words for p in selected)
    changed_budget = int(math.ceil(selected_pair_words / MAX_PACKET_WORDS) * MAX_PACKET_WORDS)
    if changed_budget > args.target_changed_budget:
        raise RuntimeError(f"selected pair words {selected_pair_words} rounded to {changed_budget} exceed target {args.target_changed_budget}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / f"selected_{args.label}_pairs.jsonl"
    increment_path = out_dir / f"selected_{args.label}_increment_pairs.jsonl"
    write_pairs(selected_path, selected)
    write_pairs(increment_path, selected_increment)

    groups = {
        "old_selected_1x": old,
        "intermediate_increment": selected_increment,
        "intermediate_old_plus_increment": selected,
        "medium_remaining_after_1x": medium_remaining,
        "war_expansion_accepted": expansion,
    }
    summaries = {name: summarize_group(mod, name, rows) for name, rows in groups.items()}
    divergences = {name: divergence(mod, old, rows) for name, rows in groups.items() if name != "old_selected_1x"}
    meta = {
        "status": "INTERMEDIATE_DOSE_PAIRS_SELECTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "scientific_purpose": "Prepare a third fixed-budget dose point before MAX scores land, because the two-point 1x-vs-MAX axis cannot locate the restructuring turnover fraction.",
        "target": {
            "target_changed_budget_words": args.target_changed_budget,
            "target_restructured_fraction_of_10M": args.target_changed_budget / TOTAL_WORDS,
            "target_dose_multiple_vs_1x_budget": args.target_changed_budget / BASE_CHANGED_BLOCK_WORDS,
            "target_increment_pair_words": target_increment_words,
            "midpoint_of_1x_and_step256_max_changed_budgets": (BASE_CHANGED_BLOCK_WORDS + MAX_CHANGED_BLOCK_WORDS) / 2,
        },
        "inputs": {
            "selector": str(SELECTOR),
            "old_selected": str(mod.OLD_SELECTED_DEFAULT),
            "medium_accepted": str(mod.MEDIUM_ACCEPTED_DEFAULT),
            "expansion_accepted": str(mod.EXPANSION_ACCEPTED_DEFAULT),
        },
        "selection": select_meta,
        "groups": summaries,
        "divergence_vs_old_selected": divergences,
        "actual": {
            "selected_pairs": len(selected),
            "selected_increment_pairs": len(selected_increment),
            "selected_pair_words": selected_pair_words,
            "changed_block_budget_words_after_row_rounding": changed_budget,
            "topup_words": changed_budget - selected_pair_words,
            "restructured_fraction_of_10M": changed_budget / TOTAL_WORDS,
            "dose_multiple_vs_1x_budget": changed_budget / BASE_CHANGED_BLOCK_WORDS,
            "old_selected_is_inner_prefix": [p.key for p in selected[:len(old)]] == [p.key for p in old],
            "increment_origin_pair_words": summaries["intermediate_increment"]["origin_pair_words"],
        },
        "files": {
            "selected_pairs": str(selected_path),
            "selected_increment": str(increment_path),
        },
        "sha256": {
            "selected_pairs": sha256_file(selected_path),
            "selected_increment": sha256_file(increment_path),
        },
        "interpretation": "This is only a prepared mechanism instrument. Whether it should be trained before a higher-dose arm depends on the delivered MAX profile: a declining MAX semantic leg asks for this intermediate point; a growing MAX semantic leg asks first for higher dose or second-architecture transfer.",
        "no_training_or_evaluation": True,
    }
    meta_path = out_dir / f"{args.label}_selection_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# research intermediate dose selection: {args.label}",
        "",
        "Purpose: prepare a third dose point without new generation, because 1x and MAX alone cannot locate the fixed-budget restructuring turnover.",
        f"Target changed block: {args.target_changed_budget:,} words (rho={args.target_changed_budget / TOTAL_WORDS:.5f}, dose={args.target_changed_budget / BASE_CHANGED_BLOCK_WORDS:.4f}x).",
        f"Selected: {len(selected):,} pairs, {selected_pair_words:,} pair words, rounded changed block {changed_budget:,}, topup {changed_budget - selected_pair_words}.",
        f"Old 1x block preserved as inner prefix: {meta['actual']['old_selected_is_inner_prefix']}",
        f"Increment origins: {json.dumps(meta['actual']['increment_origin_pair_words'], ensure_ascii=False)}",
        "",
        "## Mean shifts vs inherited 1x selected block (old-block SD units)",
    ]
    for group in ["intermediate_increment", "intermediate_old_plus_increment"]:
        lines.append(f"### {group}")
        for feat, rec in divergences[group].items():
            lines.append(f"- {feat}: {rec['mean_shift_in_old_sd']:+.3f} (old {rec['old_mean']:.4f}, group {rec['other_mean']:.4f})")
    lines += ["", f"Metadata: `{meta_path}`", f"Selected pairs: `{selected_path}`"]
    (out_dir / f"{args.label}_selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": meta["status"],
        "label": args.label,
        "target_changed_budget_words": args.target_changed_budget,
        "selected_pair_words": selected_pair_words,
        "changed_block_budget_words": changed_budget,
        "topup_words": changed_budget - selected_pair_words,
        "dose_multiple_vs_1x_budget": meta["actual"]["dose_multiple_vs_1x_budget"],
        "old_selected_is_inner_prefix": meta["actual"]["old_selected_is_inner_prefix"],
        "increment_origin_pair_words": meta["actual"]["increment_origin_pair_words"],
        "metadata": str(meta_path),
        "selected_pairs": str(selected_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
