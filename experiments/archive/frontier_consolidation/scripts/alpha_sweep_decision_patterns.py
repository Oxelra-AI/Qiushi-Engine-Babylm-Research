#!/usr/bin/env python3
"""Decision-pattern analysis across coherent86 private adapter scales.

Alphas: 0.0 (protected chck82 anchor), 0.5, 0.75, 1.0.  Uses saved
official-compatible prediction payloads only.  It asks whether smaller alpha
moderates the learned private residual smoothly or merely moves a different small
set of official decisions.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, uid_group  # noqa: E402

ALPHA_PATHS = OrderedDict([
    ("a0", _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')),
    ("a0p5", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json')),
    ("a0p75", _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json')),
    ("a1", _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')),
])
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/alpha_sweep_decision_patterns')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_maps(column: str) -> dict[str, dict[str, Any]]:
    out = {}
    for label, path in ALPHA_PATHS.items():
        rows, _ = PayloadLoader(path).load_column(column)
        out[label] = {r.item_id: r for r in rows}
    return out


def analyze_column(column: str) -> dict[str, Any]:
    labels = list(ALPHA_PATHS)
    maps = load_maps(column)
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    pattern_counts = Counter()
    group_patterns: dict[str, Counter] = defaultdict(Counter)
    anchor_wrong_patterns = Counter()
    anchor_correct_damage_patterns = Counter()
    changes_between = Counter()
    nonmonotonic_gain_candidates = 0
    monotonic_activation_gain_candidates = 0
    monotonic_damage_candidates = 0
    nonmonotonic_damage_candidates = 0
    best_alpha_item_vote = Counter()

    for item_id in common:
        vals = {lab: bool(maps[lab][item_id].correct) for lab in labels}
        bits = "".join("1" if vals[lab] else "0" for lab in labels)
        pattern_counts[bits] += 1
        group = uid_group(column, maps["a0"][item_id])
        group_patterns[str(group)][bits] += 1
        # counts of pairwise correctness changes along alpha path
        for lo, hi in [("a0", "a0p5"), ("a0p5", "a0p75"), ("a0p75", "a1")]:
            if vals[lo] != vals[hi]:
                changes_between[f"{lo}->{hi}"] += 1
        if not vals["a0"]:
            sub = bits[1:]
            anchor_wrong_patterns[sub] += 1
            if sub in {"001", "011", "111"}:  # once correct at a later alpha, stays correct
                monotonic_activation_gain_candidates += 1
            elif "1" in sub:
                nonmonotonic_gain_candidates += 1
            # For a wrong anchor, every correct alpha is equally good; count all correct labels.
            for lab in ["a0p5", "a0p75", "a1"]:
                if vals[lab]:
                    best_alpha_item_vote[lab] += 1
        else:
            damage_bits = "".join("1" if not vals[lab] else "0" for lab in ["a0p5", "a0p75", "a1"])
            anchor_correct_damage_patterns[damage_bits] += 1
            if damage_bits in {"001", "011", "111"}:  # once damaged, stays damaged with larger alpha
                monotonic_damage_candidates += 1
            elif "1" in damage_bits:
                nonmonotonic_damage_candidates += 1
            for lab in ["a0p5", "a0p75", "a1"]:
                if vals[lab]:
                    best_alpha_item_vote[lab] += 1

    def top_groups_for(pattern: str, limit: int = 12) -> list[dict[str, Any]]:
        rows = []
        for g, cnt in group_patterns.items():
            n = sum(cnt.values())
            c = cnt.get(pattern, 0)
            if c:
                rows.append({"group": g, "pattern": pattern, "count": int(c), "n": int(n), "pct": 100.0 * c / n if n else 0.0})
        return sorted(rows, key=lambda r: (r["count"], r["pct"]), reverse=True)[:limit]

    # Useful pattern labels, bits are [a0,a0p5,a0p75,a1].
    named = {
        "anchor_wrong_all_scaled_gain_0111": pattern_counts.get("0111", 0),
        "anchor_wrong_gain_only_0p5_0100": pattern_counts.get("0100", 0),
        "anchor_wrong_gain_0p5_and_0p75_not_1_0110": pattern_counts.get("0110", 0),
        "anchor_wrong_gain_only_0p75_0010": pattern_counts.get("0010", 0),
        "anchor_wrong_gain_0p75_and_1_not_0p5_0011": pattern_counts.get("0011", 0),
        "anchor_wrong_gain_only_1_0001": pattern_counts.get("0001", 0),
        "anchor_correct_preserved_all_1111": pattern_counts.get("1111", 0),
        "anchor_correct_lost_only_at_1_1110": pattern_counts.get("1110", 0),
        "anchor_correct_lost_at_0p75_and_1_1100": pattern_counts.get("1100", 0),
        "anchor_correct_lost_at_all_scaled_1000": pattern_counts.get("1000", 0),
        "anchor_correct_lost_only_0p5_1011": pattern_counts.get("1011", 0),
        "anchor_correct_lost_only_0p75_1101": pattern_counts.get("1101", 0),
    }
    return {
        "column": column,
        "n_common": len(common),
        "pattern_counts_bits_a0_a0p5_a0p75_a1": dict(pattern_counts.most_common()),
        "named_patterns": named,
        "anchor_wrong_scaled_patterns_bits_a0p5_a0p75_a1": dict(anchor_wrong_patterns.most_common()),
        "anchor_correct_damage_patterns_bits_a0p5_a0p75_a1": dict(anchor_correct_damage_patterns.most_common()),
        "changes_between_adjacent_alphas": dict(changes_between),
        "monotonic_activation_gain_candidates": monotonic_activation_gain_candidates,
        "nonmonotonic_gain_candidates": nonmonotonic_gain_candidates,
        "monotonic_damage_candidates": monotonic_damage_candidates,
        "nonmonotonic_damage_candidates": nonmonotonic_damage_candidates,
        "best_alpha_item_vote_correctness_count": dict(best_alpha_item_vote),
        "top_groups_gain_only_0p5": top_groups_for("0100"),
        "top_groups_gain_0p5_0p75_not_1": top_groups_for("0110"),
        "top_groups_lost_only_at_1": top_groups_for("1110"),
        "top_groups_lost_at_0p75_and_1": top_groups_for("1100"),
        "top_groups_lost_all_scaled": top_groups_for("1000"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, path in ALPHA_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)

    cols = OrderedDict((col, analyze_column(col)) for col in DISCRETE_COLUMNS)
    aggregate_counts = Counter()
    adjacent = Counter()
    for col, c in cols.items():
        aggregate_counts.update(c["pattern_counts_bits_a0_a0p5_a0p75_a1"])
        adjacent.update(c["changes_between_adjacent_alphas"])
    total_common = sum(c["n_common"] for c in cols.values())
    total_nonmono_gain = sum(c["nonmonotonic_gain_candidates"] for c in cols.values())
    total_mono_gain = sum(c["monotonic_activation_gain_candidates"] for c in cols.values())
    total_nonmono_damage = sum(c["nonmonotonic_damage_candidates"] for c in cols.values())
    total_mono_damage = sum(c["monotonic_damage_candidates"] for c in cols.values())
    scientific = [
        f"Across {total_common} common discrete items, adjacent alpha changes are {dict(adjacent)}; most decisions are stable, but the private residual has thousands of threshold-sensitive decisions.",
        f"Anchor-wrong gains include {total_mono_gain} monotonic activation candidates and {total_nonmono_gain} nonmonotonic candidates; anchor-correct damage includes {total_mono_damage} monotonic larger-alpha damage candidates and {total_nonmono_damage} nonmonotonic damage candidates.",
        "Alpha0.5/0.75 improving cheap7 while reducing changed items is consistent with amplitude moderation, but binary item patterns remain mixed rather than a clean capability threshold.",
    ]
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "alpha_payloads": {k: rel(v) for k, v in ALPHA_PATHS.items()},
        "columns": cols,
        "aggregate": {
            "total_common_items": total_common,
            "pattern_counts_bits_a0_a0p5_a0p75_a1": dict(aggregate_counts.most_common()),
            "changes_between_adjacent_alphas": dict(adjacent),
            "monotonic_activation_gain_candidates": total_mono_gain,
            "nonmonotonic_gain_candidates": total_nonmono_gain,
            "monotonic_damage_candidates": total_mono_damage,
            "nonmonotonic_damage_candidates": total_nonmono_damage,
        },
        "scientific_reading": scientific,
    }
    out_json = out_dir / "alpha_sweep_decision_patterns.json"
    out_md = out_dir / "alpha_sweep_decision_patterns.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research alpha-sweep decision patterns",
        "",
        f"Status: **{out['status']}**",
        "",
        "Bits are ordered `[alpha0, alpha0.5, alpha0.75, alpha1]`, where alpha0 is the protected anchor.",
        "",
        "## Aggregate",
        "",
        f"- total common discrete items: `{total_common}`",
        f"- adjacent changes: `{dict(adjacent)}`",
        f"- monotonic activation gain candidates: `{total_mono_gain}`; nonmonotonic gain candidates: `{total_nonmono_gain}`",
        f"- monotonic damage candidates: `{total_mono_damage}`; nonmonotonic damage candidates: `{total_nonmono_damage}`",
        "",
        "Top aggregate patterns:",
    ]
    for patt, count in aggregate_counts.most_common(12):
        lines.append(f"- `{patt}`: {count}")
    lines += ["", "## By column", "", "| column | n | adjacent changes | all stable correct 1111 | all stable wrong 0000 | gain all scaled 0111 | gain only 0.5 0100 | gain 0.5/0.75 not 1 0110 | lost only at 1 1110 | lost at 0.75/1 1100 | lost all scaled 1000 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for col, c in cols.items():
        pc = c["pattern_counts_bits_a0_a0p5_a0p75_a1"]
        nm = c["named_patterns"]
        lines.append(f"| {col} | {c['n_common']} | `{c['changes_between_adjacent_alphas']}` | {pc.get('1111',0)} | {pc.get('0000',0)} | {nm['anchor_wrong_all_scaled_gain_0111']} | {nm['anchor_wrong_gain_only_0p5_0100']} | {nm['anchor_wrong_gain_0p5_and_0p75_not_1_0110']} | {nm['anchor_correct_lost_only_at_1_1110']} | {nm['anchor_correct_lost_at_0p75_and_1_1100']} | {nm['anchor_correct_lost_at_all_scaled_1000']} |")
    lines += ["", "## Scientific reading", ""]
    for x in scientific:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(out_json), "out_md": rel(out_md), "aggregate": out["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
