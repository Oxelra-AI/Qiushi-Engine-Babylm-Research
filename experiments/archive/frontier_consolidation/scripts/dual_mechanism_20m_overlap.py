#!/usr/bin/env python3
"""research: CPU-only overlap analysis for two active 20M mechanisms.

Compare saved official-compatible predictions for:
  base   = research legal compact-view reinvest at 20M
  scale  = adapter128 train-time scale1.75 at 20M
  u256   = faithful stream-object U256 at 20M

This does not run model inference. It imports the already validated research
payload parser and computes per-item repair/damage overlap, group-level
correlation, and a non-decision-theoretic oracle-union bound to judge whether
the two mechanisms are probably redundant or complementary before endpoint
results arrive.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(".")
PARSER = ROOT / "experiments/archive/frontier_consolidation/scripts/pairwise_item_flip_analysis.py"
BASE = ROOT / "experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json"
SCALE = ROOT / "experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p75/eval/per_target/adapter128_scale1p75_h100M20M_seed43022.json"
U256 = ROOT / "experiments/archive/frontier_consolidation/data/eu_U256_20M_eval/per_target/eu_U256_20M_seed43022.json"
OUT_DIR = ROOT / "experiments/archive/frontier_consolidation/data/dual_mechanism_20m_overlap"
NOTE = ROOT / "research/notes/frontier_consolidation/dual_mechanism_20m_overlap.md"


def load_parser():
    spec = importlib.util.spec_from_file_location("parser", PARSER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser from {PARSER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def pct(x: float, n: float) -> float:
    return 100.0 * x / n if n else 0.0


def item_map(rows: List[Any]) -> Dict[str, Any]:
    return {r.item_id: r for r in rows}


def safe_jaccard(a: set, b: set) -> float | None:
    u = len(a | b)
    if u == 0:
        return None
    return len(a & b) / u


def group_sorted(group_rows: Dict[str, Dict[str, Any]], key: str, reverse: bool = True, limit: int = 8) -> List[Dict[str, Any]]:
    vals = list(group_rows.values())
    vals.sort(key=lambda d: (d.get(key, 0), d.get("n", 0)), reverse=reverse)
    return vals[:limit]


def analyze_column(mod: Any, loaders: Dict[str, Any], column: str) -> Dict[str, Any]:
    loaded = {name: loader.load_column(column)[0] for name, loader in loaders.items()}
    maps = {name: item_map(rows) for name, rows in loaded.items()}
    common = set(maps["base"]) & set(maps["scale"]) & set(maps["u256"])
    if not common:
        raise RuntimeError(f"No common items for {column}")

    counts: Counter[str] = Counter()
    repair_scale: set[str] = set()
    repair_u256: set[str] = set()
    damage_scale: set[str] = set()
    damage_u256: set[str] = set()
    group: Dict[str, Counter[str]] = defaultdict(Counter)

    for item_id in sorted(common):
        b = bool(maps["base"][item_id].correct)
        s = bool(maps["scale"][item_id].correct)
        u = bool(maps["u256"][item_id].correct)
        g = mod.uid_group(column, maps["base"][item_id])
        group[g]["n"] += 1
        group[g]["base_correct"] += int(b)
        group[g]["scale_correct"] += int(s)
        group[g]["u256_correct"] += int(u)
        if not b:
            group[g]["base_wrong"] += 1
            if s and u:
                counts["both_repair"] += 1; group[g]["both_repair"] += 1
                repair_scale.add(item_id); repair_u256.add(item_id)
            elif s and not u:
                counts["scale_only_repair"] += 1; group[g]["scale_only_repair"] += 1
                repair_scale.add(item_id)
            elif u and not s:
                counts["u256_only_repair"] += 1; group[g]["u256_only_repair"] += 1
                repair_u256.add(item_id)
            else:
                counts["both_miss_base_wrong"] += 1; group[g]["both_miss_base_wrong"] += 1
        else:
            group[g]["base_correct_items"] += 1
            if not s and not u:
                counts["both_damage"] += 1; group[g]["both_damage"] += 1
                damage_scale.add(item_id); damage_u256.add(item_id)
            elif not s and u:
                counts["scale_only_damage"] += 1; group[g]["scale_only_damage"] += 1
                damage_scale.add(item_id)
            elif s and not u:
                counts["u256_only_damage"] += 1; group[g]["u256_only_damage"] += 1
                damage_u256.add(item_id)
            else:
                counts["both_preserve_base_correct"] += 1; group[g]["both_preserve_base_correct"] += 1

    n = len(common)
    base_wrong = counts["both_repair"] + counts["scale_only_repair"] + counts["u256_only_repair"] + counts["both_miss_base_wrong"]
    base_correct = counts["both_damage"] + counts["scale_only_damage"] + counts["u256_only_damage"] + counts["both_preserve_base_correct"]
    assert base_wrong + base_correct == n

    base_score = mod.official_score([maps["base"][i] for i in common], column)
    scale_score = mod.official_score([maps["scale"][i] for i in common], column)
    u256_score = mod.official_score([maps["u256"][i] for i in common], column)

    # Raw-item union is not an achievable model-combination procedure. It is a
    # diagnostic upper bound on complementarity if repairs could be retained
    # without carrying damages.
    raw_base = 100.0 * base_correct / n
    raw_scale = 100.0 * (base_correct - len(damage_scale) + len(repair_scale)) / n
    raw_u256 = 100.0 * (base_correct - len(damage_u256) + len(repair_u256)) / n
    raw_union = 100.0 * (base_correct + len(repair_scale | repair_u256)) / n
    raw_intersection_preserve = 100.0 * (base_correct - len(damage_scale | damage_u256) + len(repair_scale & repair_u256)) / n

    group_rows: Dict[str, Dict[str, Any]] = {}
    xs: List[float] = []
    ys: List[float] = []
    for g, c in group.items():
        gn = int(c["n"])
        scale_net = int(c["both_repair"] + c["scale_only_repair"] - c["both_damage"] - c["scale_only_damage"])
        u256_net = int(c["both_repair"] + c["u256_only_repair"] - c["both_damage"] - c["u256_only_damage"])
        union_repair = int(c["both_repair"] + c["scale_only_repair"] + c["u256_only_repair"])
        shared_repair = int(c["both_repair"])
        shared_damage = int(c["both_damage"])
        row = {
            "group": g,
            "n": gn,
            "base_item_pct": pct(c["base_correct"], gn),
            "scale_item_pct": pct(c["scale_correct"], gn),
            "u256_item_pct": pct(c["u256_correct"], gn),
            "scale_net_items": scale_net,
            "u256_net_items": u256_net,
            "scale_net_pct": pct(scale_net, gn),
            "u256_net_pct": pct(u256_net, gn),
            "both_repair": shared_repair,
            "scale_only_repair": int(c["scale_only_repair"]),
            "u256_only_repair": int(c["u256_only_repair"]),
            "both_damage": shared_damage,
            "scale_only_damage": int(c["scale_only_damage"]),
            "u256_only_damage": int(c["u256_only_damage"]),
            "union_repair": union_repair,
            "union_repair_pct_of_base_wrong": pct(union_repair, c["base_wrong"]),
            "shared_repair_pct_of_union_repair": pct(shared_repair, union_repair),
            "sign_relation": "same_positive" if scale_net > 0 and u256_net > 0 else ("same_negative" if scale_net < 0 and u256_net < 0 else ("opposite" if scale_net * u256_net < 0 else "zero_mixed")),
        }
        group_rows[g] = row
        xs.append(row["scale_net_pct"])
        ys.append(row["u256_net_pct"])

    result = {
        "column": column,
        "n_common": n,
        "official_reconstructed_scores": {"base": base_score, "scale1p75": scale_score, "u256": u256_score},
        "official_reconstructed_deltas_vs_base": {"scale1p75": scale_score - base_score, "u256": u256_score - base_score},
        "raw_item_scores": {"base": raw_base, "scale1p75": raw_scale, "u256": raw_u256, "oracle_union_preserve_base_and_any_repair": raw_union, "intersection_preserve_after_any_damage": raw_intersection_preserve},
        "counts": dict(counts),
        "base_wrong_items": base_wrong,
        "base_correct_items": base_correct,
        "repair_overlap": {
            "scale_repaired_items": len(repair_scale),
            "u256_repaired_items": len(repair_u256),
            "shared_repaired_items": len(repair_scale & repair_u256),
            "union_repaired_items": len(repair_scale | repair_u256),
            "shared_fraction_of_scale_repairs": len(repair_scale & repair_u256) / len(repair_scale) if repair_scale else None,
            "shared_fraction_of_u256_repairs": len(repair_scale & repair_u256) / len(repair_u256) if repair_u256 else None,
            "jaccard": safe_jaccard(repair_scale, repair_u256),
            "oracle_extra_repairs_over_best_single": len(repair_scale | repair_u256) - max(len(repair_scale), len(repair_u256)),
        },
        "damage_overlap": {
            "scale_damaged_items": len(damage_scale),
            "u256_damaged_items": len(damage_u256),
            "shared_damaged_items": len(damage_scale & damage_u256),
            "union_damaged_items": len(damage_scale | damage_u256),
            "shared_fraction_of_scale_damages": len(damage_scale & damage_u256) / len(damage_scale) if damage_scale else None,
            "shared_fraction_of_u256_damages": len(damage_scale & damage_u256) / len(damage_u256) if damage_u256 else None,
            "jaccard": safe_jaccard(damage_scale, damage_u256),
        },
        "group_net_pct_pearson": pearson(xs, ys),
        "group_rows": group_rows,
        "top_both_positive_groups": [r for r in sorted(group_rows.values(), key=lambda r: (min(r["scale_net_pct"], r["u256_net_pct"]), r["n"]), reverse=True) if r["scale_net_items"] > 0 and r["u256_net_items"] > 0][:8],
        "top_opposite_scale_positive_u256_negative": [r for r in sorted(group_rows.values(), key=lambda r: (r["scale_net_pct"] - r["u256_net_pct"], r["n"]), reverse=True) if r["scale_net_items"] > 0 and r["u256_net_items"] < 0][:8],
        "top_opposite_u256_positive_scale_negative": [r for r in sorted(group_rows.values(), key=lambda r: (r["u256_net_pct"] - r["scale_net_pct"], r["n"]), reverse=True) if r["u256_net_items"] > 0 and r["scale_net_items"] < 0][:8],
        "top_shared_repairs": group_sorted(group_rows, "both_repair", True, 8),
        "worst_shared_damages": group_sorted(group_rows, "both_damage", True, 8),
    }
    return result


def main() -> None:
    mod = load_parser()
    loaders = {
        "base": mod.PayloadLoader(BASE),
        "scale": mod.PayloadLoader(SCALE),
        "u256": mod.PayloadLoader(U256),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)

    columns = {col: analyze_column(mod, loaders, col) for col in mod.DISCRETE_COLUMNS}
    official_mean_scale = sum(columns[c]["official_reconstructed_deltas_vs_base"]["scale1p75"] for c in mod.DISCRETE_COLUMNS) / len(mod.DISCRETE_COLUMNS)
    official_mean_u256 = sum(columns[c]["official_reconstructed_deltas_vs_base"]["u256"] for c in mod.DISCRETE_COLUMNS) / len(mod.DISCRETE_COLUMNS)
    total = Counter()
    repair_shared_num = 0
    repair_union_num = 0
    damage_shared_num = 0
    damage_union_num = 0
    for c in columns.values():
        total.update(c["counts"])
        repair_shared_num += c["repair_overlap"]["shared_repaired_items"]
        repair_union_num += c["repair_overlap"]["union_repaired_items"]
        damage_shared_num += c["damage_overlap"]["shared_damaged_items"]
        damage_union_num += c["damage_overlap"]["union_damaged_items"]

    summary = {
        "status": "DUAL_MECHANISM_20M_OVERLAP",
        "base_payload": str(BASE),
        "scale1p75_payload": str(SCALE),
        "u256_payload": str(U256),
        "columns": columns,
        "six_discrete_mean_deltas_vs_base": {"scale1p75": official_mean_scale, "u256": official_mean_u256},
        "aggregate_counts_over_discrete_items": dict(total),
        "aggregate_repair_shared_fraction_of_union": repair_shared_num / repair_union_num if repair_union_num else None,
        "aggregate_damage_shared_fraction_of_union": damage_shared_num / damage_union_num if damage_union_num else None,
        "interpretation": [
            "The two 20M mechanisms are not interchangeable if repair-overlap fractions are low and group-net correlations differ across columns.",
            "The oracle-union rows are upper-bound diagnostics only; they are not an implementable model selection procedure and must not justify ensembling or endpoint promotion without legal official evaluation.",
        ],
    }
    out_json = OUT_DIR / "dual_mechanism_20m_overlap.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# research — 20M overlap of residual-side-capacity and faithful-visibility mechanisms")
    lines.append("")
    lines.append("CPU-only analysis of saved predictions. It compares research legal 20M, adapter128 scale1.75 20M, and U256 faithful-visibility 20M on common discrete items. Reading is not included because the current saved-prediction parser handles only the six discrete zero-shot columns.")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Six-discrete reconstructed mean Δ vs research: scale1.75 {official_mean_scale:+.4f}; U256 {official_mean_u256:+.4f}.")
    lines.append(f"- Aggregate repair shared fraction of union: {summary['aggregate_repair_shared_fraction_of_union']:.4f}.")
    lines.append(f"- Aggregate damage shared fraction of union: {summary['aggregate_damage_shared_fraction_of_union']:.4f}.")
    lines.append(f"- Aggregate counts: `{dict(total)}`.")
    lines.append("")
    lines.append("## Column overlap")
    lines.append("")
    lines.append("| column | Δ scale | Δ U256 | repair Jaccard | damage Jaccard | shared repairs / union | shared damages / union | group-net r | oracle extra repairs over best | strongest opposite groups |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for col, c in columns.items():
        ro = c["repair_overlap"]
        do = c["damage_overlap"]
        opp = c["top_opposite_scale_positive_u256_negative"][:2] + c["top_opposite_u256_positive_scale_negative"][:2]
        opp_s = "; ".join(f"{r['group']} s={r['scale_net_items']:+d} u={r['u256_net_items']:+d}" for r in opp)
        shared_rep_frac = ro["shared_repaired_items"] / ro["union_repaired_items"] if ro["union_repaired_items"] else 0.0
        shared_dam_frac = do["shared_damaged_items"] / do["union_damaged_items"] if do["union_damaged_items"] else 0.0
        lines.append(f"| {col} | {c['official_reconstructed_deltas_vs_base']['scale1p75']:+.3f} | {c['official_reconstructed_deltas_vs_base']['u256']:+.3f} | {ro['jaccard'] if ro['jaccard'] is not None else 0:.3f} | {do['jaccard'] if do['jaccard'] is not None else 0:.3f} | {shared_rep_frac:.3f} | {shared_dam_frac:.3f} | {c['group_net_pct_pearson'] if c['group_net_pct_pearson'] is not None else 0:.3f} | {ro['oracle_extra_repairs_over_best_single']} | {opp_s} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("- Low repair/damage overlap means the two early gains are mostly different item decisions rather than one common easy-score movement. This can support later combined-mechanism reasoning only after the 100M endpoints show that the individual mature effects are real, because several previous early gains reversed at maturity.")
    lines.append("- Strong opposite groups are especially important: a combined route is scientifically plausible only if one mechanism repairs the other's mature losing families without destroying the winning families. These 20M overlaps therefore generate hypotheses for interpreting the arriving 100M endpoints; they are not evidence that a combined training run would succeed.")
    lines.append("- The scale1.75 100M endpoint has now finished training; the managed hardened evaluator should consume it. Do not duplicate its GPU evaluation. If full evaluation clears 41.8, independent legality/reproducibility/submission checks are needed before treating it as the session result.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "mean_delta_scale1p75": official_mean_scale,
        "mean_delta_u256": official_mean_u256,
        "repair_shared_fraction_union": summary["aggregate_repair_shared_fraction_of_union"],
        "damage_shared_fraction_union": summary["aggregate_damage_shared_fraction_of_union"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
