#!/usr/bin/env python3
"""research multi-arm decision partition for frozen-anchor fast-path endpoints.

Uses only saved official-compatible prediction payloads and evaluation data.  It
partitions item decisions across the protected anchor, ordinary 86M continuation,
shuffled private tail, coherent private tail, and spanbreak private tail.  The
purpose is scientific localization: are coherent gains unique, shared with ordinary
continued training, or merely part of broad endpoint disagreement?
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
from dataclasses import replace
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import (  # noqa: E402
    CHEAP_COLS,
    DISCRETE_COLUMNS,
    ItemRow,
    PayloadLoader,
    official_score,
    uid_group,
)

ARM_PATHS = OrderedDict([
    (
        "anchor82",
        _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    ),
    (
        "ordinary86",
        _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json'),
    ),
    (
        "shuffled86",
        _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json'),
    ),
    (
        "coherent86",
        _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json'),
    ),
    (
        "spanbreak86",
        _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json'),
    ),
])
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/multiarm_decision_partition')


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


def scores_for(path: pathlib.Path) -> dict[str, float | None]:
    scores = read_json(path).get("official_overall", {}).get("scores", {})
    return {c: (None if scores.get(c) is None else float(scores.get(c))) for c in CHEAP_COLS}


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def load_column_maps(arms: OrderedDict[str, pathlib.Path], column: str) -> dict[str, dict[str, ItemRow]]:
    out = {}
    for name, path in arms.items():
        rows, _ = PayloadLoader(path).load_column(column)
        out[name] = {r.item_id: r for r in rows}
    return out


def pattern_key(names: list[str], vals: dict[str, bool]) -> str:
    yes = [n for n in names if vals[n]]
    return "+".join(yes) if yes else "none"


def majority_vote_row(column: str, item_id: str, rows: dict[str, ItemRow], ordered_arms: list[str]) -> ItemRow:
    # Label-free hard vote among prediction strings, with anchor tie preference.
    anchor = rows["anchor82"]
    counts = Counter(str(rows[a].pred) for a in ordered_arms)
    max_count = max(counts.values())
    candidates = {p for p, c in counts.items() if c == max_count}
    anchor_pred = str(anchor.pred)
    if anchor_pred in candidates:
        pred = anchor_pred
    else:
        # deterministic stable tie break by first arm order
        pred = next(str(rows[a].pred) for a in ordered_arms if str(rows[a].pred) in candidates)
    return replace(anchor, pred=pred, correct=(pred == str(anchor.gold)))


def analyze_column(column: str, arms: OrderedDict[str, pathlib.Path]) -> dict[str, Any]:
    ordered = list(arms.keys())
    maps = load_column_maps(arms, column)
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    arm_scores = {}
    for name in ordered:
        arm_scores[name] = official_score([maps[name][i] for i in common], column)

    anchor_wrong_patterns = Counter()
    anchor_correct_damage_patterns = Counter()
    group_anchor_wrong_patterns: dict[str, Counter] = defaultdict(Counter)
    group_anchor_correct_damage: dict[str, Counter] = defaultdict(Counter)
    coherent_unique_gain = []
    coherent_shared_gain_with_ordinary = []
    coherent_shared_gain_with_shuffled = []
    coherent_unique_loss = []
    coherent_loss_ordinary_preserves = []
    coherent_gain_groups = Counter()
    coherent_loss_groups = Counter()

    pair_sets = {name: {"gain": set(), "loss": set()} for name in ordered if name != "anchor82"}
    majority_rows = []

    non_anchor = [n for n in ordered if n != "anchor82"]
    for item_id in common:
        rows = {name: maps[name][item_id] for name in ordered}
        corr = {name: bool(rows[name].correct) for name in ordered}
        group = uid_group(column, rows["anchor82"])
        for name in non_anchor:
            if (not corr["anchor82"]) and corr[name]:
                pair_sets[name]["gain"].add(item_id)
            elif corr["anchor82"] and (not corr[name]):
                pair_sets[name]["loss"].add(item_id)
        if not corr["anchor82"]:
            key = pattern_key(non_anchor, corr)
            anchor_wrong_patterns[key] += 1
            group_anchor_wrong_patterns[str(group)][key] += 1
            if corr["coherent86"]:
                coherent_gain_groups[str(group)] += 1
                other_correct = [n for n in non_anchor if n != "coherent86" and corr[n]]
                if not other_correct:
                    coherent_unique_gain.append(item_id)
                if corr.get("ordinary86"):
                    coherent_shared_gain_with_ordinary.append(item_id)
                if corr.get("shuffled86"):
                    coherent_shared_gain_with_shuffled.append(item_id)
        else:
            damaged = {n: (not corr[n]) for n in non_anchor}
            key = pattern_key(non_anchor, damaged)
            anchor_correct_damage_patterns[key] += 1
            group_anchor_correct_damage[str(group)][key] += 1
            if not corr["coherent86"]:
                coherent_loss_groups[str(group)] += 1
                other_preserve = [n for n in non_anchor if n != "coherent86" and corr[n]]
                if len(other_preserve) == len(non_anchor) - 1:
                    coherent_unique_loss.append(item_id)
                if corr.get("ordinary86"):
                    coherent_loss_ordinary_preserves.append(item_id)
        majority_rows.append(majority_vote_row(column, item_id, rows, ordered))

    majority_score = official_score(majority_rows, column)

    overlaps: dict[str, Any] = {}
    for a in non_anchor:
        for b in non_anchor:
            if a >= b:
                continue
            for typ in ["gain", "loss"]:
                A = pair_sets[a][typ]
                B = pair_sets[b][typ]
                overlaps[f"{a}_{b}_{typ}"] = {
                    "a": len(A),
                    "b": len(B),
                    "intersection": len(A & B),
                    "union": len(A | B),
                    "jaccard": None if not (A | B) else len(A & B) / len(A | B),
                }

    def top_group_patterns(gdict: dict[str, Counter], pattern: str | None = None, limit: int = 15) -> list[dict[str, Any]]:
        rows = []
        for group, cnt in gdict.items():
            n = sum(cnt.values())
            if pattern is None:
                value = cnt.most_common(1)[0][1] if cnt else 0
                patt = cnt.most_common(1)[0][0] if cnt else "none"
            else:
                value = cnt.get(pattern, 0)
                patt = pattern
            if value:
                rows.append({"group": group, "pattern": patt, "count": int(value), "n": int(n), "pct": 100.0 * value / n if n else 0.0})
        return sorted(rows, key=lambda x: (x["count"], x["pct"]), reverse=True)[:limit]

    return {
        "column": column,
        "n_common": len(common),
        "arm_scores_common": arm_scores,
        "majority_anchor_tie_score_common": majority_score,
        "majority_delta_vs_anchor_common": majority_score - arm_scores["anchor82"],
        "majority_delta_vs_coherent_common": majority_score - arm_scores["coherent86"],
        "anchor_wrong_patterns": dict(anchor_wrong_patterns.most_common()),
        "anchor_correct_damage_patterns": dict(anchor_correct_damage_patterns.most_common()),
        "coherent_unique_gain": len(coherent_unique_gain),
        "coherent_shared_gain_with_ordinary": len(coherent_shared_gain_with_ordinary),
        "coherent_shared_gain_with_shuffled": len(coherent_shared_gain_with_shuffled),
        "coherent_unique_loss": len(coherent_unique_loss),
        "coherent_loss_ordinary_preserves": len(coherent_loss_ordinary_preserves),
        "coherent_gain_groups_top": [{"group": g, "count": c} for g, c in coherent_gain_groups.most_common(15)],
        "coherent_loss_groups_top": [{"group": g, "count": c} for g, c in coherent_loss_groups.most_common(15)],
        "top_unique_coherent_gain_groups": top_group_patterns(group_anchor_wrong_patterns, "coherent86"),
        "top_coherent_damage_groups": top_group_patterns(group_anchor_correct_damage, "coherent86"),
        "overlaps": overlaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = OrderedDict((k, v) for k, v in ARM_PATHS.items() if v.exists())
    if list(arms.keys()) != list(ARM_PATHS.keys()):
        missing = [k for k, v in ARM_PATHS.items() if not v.exists()]
        raise SystemExit(f"missing default arms: {missing}")

    payload_scores = OrderedDict()
    for name, path in arms.items():
        sc = scores_for(path)
        payload_scores[name] = {"payload": rel(path), "scores": sc, "cheap7": cheap7(sc)}

    columns = OrderedDict()
    for col in DISCRETE_COLUMNS:
        columns[col] = analyze_column(col, arms)

    # Aggregate across common discrete item rows.  Scores are column-wise official score means, not row means.
    coherent_unique_gain = sum(columns[c]["coherent_unique_gain"] for c in DISCRETE_COLUMNS)
    coherent_unique_loss = sum(columns[c]["coherent_unique_loss"] for c in DISCRETE_COLUMNS)
    coherent_loss_ordinary_preserves = sum(columns[c]["coherent_loss_ordinary_preserves"] for c in DISCRETE_COLUMNS)
    majority_scores = {c: columns[c]["majority_anchor_tie_score_common"] for c in DISCRETE_COLUMNS}
    majority_discrete_mean = float(mean(majority_scores.values()))
    anchor_discrete_mean = float(mean(columns[c]["arm_scores_common"]["anchor82"] for c in DISCRETE_COLUMNS))
    coherent_discrete_mean = float(mean(columns[c]["arm_scores_common"]["coherent86"] for c in DISCRETE_COLUMNS))
    ordinary_discrete_mean = float(mean(columns[c]["arm_scores_common"]["ordinary86"] for c in DISCRETE_COLUMNS))

    scientific_reading = [
        f"Across the six discrete columns, coherent86 has {coherent_unique_gain} unique gains and {coherent_unique_loss} unique losses relative to the anchor when compared with ordinary86, shuffled86, and spanbreak86.",
        f"Coherent losses that ordinary86 preserves total {coherent_loss_ordinary_preserves}; these are the decisions a retention-oriented route would need to protect without using benchmark labels.",
        f"Anchor-tie hard prediction majority has discrete mean {majority_discrete_mean:.6f} versus anchor {anchor_discrete_mean:.6f}, coherent {coherent_discrete_mean:.6f}, and ordinary86 {ordinary_discrete_mean:.6f}. This is analysis of endpoint diversity, not a submission procedure or a training signal.",
    ]
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "arms": payload_scores,
        "columns": columns,
        "aggregate": {
            "coherent_unique_gain_total": int(coherent_unique_gain),
            "coherent_unique_loss_total": int(coherent_unique_loss),
            "coherent_loss_ordinary_preserves_total": int(coherent_loss_ordinary_preserves),
            "majority_discrete_mean_common": majority_discrete_mean,
            "anchor_discrete_mean_common": anchor_discrete_mean,
            "coherent_discrete_mean_common": coherent_discrete_mean,
            "ordinary86_discrete_mean_common": ordinary_discrete_mean,
            "majority_delta_vs_anchor_discrete_mean": majority_discrete_mean - anchor_discrete_mean,
            "majority_delta_vs_coherent_discrete_mean": majority_discrete_mean - coherent_discrete_mean,
        },
        "scientific_reading": scientific_reading,
        "note": "Uses official labels only to analyze already evaluated endpoints. Do not use item labels to tune pretraining or submission logic.",
    }
    out_json = out_dir / "multiarm_decision_partition.json"
    out_md = out_dir / "multiarm_decision_partition.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research multi-arm decision partition",
        "",
        f"Status: **{out['status']}**",
        "",
        "Saved official-compatible prediction payloads are compared at common item level. This is a scientific reading of endpoint diversity, not a training or submission procedure.",
        "",
        "## Payload scores",
        "",
        "| arm | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | payload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in payload_scores.items():
        sc = row["scores"]
        lines.append(f"| {name} | {row['cheap7']} | {sc.get('BLiMP')} | {sc.get('Supplement')} | {sc.get('EWoK')} | {sc.get('Entity')} | {sc.get('COMPS')} | {sc.get('GlobalPIQA')} | {sc.get('Reading')} | `{row['payload']}` |")
    ag = out["aggregate"]
    lines += [
        "",
        "## Aggregate decision partition",
        "",
        f"- coherent unique gains vs anchor and all other arms: **{ag['coherent_unique_gain_total']}**",
        f"- coherent unique losses vs anchor while all other arms preserve: **{ag['coherent_unique_loss_total']}**",
        f"- coherent losses vs anchor that ordinary86 preserves: **{ag['coherent_loss_ordinary_preserves_total']}**",
        f"- anchor-tie hard majority discrete mean: **{ag['majority_discrete_mean_common']:.6f}** (Δ vs anchor {ag['majority_delta_vs_anchor_discrete_mean']:+.6f}; Δ vs coherent {ag['majority_delta_vs_coherent_discrete_mean']:+.6f})",
        "",
        "## By-column compact view",
        "",
        "| column | n common | coherent unique gains | coherent unique losses | coherent losses ordinary preserves | majority score | majority Δ vs anchor | majority Δ vs coherent | top anchor-wrong patterns | top anchor-correct damage patterns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for col, c in columns.items():
        aw = "; ".join(f"{k}:{v}" for k, v in list(c["anchor_wrong_patterns"].items())[:5])
        dmg = "; ".join(f"{k}:{v}" for k, v in list(c["anchor_correct_damage_patterns"].items())[:5])
        lines.append(f"| {col} | {c['n_common']} | {c['coherent_unique_gain']} | {c['coherent_unique_loss']} | {c['coherent_loss_ordinary_preserves']} | {c['majority_anchor_tie_score_common']:.4f} | {c['majority_delta_vs_anchor_common']:+.4f} | {c['majority_delta_vs_coherent_common']:+.4f} | {aw} | {dmg} |")
    lines += ["", "## Scientific reading", ""]
    for x in scientific_reading:
        lines.append(f"- {x}")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(out_json), "out_md": rel(out_md), "aggregate": out["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
