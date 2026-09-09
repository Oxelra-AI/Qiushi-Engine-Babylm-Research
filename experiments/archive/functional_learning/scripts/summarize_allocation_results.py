#!/usr/bin/env python3
"""Summarize research balanced allocation comparison across runs.

Combines the two research run roots, computes arm means over six seeds, source-pair
level common-target decompositions, and concise scientific contrasts for the compact
allocation route. This is an analysis script only; it does not train models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
RUN_ROOTS = [
    _public_path('experiments/archive/functional_learning/data/allocation_balanced_comparison'),
    _public_path('experiments/archive/functional_learning/data/allocation_balanced_comparison_rep2'),
]
OUT = _public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmean(xs: Iterable[Any]) -> Optional[float]:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def fmedian(xs: Iterable[Any]) -> Optional[float]:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return statistics.median(vals) if vals else None


def fstd(xs: Iterable[Any]) -> Optional[float]:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None


def agg(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [r.get(key) for r in rows]
    nums = []
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            nums.append(f)
    return {"n": len(nums), "mean": fmean(nums), "median": fmedian(nums), "stdev": fstd(nums), "min": min(nums) if nums else None, "max": max(nums) if nums else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = [load_json(root / "summary.json") for root in RUN_ROOTS]
    metric_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    schedule_rows: List[Dict[str, Any]] = []
    for root, summ in zip(RUN_ROOTS, summaries):
        run_name = root.name
        metric_rows.extend([{**r, "run_root": rel(root)} for r in summ["aggregate"]["per_seed_arm_metrics"]])
        plan = load_json(root / "plan.json")
        for seed, sched in plan["repaired_schedules"].items():
            rec = sched["recurrence_summary"]
            sub = sched["current_substitution_summary"]
            schedule_rows.append({
                "run_root": rel(root), "seed": int(seed),
                "rec_used_extra_words": rec["used_extra_words_total"],
                "rec_unused_saved_words": rec["unused_saved_words_total"],
                "rec_presentation_min": rec["presentation_count_range"][0],
                "rec_presentation_max": rec["presentation_count_range"][1],
                "rec_presentation_stdev": rec["presentation_count_stdev"],
                "current_sub_total_row_words": sub["total_row_words"],
                "current_sub_unused_words": sub["unused_capacity_total"],
                "current_sub_drop_min": sub["drop_count_range"][0],
                "current_sub_drop_max": sub["drop_count_range"][1],
                "current_sub_drop_stdev": sub["drop_count_stdev"],
            })
        for seed_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
            seed = int(seed_dir.name.split("_")[-1])
            for arm_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir()]):
                arm = arm_dir.name
                delta_path = arm_dir / "common_delta_rows.jsonl"
                follow_path = arm_dir / "common_source_follow_rows.jsonl"
                if delta_path.exists():
                    for r in load_jsonl(delta_path):
                        pair_rows.append({**r, "seed": seed, "arm": arm, "run_root": rel(root)})
                if follow_path.exists():
                    for r in load_jsonl(follow_path):
                        source_rows.append({**r, "seed": seed, "arm": arm, "run_root": rel(root)})

    metric_keys = [k for k in metric_rows[0] if k not in {"seed", "arm", "run_root"}] if metric_rows else []
    by_arm = {}
    for arm in sorted(set(r["arm"] for r in metric_rows)):
        ar = [r for r in metric_rows if r["arm"] == arm]
        by_arm[arm] = {k: agg(ar, k) for k in metric_keys}

    def arm_mean(arm: str, key: str) -> Optional[float]:
        return by_arm.get(arm, {}).get(key, {}).get("mean")

    contrasts = {
        "compact_aux_vs_compact_unspent": {
            "base_common_g": arm_mean("compact_aux_support", "base_common_g") - arm_mean("compact_base80_unspent", "base_common_g"),
            "aux_common_g": arm_mean("compact_aux_support", "aux_common_g") - arm_mean("compact_base80_unspent", "aux_common_g"),
            "base_compact_surface_delta_nll": arm_mean("compact_aux_support", "base_compact_with_source_delta_nll") - arm_mean("compact_base80_unspent", "base_compact_with_source_delta_nll"),
            "aux_compact_surface_delta_nll": arm_mean("compact_aux_support", "aux_compact_with_source_delta_nll") - arm_mean("compact_base80_unspent", "aux_compact_with_source_delta_nll"),
        },
        "compact_aux_vs_recurrence": {
            "base_common_g": arm_mean("compact_aux_support", "base_common_g") - arm_mean("compact_interleaved_recurrence", "base_common_g"),
            "aux_common_g": arm_mean("compact_aux_support", "aux_common_g") - arm_mean("compact_interleaved_recurrence", "aux_common_g"),
            "base_compact_surface_delta_nll": arm_mean("compact_aux_support", "base_compact_with_source_delta_nll") - arm_mean("compact_interleaved_recurrence", "base_compact_with_source_delta_nll"),
            "aux_compact_surface_delta_nll": arm_mean("compact_aux_support", "aux_compact_with_source_delta_nll") - arm_mean("compact_interleaved_recurrence", "aux_compact_with_source_delta_nll"),
        },
        "compact_aux_vs_current_aux_substitution": {
            "base_common_g": arm_mean("compact_aux_support", "base_common_g") - arm_mean("current_aux_substitution", "base_common_g"),
            "aux_common_g": arm_mean("compact_aux_support", "aux_common_g") - arm_mean("current_aux_substitution", "aux_common_g"),
            "base_compact_surface_delta_nll": arm_mean("compact_aux_support", "base_compact_with_source_delta_nll") - arm_mean("current_aux_substitution", "base_compact_with_source_delta_nll"),
            "base_current_surface_delta_nll": arm_mean("compact_aux_support", "base_current_with_source_delta_nll") - arm_mean("current_aux_substitution", "base_current_with_source_delta_nll"),
            "aux_compact_surface_delta_nll": arm_mean("compact_aux_support", "aux_compact_with_source_delta_nll") - arm_mean("current_aux_substitution", "aux_compact_with_source_delta_nll"),
            "aux_current_surface_delta_nll": arm_mean("compact_aux_support", "aux_current_with_source_delta_nll") - arm_mean("current_aux_substitution", "aux_current_with_source_delta_nll"),
        },
    }

    # Pair-level distribution for common-target g.
    pair_summary = []
    for arm in sorted(set(r["arm"] for r in pair_rows)):
        for group in sorted(set(r["task_group"] for r in pair_rows)):
            for pid in sorted(set(r["pair_id"] for r in pair_rows if r["arm"] == arm and r["task_group"] == group)):
                g = [r for r in pair_rows if r["arm"] == arm and r["task_group"] == group and r["pair_id"] == pid]
                pair_summary.append({
                    "arm": arm,
                    "task_group": group,
                    "pair_id": pid,
                    "n_task_seed_rows": len(g),
                    "n_seeds": len(set(r["seed"] for r in g)),
                    "mean_g": fmean(r["g_source_follow"] for r in g),
                    "median_g": fmedian(r["g_source_follow"] for r in g),
                    "mean_b": fmean(r["b_direction_imbalance"] for r in g),
                    "mean_p": fmean(r.get("p_no_source_original_prior") for r in g),
                    "seeds_positive_g": sum(1 for r in g if float(r["g_source_follow"]) > 0),
                })

    # For pair-level contrast compact_aux - current_aux and compact_aux - recurrence.
    ps_key = {(r["arm"], r["task_group"], r["pair_id"]): r for r in pair_summary}
    pair_contrasts = []
    for group in sorted(set(r["task_group"] for r in pair_summary)):
        pids = sorted(set(r["pair_id"] for r in pair_summary if r["task_group"] == group))
        for pid in pids:
            ca = ps_key.get(("compact_aux_support", group, pid))
            cu = ps_key.get(("current_aux_substitution", group, pid))
            cr = ps_key.get(("compact_interleaved_recurrence", group, pid))
            cb = ps_key.get(("compact_base80_unspent", group, pid))
            pair_contrasts.append({
                "task_group": group,
                "pair_id": pid,
                "compact_aux_g": ca.get("mean_g") if ca else None,
                "current_aux_g": cu.get("mean_g") if cu else None,
                "recurrence_g": cr.get("mean_g") if cr else None,
                "compact_unspent_g": cb.get("mean_g") if cb else None,
                "ca_minus_current_aux_g": (ca["mean_g"] - cu["mean_g"]) if ca and cu and ca.get("mean_g") is not None and cu.get("mean_g") is not None else None,
                "ca_minus_recurrence_g": (ca["mean_g"] - cr["mean_g"]) if ca and cr and ca.get("mean_g") is not None and cr.get("mean_g") is not None else None,
                "ca_minus_unspent_g": (ca["mean_g"] - cb["mean_g"]) if ca and cb and ca.get("mean_g") is not None and cb.get("mean_g") is not None else None,
            })

    # Success counts by source-follow correctness from raw model margins (not deltas).
    success_summary = []
    for arm in sorted(set(r["arm"] for r in source_rows)):
        for group in sorted(set(r["task_group"] for r in source_rows)):
            g = [r for r in source_rows if r["arm"] == arm and r["task_group"] == group]
            success_summary.append({
                "arm": arm,
                "task_group": group,
                "n_task_seed_rows": len(g),
                "n_pairs": len(set(r["pair_id"] for r in g)),
                "both_correct": sum(1 for r in g if r.get("both_source_conditions_correct")),
                "both_correct_rate": sum(1 for r in g if r.get("both_source_conditions_correct")) / len(g) if g else None,
                "mean_swing": fmean(r["source_follow_swing"] for r in g),
            })

    # Write tables.
    def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_per_seed_arm_metrics.csv'), metric_rows)
    write_csv(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_pair_g_summary.csv'), pair_summary)
    write_csv(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_pair_contrasts.csv'), pair_contrasts)
    write_csv(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_source_follow_success.csv'), success_summary)
    write_csv(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/balanced_schedule_summary.csv'), schedule_rows)

    # Helpful counts: across aux pairs, how often compact_aux has higher g.
    aux_pc = [r for r in pair_contrasts if r["task_group"] == "aux_common"]
    base_pc = [r for r in pair_contrasts if r["task_group"] == "base_common"]
    distribution = {
        "aux_pairs_ca_gt_current_aux": sum(1 for r in aux_pc if r.get("ca_minus_current_aux_g") is not None and r["ca_minus_current_aux_g"] > 0),
        "aux_pairs_total_compared_current_aux": sum(1 for r in aux_pc if r.get("ca_minus_current_aux_g") is not None),
        "aux_pairs_ca_gt_recurrence": sum(1 for r in aux_pc if r.get("ca_minus_recurrence_g") is not None and r["ca_minus_recurrence_g"] > 0),
        "aux_pairs_total_compared_recurrence": sum(1 for r in aux_pc if r.get("ca_minus_recurrence_g") is not None),
        "base_pairs_ca_gt_current_aux": sum(1 for r in base_pc if r.get("ca_minus_current_aux_g") is not None and r["ca_minus_current_aux_g"] > 0),
        "base_pairs_total_compared_current_aux": sum(1 for r in base_pc if r.get("ca_minus_current_aux_g") is not None),
        "base_pairs_ca_gt_unspent": sum(1 for r in base_pc if r.get("ca_minus_unspent_g") is not None and r["ca_minus_unspent_g"] > 0),
        "base_pairs_total_compared_unspent": sum(1 for r in base_pc if r.get("ca_minus_unspent_g") is not None),
    }

    summary = {
        "status": "ALLOCATION_BALANCED_SYNTHESIS",
        "run_roots": [rel(p) for p in RUN_ROOTS],
        "n_metric_rows": len(metric_rows),
        "n_pair_delta_rows": len(pair_rows),
        "n_source_follow_rows": len(source_rows),
        "by_arm_metric": by_arm,
        "mean_arm_contrasts": contrasts,
        "pairwise_distribution": distribution,
        "key_interpretation": {
            "compact_aux_joint": "compact_aux_support learns auxiliary compact surface about as strongly as current_aux_substitution, but has higher aux common source-following g and far better compact/base common preservation than current_aux; it does not beat compact recurrence on base preservation.",
            "current_aux_substitution": "current_aux learns the auxiliary compact surface well but sacrifices current-base presentations and keeps current-surface NLL gains; its base common g remains near current_base rather than compact arms.",
            "recurrence": "balanced recurrence improves base compact surface the most and base common g similarly to compact_aux, but does not receive auxiliary rows and shows only weak aux common g movement from general spillover.",
            "limit": "All measurements are concentrated private-adapter diagnostics on 17 base and 7 auxiliary reviewed rows; they are not a legal BabyLM endpoint or a full principle proof.",
        },
        "artifacts": {
            "combined_metrics_csv": rel(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_per_seed_arm_metrics.csv')),
            "pair_g_summary_csv": rel(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_pair_g_summary.csv')),
            "pair_contrasts_csv": rel(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_pair_contrasts.csv')),
            "source_follow_success_csv": rel(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/combined_source_follow_success.csv')),
            "schedule_summary_csv": rel(_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/balanced_schedule_summary.csv')),
        },
    }
    (_public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
