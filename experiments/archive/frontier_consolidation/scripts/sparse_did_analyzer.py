#!/usr/bin/env python3
"""research: difference-in-differences analyzer for sparse temporal probes.

Given a compact_view_reinvest sparse temporal summary and a clean-Qwen sparse
summary with the same seeds/exposures/task groups, compare seed gaps:

    excess_gap = (reinvest seed43122 - reinvest seed43022)
               - (clean seed43122 - clean seed43022)

Positive excess_gap means seed43122 is less disadvantaged (or more advantaged)
under reinvestment than under clean-Qwen for that slice. Negative excess_gap means
the reinvestment route enlarges the seed43122 shortfall for that slice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reinvest", required=True, help="compact_view_reinvest sparse summary JSON")
    ap.add_argument("--clean", required=True, help="clean-Qwen sparse summary JSON")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--label", default="sparse_did", help="label for output filenames")
    return ap.parse_args()


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in summary.get("rows", []):
        out[(str(row["seed"]), int(row["exposure_m"]))] = row
    return out


def get_score(row: dict[str, Any], task_group: str) -> float | None:
    rec = row.get("scores", {}).get(task_group)
    if rec is None:
        return None
    return float(rec["average_score"])


def get_subscores(row: dict[str, Any], task_group: str) -> dict[str, float]:
    rec = row.get("scores", {}).get(task_group)
    if rec is None:
        return {}
    return {str(k): float(v) for k, v in rec.get("subtask_scores", {}).items()}


def signed_gap(summary: dict[str, Any], exposure: int, task_group: str) -> float | None:
    ix = row_index(summary)
    a = ix.get(("43022", exposure))
    b = ix.get(("43122", exposure))
    if not a or not b:
        return None
    s0 = get_score(a, task_group)
    s1 = get_score(b, task_group)
    if s0 is None or s1 is None:
        return None
    return s1 - s0


def signed_subtask_gap(summary: dict[str, Any], exposure: int, task_group: str) -> dict[str, float]:
    ix = row_index(summary)
    a = ix.get(("43022", exposure))
    b = ix.get(("43122", exposure))
    if not a or not b:
        return {}
    s0 = get_subscores(a, task_group)
    s1 = get_subscores(b, task_group)
    return {k: s1[k] - s0[k] for k in sorted(set(s0) & set(s1))}


def main() -> None:
    ns = parse_args()
    reinvest_path = pathlib.Path(ns.reinvest)
    clean_path = pathlib.Path(ns.clean)
    out_dir = pathlib.Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reinvest = load(reinvest_path)
    clean = load(clean_path)
    exposures = sorted(set(map(int, reinvest.get("exposures_m", []))) & set(map(int, clean.get("exposures_m", []))))
    task_groups = sorted(set(reinvest.get("task_groups", {})) & set(clean.get("task_groups", {})))

    by_exposure: dict[str, Any] = {}
    subtask_excess: dict[str, Any] = {}
    all_group_excess = []
    for exp in exposures:
        by_exposure[str(exp)] = {}
        subtask_excess[str(exp)] = {}
        for tg in task_groups:
            rgap = signed_gap(reinvest, exp, tg)
            cgap = signed_gap(clean, exp, tg)
            if rgap is None or cgap is None:
                continue
            excess = rgap - cgap
            by_exposure[str(exp)][tg] = {
                "reinvest_seed_gap_43122_minus_43022": rgap,
                "clean_seed_gap_43122_minus_43022": cgap,
                "excess_reinvest_minus_clean": excess,
            }
            all_group_excess.append(excess)
            rsub = signed_subtask_gap(reinvest, exp, tg)
            csub = signed_subtask_gap(clean, exp, tg)
            common = sorted(set(rsub) & set(csub))
            recs = []
            for key in common:
                recs.append({
                    "subtask": key,
                    "reinvest_seed_gap": rsub[key],
                    "clean_seed_gap": csub[key],
                    "excess_reinvest_minus_clean": rsub[key] - csub[key],
                })
            subtask_excess[str(exp)][tg] = {
                "n_common_subtasks": len(recs),
                "mean_excess": statistics.mean([r["excess_reinvest_minus_clean"] for r in recs]) if recs else None,
                "most_negative_excess": sorted(recs, key=lambda x: x["excess_reinvest_minus_clean"])[:10],
                "most_positive_excess": sorted(recs, key=lambda x: x["excess_reinvest_minus_clean"], reverse=True)[:10],
                "all": recs,
            }

    task_group_summary: dict[str, Any] = {}
    for tg in task_groups:
        vals = []
        for exp in exposures:
            rec = by_exposure.get(str(exp), {}).get(tg)
            if rec:
                vals.append((exp, rec["excess_reinvest_minus_clean"]))
        if vals:
            task_group_summary[tg] = {
                "excess_by_exposure": {str(exp): val for exp, val in vals},
                "mean_excess": statistics.mean([v for _e, v in vals]),
                "final_excess": dict(vals).get(100, vals[-1][1]),
                "max_abs_excess": max(abs(v) for _e, v in vals),
            }

    result = {
        "status": "SPARSE_DID_ANALYZER",
        "label": ns.label,
        "purpose": "Compare reinvest seed-gap trajectories against clean-Qwen seed-gap trajectories on matching sparse slices.",
        "reinvest_summary": str(reinvest_path),
        "clean_summary": str(clean_path),
        "exposures_used_m": exposures,
        "task_groups_used": task_groups,
        "by_exposure": by_exposure,
        "task_group_summary": task_group_summary,
        "subtask_excess": subtask_excess,
        "overall_mean_excess_across_task_groups_and_exposures": statistics.mean(all_group_excess) if all_group_excess else None,
        "interpretation_help": {
            "near_zero_excess": "similar seed spread in reinvest and clean-Qwen for that slice",
            "negative_excess": "seed43122 shortfall is larger under reinvestment for that slice",
            "positive_excess": "seed43122 shortfall is smaller under reinvestment for that slice",
        },
    }
    out_json = out_dir / f"{ns.label}_sparse_did.json"
    out_md = out_dir / f"{ns.label}_sparse_did.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# research sparse difference-in-differences: {ns.label}\n\n")
    lines.append("This compares seed43122-minus-seed43022 under compact_view_reinvest with the same seed gap under clean-Qwen.\n\n")
    lines.append("## Task-group excess gaps\n")
    for exp in exposures:
        lines.append(f"### {exp}M\n")
        for tg, rec in sorted(by_exposure.get(str(exp), {}).items()):
            lines.append(f"- {tg}: reinvest_gap={rec['reinvest_seed_gap_43122_minus_43022']:.3f}, clean_gap={rec['clean_seed_gap_43122_minus_43022']:.3f}, excess={rec['excess_reinvest_minus_clean']:.3f}\n")
    lines.append("\n## Summary by task group\n")
    for tg, rec in sorted(task_group_summary.items()):
        lines.append(f"- {tg}: excess_by_exposure={rec['excess_by_exposure']}, mean={rec['mean_excess']:.3f}, final={rec['final_excess']:.3f}, max_abs={rec['max_abs_excess']:.3f}\n")
    lines.append("\nInterpretation: excess near zero means the seed spread is inherited; negative excess means the compact-view reinvestment route enlarges seed43122's shortfall on that slice; positive excess means the seed gap is smaller under reinvestment.\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "exposures": exposures, "task_groups": task_groups}, indent=2))


if __name__ == "__main__":
    main()
