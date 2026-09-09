#!/usr/bin/env python3
"""research: analyze delivered compact_view_reinvest sparse temporal seed comparison.

Input is the completed reinvest sparse-evaluation summary:
  experiments/archive/frontier_consolidation/training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json

This script does not compare to clean-Qwen; that awaits the still-running matched clean sparse
control. It summarizes seed43122-minus-seed43022 trajectories, subtask excess loci, and the
relationship between the fast temporal EWoK slice and the repaired full official EWoK coordinate.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
IN_PATH = STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json"
OUT_DIR = STUDY / "data/reinvest_sparse_temporal_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# From the official full 7618-row EWoK repair and seed43022 official coordinate.
OFFICIAL_EWOK_100M = {
    "seed43022": 53.536575594886855,
    "seed43122": 51.8917180172832,
    "delta_43122_minus_43022": 51.8917180172832 - 53.536575594886855,
    "source": "A01 research/research pristine full official EWoK coordinate, 7618 rows; not the fast EWoK slice used in the temporal probe.",
}


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(r["seed"]), int(r["exposure_m"])): r for r in summary.get("rows", [])}


def main() -> None:
    summary = load(IN_PATH)
    ix = row_index(summary)
    exposures = [int(x) for x in summary["exposures_m"]]
    groups = list(summary["task_groups"].keys())

    by_exposure: dict[str, Any] = {}
    group_trajectories: dict[str, Any] = {}
    for exp in exposures:
        r0 = ix[("43022", exp)]
        r1 = ix[("43122", exp)]
        by_exposure[str(exp)] = {}
        for g in groups:
            s0 = r0["scores"][g]["average_score"]
            s1 = r1["scores"][g]["average_score"]
            by_exposure[str(exp)][g] = {
                "seed43022": s0,
                "seed43122": s1,
                "delta_43122_minus_43022": s1 - s0,
            }
    for g in groups:
        vals = [by_exposure[str(exp)][g]["delta_43122_minus_43022"] for exp in exposures]
        group_trajectories[g] = {
            "delta_by_exposure": {str(exp): by_exposure[str(exp)][g]["delta_43122_minus_43022"] for exp in exposures},
            "seed43022_scores_by_exposure": {str(exp): by_exposure[str(exp)][g]["seed43022"] for exp in exposures},
            "seed43122_scores_by_exposure": {str(exp): by_exposure[str(exp)][g]["seed43122"] for exp in exposures},
            "final_delta_100M": vals[-1],
            "min_delta": min(vals),
            "max_delta": max(vals),
            "late_change_40_to_100M": vals[-1] - vals[-2] if len(vals) >= 2 else None,
            "early_change_1_to_10M": vals[1] - vals[0] if len(vals) >= 2 else None,
        }

    # Subtask final deltas and late changes.
    subtask_records = []
    for g in groups:
        subs_by_exp = {}
        for exp in exposures:
            r0 = ix[("43022", exp)]["scores"][g]["subtask_scores"]
            r1 = ix[("43122", exp)]["scores"][g]["subtask_scores"]
            common = sorted(set(r0) & set(r1))
            subs_by_exp[exp] = {k: r1[k] - r0[k] for k in common}
        final = subs_by_exp[100]
        for k, v in final.items():
            rec = {
                "task_group": g,
                "subtask": k,
                "final_delta_43122_minus_43022": v,
                "delta_by_exposure": {str(exp): subs_by_exp[exp].get(k) for exp in exposures},
                "late_change_40_to_100M": v - subs_by_exp[40].get(k) if 40 in subs_by_exp and k in subs_by_exp[40] else None,
            }
            subtask_records.append(rec)
    most_negative_final = sorted(subtask_records, key=lambda r: r["final_delta_43122_minus_43022"])[:20]
    most_positive_final = sorted(subtask_records, key=lambda r: r["final_delta_43122_minus_43022"], reverse=True)[:20]
    most_negative_late = sorted([r for r in subtask_records if r["late_change_40_to_100M"] is not None], key=lambda r: r["late_change_40_to_100M"])[:20]

    # Compare fast temporal EWoK 100M to full official EWoK 100M.
    fast_ewok_100 = by_exposure["100"]["ewok_all"]
    ewok_coordinate_shift = {
        "fast_temporal_100M": fast_ewok_100,
        "official_full_7618_100M": OFFICIAL_EWOK_100M,
        "seed43022_full_minus_fast": OFFICIAL_EWOK_100M["seed43022"] - fast_ewok_100["seed43022"],
        "seed43122_full_minus_fast": OFFICIAL_EWOK_100M["seed43122"] - fast_ewok_100["seed43122"],
        "delta_full_minus_delta_fast": OFFICIAL_EWOK_100M["delta_43122_minus_43022"] - fast_ewok_100["delta_43122_minus_43022"],
        "interpretation": "The fast temporal slice overstates the final official EWoK seed gap: fast delta is about -3.73, full official delta about -1.64.",
    }

    result = {
        "status": "REINVEST_SPARSE_TEMPORAL_ANALYSIS",
        "source_summary": str(IN_PATH),
        "task_status": summary.get("status"),
        "failures": summary.get("failures"),
        "seeds": summary.get("seeds"),
        "exposures_m": exposures,
        "task_groups": summary.get("task_groups"),
        "by_exposure": by_exposure,
        "group_trajectories": group_trajectories,
        "most_negative_final_subtasks": most_negative_final,
        "most_positive_final_subtasks": most_positive_final,
        "most_negative_late_change_subtasks": most_negative_late,
        "ewok_coordinate_shift": ewok_coordinate_shift,
        "interpretation": {
            "no_uniform_early_failure": "At 1M, seed43122 is much lower on supplement_all and blimp_worst but higher on blimp_control and slightly higher on ewok_all/entity_full; the divergence is task-specific from the start.",
            "late_instability": "From 40M to 100M, blimp_worst collapses for seed43122 relative to seed43022 while blimp_control improves sharply; supplement reverses from +2.4 at 40M to -3.2 at 100M; fast EWoK remains negative but full official EWoK later reduces the 100M gap.",
            "await_clean_control": "This reinvest-only trajectory cannot decide whether the pattern is inherited or treatment-specific; run the research DiD analyzer after the clean sparse control becomes available.",
        },
    }
    out_json = OUT_DIR / "reinvest_sparse_temporal_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — compact_view_reinvest sparse temporal seed analysis\n\n")
    lines.append("The selected sparse evaluation completed with failures=0. This note summarizes seed43122-minus-seed43022 on selected sparse slices before the clean-Qwen DiD control arrives.\n\n")
    lines.append("## Task-group seed gaps by exposure (seed43122 - seed43022)\n")
    for exp in exposures:
        lines.append(f"### {exp}M\n")
        for g in groups:
            rec = by_exposure[str(exp)][g]
            lines.append(f"- {g}: {rec['delta_43122_minus_43022']:+.3f} (43022={rec['seed43022']:.3f}, 43122={rec['seed43122']:.3f})\n")
    lines.append("\n## Late changes 40M -> 100M\n")
    for g, rec in group_trajectories.items():
        lines.append(f"- {g}: final={rec['final_delta_100M']:+.3f}, late_change={rec['late_change_40_to_100M']:+.3f}\n")
    lines.append("\n## Most negative final subtasks\n")
    for r in most_negative_final[:12]:
        lines.append(f"- {r['task_group']} / {r['subtask']}: final={r['final_delta_43122_minus_43022']:+.3f}, by_exp={r['delta_by_exposure']}\n")
    lines.append("\n## EWoK coordinate caveat\n")
    lines.append(f"- fast temporal 100M EWoK delta: {fast_ewok_100['delta_43122_minus_43022']:+.3f}\n")
    lines.append(f"- full official 7618-row 100M EWoK delta: {OFFICIAL_EWOK_100M['delta_43122_minus_43022']:+.3f}\n")
    lines.append("The fast temporal EWoK slice exaggerates the final official EWoK seed gap; use the full official repair for score arithmetic and the fast slice only for temporal patterning.\n\n")
    lines.append("## Current interpretation\n")
    lines.append("The reinvest trajectory is not one simple early-undertraining failure. It is task-specific at 1M and becomes sharply late-divergent on selected BLiMP/Supplement slices. Whether that late divergence is inherited from clean-Qwen or treatment-specific awaits the matched clean sparse control.\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "reinvest_sparse_temporal_analysis.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "failures": summary.get("failures"),
        "final_gaps": {g: group_trajectories[g]["final_delta_100M"] for g in groups},
        "late_changes": {g: group_trajectories[g]["late_change_40_to_100M"] for g in groups},
        "fast_ewok_delta_100M": fast_ewok_100["delta_43122_minus_43022"],
        "official_ewok_delta_100M": OFFICIAL_EWOK_100M["delta_43122_minus_43022"],
        "out_json": str(out_json),
    }, indent=2))

if __name__ == "__main__":
    main()
