#!/usr/bin/env python3
"""research: sparse temporal treatment-effect trajectories.

Compute compact_view_reinvest minus clean-Qwen within each matched seed across
1/10/40/100M for the delivered sparse probes. This complements the seed-gap DiD
by showing whether treatment benefits appear, vanish, or reverse within each seed.
Includes a fixed 18-subtask BLiMP union to avoid overreading outcome-selected
worst/control BLiMP groups.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
REINVEST = STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json"
CLEAN = STUDY / "training/runs/clean_sparse_temporal_probe_main_direct/clean_sparse_temporal_probe_summary.json"
OUT_DIR = STUDY / "data/temporal_treatment_effects"
SEEDS = ["43022", "43122"]


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(r["seed"]), int(r["exposure_m"])): r for r in summary["rows"]}


def group_score(row: dict[str, Any], group: str) -> float:
    return float(row["scores"][group]["average_score"])


def group_subscores(row: dict[str, Any], group: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in row["scores"][group].get("subtask_scores", {}).items()}


def blimp_union_score(row: dict[str, Any]) -> float:
    vals = []
    for group in ["blimp_worst", "blimp_control"]:
        vals.extend(group_subscores(row, group).values())
    if len(vals) != 18:
        raise RuntimeError(f"expected 18 BLiMP union subtasks, got {len(vals)}")
    return statistics.mean(vals)


def score(summary: dict[str, Any], ix: dict[tuple[str, int], dict[str, Any]], seed: str, exposure: int, group: str) -> float:
    row = ix[(seed, exposure)]
    if group == "blimp_union_18_subtasks":
        return blimp_union_score(row)
    return group_score(row, group)


def subtask_te(reinvest_row: dict[str, Any], clean_row: dict[str, Any], group: str) -> list[dict[str, Any]]:
    if group == "blimp_union_18_subtasks":
        recs = []
        for source_group in ["blimp_worst", "blimp_control"]:
            rsub = group_subscores(reinvest_row, source_group)
            csub = group_subscores(clean_row, source_group)
            for k in sorted(set(rsub) & set(csub)):
                recs.append({"subtask": k, "source_group": source_group, "TE_reinvest_minus_clean": rsub[k] - csub[k]})
        return recs
    rsub = group_subscores(reinvest_row, group)
    csub = group_subscores(clean_row, group)
    return [{"subtask": k, "source_group": group, "TE_reinvest_minus_clean": rsub[k] - csub[k]} for k in sorted(set(rsub) & set(csub))]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reinvest = load(REINVEST)
    clean = load(CLEAN)
    rix = row_index(reinvest)
    cix = row_index(clean)
    exposures = sorted(set(map(int, reinvest["exposures_m"])) & set(map(int, clean["exposures_m"])))
    groups = sorted(set(reinvest["task_groups"]) & set(clean["task_groups"])) + ["blimp_union_18_subtasks"]

    trajectories: dict[str, Any] = {}
    subtask_final: dict[str, Any] = {}
    for group in groups:
        trajectories[group] = {}
        for exp in exposures:
            trajectories[group][str(exp)] = {}
            for seed in SEEDS:
                rv = score(reinvest, rix, seed, exp, group)
                cv = score(clean, cix, seed, exp, group)
                trajectories[group][str(exp)][seed] = {
                    "reinvest": rv,
                    "clean": cv,
                    "TE_reinvest_minus_clean": rv - cv,
                }
            te430 = trajectories[group][str(exp)]["43022"]["TE_reinvest_minus_clean"]
            te431 = trajectories[group][str(exp)]["43122"]["TE_reinvest_minus_clean"]
            trajectories[group][str(exp)]["DiD_TE43122_minus_TE43022"] = te431 - te430
        final100 = trajectories[group]["100"]
        final40 = trajectories[group]["40"]
        trajectories[group]["summary"] = {
            "final_TE43022": final100["43022"]["TE_reinvest_minus_clean"],
            "final_TE43122": final100["43122"]["TE_reinvest_minus_clean"],
            "final_DiD": final100["DiD_TE43122_minus_TE43022"],
            "late_change_TE43022_100_minus_40": final100["43022"]["TE_reinvest_minus_clean"] - final40["43022"]["TE_reinvest_minus_clean"],
            "late_change_TE43122_100_minus_40": final100["43122"]["TE_reinvest_minus_clean"] - final40["43122"]["TE_reinvest_minus_clean"],
            "late_change_DiD_100_minus_40": final100["DiD_TE43122_minus_TE43022"] - final40["DiD_TE43122_minus_TE43022"],
            "mean_DiD_over_exposures": statistics.mean(trajectories[group][str(exp)]["DiD_TE43122_minus_TE43022"] for exp in exposures),
        }
        # final subtask TEs and DiD
        r430 = rix[("43022", 100)]
        r431 = rix[("43122", 100)]
        c430 = cix[("43022", 100)]
        c431 = cix[("43122", 100)]
        s430 = {x["subtask"]: x for x in subtask_te(r430, c430, group)}
        s431 = {x["subtask"]: x for x in subtask_te(r431, c431, group)}
        recs = []
        for k in sorted(set(s430) & set(s431)):
            recs.append({
                "subtask": k,
                "source_group": s430[k]["source_group"],
                "TE43022": s430[k]["TE_reinvest_minus_clean"],
                "TE43122": s431[k]["TE_reinvest_minus_clean"],
                "DiD_TE43122_minus_TE43022": s431[k]["TE_reinvest_minus_clean"] - s430[k]["TE_reinvest_minus_clean"],
            })
        subtask_final[group] = {
            "n_common_subtasks": len(recs),
            "most_negative_DiD": sorted(recs, key=lambda x: x["DiD_TE43122_minus_TE43022"])[:10],
            "most_positive_DiD": sorted(recs, key=lambda x: x["DiD_TE43122_minus_TE43022"], reverse=True)[:10],
            "all": recs,
        }

    result = {
        "status": "TEMPORAL_TREATMENT_EFFECTS",
        "purpose": "Treatment-effect trajectories reinvest-clean within seed on exact sparse summaries, plus fixed BLiMP union.",
        "inputs": {"reinvest": str(REINVEST), "clean": str(CLEAN)},
        "exposures_m": exposures,
        "groups": groups,
        "trajectories": trajectories,
        "final_100M_subtask_treatment_interactions": subtask_final,
        "interpretation": {
            "supplement": "Final Supplement DiD is negative after a positive 40M DiD, consistent with a late treatment-by-seed reversal rather than only early acquisition failure.",
            "ewok": "Fast EWoK DiD is persistently negative after 10M, while current-official full EWoK confirms positive but small seed43122 treatment effect; EWoK remains the strongest semantic/domain instability signal.",
            "blimp": "The fixed BLiMP union has much smaller final and late DiD than the selected groups, so broad BLiMP repair is not the main target.",
        },
    }
    out_json = OUT_DIR / "temporal_treatment_effects.json"
    out_md = OUT_DIR / "temporal_treatment_effects.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — sparse temporal treatment effects\n\n"]
    lines.append("Treatment effect is compact_view_reinvest minus clean-Qwen within the same seed. DiD is TE43122 - TE43022.\n\n")
    for group in groups:
        summ = trajectories[group]["summary"]
        lines.append(f"## {group}\n")
        lines.append(f"- final TE43022 {summ['final_TE43022']:+.3f}; final TE43122 {summ['final_TE43122']:+.3f}; final DiD {summ['final_DiD']:+.3f}\n")
        lines.append(f"- late change TE43022 {summ['late_change_TE43022_100_minus_40']:+.3f}; TE43122 {summ['late_change_TE43122_100_minus_40']:+.3f}; DiD {summ['late_change_DiD_100_minus_40']:+.3f}\n")
        lines.append("- TE by exposure: " + "; ".join([f"{exp}M: seed43022 {trajectories[group][str(exp)]['43022']['TE_reinvest_minus_clean']:+.2f}, seed43122 {trajectories[group][str(exp)]['43122']['TE_reinvest_minus_clean']:+.2f}, DiD {trajectories[group][str(exp)]['DiD_TE43122_minus_TE43022']:+.2f}" for exp in exposures]) + "\n\n")
    lines.append("## 100M subtask interaction highlights\n")
    for group in ["supplement_all", "ewok_all", "entity_full", "blimp_union_18_subtasks"]:
        block = subtask_final[group]
        lines.append(f"### {group}\n")
        lines.append("Most negative: " + "; ".join([f"{r['subtask']} {r['DiD_TE43122_minus_TE43022']:+.2f}" for r in block["most_negative_DiD"][:6]]) + "\n")
        lines.append("Most positive: " + "; ".join([f"{r['subtask']} {r['DiD_TE43122_minus_TE43022']:+.2f}" for r in block["most_positive_DiD"][:6]]) + "\n")
    lines.append("\n## Scientific read\n")
    for v in result["interpretation"].values():
        lines.append(f"- {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "summary": {g: trajectories[g]["summary"] for g in groups},
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
