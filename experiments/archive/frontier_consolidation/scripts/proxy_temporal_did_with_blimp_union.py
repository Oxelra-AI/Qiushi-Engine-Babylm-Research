#!/usr/bin/env python3
"""research: proxy temporal DiD using existing clean-Qwen full reports, with
post-hoc BLiMP selection handled by an explicit 18-file union.

The exact clean sparse control remains the
authoritative control once delivered. This script is intentionally cheaper: it
uses the already available COMPACT_EXPERIENCE clean-Qwen per-checkpoint full-report summaries
at 10/40/100M and compares them to the delivered reinvest sparse temporal
summary. It corrects the research proxy script's main interpretive weakness by
adding `blimp_union = blimp_worst ∪ blimp_control`.

Scientific purpose:
  - avoid overreading selected BLiMP groups whose 100M gaps are inflated because
    the groups were defined from the reinvest seed43122 100M outcome;
  - provide a provisional treatment-specific-vs-inherited read while the exact
    clean sparse control is still running;
  - keep EWoK coordinate caveats explicit (fast temporal vs full official EWoK).
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = STUDY / "data/proxy_temporal_did_clean_full_union"
REINVEST_SUMMARY = STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json"
REINVEST_ANALYSIS = STUDY / "data/reinvest_sparse_temporal_analysis/reinvest_sparse_temporal_analysis.json"
CLEAN_BASE = ROOT / "experiments/archive/compact_experience/data/trajectory_screen/official_outputs"
SEED_DIR = {"43022": "clean_qwen_seed43022", "43122": "clean_qwen_seed43122"}
EXPOSURES = [10, 40, 100]

BLIMP_WORST = [
    "principle_A_reconstruction", "superlative_quantifiers_1", "principle_A_c_command",
    "wh_questions_object_gap", "sentential_subject_island", "distractor_agreement_relative_clause",
    "left_branch_island_simple_question", "sentential_negation_npi_scope",
    "matrix_question_npi_licensor_present", "principle_A_case_2",
]
BLIMP_CONTROL = [
    "principle_A_domain_1", "left_branch_island_echo_question", "wh_island",
    "only_npi_licensor_present", "drop_argument", "superlative_quantifiers_2",
    "principle_A_domain_2", "wh_vs_that_with_gap",
]
BLIMP_UNION = BLIMP_WORST + BLIMP_CONTROL
SUPPLEMENT_ALL = ["qa_congruence_easy", "qa_congruence_tricky", "hypernym", "subject_aux_inversion", "turn_taking"]
EWOK_ALL = [
    "agent-properties", "material-dynamics", "material-properties", "physical-dynamics",
    "physical-interactions", "physical-relations", "quantitative-properties", "social-interactions",
    "social-properties", "social-relations", "spatial-relations",
]
ENTITY_FULL = [
    "regular_0_ops", "regular_1_ops", "regular_2_ops", "regular_3_ops", "regular_4_ops", "regular_5_ops",
    "ambiref_0_ops", "ambiref_1_ops", "ambiref_2_ops", "ambiref_3_ops", "ambiref_4_ops", "ambiref_5_ops",
    "move_contents_0_ops", "move_contents_1_ops", "move_contents_2_ops", "move_contents_3_ops", "move_contents_4_ops", "move_contents_5_ops",
    "regular", "ambiref", "move_contents",
]
GROUPS = {
    # BLiMP selected groups are kept for transparency, but final interpretation should emphasize union.
    "blimp_union": ("BLiMP", BLIMP_UNION, "unselected union of post-hoc worst/control BLiMP files; preferred BLiMP read"),
    "blimp_worst_selected": ("BLiMP", BLIMP_WORST, "post-hoc selected by reinvest seed43122 100M weakness; selection-inflated"),
    "blimp_control_selected": ("BLiMP", BLIMP_CONTROL, "post-hoc selected by reinvest seed43122 100M strength; selection-inflated"),
    "supplement_all": ("Supplement", SUPPLEMENT_ALL, "all fast Supplement slices"),
    "ewok_all_fast": ("EWoK", EWOK_ALL, "fast EWoK temporal coordinate; not final official EWoK arithmetic"),
    "entity_full": ("Entity", ENTITY_FULL, "full Entity Tracking slices from reports"),
}


def parse_report(path: pathlib.Path) -> dict[str, float]:
    d: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^([^:#]+):\s*([-+0-9.]+)\s*$", line.strip())
        if m:
            d[m.group(1).strip()] = float(m.group(2))
    return d


def find_report(seed: str, exposure: int, column: str) -> pathlib.Path:
    root = CLEAN_BASE / SEED_DIR[seed] / f"chck_{exposure}M" / column
    matches = sorted(root.rglob("best_temperature_report.txt"))
    if not matches:
        raise FileNotFoundError(f"No report under {root}")
    return matches[0]


def get_reinvest_row_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["seed"]), int(row["exposure_m"])): row for row in summary.get("rows", [])}


def reinvest_subscores(row: dict[str, Any], group: str) -> dict[str, float]:
    """Return subtask scores for a logical research group from research row."""
    if group == "blimp_union":
        out = {}
        out.update({k: float(v) for k, v in row["scores"]["blimp_worst"]["subtask_scores"].items()})
        out.update({k: float(v) for k, v in row["scores"]["blimp_control"]["subtask_scores"].items()})
        return out
    if group == "blimp_worst_selected":
        src = "blimp_worst"
    elif group == "blimp_control_selected":
        src = "blimp_control"
    elif group == "ewok_all_fast":
        src = "ewok_all"
    else:
        src = group
    return {k: float(v) for k, v in row["scores"][src]["subtask_scores"].items()}


def clean_subscores(seed: str, exposure: int, group: str, column: str, stems: list[str], reports: dict) -> dict[str, float]:
    rpt = find_report(seed, exposure, column)
    reports.setdefault(seed, {}).setdefault(str(exposure), {})[group] = str(rpt)
    vals = parse_report(rpt)
    missing = [s for s in stems if s not in vals]
    if missing:
        raise KeyError(f"Missing {missing} in {rpt}")
    return {s: vals[s] for s in stems}


def mean(d: dict[str, float], stems: list[str]) -> float:
    return statistics.mean([d[s] for s in stems])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reinvest = json.loads(REINVEST_SUMMARY.read_text(encoding="utf-8"))
    reinvest_analysis = json.loads(REINVEST_ANALYSIS.read_text(encoding="utf-8")) if REINVEST_ANALYSIS.exists() else {}
    ix = get_reinvest_row_index(reinvest)
    reports: dict[str, Any] = {}

    by_exposure: dict[str, Any] = {}
    subtask_excess: dict[str, Any] = {}
    for exp in EXPOSURES:
        by_exposure[str(exp)] = {}
        subtask_excess[str(exp)] = {}
        r0row = ix[("43022", exp)]
        r1row = ix[("43122", exp)]
        for group, (column, stems, description) in GROUPS.items():
            r0sub = reinvest_subscores(r0row, group)
            r1sub = reinvest_subscores(r1row, group)
            c0sub = clean_subscores("43022", exp, group, column, stems, reports)
            c1sub = clean_subscores("43122", exp, group, column, stems, reports)
            r0 = mean(r0sub, stems)
            r1 = mean(r1sub, stems)
            c0 = mean(c0sub, stems)
            c1 = mean(c1sub, stems)
            rgap = r1 - r0
            cgap = c1 - c0
            excess = rgap - cgap
            by_exposure[str(exp)][group] = {
                "description": description,
                "n_subtasks": len(stems),
                "reinvest_seed43022": r0,
                "reinvest_seed43122": r1,
                "reinvest_gap_43122_minus_43022": rgap,
                "clean_seed43022_proxy": c0,
                "clean_seed43122_proxy": c1,
                "clean_proxy_gap_43122_minus_43022": cgap,
                "proxy_excess_reinvest_minus_clean": excess,
            }
            recs = []
            for s in stems:
                recs.append({
                    "subtask": s,
                    "reinvest_gap": r1sub[s] - r0sub[s],
                    "clean_proxy_gap": c1sub[s] - c0sub[s],
                    "proxy_excess": (r1sub[s] - r0sub[s]) - (c1sub[s] - c0sub[s]),
                })
            subtask_excess[str(exp)][group] = {
                "mean_proxy_excess": statistics.mean([r["proxy_excess"] for r in recs]),
                "most_negative_proxy_excess": sorted(recs, key=lambda r: r["proxy_excess"])[:10],
                "most_positive_proxy_excess": sorted(recs, key=lambda r: r["proxy_excess"], reverse=True)[:10],
                "all": recs,
            }

    group_summary: dict[str, Any] = {}
    for group in GROUPS:
        vals = [by_exposure[str(exp)][group]["proxy_excess_reinvest_minus_clean"] for exp in EXPOSURES]
        rgaps = [by_exposure[str(exp)][group]["reinvest_gap_43122_minus_43022"] for exp in EXPOSURES]
        cgaps = [by_exposure[str(exp)][group]["clean_proxy_gap_43122_minus_43022"] for exp in EXPOSURES]
        group_summary[group] = {
            "proxy_excess_by_exposure": {str(exp): by_exposure[str(exp)][group]["proxy_excess_reinvest_minus_clean"] for exp in EXPOSURES},
            "reinvest_gap_by_exposure": {str(exp): by_exposure[str(exp)][group]["reinvest_gap_43122_minus_43022"] for exp in EXPOSURES},
            "clean_proxy_gap_by_exposure": {str(exp): by_exposure[str(exp)][group]["clean_proxy_gap_43122_minus_43022"] for exp in EXPOSURES},
            "mean_proxy_excess": statistics.mean(vals),
            "final_proxy_excess_100M": vals[-1],
            "late_proxy_excess_change_40_to_100M": vals[-1] - vals[-2],
            "late_reinvest_gap_change_40_to_100M": rgaps[-1] - rgaps[-2],
            "late_clean_proxy_gap_change_40_to_100M": cgaps[-1] - cgaps[-2],
        }

    result = {
        "status": "PROXY_TEMPORAL_DID_WITH_BLIMP_UNION",
        "purpose": "Provisional DiD against existing clean-Qwen full reports, correcting post-hoc BLiMP selected-group interpretation via an 18-file union.",
        "caveat": "Proxy only: clean reports are existing full-report per-checkpoint outputs and 1M is unavailable; managed exact clean sparse control remains authoritative when delivered.",
        "sources": {
            "reinvest_sparse_summary": str(REINVEST_SUMMARY),
            "reinvest_sparse_analysis": str(REINVEST_ANALYSIS),
            "clean_full_reports_root": str(CLEAN_BASE),
            "clean_reports_used": reports,
        },
        "exposures_m": EXPOSURES,
        "groups": {g: {"column": col, "stems": stems, "description": desc} for g, (col, stems, desc) in GROUPS.items()},
        "by_exposure": by_exposure,
        "group_summary": group_summary,
        "subtask_excess": subtask_excess,
        "interpretation": {
            "BLiMP": "Use blimp_union for the main BLiMP read. The selected worst/control groups are shown only to expose the post-hoc redistribution artifact.",
            "EWoK": "Uses fast temporal EWoK domain scores. Full official 7618-row EWoK at 100M has a smaller seed gap (-1.6449) and positive seed43122 treatment effect (+1.4617), so do not use this proxy for final EWoK arithmetic.",
            "replacement": "Replace this proxy with the exact research clean sparse DiD when the exact clean sparse control is available.",
        },
    }
    out_json = OUT_DIR / "proxy_temporal_did_with_blimp_union.json"
    out_md = OUT_DIR / "proxy_temporal_did_with_blimp_union.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — proxy temporal DiD with unbiased BLiMP union\n\n"]
    lines.append("This provisional control compares delivered compact_view_reinvest sparse seed gaps against existing COMPACT_EXPERIENCE clean-Qwen full-report seed gaps at 10/40/100M. It is weaker than the exact managed clean sparse control, but it corrects the post-hoc BLiMP selected-group artifact by adding the 18-file union.\n\n")
    lines.append("## Main proxy excess gaps (reinvest seed gap − clean seed gap)\n")
    for exp in EXPOSURES:
        lines.append(f"### {exp}M\n")
        for group in ["blimp_union", "supplement_all", "ewok_all_fast", "entity_full", "blimp_worst_selected", "blimp_control_selected"]:
            rec = by_exposure[str(exp)][group]
            lines.append(
                f"- {group}: reinvest_gap={rec['reinvest_gap_43122_minus_43022']:+.3f}, "
                f"clean_proxy_gap={rec['clean_proxy_gap_43122_minus_43022']:+.3f}, "
                f"proxy_excess={rec['proxy_excess_reinvest_minus_clean']:+.3f}\n"
            )
    lines.append("\n## Group summary\n")
    for group in ["blimp_union", "supplement_all", "ewok_all_fast", "entity_full", "blimp_worst_selected", "blimp_control_selected"]:
        rec = group_summary[group]
        lines.append(
            f"- {group}: final_100M_excess={rec['final_proxy_excess_100M']:+.3f}, "
            f"mean_excess={rec['mean_proxy_excess']:+.3f}, "
            f"late_excess_change_40_to_100M={rec['late_proxy_excess_change_40_to_100M']:+.3f}\n"
        )
    lines.append("\n## Interpretive notes\n")
    for k, v in result["interpretation"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "final_proxy_excess_100M": {g: group_summary[g]["final_proxy_excess_100M"] for g in ["blimp_union", "supplement_all", "ewok_all_fast", "entity_full"]},
        "late_proxy_excess_change_40_to_100M": {g: group_summary[g]["late_proxy_excess_change_40_to_100M"] for g in ["blimp_union", "supplement_all", "ewok_all_fast", "entity_full"]},
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
