#!/usr/bin/env python3
"""research: proxy temporal DiD using delivered reinvest sparse probes and existing clean-Qwen full reports.

The exact clean sparse control remains pending. This script uses the already
available COMPACT_EXPERIENCE clean-Qwen per-checkpoint full-report summaries at 10/40/100M to form a proxy
control on the same named subtask groups. It is explicitly weaker than the matched sparse control:
full-report task slices are not identical to the fast slices used by research and 1M is unavailable.

Scientific purpose: determine whether the huge reinvest late divergences (selected BLiMP/Supp/EWoK)
look qualitatively new or have analogues in clean-Qwen seed dynamics before the exact control arrives.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = STUDY / "data/proxy_temporal_did_clean_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REINVEST = STUDY / "data/reinvest_sparse_temporal_analysis/reinvest_sparse_temporal_analysis.json"
CLEAN_BASE = ROOT / "experiments/archive/compact_experience/data/trajectory_screen/official_outputs"

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
    "blimp_worst": ("BLiMP", BLIMP_WORST),
    "blimp_control": ("BLiMP", BLIMP_CONTROL),
    "supplement_all": ("Supplement", SUPPLEMENT_ALL),
    "ewok_all": ("EWoK", EWOK_ALL),
    "entity_full": ("Entity", ENTITY_FULL),
}
EXPOSURES = [10, 40, 100]
SEED_DIR = {"43022": "clean_qwen_seed43022", "43122": "clean_qwen_seed43122"}


def parse_report(path: pathlib.Path) -> dict[str, float]:
    d = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^([^:#]+):\s*([-+0-9.]+)\s*$", line.strip())
        if m:
            d[m.group(1).strip()] = float(m.group(2))
    return d


def find_report(seed: str, exposure: int, column: str) -> pathlib.Path:
    root = CLEAN_BASE / SEED_DIR[seed] / f"chck_{exposure}M" / column
    matches = sorted(root.rglob("best_temperature_report.txt"))
    if not matches:
        raise FileNotFoundError(root)
    return matches[0]


def clean_scores() -> dict[str, Any]:
    out = {}
    reports = {}
    for seed in SEED_DIR:
        out[seed] = {}
        reports[seed] = {}
        for exp in EXPOSURES:
            out[seed][str(exp)] = {}
            reports[seed][str(exp)] = {}
            for group, (col, stems) in GROUPS.items():
                rpt = find_report(seed, exp, col)
                vals = parse_report(rpt)
                missing = [s for s in stems if s not in vals]
                # Entity aggregate labels in reports may be under UID or NUMOPS; all requested labels exist in earlier reads.
                if missing:
                    raise KeyError(f"Missing {missing} in {rpt}")
                sub = {s: vals[s] for s in stems}
                out[seed][str(exp)][group] = {"average_score": statistics.mean(sub.values()), "subtask_scores": sub}
                reports[seed][str(exp)][group] = str(rpt)
    return {"scores": out, "reports": reports}


def main() -> None:
    rein = json.loads(REINVEST.read_text(encoding="utf-8"))
    clean = clean_scores()
    by_exposure = {}
    for exp in EXPOSURES:
        by_exposure[str(exp)] = {}
        for group in GROUPS:
            r0 = rein["by_exposure"][str(exp)][group]["seed43022"]
            r1 = rein["by_exposure"][str(exp)][group]["seed43122"]
            c0 = clean["scores"]["43022"][str(exp)][group]["average_score"]
            c1 = clean["scores"]["43122"][str(exp)][group]["average_score"]
            by_exposure[str(exp)][group] = {
                "reinvest_gap_43122_minus_43022": r1 - r0,
                "clean_proxy_gap_43122_minus_43022": c1 - c0,
                "proxy_excess_reinvest_minus_clean": (r1 - r0) - (c1 - c0),
                "reinvest_seed43022": r0,
                "reinvest_seed43122": r1,
                "clean_seed43022_proxy": c0,
                "clean_seed43122_proxy": c1,
            }
    group_summary = {}
    for group in GROUPS:
        vals = [by_exposure[str(e)][group]["proxy_excess_reinvest_minus_clean"] for e in EXPOSURES]
        group_summary[group] = {
            "proxy_excess_by_exposure": {str(e): by_exposure[str(e)][group]["proxy_excess_reinvest_minus_clean"] for e in EXPOSURES},
            "mean_proxy_excess": statistics.mean(vals),
            "final_proxy_excess_100M": vals[-1],
            "late_proxy_excess_change_40_to_100M": vals[-1] - vals[-2],
        }

    subtask_excess = {}
    for exp in EXPOSURES:
        subtask_excess[str(exp)] = {}
        for group, (_col, stems) in GROUPS.items():
            recs = []
            r0sub = None
            # get reinvest subtask scores from original summary rows
            for row in json.loads((STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json").read_text(encoding="utf-8"))["rows"]:
                if row["seed"] == "43022" and int(row["exposure_m"]) == exp:
                    r0sub = row["scores"][group]["subtask_scores"]
                if row["seed"] == "43122" and int(row["exposure_m"]) == exp:
                    r1sub = row["scores"][group]["subtask_scores"]
            c0sub = clean["scores"]["43022"][str(exp)][group]["subtask_scores"]
            c1sub = clean["scores"]["43122"][str(exp)][group]["subtask_scores"]
            for s in stems:
                recs.append({
                    "subtask": s,
                    "reinvest_gap": r1sub[s] - r0sub[s],
                    "clean_proxy_gap": c1sub[s] - c0sub[s],
                    "proxy_excess": (r1sub[s] - r0sub[s]) - (c1sub[s] - c0sub[s]),
                })
            subtask_excess[str(exp)][group] = {
                "mean_proxy_excess": statistics.mean(r["proxy_excess"] for r in recs),
                "most_negative_proxy_excess": sorted(recs, key=lambda r: r["proxy_excess"])[:8],
                "most_positive_proxy_excess": sorted(recs, key=lambda r: r["proxy_excess"], reverse=True)[:8],
            }

    result = {
        "status": "PROXY_TEMPORAL_DID_FROM_CLEAN_FULL_REPORTS",
        "purpose": "Use existing clean-Qwen full per-checkpoint reports as a provisional control for the delivered reinvest sparse temporal trajectory while the exact clean sparse control is still running.",
        "caveat": "Proxy only: clean reports are full task files and no 1M exposure; research managed clean sparse control remains authoritative for exact DiD.",
        "sources": {
            "reinvest_sparse_analysis": str(REINVEST),
            "clean_full_reports_root": str(CLEAN_BASE),
            "clean_reports_used": clean["reports"],
        },
        "exposures_m": EXPOSURES,
        "by_exposure": by_exposure,
        "group_summary": group_summary,
        "subtask_excess": subtask_excess,
        "interpretation": {
            "100M_BLiMP_worst": "If proxy excess is strongly negative, the selected late BLiMP collapse is not clean-like.",
            "100M_BLiMP_control": "Positive proxy excess means seed43122's advantage in control slices is larger under reinvest than clean, so the BLiMP issue is slice-selective rather than global syntax collapse.",
            "EWoK": "This proxy uses fast EWoK temporal scores; the repaired full official EWoK seed gap is smaller and must be used for final official arithmetic.",
        },
    }
    out_json = OUT_DIR / "proxy_temporal_did_from_clean_full_reports.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = []
    lines.append("# research — proxy temporal DiD from clean full reports\n\n")
    lines.append("This is a provisional control while the exact clean sparse probe runs. It compares delivered reinvest sparse seed gaps against existing COMPACT_EXPERIENCE clean-Qwen full-report seed gaps on the same named subtask groups at 10/40/100M.\n\n")
    lines.append("## Proxy excess gaps (reinvest seed gap − clean seed gap)\n")
    for exp in EXPOSURES:
        lines.append(f"### {exp}M\n")
        for group in GROUPS:
            rec = by_exposure[str(exp)][group]
            lines.append(f"- {group}: reinvest_gap={rec['reinvest_gap_43122_minus_43022']:+.3f}, clean_proxy_gap={rec['clean_proxy_gap_43122_minus_43022']:+.3f}, proxy_excess={rec['proxy_excess_reinvest_minus_clean']:+.3f}\n")
    lines.append("\n## Group summary\n")
    for group, rec in group_summary.items():
        lines.append(f"- {group}: mean_proxy_excess={rec['mean_proxy_excess']:+.3f}, final_100M={rec['final_proxy_excess_100M']:+.3f}, late_change={rec['late_proxy_excess_change_40_to_100M']:+.3f}\n")
    lines.append("\nUse this only as a provisional read; the exact research clean sparse control should replace it when delivered.\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "proxy_temporal_did_from_clean_full_reports.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "group_final_proxy_excess_100M": {g: group_summary[g]["final_proxy_excess_100M"] for g in GROUPS},
        "out_json": str(out_json),
    }, indent=2))

if __name__ == "__main__":
    main()
