#!/usr/bin/env python3
"""research: synthesize state-update replacement evidence for route judgment.

This is a CPU-only analysis over research/052 T/U/N margin summaries and research
Entity strata. It writes a research-facing note and small tables; it does not run
training or benchmark evaluation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path("experiments/archive/relation_learning")
OUT_DIR = ROOT / "data" / "route_synthesis"
NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/relation_learning/state_update_route_synthesis.md')
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE.parent.mkdir(parents=True, exist_ok=True)

ENTITY_PATHS = {
    43022: ROOT / "data/state_update_entity_eval_seed43022/entity_intervention_minus_base_by_group.csv",
    43122: ROOT / "data/state_update_entity_eval_seed43122/entity_intervention_minus_base_by_group.csv",
}
MARGIN_PATHS = {
    43022: ROOT / "data/state_margin_scores_seed43022/relation_composite_delta_summary.csv",
    43122: ROOT / "data/state_margin_scores_seed43122/relation_composite_delta_summary.csv",
}
RAW_PATHS = {
    43022: ROOT / "data/state_margin_scores_seed43022/raw_TUN_margin_summary.csv",
    43122: ROOT / "data/state_margin_scores_seed43122/raw_TUN_margin_summary.csv",
}
KEY_GROUPS = [
    "ALL",
    "rel_eq0",
    "rel_eq0_irrelevant_ops_gt0",
    "rel_eq0_irrelevant_ops_ge7",
    "rel_ge1",
    "rel_ge1_postrel_ops0",
    "rel_ge1_postrel_ops_gt0",
    "rel_ge3",
    "rel_ge3_postrel_ops0",
    "rel_ge3_postrel_ops_gt0",
    "stale_available_not_gold",
]
CHECKPOINTS = ["chck_86M", "final"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    if x is None or x == "" or str(x).lower() == "nan":
        return None
    return float(x)


entity_rows = []
for seed, path in ENTITY_PATHS.items():
    for r in read_csv(path):
        if r["checkpoint"] in CHECKPOINTS and r["group"] in KEY_GROUPS:
            entity_rows.append({
                "seed": seed,
                "checkpoint": r["checkpoint"],
                "group": r["group"],
                "n": int(r["n"]),
                "base": fnum(r["base_accuracy_pct"]),
                "intervention": fnum(r["intervention_accuracy_pct"]),
                "delta": fnum(r["delta_accuracy_pct_intervention_minus_base"]),
            })

# Wide table by group/checkpoint for the two seeds.
wide_rows = []
for ck in CHECKPOINTS:
    for group in KEY_GROUPS:
        r430 = next(r for r in entity_rows if r["seed"] == 43022 and r["checkpoint"] == ck and r["group"] == group)
        r431 = next(r for r in entity_rows if r["seed"] == 43122 and r["checkpoint"] == ck and r["group"] == group)
        base_range = abs(r430["base"] - r431["base"])
        int_range = abs(r430["intervention"] - r431["intervention"])
        mean_delta = mean([r430["delta"], r431["delta"]])
        wide_rows.append({
            "checkpoint": ck,
            "group": group,
            "n": r430["n"],
            "base_43022": r430["base"],
            "int_43022": r430["intervention"],
            "delta_43022": r430["delta"],
            "base_43122": r431["base"],
            "int_43122": r431["intervention"],
            "delta_43122": r431["delta"],
            "mean_delta": mean_delta,
            "base_seed_range": base_range,
            "intervention_seed_range": int_range,
            "seed_range_change_int_minus_base": int_range - base_range,
        })

with (OUT_DIR / "entity_side_by_side_key_groups.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
    writer.writeheader(); writer.writerows(wide_rows)

# Selected margin metrics from relation composites, including neutral-anchored terms.
margin_rows = []
for seed, path in MARGIN_PATHS.items():
    for r in read_csv(path):
        if (r["checkpoint"] in CHECKPOINTS and r["probe_set"] == "extended_nontrain"
                and r["slot_mode"] == "template" and r["packet_type"] in {"UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"}
                and r["grouping"] == "seed+checkpoint+probe_set+slot_mode+packet_type"):
            margin_rows.append({
                "seed": seed,
                "checkpoint": r["checkpoint"],
                "packet_type": r["packet_type"],
                "n_packets": int(float(r["n_packets"])),
                "full_delta_T_update_use": fnum(r.get("mean_full_delta_T_update_use")),
                "se_full_delta_T_update_use": fnum(r.get("se_full_delta_T_update_use")),
                "full_delta_U_retention": fnum(r.get("mean_full_delta_U_retention")),
                "se_full_delta_U_retention": fnum(r.get("se_full_delta_U_retention")),
                "full_delta_T_retention": fnum(r.get("mean_full_delta_T_retention")),
                "se_full_delta_T_retention": fnum(r.get("se_full_delta_T_retention")),
                "full_binding_pair_sum_Tupdate_Uretention": fnum(r.get("mean_full_binding_pair_sum_Tupdate_Uretention")),
                "se_full_binding_pair_sum_Tupdate_Uretention": fnum(r.get("se_full_binding_pair_sum_Tupdate_Uretention")),
                "full_binding_pair_sum_Tretention_Uretention": fnum(r.get("mean_full_binding_pair_sum_Tretention_Uretention")),
                "se_full_binding_pair_sum_Tretention_Uretention": fnum(r.get("se_full_binding_pair_sum_Tretention_Uretention")),
                "full_net_T_minus_N_update_use": fnum(r.get("mean_full_net_T_minus_N_update_use")),
                "se_full_net_T_minus_N_update_use": fnum(r.get("se_full_net_T_minus_N_update_use")),
                "full_net_T_minus_U_update_use": fnum(r.get("mean_full_net_T_minus_U_update_use")),
                "se_full_net_T_minus_U_update_use": fnum(r.get("se_full_net_T_minus_U_update_use")),
                "full_net_T_minus_N_retention": fnum(r.get("mean_full_net_T_minus_N_retention")),
                "se_full_net_T_minus_N_retention": fnum(r.get("se_full_net_T_minus_N_retention")),
                "full_net_T_minus_U_retention": fnum(r.get("mean_full_net_T_minus_U_retention")),
                "se_full_net_T_minus_U_retention": fnum(r.get("se_full_net_T_minus_U_retention")),
                "content_delta_T_update_use": fnum(r.get("mean_content_delta_T_update_use")),
                "content_delta_U_retention": fnum(r.get("mean_content_delta_U_retention")),
                "content_delta_T_retention": fnum(r.get("mean_content_delta_T_retention")),
            })

with (OUT_DIR / "state_margin_selected_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys()))
    writer.writeheader(); writer.writerows(margin_rows)

# Raw base recency numbers for the powered UPDATED set and co-established distractor.
raw_selected = []
for seed, path in RAW_PATHS.items():
    for r in read_csv(path):
        if (r["seed"] == str(seed) and r["arm"] == "base" and r["checkpoint"] in CHECKPOINTS
                and r["probe_set"] == "extended_nontrain" and r["slot_mode"] == "template"):
            if r["packet_type"] == "UPDATED_USE" and r["condition"] in {"T", "U"} and r["grouping"].endswith("condition"):
                raw_selected.append({
                    "seed": seed,
                    "checkpoint": r["checkpoint"],
                    "packet_type": r["packet_type"],
                    "condition": r["condition"],
                    "n_packets": int(float(r["n_packets"])),
                    "base_full_orig_new_minus_source": fnum(r["mean_full_m_orig_new_minus_source"]),
                    "se": fnum(r["se_full_m_orig_new_minus_source"]),
                })
            if r["packet_type"] == "UNCHANGED_DISTRACTOR_USE" and r["condition"] == "T" and r["grouping"].endswith("condition"):
                raw_selected.append({
                    "seed": seed,
                    "checkpoint": r["checkpoint"],
                    "packet_type": r["packet_type"],
                    "condition": r["condition"],
                    "n_packets": int(float(r["n_packets"])),
                    "base_full_orig_new_minus_source": fnum(r["mean_full_m_orig_new_minus_source"]),
                    "se": fnum(r["se_full_m_orig_new_minus_source"]),
                })

with (OUT_DIR / "raw_base_recency_selected.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(raw_selected[0].keys()))
    writer.writeheader(); writer.writerows(raw_selected)

# Compact JSON result for later programmatic lookup.
summary = {
    "status": "STATE_UPDATE_SYNTHESIS_READY",
    "entity_side_by_side": str(OUT_DIR / "entity_side_by_side_key_groups.csv"),
    "margin_selected_metrics": str(OUT_DIR / "state_margin_selected_metrics.csv"),
    "raw_base_recency_selected": str(OUT_DIR / "raw_base_recency_selected.csv"),
    "key_decision": "Do not advance the ALN-replacing state-update/plain-use arm to coherent replay or full Overall. Treat it as evidence that weak natural packets induced a coarse state-retention/anti-recency policy rather than entity-gated binding.",
    "word_exposure": 99909920,
    "stream_confounds": ["0.09% underexposure", "displaced inherited ALN qwen_pair_packed companions"],
}
(OUT_DIR / "state_update_synthesis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Human-readable note.
def fmt(x):
    if x is None:
        return "NA"
    return f"{x:+.3f}" if abs(x) < 10 else f"{x:.2f}"

def pct(x):
    return f"{x:.2f}"

lines = []
lines.append("# research synthesis: state-update replacement route and next screens\n")
lines.append("This note integrates the research/052 held-out T/U/N margin readout with the research official Entity deployment strata. It is a research-facing route note; it does not run new training or official evaluation.\n")
lines.append("## Main route judgment\n")
lines.append("The plain-use state-update arm should not be advanced as a direct practical SOTA route. It replaced inherited `qwen_pair_packed` ALN companions and trained for only `99,909,920` words, and its strongest two-seed signal is not entity-gated state revision. The powered template margins move target-update use flat/negative while raising source-state retention in co-established distractor packets. Entity strata then show seed-basin-dependent policy overwrite rather than additive competence.\n")
lines.append("## Entity side-by-side: arm pulls different bases toward a shallower policy\n")
lines.append("At `chck_86M`, the intervention compresses large between-seed differences in relevant-update strata but does so by damaging the seed that had more relevant-update competence.\n")
lines.append("| group | n | base 43022 -> int | base 43122 -> int | delta 43022 | delta 43122 | seed-range change |\n")
lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
for group in ["ALL","rel_eq0","rel_eq0_irrelevant_ops_gt0","rel_ge1","rel_ge3","stale_available_not_gold"]:
    r = next(x for x in wide_rows if x["checkpoint"] == "chck_86M" and x["group"] == group)
    lines.append(f"| `{group}` | {r['n']} | {pct(r['base_43022'])} -> {pct(r['int_43022'])} | {pct(r['base_43122'])} -> {pct(r['int_43122'])} | {fmt(r['delta_43022'])} | {fmt(r['delta_43122'])} | {fmt(r['seed_range_change_int_minus_base'])} |\n")
lines.append("\nNegative seed-range change means the two seeds became more similar after intervention; for `rel_ge1` and `rel_ge3` this convergence is not competence gain, because seed43022 drops sharply and seed43122 is only flat/small-positive. The same qualitative pattern persists at final, especially for `rel_ge3` (seed43022 28.49 -> 25.74; seed43122 26.45 -> 26.34).\n")
lines.append("## Margin evidence\n")
lines.append("| seed | ck | packet | n | T update | U retention | T retention | Tupdate+Uret | Tret+Uret | T-N update | T-U update | T-N retention |\n")
lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
for r in margin_rows:
    lines.append(f"| {r['seed']} | {r['checkpoint']} | {r['packet_type']} | {r['n_packets']} | {fmt(r['full_delta_T_update_use'])} | {fmt(r['full_delta_U_retention'])} | {fmt(r['full_delta_T_retention'])} | {fmt(r['full_binding_pair_sum_Tupdate_Uretention'])} | {fmt(r['full_binding_pair_sum_Tretention_Uretention'])} | {fmt(r['full_net_T_minus_N_update_use'])} | {fmt(r['full_net_T_minus_U_update_use'])} | {fmt(r['full_net_T_minus_N_retention'])} |\n")
lines.append("\nThe neutral-anchored terms reinforce the caution: relation-specific update use (`T-N update`) is negative in both seeds for UPDATED packets, while relation-specific retention (`T-N retention`) is positive on the co-established distractor packets. A scalar average can hide the tradeoff.\n")
lines.append("## Consequences for future practical arms\n")
lines.append("1. Do not spend coherent replay, full Overall, or another two-seed 100M run on a larger version of this replacement packet family. A distractor-heavy version would make the same coarse source-retention relation denser and risks further damage to legitimate target updates.\n")
lines.append("2. Entity aggregate should not be the primary screen for one-percent relation interventions. Its between-seed dispersion is larger than the expected companion-dose effect. It remains a deployment readout after a low-noise source-use or broad-fit signal exists.\n")
lines.append("3. The best evidence-backed practical route is to preserve existing ALN and add more correctly aligned local restatement from filler. Existing ALN improves ordinary held-out MLM at two seeds (about -0.0125/-0.0128 nats) and has large in-family source-recurring T/U/N margins; SHUF shows correctness of correspondence matters; split controls show locality matters.\n")
lines.append("4. The best mechanism-science route is a same-source contrastive entity-binding packet family, added from filler rather than replacing ALN, where the only sufficient predictor is which entity receives the update. Its first result should be the vector of neutral-adjusted update and retention margins, not Entity aggregate.\n")
lines.append("\n## Output tables\n")
lines.append(f"- Entity side-by-side CSV: `{summary['entity_side_by_side']}`\n")
lines.append(f"- Margin selected metrics CSV: `{summary['margin_selected_metrics']}`\n")
lines.append(f"- Raw base recency selected CSV: `{summary['raw_base_recency_selected']}`\n")
NOTE.write_text("".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
print(str(NOTE), flush=True)
