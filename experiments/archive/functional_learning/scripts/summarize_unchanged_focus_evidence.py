#!/usr/bin/env python3
"""research: synthesize unchanged-tail weighted-focus evidence.

This script collates the completed research train/eval results and the research
source-perturb readout into a compact evidence object.  It is research-facing: the
purpose is to decide what the unchanged inherited Qwen-pair objective actually
changed, not to prepare a final deliverable.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
TRAIN_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/comparison_train_summary.json')
EVAL_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_eval/eval_summary.json')
PERTURB_SUMMARY = _public_path('experiments/archive/functional_learning/data/qwen_source_perturb_probe/summary.json')
CREDIT_SUMMARY = _public_path('experiments/archive/functional_learning/data/unchanged_credit_audit/credit_audit_summary.json')
PARENT_FAST = _public_path('experiments/archive/functional_learning/data/common_eval/collated_common_eval.json')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/unchanged_focus_synthesis')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_parent_scores() -> Dict[str, float]:
    obj = read_json(PARENT_FAST)
    for row in obj.get("rows", []):
        if row.get("tag") == "coherent86_alpha075":
            return row["scores"]
    raise RuntimeError("coherent86_alpha075 parent scores not found")


def cheap_scores(eval_summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for rec in eval_summary.get("cheap7_results", []):
        out[rec["name"]] = rec.get("scores", {})
    return out


def delta_dict(scores: Dict[str, float], ref: Dict[str, float], keys: List[str]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for k in keys:
        out[k] = float(scores[k]) - float(ref[k]) if k in scores and k in ref else None
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_json(TRAIN_SUMMARY)
    ev = read_json(EVAL_SUMMARY)
    perturb = read_json(PERTURB_SUMMARY)
    credit = read_json(CREDIT_SUMMARY)
    parent_scores = get_parent_scores()
    objectives = {s["objective"]: s for s in train.get("summaries", [])}
    inherited = objectives["inherited_wwm"]
    focus = objectives["correspondence_focus_weighted"]
    credit_tail = credit["tail_summaries"][0]

    common = ev["common_target"]["deltas_vs_parent"]
    qwen = ev["qwen_view_surface"]["model_summaries"]
    pert = perturb["model_summaries"]
    cheap = cheap_scores(ev)
    cheap_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading", "equal_valid_mean"]

    models = ["unchanged_inherited_wwm_u0080", "unchanged_correspondence_focus_weighted_u0080"]
    rows = []
    for m in models:
        cheap_row = cheap.get(m, {})
        source_spec = pert[m]["source_specificity"]
        qwen_s = qwen[m]
        common_m = common[m]
        rows.append({
            "model": m,
            "cheap_equal_valid_mean": cheap_row.get("equal_valid_mean"),
            "cheap_delta_equal_vs_parent": (cheap_row.get("equal_valid_mean") - parent_scores.get("equal_valid_mean")) if cheap_row.get("equal_valid_mean") is not None else None,
            "cheap_entity": cheap_row.get("Entity"),
            "cheap_globalpiqa_mean": cheap_row.get("GlobalPIQA_mean"),
            "cheap_reading": cheap_row.get("Reading"),
            "common_mean_delta_vs_parent": common_m.get("mean_delta_expected_margin_vs_parent"),
            "common_source_original_delta": common_m.get("by_condition", {}).get("source_original", {}).get("mean_delta"),
            "common_source_altered_delta": common_m.get("by_condition", {}).get("source_altered", {}).get("mean_delta"),
            "common_no_source_delta": common_m.get("by_condition", {}).get("no_source", {}).get("mean_delta"),
            "common_held_delta": common_m.get("by_split", {}).get("held_source", {}).get("mean_delta"),
            "qwen_view_only_delta_nll": qwen_s.get("by_condition", {}).get("view_only", {}).get("mean_delta_nll_vs_parent"),
            "qwen_with_source_delta_nll": qwen_s.get("by_condition", {}).get("with_source", {}).get("mean_delta_nll_vs_parent"),
            "qwen_source_help_delta": qwen_s.get("source_help", {}).get("mean_delta_source_help_vs_parent"),
            "perturb_specific_source_delta": source_spec.get("mean_delta_specific_source_advantage_vs_parent"),
            "perturb_specific_source_delta_median": source_spec.get("median_delta_specific_source_advantage_vs_parent"),
            "perturb_specific_positive_share": source_spec.get("share_delta_specific_positive_vs_parent"),
            "perturb_total_source_advantage_delta": source_spec.get("mean_delta_total_source_advantage_vs_parent"),
            "perturb_correct_source_nll_delta": source_spec.get("mean_delta_correct_source_nll_vs_parent"),
            "perturb_wrong_source_nll_delta": source_spec.get("mean_delta_wrong_source_nll_vs_parent"),
            "perturb_view_only_nll_delta": source_spec.get("mean_delta_view_only_nll_vs_parent"),
        })

    csv_path = _public_path('experiments/archive/functional_learning/data/unchanged_focus_synthesis/model_comparison_table.csv')
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    focus_vs_inherited = {}
    ri, rf = rows[0], rows[1]
    for k in rows[0].keys():
        if k == "model":
            continue
        if isinstance(ri.get(k), (int, float)) and isinstance(rf.get(k), (int, float)):
            focus_vs_inherited[k] = float(rf[k]) - float(ri[k])

    inherited_qwen_fraction = credit_tail["aggregate_inherited_wwm_qwen_target_fraction"]
    focus_label_fraction = focus["aggregate_focus_target_fraction"]
    effective_focus_lambda = focus["mean_lambda_focus_optimized"]
    # Approximate per-token relative emphasis under the macro-normalized weighted objective.
    total_focus_targets = focus["total_focus_targets"]
    total_ordinary_targets = focus["total_ordinary_targets"]
    per_focus_to_ordinary_weight = (effective_focus_lambda / total_focus_targets) / ((1.0 - effective_focus_lambda) / total_ordinary_targets)

    final = {
        "status": "UNCHANGED_FOCUS_SYNTHESIS_DONE",
        "created_utc": now(),
        "inputs": {
            "train_summary": rel(TRAIN_SUMMARY),
            "eval_summary": rel(EVAL_SUMMARY),
            "perturb_summary": rel(PERTURB_SUMMARY),
            "credit_summary": rel(CREDIT_SUMMARY),
            "parent_fast": rel(PARENT_FAST),
        },
        "prefix": focus["prefix_info"],
        "training_credit": {
            "inherited_wwm_total_targets": inherited["total_targets"],
            "inherited_wwm_qwen_targets_credit_audit": credit_tail["totals"]["inherited_wwm_qwen_targets"],
            "inherited_wwm_qwen_target_fraction_credit_audit": inherited_qwen_fraction,
            "focus_total_targets": focus["total_targets"],
            "focus_qwen_targets": focus["total_focus_targets"],
            "focus_label_fraction": focus_label_fraction,
            "focus_objective_lambda": effective_focus_lambda,
            "focus_targets_vs_wwm_qwen_targets_ratio_credit_audit": credit_tail["aggregate_qwen_target_ratio_focus_vs_wwm"],
            "approx_per_focus_token_weight_vs_ordinary_token": per_focus_to_ordinary_weight,
            "interpretation": "Weighted objective restored total qwen objective mass to roughly the WWM qwen share (0.15), while using far fewer qwen labels; each focus target therefore had several-fold larger per-token weight than an ordinary target.",
        },
        "parent_fast_scores": parent_scores,
        "model_rows": rows,
        "focus_minus_inherited": focus_vs_inherited,
        "interpretation": {
            "source_conditioned_behavior": "Weighted focus improved NLL on Qwen second-view tokens but did not improve source-help or correct-vs-wrong source specificity. Common-target altered-source margins moved downward while no-source and original-source margins moved upward, indicating surface/familiar-answer strengthening rather than stronger use of changed source evidence.",
            "broad_behavior": "At update80, ordinary unchanged WWM has a slightly higher fast mean than coherent86 but damages Entity. Weighted focus has Entity near parent and slightly better COMPS/Reading, but lower equal-valid mean than parent and ordinary WWM, mainly with lower EWoK relative to parent. This is not a promotion candidate.",
            "scientific_consequence": "The clean unchanged-text result rejects the idea that simply making inherited Qwen source/rewrite pairs source-visible and assigning them matched aggregate objective mass is enough to produce reusable source-conditioned competence in this late private-adapter continuation. The intervention mostly fits the practiced view surface.",
            "next_experimental_pressure": "Future work should not scale this exact objective blindly. If Qwen correspondence remains the route, it needs an objective that forces contrastive source dependence (correct source versus wrong/no source, altered-source response) or a different data policy where saved budget brings genuinely new support; otherwise route selection should return to mechanisms beyond unchanged-pair view reconstruction.",
        },
        "outputs": {
            "comparison_csv": rel(csv_path),
        },
    }
    out_path = _public_path('experiments/archive/functional_learning/data/unchanged_focus_synthesis/unchanged_focus_evidence_summary.json')
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
