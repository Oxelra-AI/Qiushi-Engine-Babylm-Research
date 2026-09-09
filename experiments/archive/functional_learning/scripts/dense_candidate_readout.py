#!/usr/bin/env python3
"""research: summarize dense-focus candidate evidence around the official decision.

This script performs CPU-only analysis while dense official evaluation and replication
run on the GPUs.  It uses already saved payloads to preserve the exact reference
coordinate and the fast-screen item movement, then writes a compact readout that can
be updated once the official dense payload exists.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

ROOT = pathlib.Path(".").resolve()

COH_ZERO = pathlib.Path("experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json")
COH_SG = pathlib.Path("experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json")
FAST_TRANSITION = pathlib.Path("experiments/archive/functional_learning/data/dense_fast_transition_compare/dense_vs_coherent86_fast_official_transition.json")
DENSE_TRAIN = pathlib.Path("experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json")
SPARSE_TRAIN = pathlib.Path("experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/train_summary.json")
DENSE_EVAL = pathlib.Path("experiments/archive/functional_learning/data/unchanged_dense_focus_eval/eval_summary.json")
SPARSE_SYNTH = pathlib.Path("experiments/archive/functional_learning/data/unchanged_focus_synthesis/model_comparison_table.csv")
DENSE_OFFICIAL = pathlib.Path("experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json")

OFFICIAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
CHEAP7_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_score(payload: dict[str, Any], col: str) -> float | None:
    overall = payload.get("official_overall", {}).get("scores", {})
    if col in overall and overall[col] is not None:
        return float(overall[col])
    tasks = payload.get("tasks", {})
    if col == "Reading" and isinstance(tasks.get("Reading"), dict):
        scores = tasks["Reading"].get("scores", {})
        if scores.get("Reading") is not None:
            return float(scores["Reading"])
    if col == "SuperGLUE" and isinstance(tasks.get("SuperGLUE"), dict):
        v = tasks["SuperGLUE"].get("superglue_mean")
        if v is not None:
            return float(v)
    if col == "GlobalPIQA":
        a = get_score(payload, "GlobalPIQA_parallel")
        b = get_score(payload, "GlobalPIQA_nonparallel")
        if a is not None and b is not None:
            return (a + b) / 2.0
    if isinstance(tasks.get(col), dict):
        if tasks[col].get("score") is not None:
            return float(tasks[col]["score"])
    return None


def combine_scores(zero_payload: dict[str, Any], sg_payload: dict[str, Any] | None = None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for col in OFFICIAL_COLUMNS:
        out[col] = get_score(zero_payload, col)
    if sg_payload is not None:
        sg = get_score(sg_payload, "SuperGLUE")
        if sg is not None:
            out["SuperGLUE"] = sg
    if out.get("GlobalPIQA") is None:
        gp_p = get_score(zero_payload, "GlobalPIQA_parallel")
        gp_np = get_score(zero_payload, "GlobalPIQA_nonparallel")
        if gp_p is not None and gp_np is not None:
            out["GlobalPIQA"] = (gp_p + gp_np) / 2.0
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(k) for k in CHEAP7_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(sum(v for v in vals if v is not None) / len(vals))


def projected_overall_aoa0(scores: dict[str, float | None]) -> float | None:
    tmp = dict(scores)
    if tmp.get("AoA") is None:
        tmp["AoA"] = 0.0
    vals = [tmp.get(k) for k in OFFICIAL_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(sum(v for v in vals if v is not None) / len(vals))


def task_status(payload_path: pathlib.Path) -> dict[str, Any]:
    if not payload_path.exists():
        return {"path": rel(payload_path), "exists": False, "status": "missing"}
    try:
        payload = load_json(payload_path)
    except Exception as e:
        return {"path": rel(payload_path), "exists": True, "status": "unreadable", "error": repr(e), "size": payload_path.stat().st_size}
    tasks = payload.get("tasks", {})
    present = sorted(tasks.keys())
    finished = []
    unfinished = []
    for k, v in tasks.items():
        if not isinstance(v, dict):
            unfinished.append(k)
        elif k == "AoA":
            # AoA can be recorded rather than run if checkpoint ladder is absent.
            finished.append(k)
        elif v.get("returncode") == 0 or (k == "SuperGLUE" and v.get("superglue_mean") is not None):
            finished.append(k)
        else:
            unfinished.append(k)
    scores = combine_scores(payload)
    return {
        "path": rel(payload_path),
        "exists": True,
        "size": payload_path.stat().st_size,
        "target": payload.get("target"),
        "present_tasks": present,
        "finished_tasks": sorted(finished),
        "unfinished_or_incomplete_tasks": sorted(unfinished),
        "scores_seen": scores,
        "cheap7_seen": cheap7(scores),
        "projected_overall_aoa0_seen": projected_overall_aoa0(scores),
        "official_overall_field": payload.get("official_overall"),
    }


def top_subtasks(transition: dict[str, Any], col: str, n: int = 8) -> dict[str, list[dict[str, Any]]]:
    rec = transition["column_comparisons"].get(col, {})
    rows = rec.get("subtasks_by_abs_official_delta", [])
    gains = [r for r in rows if r.get("delta_acc_b_minus_a", 0) > 0]
    losses = [r for r in rows if r.get("delta_acc_b_minus_a", 0) < 0]
    gains = sorted(gains, key=lambda r: r["delta_acc_b_minus_a"], reverse=True)[:n]
    losses = sorted(losses, key=lambda r: r["delta_acc_b_minus_a"])[:n]
    keep = ["subtask", "n", "a_acc", "b_acc", "delta_acc_b_minus_a", "net_item_delta_b_minus_a"]
    return {
        "largest_gains": [{k: r.get(k) for k in keep} for r in gains],
        "largest_losses": [{k: r.get(k) for k in keep} for r in losses],
    }


def score_rows_to_dict(transition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["column"]: r for r in transition["score_summary"]["rows"]}


def parse_sparse_csv(path: pathlib.Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split(",")
    out = {}
    for ln in lines[1:]:
        parts = ln.split(",")
        row = dict(zip(header, parts))
        out[row.get("model", f"row{len(out)}")] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="experiments/archive/functional_learning/data/dense_candidate_readout")
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    coh_zero = load_json(COH_ZERO)
    coh_sg = load_json(COH_SG)
    transition = load_json(FAST_TRANSITION)
    dense_train = load_json(DENSE_TRAIN)
    sparse_train = load_json(SPARSE_TRAIN)
    dense_eval = load_json(DENSE_EVAL)
    sparse_rows = parse_sparse_csv(SPARSE_SYNTH)

    coh_scores = combine_scores(coh_zero, coh_sg)
    coh_scores["AoA"] = coh_scores.get("AoA") if coh_scores.get("AoA") is not None else 0.0
    coh_summary = {
        "zero_reading_payload": rel(COH_ZERO),
        "superglue_payload": rel(COH_SG),
        "scores": coh_scores,
        "cheap7": cheap7(coh_scores),
        "projected_overall_aoa0": projected_overall_aoa0(coh_scores),
    }

    score_rows = score_rows_to_dict(transition)
    fast_score_deltas = {k: score_rows[k].get("computed_delta_b_minus_a") for k in score_rows if score_rows[k].get("computed_delta_b_minus_a") is not None}
    fast_payload_deltas = {k: score_rows[k].get("payload_delta_b_minus_a") for k in score_rows if score_rows[k].get("payload_delta_b_minus_a") is not None}
    fast_transition_focus = {
        "path": rel(FAST_TRANSITION),
        "computed_score_deltas": fast_score_deltas,
        "payload_score_deltas": fast_payload_deltas,
        "net_item_deltas": {k: v.get("net_item_delta_b_minus_a") for k, v in transition["column_comparisons"].items()},
        "top_subtasks": {col: top_subtasks(transition, col) for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"] if col in transition["column_comparisons"]},
    }

    dense_common = dense_eval.get("common_target", {}).get("deltas_vs_parent", {}).get("unchanged_correspondence_focus_weighted_u0080", {})
    sparse_row = sparse_rows.get("correspondence_focus_weighted", {})
    dense_mechanism = {
        "dense_train_summary": rel(DENSE_TRAIN),
        "sparse_train_summary": rel(SPARSE_TRAIN),
        "focus_lambda_both": {
            "dense": dense_train.get("focus_lambda_arg"),
            "sparse": sparse_train.get("focus_lambda_arg"),
        },
        "target_counts": {
            "dense_total_targets": dense_train.get("total_targets"),
            "dense_focus_targets": dense_train.get("total_focus_targets"),
            "dense_ordinary_targets": dense_train.get("total_ordinary_targets"),
            "dense_focus_target_fraction": dense_train.get("aggregate_focus_target_fraction"),
            "dense_selected_groups": dense_train.get("focus_selected_groups_total"),
            "sparse_total_targets": sparse_train.get("total_targets"),
            "sparse_focus_targets": sparse_train.get("total_focus_targets"),
            "sparse_ordinary_targets": sparse_train.get("total_ordinary_targets"),
            "sparse_focus_target_fraction": sparse_train.get("aggregate_focus_target_fraction"),
            "sparse_selected_groups": sparse_train.get("focus_selected_groups_total"),
        },
        "dense_common_target_delta": dense_common,
        "sparse_synthesis_row": sparse_row,
        "mechanism_note": "Dense and sparse optimize the same focus-lambda coefficient on mean focus loss. The material change is which second-view content groups become targets and unavailable as local clues while the source remains visible; this changes the effective prediction problem and may make cross-span evidence more useful.",
    }

    # Sensitivity of the official decision: AoA is currently carried as zero for the coherent86 projection.
    coh_overall = projected_overall_aoa0(coh_scores)
    dense_official_state = task_status(DENSE_OFFICIAL)
    sensitivity = {
        "coherent86_projected_overall_aoa0": coh_overall,
        "rule": "With AoA fixed at 0, each +1.0 summed point across the eight non-AoA official columns changes Overall by +1/9 = +0.111111. Dense beats coherent86 when (dense seven non-SG columns - coherent seven non-SG columns) + (dense SuperGLUE - coherent SuperGLUE) > 0.",
        "if_dense_official_non_sg_matches_fast_delta_and_superglue_unchanged": None,
    }
    fast_cheap7_delta = fast_score_deltas.get("cheap7_mean")
    if fast_cheap7_delta is not None and coh_overall is not None:
        # seven-column mean delta times seven official columns, divided by nine.
        sensitivity["if_dense_official_non_sg_matches_fast_delta_and_superglue_unchanged"] = coh_overall + (float(fast_cheap7_delta) * 7.0) / 9.0

    result = {
        "status": "DENSE_CANDIDATE_READOUT_READY",
        "coherent86_reference": coh_summary,
        "dense_official_payload_state_at_script_time": dense_official_state,
        "official_arithmetic_sensitivity": sensitivity,
        "fast_transition_summary": fast_transition_focus,
        "dense_vs_sparse_training_and_common_target": dense_mechanism,
        "pending_runtime_tasks_known_from_step73": {
            "dense_seed62064_official_evaluation": "dense seed62064 official-compatible evaluation",
            "dense_seed62065_replication": "dense focus fixed-policy seed62065 replication",
        },
    }

    out_json = out_root / "dense_candidate_readout.json"
    out_md = out_root / "dense_candidate_readout.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research dense-focus candidate readout\n")
    lines.append("## True coherent86 reference\n")
    lines.append(f"- Zero-shot/Reading payload: `{rel(COH_ZERO)}`")
    lines.append(f"- SuperGLUE payload: `{rel(COH_SG)}`")
    lines.append(f"- cheap7: {coh_summary['cheap7']:.12f}")
    lines.append(f"- projected Overall with AoA=0: {coh_summary['projected_overall_aoa0']:.12f}\n")
    lines.append("## Fast-screen dense movement\n")
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA", "Reading", "cheap7_mean"]:
        if k in fast_score_deltas:
            lines.append(f"- {k}: computed delta {fast_score_deltas[k]:+.6f}; payload delta {fast_payload_deltas.get(k)}")
    lines.append("\n## Dense/sparse objective distinction\n")
    lines.append(f"- Dense focus targets: {dense_train.get('total_focus_targets')} of {dense_train.get('total_targets')} total targets; selected groups {dense_train.get('focus_selected_groups_total')}; lambda_focus {dense_train.get('focus_lambda_arg')}.")
    lines.append(f"- Sparse focus targets: {sparse_train.get('total_focus_targets')} of {sparse_train.get('total_targets')} total targets; selected groups {sparse_train.get('focus_selected_groups_total')}; lambda_focus {sparse_train.get('focus_lambda_arg')}.")
    lines.append("- Interpretation: the intervention broadens second-view content masking and removes local content clues while keeping the paired source visible; it does not multiply the aggregate focus coefficient.\n")
    lines.append("## Common-target movement for dense u0080\n")
    if dense_common:
        lines.append(f"- mean delta: {dense_common.get('mean_delta_expected_margin_vs_parent'):+.6f}")
        for cond, rec in dense_common.get("by_condition", {}).items():
            lines.append(f"- {cond}: mean {rec.get('mean_delta'):+.6f}, median {rec.get('median_delta'):+.6f}, n={rec.get('n')}")
        for split, rec in dense_common.get("by_split", {}).items():
            lines.append(f"- {split}: mean {rec.get('mean_delta'):+.6f}, median {rec.get('median_delta'):+.6f}, n={rec.get('n')}")
    lines.append("\n## Official arithmetic sensitivity\n")
    lines.append(sensitivity["rule"])
    if sensitivity["if_dense_official_non_sg_matches_fast_delta_and_superglue_unchanged"] is not None:
        lines.append(f"If the official non-SuperGLUE seven-column mean moved like the fast screen and SuperGLUE were unchanged, projected Overall would be approximately {sensitivity['if_dense_official_non_sg_matches_fast_delta_and_superglue_unchanged']:.12f}. This is only a sensitivity calculation, not evidence about full official columns.")
    lines.append("\n## Dense official payload state at script time\n")
    lines.append(json.dumps(dense_official_state, indent=2, ensure_ascii=False))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "dense_official_state": dense_official_state.get("status")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
