#!/usr/bin/env python3
"""research: Trusted comparison with measured-AoA evidence objects.

This is a small evolution of trusted_comparison.py. It keeps strict
SuperGLUE validation and the projected/measured coordinate separation, but it
fixes one important semantics bug: AoA is measured when there is a complete
research surprisal+scoring manifest, even if the official AoA score is exactly
0.0. Missing-checkpoint placeholder zero remains unmeasured.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUT_ROOT = Path("experiments/archive/functional_learning/data/trusted_comparison_with_aoa")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SUPERGLUE_REQUIRED_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]

MODEL_PATHS = {
    "coherent86": {
        "zero_reading": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/shared_aoa/assembled/coherent86/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
    "dense_seed62064": {
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/shared_aoa/assembled/dense_seed62064/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
    "dense_seed62065": {
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
        "aoa_manifest": "experiments/archive/functional_learning/data/shared_aoa/assembled/dense_seed62065/full/aoa_manifest.json",
        "expected_aoa_steps": 18,
    },
}


def load_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def validate_superglue(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"valid": False, "reason": "payload_missing"}
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    if not sg:
        return {"valid": False, "reason": "no_superglue_task"}
    details = sg.get("superglue_primary_metric_details", [])
    if not isinstance(details, list) or not details:
        return {"valid": False, "reason": "no_primary_metric_details"}
    detail_tasks = {}
    for d in details:
        if not isinstance(d, dict):
            continue
        task = d.get("task")
        score = d.get("score")
        metric = d.get("metric")
        if task and metric and score is not None:
            detail_tasks[task] = {"score": float(score), "metric": metric, "results_txt": d.get("results_txt")}
    missing = [t for t in SUPERGLUE_REQUIRED_TASKS if t not in detail_tasks]
    if missing:
        return {"valid": False, "reason": "missing_tasks", "missing": missing, "present": sorted(detail_tasks)}
    sg_mean = sg.get("superglue_mean")
    if sg_mean is None:
        sg_mean = sum(detail_tasks[t]["score"] for t in SUPERGLUE_REQUIRED_TASKS) / 7.0
    # Check returned task records too, when available, to preserve provenance.
    task_records = sg.get("tasks", [])
    bad_returncodes = [r.get("task") for r in task_records if isinstance(r, dict) and r.get("returncode") not in (0, None)]
    if bad_returncodes:
        return {"valid": False, "reason": "nonzero_subtask_returncode", "tasks": bad_returncodes}
    return {"valid": True, "superglue_mean": float(sg_mean), "per_task": detail_tasks, "n_tasks": 7}


def extract_zero_reading_scores(payload: dict[str, Any] | None) -> dict[str, float]:
    if payload is None:
        return {}
    tasks = payload.get("tasks", {})
    scores: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        t = tasks.get(col, {})
        if t.get("returncode") == 0 and t.get("score") is not None:
            scores[col] = float(t["score"])
    gp_par = tasks.get("GlobalPIQA_parallel", {}).get("score")
    gp_non = tasks.get("GlobalPIQA_nonparallel", {}).get("score")
    if gp_par is not None and gp_non is not None:
        scores["GlobalPIQA_parallel"] = float(gp_par)
        scores["GlobalPIQA_nonparallel"] = float(gp_non)
        scores["GlobalPIQA"] = (float(gp_par) + float(gp_non)) / 2.0
    reading = tasks.get("Reading", {}).get("scores", {}).get("Reading")
    if reading is not None:
        scores["Reading"] = float(reading)
    return scores


def validate_step081_aoa(path: str, expected_steps: int = 18) -> dict[str, Any]:
    man = load_json(path)
    if man is None:
        return {"status": "missing", "measured": False, "score": None, "source": path}
    payload = man.get("aoa_payload_for_comparison", {})
    score = man.get("score", {})
    assembled = man.get("assembled", {})
    summary = assembled.get("summary", {})
    complete = bool(man.get("complete_measured_evidence"))
    measured = bool(payload.get("measured") or score.get("measured"))
    n_steps = int(payload.get("n_steps") or len(assembled.get("steps", [])) or 0)
    wrong_counts = summary.get("steps_with_wrong_counts", {})
    missing_steps = summary.get("missing_steps", [])
    if not complete or not measured:
        return {"status": "present_not_measured", "measured": False, "score": None, "source": path, "reason": "manifest incomplete or scorer not run", "manifest_status": man.get("status")}
    if n_steps != expected_steps:
        return {"status": "wrong_step_count", "measured": False, "score": None, "source": path, "n_steps": n_steps, "expected_steps": expected_steps}
    if missing_steps or wrong_counts:
        return {"status": "incomplete_steps", "measured": False, "score": None, "source": path, "missing_steps": missing_steps, "wrong_counts": wrong_counts}
    # Crucial distinction: a zero score is measured if the manifest says the extraction+official scorer completed.
    estimator = man.get("aoa_estimator") or score.get("aoa_estimator") or {}
    return {
        "status": "measured",
        "measured": True,
        "score": float(payload.get("aoa_leaderboard_score", score.get("aoa_leaderboard_score"))),
        "raw_correlation": payload.get("aoa_raw_correlation", score.get("aoa_raw_correlation")),
        "legitimate_zero": bool(payload.get("legitimate_zero", score.get("legitimate_zero"))),
        "estimator_id": payload.get("estimator_id") or estimator.get("estimator_id"),
        "estimator_variant": payload.get("estimator_variant") or estimator.get("variant_name"),
        "estimator_provenance_path": payload.get("estimator_provenance_path") or estimator.get("provenance_path"),
        "leaderboard_snapshot_path": payload.get("leaderboard_snapshot_path") or estimator.get("leaderboard_snapshot_path"),
        "source": path,
        "n_results": payload.get("n_results", summary.get("n_results")),
        "n_steps": n_steps,
        "n_words_evaluated": payload.get("n_words_evaluated"),
        "n_contexts_evaluated": payload.get("n_contexts_evaluated"),
        "n_valid_words": score.get("n_valid_words"),
        "interpretation": payload.get("interpretation"),
    }


def compute_overall(scores: dict[str, float], aoa_score: float | None, aoa_type: str) -> dict[str, Any]:
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading"]
    vals = {k: scores.get(k) for k in cols}
    vals["AoA"] = aoa_score
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        return {"complete": False, "missing": missing, "aoa_type": aoa_type, "components": vals}
    return {"complete": True, "overall": sum(float(v) for v in vals.values()) / 9.0, "components": vals, "aoa_type": aoa_type}


def build_record(label: str, cfg: dict[str, Any]) -> dict[str, Any]:
    zr_payload = load_json(cfg["zero_reading"])
    sg_payload = load_json(cfg["superglue"])
    zr_scores = extract_zero_reading_scores(zr_payload)
    sg = validate_superglue(sg_payload)
    scores = dict(zr_scores)
    if sg.get("valid"):
        scores["SuperGLUE"] = sg["superglue_mean"]
    aoa = validate_step081_aoa(cfg["aoa_manifest"], cfg.get("expected_aoa_steps", 18))
    rec = {
        "label": label,
        "sources": cfg,
        "zero_reading_scores": zr_scores,
        "superglue_validation": sg,
        "aoa_validation": aoa,
        "scores_without_aoa": scores,
        "projected_overall_aoa0": compute_overall(scores, 0.0, "projected_shared_zero_placeholder"),
        "measured_overall": compute_overall(scores, aoa["score"] if aoa.get("measured") else None, "measured_step081_aoa"),
    }
    rec["submission_ready"] = bool(sg.get("valid") and aoa.get("measured") and rec["measured_overall"].get("complete"))
    rec["aoa_estimator_id"] = aoa.get("estimator_id")
    return rec


def compare(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = {"parent_label": parent["label"], "child_label": child["label"]}
    if not (parent["superglue_validation"].get("valid") and child["superglue_validation"].get("valid")):
        out["status"] = "incomplete_superglue"
        out["parent_superglue"] = parent["superglue_validation"]
        out["child_superglue"] = child["superglue_validation"]
        return out
    comp_deltas = {}
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading"]:
        pv = parent["scores_without_aoa"].get(k)
        cv = child["scores_without_aoa"].get(k)
        if pv is not None and cv is not None:
            comp_deltas[k] = cv - pv
    out["component_deltas_without_aoa"] = comp_deltas
    sg_deltas = {}
    for task in SUPERGLUE_REQUIRED_TASKS:
        p = parent["superglue_validation"].get("per_task", {}).get(task, {}).get("score")
        c = child["superglue_validation"].get("per_task", {}).get(task, {}).get("score")
        if p is not None and c is not None:
            sg_deltas[task] = {"parent": p, "child": c, "delta": c - p}
    out["superglue_per_task"] = sg_deltas
    pp = parent["projected_overall_aoa0"]
    cp = child["projected_overall_aoa0"]
    if pp.get("complete") and cp.get("complete"):
        out["projected_aoa0"] = {"complete": True, "parent_overall": pp["overall"], "child_overall": cp["overall"], "delta": cp["overall"] - pp["overall"], "note": "conditional arithmetic only; AoA=0 placeholder"}
    else:
        out["projected_aoa0"] = {"complete": False, "parent": pp, "child": cp}
    pm = parent["measured_overall"]
    cm = child["measured_overall"]
    if pm.get("complete") and cm.get("complete"):
        out["measured_overall"] = {"complete": True, "parent_overall": pm["overall"], "child_overall": cm["overall"], "delta": cm["overall"] - pm["overall"], "parent_aoa": parent["aoa_validation"], "child_aoa": child["aoa_validation"]}
    else:
        out["measured_overall"] = {"complete": False, "parent_aoa": parent["aoa_validation"], "child_aoa": child["aoa_validation"]}
    out["status"] = "complete"
    if out["measured_overall"].get("complete") and out["measured_overall"].get("delta", 0) > 0:
        out["comparison_state"] = "measured_positive"
    elif out["projected_aoa0"].get("complete") and out["projected_aoa0"].get("delta", 0) > 0:
        out["comparison_state"] = "projected_positive_awaiting_measured_aoa"
    else:
        out["comparison_state"] = "not_positive_on_available_complete_coordinate"
    return out


def main() -> None:
    records = {label: build_record(label, cfg) for label, cfg in MODEL_PATHS.items()}
    comparisons = []
    if "coherent86" in records:
        for child in ["dense_seed62064", "dense_seed62065"]:
            if child in records:
                comparisons.append(compare(records["coherent86"], records[child]))
    result = {
        "status": "TRUSTED_COMPARISON_WITH_AOA_EVIDENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Measured AoA is determined by research surprisal+official scoring evidence, not by aoa_score != 0. Missing-checkpoint placeholder zero is unmeasured; completed scorer zero is measured.",
        "models": records,
        "comparisons": comparisons,
    }
    out_json = OUT_ROOT / "trusted_comparison_with_aoa.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/trusted_comparison_with_aoa/trusted_comparison_with_aoa.md')
    with out_json.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    lines = ["# research trusted comparison with AoA evidence\n\n", f"Created: {result['created_utc']}\n\n", result["note"] + "\n\n"]
    for label, rec in records.items():
        lines.append(f"## {label}\n\n")
        lines.append(f"- SuperGLUE valid: `{rec['superglue_validation'].get('valid')}` ({rec['superglue_validation'].get('reason')})\n")
        if rec['superglue_validation'].get('valid'):
            lines.append(f"- SuperGLUE mean: `{rec['superglue_validation'].get('superglue_mean')}`\n")
        lines.append(f"- AoA status: `{rec['aoa_validation'].get('status')}`, measured: `{rec['aoa_validation'].get('measured')}`, score: `{rec['aoa_validation'].get('score')}`\n")
        if rec['aoa_validation'].get('estimator_id'):
            lines.append(f"- AoA estimator: `{rec['aoa_validation'].get('estimator_id')}`; provenance `{rec['aoa_validation'].get('estimator_provenance_path')}`\n")
        lines.append(f"- Projected Overall(AoA0): `{rec['projected_overall_aoa0'].get('overall') if rec['projected_overall_aoa0'].get('complete') else None}`\n")
        lines.append(f"- Measured Overall: `{rec['measured_overall'].get('overall') if rec['measured_overall'].get('complete') else None}`\n\n")
    for comp in comparisons:
        lines.append(f"## {comp.get('parent_label')} vs {comp.get('child_label')}\n\n")
        lines.append(f"- Status: `{comp.get('status')}`\n")
        lines.append(f"- State: `{comp.get('comparison_state')}`\n")
        if comp.get('projected_aoa0', {}).get('complete'):
            lines.append(f"- Projected AoA0 delta: `{comp['projected_aoa0']['delta']}`\n")
        if comp.get('measured_overall', {}).get('complete'):
            lines.append(f"- Measured delta: `{comp['measured_overall']['delta']}`\n")
        lines.append("\n")
    out_md.write_text("".join(lines))
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "n_comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
