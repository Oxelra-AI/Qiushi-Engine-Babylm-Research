#!/usr/bin/env python3
"""research: Trusted Overall comparison with strict SuperGLUE validation,
separated AoA coordinates, and corrected v5 verdict logic.

Fixes research scripts: requires all 7 SuperGLUE subtasks with primary metrics,
separates projected Overall(AoA0) from measured Overall, and does not set v5
established from projected delta alone.

Three distinct evaluation coordinates:
  1. projected_overall_aoa0: uses shared AoA=0.0 placeholder for conditional arithmetic
  2. measured_overall: uses real AoA from actual surprisal curve evaluation
  3. submission_ready: all official components complete, validated, and ready for platform

The historical platform-style coherent86 record (42.1210...) used stock-model SuperGLUE.
That record is preserved as context but is not the baseline for the trusted comparison.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_ROOT = Path("experiments/archive/functional_learning/data/trusted_comparison")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# The seven official SuperGLUE subtasks that must all be present
SUPERGLUE_REQUIRED_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def validate_superglue(payload: dict) -> dict:
    """Strictly validate SuperGLUE: all 7 tasks, each with primary metric detail."""
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    if not sg:
        return {"valid": False, "reason": "no_superglue_task"}
    
    # Check for primary metric details
    details = sg.get("superglue_primary_metric_details", [])
    if not isinstance(details, list) or len(details) == 0:
        return {"valid": False, "reason": "no_primary_metric_details"}
    
    detail_tasks = {}
    for d in details:
        if isinstance(d, dict) and "task" in d:
            task = d["task"]
            score = d.get("score")
            metric = d.get("metric")
            if score is not None and metric is not None:
                detail_tasks[task] = {"score": score, "metric": metric}
    
    # Check all 7 tasks present
    missing = [t for t in SUPERGLUE_REQUIRED_TASKS if t not in detail_tasks]
    if missing:
        return {"valid": False, "reason": f"missing_tasks: {missing}", 
                "present": list(detail_tasks.keys())}
    
    # Check for reasonable scores (not None/NaN)
    for task in SUPERGLUE_REQUIRED_TASKS:
        score = detail_tasks[task]["score"]
        if score is None or (isinstance(score, float) and (score != score)):  # NaN check
            return {"valid": False, "reason": f"invalid_score_for_{task}: {score}"}
    
    # Get the mean
    sg_mean = sg.get("superglue_mean")
    if sg_mean is None:
        # Compute from details
        sg_mean = sum(detail_tasks[t]["score"] for t in SUPERGLUE_REQUIRED_TASKS) / 7.0
    
    return {
        "valid": True,
        "superglue_mean": sg_mean,
        "per_task": detail_tasks,
        "n_tasks": len(detail_tasks),
    }


def validate_aoa(payload: dict) -> dict:
    """Check AoA status: measured vs placeholder."""
    aoa_task = payload.get("tasks", {}).get("AoA", {})
    if not aoa_task:
        return {"status": "missing", "measured": False, "score": None}
    
    aoa_status = aoa_task.get("aoa_status", "")
    aoa_score = aoa_task.get("aoa_leaderboard_score", aoa_task.get("aoa_for_provisional_overall"))
    
    if "missing_checkpoints" in str(aoa_status):
        return {"status": "placeholder_zero", "measured": False, "score": 0.0,
                "reason": "Required developmental checkpoints not staged"}
    elif aoa_score is not None and aoa_score != 0.0:
        return {"status": "measured", "measured": True, "score": aoa_score}
    else:
        return {"status": "unknown_zero", "measured": False, "score": 0.0}


def extract_zero_reading_scores(payload: dict) -> dict:
    """Extract zero-shot and Reading scores."""
    tasks = payload.get("tasks", {})
    scores = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        t = tasks.get(col, {})
        if t.get("returncode") == 0 and t.get("score") is not None:
            scores[col] = t["score"]
    
    gp_par = tasks.get("GlobalPIQA_parallel", {}).get("score")
    gp_nonpar = tasks.get("GlobalPIQA_nonparallel", {}).get("score")
    if gp_par is not None and gp_nonpar is not None:
        scores["GlobalPIQA"] = (gp_par + gp_nonpar) / 2.0
        scores["GlobalPIQA_parallel"] = gp_par
        scores["GlobalPIQA_nonparallel"] = gp_nonpar
    
    reading = tasks.get("Reading", {}).get("scores", {}).get("Reading")
    if reading is not None:
        scores["Reading"] = reading
    
    return scores


def compute_projected_overall_aoa0(scores: dict) -> dict:
    """Projected Overall using shared AoA=0 convention. Not measured."""
    components = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", 
                  "SuperGLUE", "GlobalPIQA", "Reading"]
    vals = {k: scores.get(k) for k in components}
    vals["AoA"] = 0.0  # Projected placeholder, shared convention
    
    present = {k: v for k, v in vals.items() if v is not None}
    if len(present) == 9:
        return {
            "complete": True,
            "overall": sum(present.values()) / 9.0,
            "components": vals,
            "aoa_type": "projected_zero_placeholder",
        }
    return {"complete": False, "missing": [k for k in vals if vals[k] is None]}


def compute_measured_overall(scores: dict, aoa_score: float) -> dict:
    """Measured Overall using actual AoA from surprisal evaluation."""
    components = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                  "SuperGLUE", "GlobalPIQA", "Reading"]
    vals = {k: scores.get(k) for k in components}
    vals["AoA"] = aoa_score
    
    present = {k: v for k, v in vals.items() if v is not None}
    if len(present) == 9:
        return {
            "complete": True,
            "overall": sum(present.values()) / 9.0,
            "components": vals,
            "aoa_type": "measured",
        }
    return {"complete": False, "missing": [k for k in vals if vals[k] is None]}


def build_model_record(label: str, zr_path: str, sg_path: str) -> dict:
    """Build a complete model evaluation record."""
    record = {
        "label": label,
        "zero_reading_source": zr_path,
        "superglue_source": sg_path,
    }
    
    # Load zero-shot/Reading
    zr_payload = load_json(zr_path)
    if zr_payload is None:
        record["error"] = f"Zero-shot/Reading not found: {zr_path}"
        return record
    zr_scores = extract_zero_reading_scores(zr_payload)
    record["zero_reading_scores"] = zr_scores
    
    # Load and validate SuperGLUE
    sg_payload = load_json(sg_path)
    if sg_payload is None:
        record["superglue_status"] = "not_yet_available"
        record["scores"] = zr_scores
        return record
    
    sg_validation = validate_superglue(sg_payload)
    record["superglue_validation"] = sg_validation
    
    if not sg_validation["valid"]:
        record["superglue_status"] = "invalid"
        record["scores"] = zr_scores
        return record
    
    record["superglue_status"] = "valid"
    
    # Combine scores
    scores = dict(zr_scores)
    scores["SuperGLUE"] = sg_validation["superglue_mean"]
    record["scores"] = scores
    
    # AoA
    aoa_validation = validate_aoa(sg_payload)
    record["aoa_validation"] = aoa_validation
    
    # Compute projected Overall(AoA0)
    record["projected_overall_aoa0"] = compute_projected_overall_aoa0(scores)
    
    # Compute measured Overall (only if AoA is measured)
    if aoa_validation["measured"]:
        record["measured_overall"] = compute_measured_overall(scores, aoa_validation["score"])
    else:
        record["measured_overall"] = {"complete": False, "reason": "AoA not measured"}
    
    # Submission readiness
    record["submission_ready"] = (
        sg_validation["valid"] and
        aoa_validation["measured"] and
        record.get("measured_overall", {}).get("complete", False) and
        all(k in scores for k in ["BLiMP", "Supplement", "EWoK", "Entity", 
                                   "COMPS", "SuperGLUE", "GlobalPIQA", "Reading"])
    )
    
    return record


def compare_models(parent: dict, child: dict) -> dict:
    """Compare parent (coherent86) and child (dense) with proper separation."""
    comparison = {
        "parent_label": parent["label"],
        "child_label": child["label"],
    }
    
    # Check if both have valid SuperGLUE
    parent_sg_valid = parent.get("superglue_status") == "valid"
    child_sg_valid = child.get("superglue_status") == "valid"
    comparison["both_superglue_valid"] = parent_sg_valid and child_sg_valid
    
    if not comparison["both_superglue_valid"]:
        comparison["status"] = "incomplete_superglue"
        missing = []
        if not parent_sg_valid:
            missing.append(f"parent ({parent.get('superglue_status')})")
        if not child_sg_valid:
            missing.append(f"child ({child.get('superglue_status')})")
        comparison["missing"] = missing
        return comparison
    
    # Per-component deltas
    comp_deltas = {}
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", 
              "SuperGLUE", "GlobalPIQA", "Reading"]:
        pv = parent.get("scores", {}).get(k)
        cv = child.get("scores", {}).get(k)
        if pv is not None and cv is not None:
            comp_deltas[k] = cv - pv
    comparison["component_deltas"] = comp_deltas
    
    # Per-task SuperGLUE deltas
    p_sg = parent.get("superglue_validation", {}).get("per_task", {})
    c_sg = child.get("superglue_validation", {}).get("per_task", {})
    sg_deltas = {}
    for task in SUPERGLUE_REQUIRED_TASKS:
        ps = p_sg.get(task, {}).get("score")
        cs = c_sg.get(task, {}).get("score")
        if ps is not None and cs is not None:
            sg_deltas[task] = {"parent": ps, "child": cs, "delta": cs - ps}
    comparison["superglue_per_task"] = sg_deltas
    
    # Projected Overall(AoA0) comparison
    p_proj = parent.get("projected_overall_aoa0", {})
    c_proj = child.get("projected_overall_aoa0", {})
    if p_proj.get("complete") and c_proj.get("complete"):
        delta = c_proj["overall"] - p_proj["overall"]
        comparison["projected_aoa0"] = {
            "parent_overall": p_proj["overall"],
            "child_overall": c_proj["overall"],
            "delta": delta,
            "child_higher": delta > 0,
            "note": "Shared AoA=0 projected placeholder; not measured AoA",
        }
    else:
        comparison["projected_aoa0"] = {"complete": False}
    
    # Measured Overall comparison (if both have measured AoA)
    p_meas = parent.get("measured_overall", {})
    c_meas = child.get("measured_overall", {})
    if p_meas.get("complete") and c_meas.get("complete"):
        delta = c_meas["overall"] - p_meas["overall"]
        comparison["measured_overall"] = {
            "parent_overall": p_meas["overall"],
            "child_overall": c_meas["overall"],
            "delta": delta,
            "child_higher": delta > 0,
        }
    else:
        comparison["measured_overall"] = {
            "complete": False,
            "parent_aoa_measured": parent.get("aoa_validation", {}).get("measured", False),
            "child_aoa_measured": child.get("aoa_validation", {}).get("measured", False),
        }
    
    # v5 status: layered, never from projected alone
    if comparison.get("measured_overall", {}).get("child_higher"):
        if parent.get("submission_ready") and child.get("submission_ready"):
            comparison["v5_status"] = "submission_ready_positive"
        else:
            comparison["v5_status"] = "measured_positive_not_submission_ready"
    elif comparison.get("projected_aoa0", {}).get("child_higher"):
        comparison["v5_status"] = "projected_aoa0_positive_only"
    else:
        comparison["v5_status"] = "not_established"
    
    # Structural information
    comparison["status"] = "complete"
    comparison["historical_platform_v4_overall"] = 42.1210247099666
    comparison["historical_note"] = (
        "The old v4 Overall 42.121 used stock-model SuperGLUE (adapter bypass). "
        "This comparison uses adapter-model SuperGLUE for both parent and child, "
        "making it the first internally consistent Overall comparison."
    )
    
    return comparison


def main():
    results = {
        "status": "TRUSTED_COMPARISON",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "interface_validation": "experiments/archive/functional_learning/data/real_interface_validation/real_interface_validation.json",
    }
    
    # Model definitions: (label, zero_reading_path, repaired_superglue_path)
    models = {
        "coherent86": (
            "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval"
            "coherent86_private_scale_0p75/per_target/"
            "coherent86_private_scale_0p75.json",
            "experiments/archive/functional_learning/data/repaired_coherent86_eval"
            "per_target/repaired_coherent86_alpha075.json",
        ),
        "dense_seed62064": (
            "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064"
            "per_target/dense_focus_seed62064_u0080.json",
            "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064"
            "per_target/repaired_dense_seed62064_u0080.json",
        ),
        "dense_seed62065": (
            "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065"
            "per_target/dense_focus_seed62065_u0080.json",
            "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065"
            "per_target/repaired_dense_seed62065_u0080.json",
        ),
    }
    
    for label, (zr_path, sg_path) in models.items():
        print(f"\n{'='*60}")
        print(f"Building record: {label}")
        record = build_model_record(label, zr_path, sg_path)
        results["models"][label] = record
        
        if record.get("error"):
            print(f"  ERROR: {record['error']}")
            continue
        
        print(f"  SuperGLUE: {record.get('superglue_status')}")
        if record.get("superglue_status") == "valid":
            sv = record["superglue_validation"]
            print(f"    Mean: {sv['superglue_mean']:.6f}")
            for task, d in sv.get("per_task", {}).items():
                print(f"    {task}: {d['score']:.4f}")
        
        aoa = record.get("aoa_validation", {})
        print(f"  AoA: status={aoa.get('status')}, measured={aoa.get('measured')}")
        
        proj = record.get("projected_overall_aoa0", {})
        if proj.get("complete"):
            print(f"  Projected Overall(AoA0): {proj['overall']:.15f}")
    
    # Comparisons
    comparisons = []
    
    if "coherent86" in results["models"] and "dense_seed62064" in results["models"]:
        comp = compare_models(results["models"]["coherent86"], results["models"]["dense_seed62064"])
        comparisons.append(comp)
        print(f"\n{'='*60}")
        print("COMPARISON: coherent86 vs dense_seed62064")
        if comp.get("status") == "complete":
            proj = comp.get("projected_aoa0", {})
            if proj.get("complete", False) is not False and "delta" in proj:
                print(f"  Projected Overall(AoA0) delta: {proj['delta']:+.15f}")
                print(f"  Child higher: {proj.get('child_higher')}")
            meas = comp.get("measured_overall", {})
            if meas.get("complete"):
                print(f"  Measured Overall delta: {meas['delta']:+.15f}")
            else:
                print(f"  Measured Overall: incomplete (AoA not measured)")
            print(f"  v5 status: {comp.get('v5_status')}")
            
            # SuperGLUE per-task
            print(f"\n  SuperGLUE per-task deltas (dense - coherent86):")
            for task, d in comp.get("superglue_per_task", {}).items():
                print(f"    {task:8s}: {d['parent']:.4f} -> {d['child']:.4f} ({d['delta']:+.4f})")
            
            # Component deltas
            print(f"\n  Component deltas:")
            for k, v in comp.get("component_deltas", {}).items():
                print(f"    {k:12s}: {v:+.6f}")
        else:
            print(f"  Status: {comp.get('status')}")
            if comp.get("missing"):
                print(f"  Missing: {comp['missing']}")
    
    if "coherent86" in results["models"] and "dense_seed62065" in results["models"]:
        comp65 = compare_models(results["models"]["coherent86"], results["models"]["dense_seed62065"])
        comparisons.append(comp65)
    
    results["comparisons"] = comparisons
    
    # Save
    out_json = OUT_ROOT / "trusted_comparison.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/trusted_comparison/trusted_comparison.md')
    
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(out_md, "w") as f:
        f.write("# research Trusted Overall Comparison\n\n")
        f.write(f"Created: {results['created_utc']}\n\n")
        f.write("## Evaluation Coordinates\n\n")
        f.write("- **Projected Overall(AoA0)**: shared AoA=0 placeholder for conditional arithmetic\n")
        f.write("- **Measured Overall**: uses real AoA from surprisal curve evaluation\n")
        f.write("- **Submission ready**: all components complete, validated, ready for platform\n\n")
        f.write("The historical v4 Overall `42.1210247099666` used stock-model SuperGLUE ")
        f.write("(adapter bypass). This comparison uses adapter-model SuperGLUE for both, ")
        f.write("making it the first internally consistent comparison.\n\n")
        
        for label, record in results["models"].items():
            f.write(f"## {label}\n\n")
            if record.get("error"):
                f.write(f"ERROR: {record['error']}\n\n")
                continue
            f.write(f"- SuperGLUE: {record.get('superglue_status')}\n")
            if record.get("superglue_status") == "valid":
                sv = record["superglue_validation"]
                f.write(f"- SuperGLUE mean: `{sv['superglue_mean']:.6f}`\n")
            aoa = record.get("aoa_validation", {})
            f.write(f"- AoA: {aoa.get('status')} (measured: {aoa.get('measured')})\n")
            proj = record.get("projected_overall_aoa0", {})
            if proj.get("complete"):
                f.write(f"- Projected Overall(AoA0): `{proj['overall']:.15f}`\n")
            f.write(f"- Submission ready: {record.get('submission_ready', False)}\n\n")
        
        for comp in comparisons:
            f.write(f"## Comparison: {comp.get('parent_label')} vs {comp.get('child_label')}\n\n")
            if comp.get("status") == "complete":
                proj = comp.get("projected_aoa0", {})
                if "delta" in proj:
                    f.write(f"- Projected Overall(AoA0) delta: `{proj['delta']:+.15f}`\n")
                meas = comp.get("measured_overall", {})
                if meas.get("complete"):
                    f.write(f"- Measured Overall delta: `{meas['delta']:+.15f}`\n")
                else:
                    f.write(f"- Measured Overall: incomplete (AoA not measured)\n")
                f.write(f"- **v5 status: `{comp.get('v5_status')}`**\n\n")
                
                if comp.get("superglue_per_task"):
                    f.write("| Task | Parent | Child | Delta |\n")
                    f.write("|------|--------|-------|-------|\n")
                    for task, d in sorted(comp.get("superglue_per_task", {}).items()):
                        f.write(f"| {task} | {d['parent']:.4f} | {d['child']:.4f} | {d['delta']:+.4f} |\n")
                    f.write("\n")
            else:
                f.write(f"Status: {comp.get('status')}\n\n")
    
    print(f"\nSaved: {out_json}")
    print(json.dumps({"status": results["status"], 
                       "n_models": len(results["models"]),
                       "n_comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
