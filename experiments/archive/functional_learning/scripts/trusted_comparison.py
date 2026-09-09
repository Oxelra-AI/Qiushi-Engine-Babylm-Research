#!/usr/bin/env python3
"""research: Compare repaired SuperGLUE for coherent86 vs dense candidates.

Reads repaired SuperGLUE evaluation outputs, combines with valid zero-shot/Reading,
computes trusted Overall, and produces a detailed comparison showing:
1. How the adapter repair changed SuperGLUE for each model
2. Per-task SuperGLUE comparison between coherent86 and dense
3. Trusted Overall comparison with breakdown
4. Whether v5 is established

Prerequisites: completed coherent86 and dense seed62064 repaired SuperGLUE payloads.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_ROOT = Path("experiments/archive/functional_learning/data/trusted_comparison")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_superglue_tasks(payload: dict) -> dict:
    """Extract per-task SuperGLUE data from a payload."""
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    tasks = sg.get("tasks", [])
    details = sg.get("superglue_primary_metric_details", [])
    
    result = {
        "superglue_mean": sg.get("superglue_mean"),
        "superglue_mean_legacy": sg.get("superglue_mean_accuracy_only_legacy"),
        "per_task": {},
    }
    
    # From tasks list
    if isinstance(tasks, list):
        for t in tasks:
            if isinstance(t, dict):
                name = t.get("task", "")
                result["per_task"][name] = {
                    "accuracy": t.get("accuracy"),
                    "n_correct": t.get("n_correct", t.get("correct")),
                }
    
    # From details list
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                name = d.get("task", "")
                if name in result["per_task"]:
                    result["per_task"][name]["primary_metric"] = d.get("metric")
                    result["per_task"][name]["primary_score"] = d.get("score")
                else:
                    result["per_task"][name] = {
                        "primary_metric": d.get("metric"),
                        "primary_score": d.get("score"),
                    }
    
    return result


def compute_overall_from_scores(scores: dict) -> float | None:
    """Compute Overall from component scores dict."""
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
    vals = [scores.get(k) for k in keys]
    if all(v is not None for v in vals):
        return sum(vals) / 9.0
    return None


def main():
    # Paths for zero-shot/Reading sources
    coherent86_zr_path = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
    dense64_zr_path = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
    
    # Paths for repaired SuperGLUE
    coherent86_sg_path = "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json"
    dense64_sg_path = "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json"
    
    # Old (broken) SuperGLUE for comparison
    old_coherent86_sg_path = "experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json"
    old_dense64_sg_path = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
    
    results = {
        "status": "TRUSTED_COMPARISON",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    
    # Check which repaired results are available
    for label, sg_path in [("coherent86", coherent86_sg_path), ("dense_seed62064", dense64_sg_path)]:
        if os.path.exists(sg_path):
            print(f"✓ {label} repaired SuperGLUE available")
        else:
            print(f"✗ {label} repaired SuperGLUE NOT YET available")
    
    both_available = os.path.exists(coherent86_sg_path) and os.path.exists(dense64_sg_path)
    
    if not both_available:
        print("\nWaiting for both repaired evaluations to complete.")
        results["complete"] = False
        out_json = OUT_ROOT / "trusted_comparison.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved incomplete state: {out_json}")
        return
    
    # Load all data
    coherent86_zr = load_json(coherent86_zr_path)
    dense64_zr = load_json(dense64_zr_path)
    coherent86_sg = load_json(coherent86_sg_path)
    dense64_sg = load_json(dense64_sg_path)
    old_coherent86_sg = load_json(old_coherent86_sg_path)
    old_dense64_sg = load_json(old_dense64_sg_path)
    
    # Build score tables
    for label, zr, new_sg, old_sg in [
        ("coherent86", coherent86_zr, coherent86_sg, old_coherent86_sg),
        ("dense_seed62064", dense64_zr, dense64_sg, old_dense64_sg),
    ]:
        zr_tasks = zr.get("tasks", {})
        scores = {}
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
            scores[col] = zr_tasks.get(col, {}).get("score")
        gp_par = zr_tasks.get("GlobalPIQA_parallel", {}).get("score")
        gp_nonpar = zr_tasks.get("GlobalPIQA_nonparallel", {}).get("score")
        scores["GlobalPIQA"] = (gp_par + gp_nonpar) / 2.0 if gp_par and gp_nonpar else None
        scores["Reading"] = zr_tasks.get("Reading", {}).get("scores", {}).get("Reading")
        
        # Repaired SuperGLUE
        new_sg_data = extract_superglue_tasks(new_sg)
        scores["SuperGLUE"] = new_sg_data["superglue_mean"]
        scores["AoA"] = 0.0
        
        # Old SuperGLUE
        old_sg_data = extract_superglue_tasks(old_sg)
        
        overall = compute_overall_from_scores(scores)
        
        results["models"][label] = {
            "scores": scores,
            "overall": overall,
            "repaired_superglue": new_sg_data,
            "old_superglue": old_sg_data,
            "superglue_repair_delta": (new_sg_data["superglue_mean"] - old_sg_data["superglue_mean"])
                if new_sg_data["superglue_mean"] and old_sg_data["superglue_mean"] else None,
        }
    
    # Comparison
    c = results["models"]["coherent86"]
    d = results["models"]["dense_seed62064"]
    
    comparison = {
        "coherent86_overall": c["overall"],
        "dense_overall": d["overall"],
        "overall_delta": d["overall"] - c["overall"] if d["overall"] and c["overall"] else None,
    }
    
    # Per-component deltas
    comp_deltas = {}
    for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]:
        cv = c["scores"].get(k)
        dv = d["scores"].get(k)
        comp_deltas[k] = dv - cv if cv is not None and dv is not None else None
    comparison["component_deltas"] = comp_deltas
    
    # Per-task SuperGLUE comparison
    sg_task_comparison = {}
    c_sg = c["repaired_superglue"]["per_task"]
    d_sg = d["repaired_superglue"]["per_task"]
    for task in sorted(set(list(c_sg.keys()) + list(d_sg.keys()))):
        ct = c_sg.get(task, {})
        dt = d_sg.get(task, {})
        ca = ct.get("accuracy") or ct.get("primary_score")
        da = dt.get("accuracy") or dt.get("primary_score")
        sg_task_comparison[task] = {
            "coherent86": ca,
            "dense": da,
            "delta": da - ca if da is not None and ca is not None else None,
        }
    comparison["superglue_per_task"] = sg_task_comparison
    
    # Adapter repair effect
    comparison["coherent86_sg_repair_delta"] = c["superglue_repair_delta"]
    comparison["dense_sg_repair_delta"] = d["superglue_repair_delta"]
    
    # v5 verdict
    v5_established = comparison["overall_delta"] is not None and comparison["overall_delta"] > 0
    comparison["v5_established"] = v5_established
    comparison["old_v4_overall"] = 42.1210247099666
    
    results["comparison"] = comparison
    results["complete"] = True
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRUSTED OVERALL COMPARISON (repaired SuperGLUE)")
    print(f"{'='*60}")
    print(f"  coherent86 Overall: {c['overall']:.15f}" if c['overall'] else "  coherent86 Overall: INCOMPLETE")
    print(f"  dense64    Overall: {d['overall']:.15f}" if d['overall'] else "  dense64    Overall: INCOMPLETE")
    if comparison["overall_delta"]:
        print(f"  Delta (dense-v4):   {comparison['overall_delta']:+.15f}")
    print(f"  Old v4 Overall:     {comparison['old_v4_overall']:.15f} (with stock-model SuperGLUE)")
    print(f"\n  SuperGLUE repair effect:")
    print(f"    coherent86: {c['superglue_repair_delta']:+.6f}" if c['superglue_repair_delta'] else "    coherent86: N/A")
    print(f"    dense64:    {d['superglue_repair_delta']:+.6f}" if d['superglue_repair_delta'] else "    dense64:    N/A")
    print(f"\n  Per-task SuperGLUE (repaired):")
    for task, vals in sorted(sg_task_comparison.items()):
        delta_str = f"{vals['delta']:+.4f}" if vals['delta'] is not None else "N/A"
        print(f"    {task:8s}: v4={vals['coherent86']}, dense={vals['dense']}, delta={delta_str}")
    print(f"\n  v5 established: {v5_established}")
    
    # Save
    out_json = OUT_ROOT / "trusted_comparison.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/trusted_comparison/trusted_comparison.md')
    
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(out_md, "w") as f:
        f.write("# research Trusted Overall comparison\n\n")
        f.write(f"## Overall\n")
        f.write(f"- coherent86/v4 trusted: `{c['overall']}`\n" if c['overall'] else "- coherent86/v4: INCOMPLETE\n")
        f.write(f"- dense seed62064 trusted: `{d['overall']}`\n" if d['overall'] else "- dense: INCOMPLETE\n")
        if comparison["overall_delta"]:
            f.write(f"- Delta (dense - v4): `{comparison['overall_delta']:+.15f}`\n")
        f.write(f"- Old v4 Overall (stock SG): `{comparison['old_v4_overall']}`\n\n")
        f.write(f"## SuperGLUE repair effect\n")
        c_rd = c['superglue_repair_delta']
        d_rd = d['superglue_repair_delta']
        f.write(f"- coherent86: `{c_rd:+.6f}` (old: {c['old_superglue']['superglue_mean']}, new: {c['repaired_superglue']['superglue_mean']})\n" if c_rd is not None else "- coherent86: N/A (repaired SG not yet complete)\n")
        f.write(f"- dense64: `{d_rd:+.6f}` (old: {d['old_superglue']['superglue_mean']}, new: {d['repaired_superglue']['superglue_mean']})\n\n" if d_rd is not None else "- dense64: N/A (repaired SG not yet complete)\n\n")
        f.write(f"## Per-task SuperGLUE (repaired)\n")
        f.write("| Task | coherent86 | dense | delta |\n")
        f.write("|------|-----------|-------|-------|\n")
        for task, vals in sorted(sg_task_comparison.items()):
            d_val = vals.get('delta')
            f.write(f"| {task} | {vals['coherent86']} | {vals['dense']} | {d_val:+.4f if d_val is not None else 'N/A'} |\n")
        f.write(f"\n## Component deltas (dense - coherent86)\n")
        for k, v in sorted(comp_deltas.items()):
            f.write(f"- {k}: `{v:+.6f}`\n" if v else f"- {k}: N/A\n")
        f.write(f"\n**v5 established: `{v5_established}`**\n")
    
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
