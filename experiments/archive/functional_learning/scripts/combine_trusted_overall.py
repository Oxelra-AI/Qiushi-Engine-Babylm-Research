#!/usr/bin/env python3
"""research: Combine repaired SuperGLUE with valid zero-shot/Reading into trusted Overall.

This script reads:
1. Valid zero-shot/Reading from the original (old-path) evaluations
2. Repaired SuperGLUE from the research evaluations
3. AoA from the repaired evaluations (expected: 0.0 placeholder)

And produces a combined per_target payload with a trusted Overall coordinate.
Both coherent86 and dense candidates must use this combined path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_ROOT = Path("experiments/archive/functional_learning/data/trusted_overall")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_payload(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_zero_shot_reading(payload: dict) -> dict:
    """Extract zero-shot and Reading scores from a payload."""
    tasks = payload.get("tasks", {})
    result = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                 "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]:
        if col in tasks and tasks[col].get("returncode") == 0:
            result[col] = tasks[col]
    return result


def extract_superglue_aoa(payload: dict) -> dict:
    """Extract SuperGLUE and AoA from a repaired evaluation payload."""
    tasks = payload.get("tasks", {})
    result = {}
    for col in ["SuperGLUE", "AoA"]:
        if col in tasks:
            result[col] = tasks[col]
    return result


def compute_overall(scores: dict) -> dict:
    """Compute BabyLM official-like Overall from component scores."""
    # GlobalPIQA mean
    gp_par = scores.get("GlobalPIQA_parallel")
    gp_nonpar = scores.get("GlobalPIQA_nonparallel")
    if gp_par is not None and gp_nonpar is not None:
        gp_mean = (gp_par + gp_nonpar) / 2.0
    else:
        gp_mean = None

    components = {
        "BLiMP": scores.get("BLiMP"),
        "Supplement": scores.get("Supplement"),
        "EWoK": scores.get("EWoK"),
        "Entity": scores.get("Entity"),
        "COMPS": scores.get("COMPS"),
        "SuperGLUE": scores.get("SuperGLUE"),
        "GlobalPIQA": gp_mean,
        "Reading": scores.get("Reading"),
        "AoA": scores.get("AoA", 0.0),
    }

    present = {k: v for k, v in components.items() if v is not None}
    if len(present) == 9:
        overall = sum(present.values()) / 9.0
        complete = True
    else:
        overall = None
        complete = False

    return {
        "components": components,
        "n_components": len(present),
        "complete": complete,
        "Overall": overall,
    }


def combine_payloads(label: str, zero_reading_path: str, repaired_sg_path: str) -> dict:
    """Combine zero-shot/Reading with repaired SuperGLUE into a trusted payload."""
    result = {
        "label": label,
        "zero_reading_source": zero_reading_path,
        "repaired_superglue_source": repaired_sg_path,
    }

    # Load sources
    if not os.path.exists(zero_reading_path):
        result["error"] = f"Zero-shot/Reading payload not found: {zero_reading_path}"
        return result
    if not os.path.exists(repaired_sg_path):
        result["error"] = f"Repaired SuperGLUE payload not found: {repaired_sg_path}"
        return result

    zr_payload = load_payload(zero_reading_path)
    sg_payload = load_payload(repaired_sg_path)

    zr_tasks = extract_zero_shot_reading(zr_payload)
    sg_tasks = extract_superglue_aoa(sg_payload)

    # Build score table
    scores = {}
    task_details = {}

    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if col in zr_tasks:
            score = zr_tasks[col].get("score")
            if score is not None:
                scores[col] = score
                task_details[col] = {"source": "zero_shot_original", "score": score}

    # GlobalPIQA
    if "GlobalPIQA_parallel" in zr_tasks:
        scores["GlobalPIQA_parallel"] = zr_tasks["GlobalPIQA_parallel"].get("score")
        task_details["GlobalPIQA_parallel"] = {"source": "zero_shot_original", "score": scores["GlobalPIQA_parallel"]}
    if "GlobalPIQA_nonparallel" in zr_tasks:
        scores["GlobalPIQA_nonparallel"] = zr_tasks["GlobalPIQA_nonparallel"].get("score")
        task_details["GlobalPIQA_nonparallel"] = {"source": "zero_shot_original", "score": scores["GlobalPIQA_nonparallel"]}

    # Reading
    if "Reading" in zr_tasks:
        reading_scores = zr_tasks["Reading"].get("scores", {})
        scores["Reading"] = reading_scores.get("Reading")
        task_details["Reading"] = {"source": "reading_original", "score": scores["Reading"]}

    # SuperGLUE (repaired)
    if "SuperGLUE" in sg_tasks:
        sg_rec = sg_tasks["SuperGLUE"]
        sg_mean = sg_rec.get("superglue_mean")
        scores["SuperGLUE"] = sg_mean
        # Extract per-task details
        sg_task_details = {}
        for k, v in sg_rec.items():
            if isinstance(v, dict) and "primary_metric" in v:
                sg_task_details[k] = v
        task_details["SuperGLUE"] = {
            "source": "repaired_superglue",
            "superglue_mean": sg_mean,
            "tasks": sg_task_details,
        }

    # AoA
    if "AoA" in sg_tasks:
        aoa_rec = sg_tasks["AoA"]
        scores["AoA"] = aoa_rec.get("aoa_leaderboard_score", aoa_rec.get("aoa_for_provisional_overall", 0.0))
        task_details["AoA"] = {
            "source": "repaired_aoa",
            "aoa_status": aoa_rec.get("aoa_status"),
            "aoa_leaderboard_score": scores["AoA"],
        }

    # Compute Overall
    overall = compute_overall(scores)

    result["scores"] = scores
    result["task_details"] = task_details
    result["overall_computation"] = overall
    result["trusted_overall"] = overall.get("Overall")
    result["overall_complete"] = overall.get("complete")

    return result


def main():
    # Define the combination pairs
    combinations = []

    # Coherent86: zero-shot/Reading from frontier_consolidation official reference + repaired SuperGLUE
    coherent86_zr = "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"
    coherent86_sg = "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json"
    combinations.append(("coherent86_v4_trusted", coherent86_zr, coherent86_sg))

    # Dense seed62064: zero-shot/Reading from old evaluation + repaired SuperGLUE
    dense64_zr = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json"
    dense64_sg = "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json"
    combinations.append(("dense_seed62064_trusted", dense64_zr, dense64_sg))

    # Dense seed62065: zero-shot/Reading from old evaluation + repaired SuperGLUE
    dense65_zr = "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json"
    dense65_sg = "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json"
    combinations.append(("dense_seed62065_trusted", dense65_zr, dense65_sg))

    results = {"status": "TRUSTED_OVERALL_COMBINATION", "combinations": []}

    for label, zr_path, sg_path in combinations:
        print(f"\n=== {label} ===")
        combo = combine_payloads(label, zr_path, sg_path)
        results["combinations"].append(combo)

        if "error" in combo:
            print(f"  ERROR: {combo['error']}")
            continue

        if combo.get("trusted_overall") is not None:
            print(f"  Trusted Overall: {combo['trusted_overall']:.15f}")
            scores = combo["scores"]
            for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "Reading", "AoA"]:
                print(f"    {k}: {scores.get(k)}")
            gp = combo["overall_computation"]["components"].get("GlobalPIQA")
            print(f"    GlobalPIQA: {gp}")
        else:
            print(f"  Incomplete - waiting for repaired SuperGLUE results")
            available = [k for k, v in combo["scores"].items() if v is not None]
            print(f"  Available scores: {available}")

    # Save
    out_json = OUT_ROOT / "trusted_overall_combination.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/trusted_overall/trusted_overall_combination.md')

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(out_md, "w") as f:
        f.write("# research Trusted Overall combination\n\n")
        for combo in results["combinations"]:
            f.write(f"## {combo['label']}\n")
            if "error" in combo:
                f.write(f"- Error: {combo['error']}\n\n")
                continue
            if combo.get("trusted_overall") is not None:
                f.write(f"- **Trusted Overall: `{combo['trusted_overall']:.15f}`**\n")
                f.write(f"- Complete: `{combo['overall_complete']}`\n")
                scores = combo["scores"]
                for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "Reading", "AoA"]:
                    f.write(f"  - {k}: `{scores.get(k)}`\n")
                gp = combo["overall_computation"]["components"].get("GlobalPIQA")
                f.write(f"  - GlobalPIQA (mean): `{gp}`\n")
            else:
                f.write(f"- Incomplete; waiting for repaired SuperGLUE\n")
            f.write("\n")

    print(f"\nSaved: {out_json}")
    print(json.dumps({"status": results["status"]}, indent=2))


if __name__ == "__main__":
    main()
