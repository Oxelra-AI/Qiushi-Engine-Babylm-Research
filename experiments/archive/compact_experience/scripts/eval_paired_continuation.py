#!/usr/bin/env python3
"""research: Evaluate the paired continuation checkpoints (WWM vs token from chck_70M).

Uses the research-style direct-path evaluation for fair comparison with the
inherited INITIAL_MODEL_STUDIES coordinate. Evaluates: BLiMP, Supplement, EWoK, Entity, COMPS, Reading.

Usage:
  python eval_paired_continuation.py --mode all     # Run all evaluations
  python eval_paired_continuation.py --mode summary # Aggregate existing results
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
from typing import Dict, Any

ROOT = pathlib.Path(".")
RUN_BASE = ROOT / "experiments/archive/compact_experience/training/runs"
OUT_ROOT = ROOT / "experiments/archive/compact_experience/data/paired_eval"
NOTE_PATH = ROOT / "research/notes/compact_experience/paired_continuation_eval.md"

ARMS = {
    "wwm": RUN_BASE / "continuation_wwm_seed43" / "hf_model" / "chck_100M",
    "token": RUN_BASE / "continuation_token_seed43" / "hf_model" / "chck_100M",
}

# INITIAL_MODEL_STUDIES reference for comparison
INITIAL_MODEL_STUDIES_REF = ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M"

EVAL_REPO = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
FAST_EVAL_DATA = EVAL_REPO / "evaluation_data/fast_eval"
FULL_EVAL_DATA = EVAL_REPO / "evaluation_data/full_eval"

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
           "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def run_eval(model_path: pathlib.Path, column: str, gpu: int, work_tag: str) -> Dict[str, Any]:
    """Run a single BabyLM evaluation column."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    out_dir = OUT_ROOT / "eval_outputs" / f"{work_tag}_{column}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if column == "BLiMP":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "blimp", "--data_path", str(FAST_EVAL_DATA / "blimp_fast"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "Supplement":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "blimp", "--data_path", str(FAST_EVAL_DATA / "supplement_fast"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "EWoK":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "ewok", "--data_path", str(FAST_EVAL_DATA / "ewok_fast"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "Entity":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "entity_tracking",
               "--data_path", str(FULL_EVAL_DATA / "entity_tracking"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "COMPS":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "comps", "--data_path", str(FULL_EVAL_DATA / "comps"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "GlobalPIQA_parallel":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "piqa_modified",
               "--data_path", str(FAST_EVAL_DATA / "global_piqa_fast/parallel"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "GlobalPIQA_nonparallel":
        cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--task", "piqa_modified",
               "--data_path", str(FAST_EVAL_DATA / "global_piqa_fast/nonparallel"),
               "--save_predictions", "--batch_size", "64", "--output_dir", str(out_dir)]
    elif column == "Reading":
        cmd = [sys.executable, "-m", "evaluation_pipeline.reading_times.run",
               "--model_path_or_name", str(model_path), "--backend", "mlm",
               "--data_path", str(FAST_EVAL_DATA / "reading_times_fast"),
               "--batch_size", "64", "--output_dir", str(out_dir)]
    else:
        raise ValueError(f"Unknown column: {column}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EVAL_REPO), env=env)
    
    if result.returncode != 0:
        return {"column": column, "error": result.stderr[-500:], "rc": result.returncode}

    # Parse score from output
    score = parse_score(result.stdout, column, out_dir)
    return {"column": column, "score": score, "rc": 0}


def parse_score(stdout: str, column: str, out_dir: pathlib.Path) -> float:
    """Parse the evaluation score from stdout or result files."""
    if column == "Reading":
        # Reading outputs JSON with correlations
        for line in stdout.strip().split("\n"):
            if "eye_tracking" in line or "self_paced" in line:
                pass
        # Check result file
        result_files = list(out_dir.glob("*.json"))
        if result_files:
            data = json.loads(result_files[0].read_text())
            if "mean_correlation" in data:
                return round(data["mean_correlation"] * 100, 3)
        # Parse from stdout
        eye = sp = None
        for line in stdout.strip().split("\n"):
            if "eye_tracking" in line.lower() and ":" in line:
                try: eye = float(line.split(":")[-1].strip())
                except: pass
            if "self_paced" in line.lower() and ":" in line:
                try: sp = float(line.split(":")[-1].strip())
                except: pass
        if eye is not None and sp is not None:
            return round((eye + sp) / 2, 3)
        # Last resort: look for a single number
        for line in reversed(stdout.strip().split("\n")):
            parts = line.strip().split()
            if parts:
                try:
                    return float(parts[-1])
                except:
                    pass
        return 0.0
    else:
        # Sentence tasks: parse "overall_accuracy: XX.XX" or last numeric line
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            if "overall" in line.lower() and ":" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except:
                    pass
            # Try just a number
            parts = line.strip().split(":")
            if len(parts) == 2:
                try:
                    return float(parts[1].strip())
                except:
                    pass
        # Look in result files
        result_files = list(out_dir.glob("*results*.json")) + list(out_dir.glob("*.json"))
        for rf in result_files:
            try:
                data = json.loads(rf.read_text())
                if "overall_accuracy" in data:
                    return round(data["overall_accuracy"], 2)
                if "accuracy" in data:
                    return round(data["accuracy"], 2)
            except:
                pass
        return 0.0


def evaluate_all(gpu: int = 0) -> Dict[str, Dict[str, float]]:
    """Evaluate all arms on all columns."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    
    all_targets = list(ARMS.items())
    if INITIAL_MODEL_STUDIES_REF.exists():
        all_targets.append(("initial_model_studies_ref", INITIAL_MODEL_STUDIES_REF))
    
    for arm_name, model_path in all_targets:
        if not model_path.exists():
            print(f"WARNING: {arm_name} checkpoint not found at {model_path}", flush=True)
            continue
        arm_results = {}
        for column in COLUMNS:
            work_tag = f"{arm_name}_100M"
            print(json.dumps({"event": "eval_start", "arm": arm_name, "column": column}), flush=True)
            result = run_eval(model_path, column, gpu, work_tag)
            arm_results[column] = result.get("score", 0.0)
            print(json.dumps({"event": "eval_done", "arm": arm_name, "column": column,
                            "score": result.get("score"), "rc": result.get("rc")}), flush=True)
        results[arm_name] = arm_results
    
    return results


def summarize(results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Compute summary statistics and differentials."""
    summary = {"arms": results}
    
    # Compute GlobalPIQA mean and weighted fast proxy for each arm
    for arm_name, scores in results.items():
        gpiqa_p = scores.get("GlobalPIQA_parallel", 0)
        gpiqa_np = scores.get("GlobalPIQA_nonparallel", 0)
        scores["GlobalPIQA_mean"] = round((gpiqa_p + gpiqa_np) / 2, 3)
        
        # Weighted fast proxy (7 columns, equal weight)
        proxy_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
        vals = [scores.get(c, 0) for c in proxy_cols]
        scores["weighted_fast_proxy"] = round(sum(vals) / len(vals), 4)
    
    # Differential: token minus wwm
    if "wwm" in results and "token" in results:
        diff = {}
        for col in list(results["wwm"].keys()):
            diff[col] = round(results["token"].get(col, 0) - results["wwm"].get(col, 0), 4)
        summary["differential_token_minus_wwm"] = diff
    
    # Comparison with INITIAL_MODEL_STUDIES reference
    if "initial_model_studies_ref" in results and "wwm" in results:
        diff_ref = {}
        for col in list(results["wwm"].keys()):
            diff_ref[col] = round(results["wwm"].get(col, 0) - results["initial_model_studies_ref"].get(col, 0), 4)
        summary["wwm_minus_initial_model_studies_ref"] = diff_ref
    
    return summary


def write_note(summary: Dict[str, Any]):
    """Write human-readable note."""
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# research — Paired continuation evaluation (WWM vs token from chck_70M)",
        "",
        f"JSON: `{OUT_ROOT / 'paired_eval_summary.json'}`",
        "",
        "## Design",
        "Both arms loaded INITIAL_MODEL_STUDIES research chck_70M (70M WWM-trained weights),",
        "used fresh AdamW with identical LR phase (cosine from research/2442),",
        "trained on the exact INITIAL_MODEL_STUDIES post-70M example order for 30M more words.",
        "Only difference: mask_mode (wwm vs token).",
        "This isolates the masking-granularity effect from the optimizer-restart confound.",
        "",
        "## Scores",
        "",
        "| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, scores in summary.get("arms", {}).items():
        cols = [scores.get(c, 0) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                                            "GlobalPIQA_mean", "Reading", "weighted_fast_proxy"]]
        lines.append(f"| {arm_name} | " + " | ".join(f"{v:.2f}" for v in cols) + " |")
    
    if "differential_token_minus_wwm" in summary:
        diff = summary["differential_token_minus_wwm"]
        lines.extend(["", "## Differential (token - wwm)", ""])
        for col, val in diff.items():
            lines.append(f"- {col}: {val:+.3f}")
    
    if "wwm_minus_initial_model_studies_ref" in summary:
        diff = summary["wwm_minus_initial_model_studies_ref"]
        lines.extend(["", "## WWM continuation vs INITIAL_MODEL_STUDIES original chck_100M", ""])
        for col, val in diff.items():
            lines.append(f"- {col}: {val:+.3f}")
    
    lines.append("")
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Note written: {NOTE_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "summary"], default="all")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "paired_eval_summary.json"
    
    if args.mode == "all":
        results = evaluate_all(args.gpu)
        summary = summarize(results)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_note(summary)
        print(json.dumps({"status": "PAIRED_EVAL_DONE",
                         "json": str(json_path), "note": str(NOTE_PATH)}, indent=2))
    elif args.mode == "summary":
        if json_path.exists():
            summary = json.loads(json_path.read_text())
            print(json.dumps(summary, indent=2))
        else:
            print("No existing summary found. Run with --mode all first.")


if __name__ == "__main__":
    main()
