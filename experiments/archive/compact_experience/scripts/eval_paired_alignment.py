#!/usr/bin/env python3
"""research: Evaluate paired-alignment training arms on BabyLM fast eval.

Evaluates chck_100M for each completed arm on BLiMP, Supplement, EWoK, Entity,
COMPS, GlobalPIQA (parallel + nonparallel), and Reading. Uses the known-good
BabyLM strict evaluation invocation from Steps 8/10/11/15.

Contrast table:
- ALIGNED vs MISMATCHED: pure alignment effect (critical)
- ALIGNED vs SINGLE_REPEAT: multi-view diversity vs repetition
- ALIGNED vs SINGLE_ORIG: alignment vs coverage
- ALIGNED vs INITIAL_MODEL_STUDIES research: comparison to official-corpus baseline
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

EVAL_CWD = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
OUT_BASE = Path("experiments/archive/compact_experience/data/paired_alignment_eval")
NOTE_PATH = Path("research/notes/compact_experience/paired_alignment_eval.md")

TARGETS = {
    "aligned": "experiments/archive/compact_experience/training/runs/aligned_100M_seed43/hf_model/chck_100M",
    "mismatched": "experiments/archive/compact_experience/training/runs/mismatched_100M_seed43/hf_model/chck_100M",
    "single_repeat": "experiments/archive/compact_experience/training/runs/single_repeat_100M_seed43/hf_model/chck_100M",
    "single_orig": "experiments/archive/compact_experience/training/runs/single_orig_100M_seed43/hf_model/chck_100M",
    "initial_model_baseline": "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M",
}

# Evaluation tasks and their data paths (relative to EVAL_CWD)
TASKS = {
    "BLiMP": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark blimp --model {model}",
        "data": "evaluation_data/full_eval/blimp",
    },
    "Supplement": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark blimp_supplement --model {model}",
        "data": "evaluation_data/full_eval/blimp_supplement",
    },
    "EWoK": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark ewok --model {model}",
        "data": "evaluation_data/full_eval/ewok",
    },
    "Entity": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark entity_tracking --model {model}",
        "data": "evaluation_data/fast_eval/entity_tracking",
    },
    "COMPS": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark comps --model {model}",
        "data": "evaluation_data/full_eval/comps",
    },
    "GlobalPIQA_parallel": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark piqa_modified --model {model} --piqa_type parallel",
        "data": "evaluation_data/full_eval/piqa_modified",
    },
    "GlobalPIQA_nonparallel": {
        "script": "python -m evaluation_pipeline.evaluator",
        "args": "--benchmark piqa_modified --model {model} --piqa_type non-parallel",
        "data": "evaluation_data/full_eval/piqa_modified",
    },
    "Reading": {
        "script": "python -m evaluation_pipeline.reading",
        "args": "--model {model}",
        "data": "evaluation_data/full_eval/reading_times",
    },
}


def run_eval(target_name: str, model_path: str, task_name: str, gpu: int = 0) -> dict:
    """Run a single evaluation task."""
    abs_model = str(Path(model_path).resolve())
    task = TASKS[task_name]
    
    cmd = f"{task['script']} {task['args'].format(model=abs_model)}"
    
    env_str = f"CUDA_VISIBLE_DEVICES={gpu} TOKENIZERS_PARALLELISM=false"
    full_cmd = f"{env_str} {cmd}"
    
    out_dir = OUT_BASE / "eval_outputs" / target_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_BASE / "eval_logs" / f"{target_name}_{task_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        full_cmd, shell=True, capture_output=True, text=True,
        cwd=str(EVAL_CWD), timeout=600
    )
    
    log_path.write_text(f"CMD: {full_cmd}\nRC: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
    
    # Parse score from stdout
    score = None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "score" in data:
                score = data["score"]
            elif "accuracy" in data:
                score = data["accuracy"] * 100
        except (json.JSONDecodeError, ValueError):
            pass
        # Try plain float
        if score is None:
            try:
                val = float(line)
                if 0 <= val <= 100:
                    score = val
            except ValueError:
                pass
    
    # For reading times, look for correlation
    if score is None and "reading" in task_name.lower():
        for line in result.stdout.strip().split("\n"):
            try:
                data = json.loads(line)
                if "mean_delta_loglik" in data:
                    score = data["mean_delta_loglik"]
                elif "correlation" in data:
                    score = data["correlation"] * 100
            except (json.JSONDecodeError, ValueError):
                pass
    
    return {
        "score": score,
        "rc": result.returncode,
        "log": str(log_path)
    }


def parse_reading_score(log_path: str) -> float:
    """Parse reading score from the best_temperature_report pattern."""
    try:
        log_text = Path(log_path).read_text()
        for line in log_text.split("\n"):
            if "best_temperature_report" in line or "mean_delta_loglik" in line:
                try:
                    d = json.loads(line)
                    if "mean_delta_loglik" in d:
                        return d["mean_delta_loglik"]
                except:
                    pass
            # Also try the standard output format
            if "eye_tracking" in line or "self_paced" in line:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and "score" in d:
                        return d["score"]
                except:
                    pass
    except:
        pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=None,
                       help="Specific targets to evaluate (default: all available)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Check which targets are available
    available = {}
    for name, path in TARGETS.items():
        if Path(path).exists():
            available[name] = path
    
    if args.targets:
        available = {k: v for k, v in available.items() if k in args.targets}
    
    if not available:
        print("No target checkpoints found. Training may not be complete yet.")
        print("Expected paths:")
        for name, path in TARGETS.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
        return
    
    if args.summary_only:
        summary_path = OUT_BASE / "paired_alignment_eval_summary.json"
        if summary_path.exists():
            print(json.dumps(json.loads(summary_path.read_text()), indent=2))
        else:
            print("No summary found yet.")
        return
    
    print(f"Evaluating {len(available)} targets: {list(available.keys())}")
    
    # Run evaluations
    all_results = {}
    for target_name, model_path in available.items():
        print(f"\n{'='*40}")
        print(f"  Target: {target_name}")
        print(f"  Path: {model_path}")
        print(f"{'='*40}")
        
        target_scores = {}
        for task_name in TASKS:
            print(f"    {task_name}...", end=" ", flush=True)
            result = run_eval(target_name, model_path, task_name, args.gpu)
            target_scores[task_name] = result
            print(f"{'✓' if result['score'] is not None else '✗'} "
                  f"score={result['score']}", flush=True)
        
        # Compute aggregates
        blimp = target_scores.get("BLiMP", {}).get("score")
        supp = target_scores.get("Supplement", {}).get("score")
        ewok = target_scores.get("EWoK", {}).get("score")
        entity = target_scores.get("Entity", {}).get("score")
        comps = target_scores.get("COMPS", {}).get("score")
        gp_par = target_scores.get("GlobalPIQA_parallel", {}).get("score")
        gp_nonpar = target_scores.get("GlobalPIQA_nonparallel", {}).get("score")
        reading = target_scores.get("Reading", {}).get("score")
        
        gp_mean = None
        if gp_par is not None and gp_nonpar is not None:
            gp_mean = (gp_par + gp_nonpar) / 2
        
        summary_scores = {
            "BLiMP": blimp, "Supplement": supp, "EWoK": ewok,
            "Entity": entity, "COMPS": comps,
            "GlobalPIQA_parallel": gp_par,
            "GlobalPIQA_nonparallel": gp_nonpar,
            "GlobalPIQA_mean": gp_mean,
            "Reading": reading,
        }
        
        # Equal-7 mean (the 7 standard fast columns)
        cols7 = [blimp, supp, ewok, entity, comps, gp_mean, reading]
        valid7 = [c for c in cols7 if c is not None]
        equal7 = sum(valid7) / len(valid7) if valid7 else None
        
        summary_scores["equal7_mean"] = equal7
        
        all_results[target_name] = {
            "model_path": model_path,
            "scores": summary_scores,
            "raw": target_scores
        }
        
        print(f"\n    Summary: equal7={equal7:.3f}" if equal7 else "\n    Summary: incomplete")
    
    # Compute contrasts
    contrasts = {}
    if "aligned" in all_results and "mismatched" in all_results:
        a = all_results["aligned"]["scores"]
        m = all_results["mismatched"]["scores"]
        contrasts["aligned_minus_mismatched"] = {
            k: round(a[k] - m[k], 4) if a.get(k) is not None and m.get(k) is not None else None
            for k in a
        }
    
    if "aligned" in all_results and "single_repeat" in all_results:
        a = all_results["aligned"]["scores"]
        r = all_results["single_repeat"]["scores"]
        contrasts["aligned_minus_single_repeat"] = {
            k: round(a[k] - r[k], 4) if a.get(k) is not None and r.get(k) is not None else None
            for k in a
        }
    
    if "aligned" in all_results and "initial_model_baseline" in all_results:
        a = all_results["aligned"]["scores"]
        b = all_results["initial_model_baseline"]["scores"]
        contrasts["aligned_minus_initial_model_studies_baseline"] = {
            k: round(a[k] - b[k], 4) if a.get(k) is not None and b.get(k) is not None else None
            for k in a
        }
    
    # Save summary
    summary = {
        "status": "PAIRED_ALIGNMENT_EVAL",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targets_evaluated": list(all_results.keys()),
        "results": {k: v["scores"] for k, v in all_results.items()},
        "contrasts": contrasts,
    }
    
    summary_path = OUT_BASE / "paired_alignment_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
