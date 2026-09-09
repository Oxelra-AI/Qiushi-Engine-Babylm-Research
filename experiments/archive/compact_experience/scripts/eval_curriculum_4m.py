#!/usr/bin/env python3
"""Fast evaluation of the four masking-curriculum 4M arms.

Evaluates: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA (parallel+nonparallel), Reading.
Reuses the official BabyLM evaluation pipeline from INITIAL_MODEL_STUDIES.

Usage:
  # Evaluate arms 0,1 on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python eval_curriculum_4m.py --arms wwm_fixed,wwm_to_token --gpu 0
  # Evaluate arms 2,3 on GPU 1:
  CUDA_VISIBLE_DEVICES=1 python eval_curriculum_4m.py --arms amlm_hard,amlm_hard_switch --gpu 1
  # Aggregate all results:
  python eval_curriculum_4m.py --aggregate
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_REPO = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
DATA_PATH = EVAL_REPO / "evaluation_data"
RUN_BASE = Path("experiments/archive/compact_experience/training/runs")
OUT_BASE = Path("experiments/archive/compact_experience/data/curriculum_eval")

ARM_MAP = {
    "wwm_fixed": "wwm_fixed_4m_seed43",
    "wwm_to_token": "wwm_to_token_4m_seed43",
    "amlm_hard": "amlm_hard_4m_seed43",
    "amlm_hard_switch": "amlm_hard_switch_4m_seed43",
}

TASKS_ZERO_SHOT = [
    ("BLiMP", "blimp"),
    ("Supplement", "supplement"),
    ("EWoK", "ewok"),
    ("Entity", "entity_tracking"),
    ("COMPS", "comps"),
    ("GlobalPIQA_par", "global_piqa_parallel"),
    ("GlobalPIQA_nonpar", "global_piqa_nonparallel"),
]


def run_zero_shot(model_path: Path, task_id: str, output_dir: Path, gpu: int) -> float | None:
    """Run one zero-shot evaluation task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--data_path", str(DATA_PATH),
        "--task", task_id,
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--output_dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=str(EVAL_REPO), timeout=300)
    if result.returncode != 0:
        print(f"  ERROR {task_id}: {result.stderr[-500:]}", flush=True)
        return None
    # Parse score from output
    score = parse_score(output_dir, task_id)
    return score


def run_reading(model_path: Path, output_dir: Path, gpu: int) -> float | None:
    """Run reading evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--data_path", str(DATA_PATH),
        "--model_path_or_name", str(model_path),
        "--backend", "mlm",
        "--output_dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=str(EVAL_REPO), timeout=600)
    if result.returncode != 0:
        print(f"  ERROR reading: {result.stderr[-500:]}", flush=True)
        return None
    score = parse_reading_score(output_dir)
    return score


def parse_score(output_dir: Path, task_id: str) -> float | None:
    """Parse accuracy from evaluation output."""
    # Try results.json first
    results_file = output_dir / "results.json"
    if results_file.exists():
        data = json.loads(results_file.read_text())
        if "accuracy" in data:
            return float(data["accuracy"]) * 100
        if "score" in data:
            return float(data["score"]) * 100
    # Try finding score in output files
    for f in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict):
                if "accuracy" in data:
                    return float(data["accuracy"]) * 100
                if "score" in data:
                    return float(data["score"]) * 100
                # Nested format
                for v in data.values():
                    if isinstance(v, dict) and "accuracy" in v:
                        return float(v["accuracy"]) * 100
        except:
            pass
    # Try parsing from predictions
    preds_file = output_dir / "predictions.jsonl"
    if preds_file.exists():
        correct = 0
        total = 0
        with open(preds_file) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "correct" in obj:
                    total += 1
                    if obj["correct"]:
                        correct += 1
        if total > 0:
            return (correct / total) * 100
    return None


def parse_reading_score(output_dir: Path) -> float | None:
    """Parse reading evaluation score."""
    for f in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict):
                if "mean_score" in data:
                    return float(data["mean_score"])
                if "score" in data:
                    return float(data["score"])
                if "accuracy" in data:
                    return float(data["accuracy"]) * 100
        except:
            pass
    return None


def evaluate_arm(arm_key: str, gpu: int) -> dict:
    """Evaluate one arm on all fast tasks."""
    run_dir = ARM_MAP[arm_key]
    model_path = RUN_BASE / run_dir / "hf_model" / "chck_4M"
    if not model_path.exists():
        return {"arm": arm_key, "error": "checkpoint not found"}
    
    arm_out = OUT_BASE / "per_arm"
    arm_out.mkdir(parents=True, exist_ok=True)
    
    scores = {"arm": arm_key, "model_path": str(model_path)}
    start = time.time()
    
    # Zero-shot tasks
    for name, task_id in TASKS_ZERO_SHOT:
        task_out = OUT_BASE / "task_outputs" / arm_key / task_id
        print(json.dumps({"event": "eval_start", "arm": arm_key, "column": name, "gpu": gpu}), flush=True)
        score = run_zero_shot(model_path, task_id, task_out, gpu)
        scores[name] = score
        print(json.dumps({"event": "eval_done", "arm": arm_key, "column": name, "score": score, "rc": 0 if score is not None else 1}), flush=True)
    
    # Reading
    reading_out = OUT_BASE / "task_outputs" / arm_key / "reading"
    print(json.dumps({"event": "eval_start", "arm": arm_key, "column": "Reading", "gpu": gpu}), flush=True)
    reading_score = run_reading(model_path, reading_out, gpu)
    scores["Reading"] = reading_score
    print(json.dumps({"event": "eval_done", "arm": arm_key, "column": "Reading", "score": reading_score}), flush=True)
    
    # GlobalPIQA mean
    par = scores.get("GlobalPIQA_par")
    nonpar = scores.get("GlobalPIQA_nonpar")
    if par is not None and nonpar is not None:
        scores["GlobalPIQA_mean"] = (par + nonpar) / 2
    
    scores["elapsed_sec"] = round(time.time() - start, 1)
    
    # Save per-arm result
    (arm_out / f"{arm_key}.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps({"event": "arm_done", "arm": arm_key, "elapsed_sec": scores["elapsed_sec"],
                      "per_arm": str(arm_out / f'{arm_key}.json')}), flush=True)
    return scores


def aggregate():
    """Aggregate all arm results into a summary."""
    arm_out = OUT_BASE / "per_arm"
    all_scores = {}
    for f in sorted(arm_out.glob("*.json")):
        data = json.loads(f.read_text())
        arm = data.get("arm", f.stem)
        all_scores[arm] = data
    
    if not all_scores:
        print("No results to aggregate")
        return
    
    # Columns for weighted proxy (same as research)
    # BLiMP weight=1, Supplement=1, EWoK=1, Entity=1, COMPS=1, GlobalPIQA_mean=1, Reading=1
    # Simple equal-weight mean of available fast columns
    columns = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    
    table = {}
    for arm, scores in all_scores.items():
        vals = [scores.get(c) for c in columns]
        valid = [v for v in vals if v is not None]
        table[arm] = {
            "scores": {c: scores.get(c) for c in columns},
            "fast_proxy_mean": sum(valid) / len(valid) if valid else None,
            "n_valid": len(valid),
        }
    
    # Contrasts
    contrasts = {}
    if "wwm_fixed" in table and "wwm_to_token" in table:
        contrasts["wwm_to_token_minus_wwm_fixed"] = {
            c: (table["wwm_to_token"]["scores"].get(c, 0) or 0) - (table["wwm_fixed"]["scores"].get(c, 0) or 0)
            for c in columns
        }
        t1 = table["wwm_to_token"]["fast_proxy_mean"] or 0
        t2 = table["wwm_fixed"]["fast_proxy_mean"] or 0
        contrasts["wwm_to_token_minus_wwm_fixed"]["fast_proxy_delta"] = t1 - t2
    
    if "wwm_fixed" in table and "amlm_hard" in table:
        contrasts["amlm_hard_minus_wwm_fixed"] = {
            c: (table["amlm_hard"]["scores"].get(c, 0) or 0) - (table["wwm_fixed"]["scores"].get(c, 0) or 0)
            for c in columns
        }
        t1 = table["amlm_hard"]["fast_proxy_mean"] or 0
        t2 = table["wwm_fixed"]["fast_proxy_mean"] or 0
        contrasts["amlm_hard_minus_wwm_fixed"]["fast_proxy_delta"] = t1 - t2
    
    if "wwm_fixed" in table and "amlm_hard_switch" in table:
        contrasts["amlm_hard_switch_minus_wwm_fixed"] = {
            c: (table["amlm_hard_switch"]["scores"].get(c, 0) or 0) - (table["wwm_fixed"]["scores"].get(c, 0) or 0)
            for c in columns
        }
        t1 = table["amlm_hard_switch"]["fast_proxy_mean"] or 0
        t2 = table["wwm_fixed"]["fast_proxy_mean"] or 0
        contrasts["amlm_hard_switch_minus_wwm_fixed"]["fast_proxy_delta"] = t1 - t2
    
    if "amlm_hard" in table and "amlm_hard_switch" in table:
        contrasts["amlm_hard_switch_minus_amlm_hard"] = {
            c: (table["amlm_hard_switch"]["scores"].get(c, 0) or 0) - (table["amlm_hard"]["scores"].get(c, 0) or 0)
            for c in columns
        }
        t1 = table["amlm_hard_switch"]["fast_proxy_mean"] or 0
        t2 = table["amlm_hard"]["fast_proxy_mean"] or 0
        contrasts["amlm_hard_switch_minus_amlm_hard"]["fast_proxy_delta"] = t1 - t2
    
    summary = {
        "status": "CURRICULUM_EVAL_DONE",
        "arms_evaluated": list(all_scores.keys()),
        "table": table,
        "contrasts": contrasts,
    }
    
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "curriculum_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    
    # Write note
    note_path = Path("research/notes/compact_experience/masking_curriculum_eval.md")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# research: Masking Curriculum 4M Evaluation\n"]
    lines.append(f"Arms: {', '.join(all_scores.keys())}\n")
    lines.append("\n## Scores\n\n")
    lines.append(f"| Arm | " + " | ".join(columns) + " | Fast Proxy |\n")
    lines.append("|" + "---|" * (len(columns) + 2) + "\n")
    for arm in ["wwm_fixed", "wwm_to_token", "amlm_hard", "amlm_hard_switch"]:
        if arm not in table:
            continue
        row = table[arm]
        vals = [f"{row['scores'].get(c, 0):.2f}" if row['scores'].get(c) is not None else "?" for c in columns]
        lines.append(f"| {arm} | " + " | ".join(vals) + f" | {row['fast_proxy_mean']:.3f} |\n")
    lines.append("\n## Contrasts vs wwm_fixed\n\n")
    for cname, cvals in contrasts.items():
        lines.append(f"**{cname}**: proxy delta = {cvals.get('fast_proxy_delta', 0):.3f}\n")
        for c in columns:
            v = cvals.get(c, 0)
            lines.append(f"  {c}: {v:+.2f}\n")
        lines.append("\n")
    note_path.write_text("".join(lines), encoding="utf-8")
    
    print(json.dumps({"status": "aggregated", "summary": str(OUT_BASE / 'curriculum_eval_summary.json'),
                      "note": str(note_path)}))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="", help="Comma-separated arm keys to evaluate")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    
    if args.aggregate:
        aggregate()
        return
    
    if not args.arms:
        print("Specify --arms or --aggregate")
        return
    
    arm_keys = [a.strip() for a in args.arms.split(",")]
    for arm_key in arm_keys:
        if arm_key not in ARM_MAP:
            print(f"Unknown arm: {arm_key}")
            continue
        evaluate_arm(arm_key, args.gpu)


if __name__ == "__main__":
    main()
