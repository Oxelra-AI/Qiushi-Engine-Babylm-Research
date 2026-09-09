#!/usr/bin/env python3
"""research: Cheap7 evaluation for causal GPT models.

Runs official zero-shot tasks (BLiMP, BLiMP Supplement, EWoK, Entity, COMPS,
GlobalPIQA) and Reading with --backend causal, then collates into cheap7.

Usage:
  python causal_cheap7_eval.py \
    --model-dir training/runs/causal_compact_seed43022/hf_model/chck_20M \
    --output-dir data/causal_eval/compact_chck_20M \
    --gpu 0
"""
import argparse, json, os, pathlib, subprocess, sys, time, hashlib

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
EVAL_REPO = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline")
ZERO_SHOT = EVAL_REPO / "sentence_zero_shot" / "run.py"
READING = EVAL_REPO / "reading" / "run.py"

# Official task definitions matching evaluate_compliant_endpoint.py
BLIMP_ROOT = EVAL_REPO / "full_eval" / "blimp"
SUPP_ROOT = EVAL_REPO / "full_eval" / "blimp_supplement"
EWOK_ROOT = EVAL_REPO / "full_eval" / "ewok"
ENTITY_ROOT = EVAL_REPO / "full_eval" / "entity_tracking"
COMPS_ROOT = EVAL_REPO / "full_eval" / "comps" / "comps"
GLOBALPIQA_FULL = EVAL_REPO / "full_eval" / "global_piqa"
READING_DATA = EVAL_REPO / "full_eval" / "reading" / "reading_data.csv"

ZERO_SHOT_TASKS = [
    {"column": "BLiMP",       "task": "blimp",            "data_path": str(BLIMP_ROOT.resolve()),    "batch_size": 128},
    {"column": "Supplement",  "task": "supplement",        "data_path": str(SUPP_ROOT.resolve()),     "batch_size": 128},
    {"column": "EWoK",        "task": "ewok",             "data_path": str(EWOK_ROOT.resolve()),     "batch_size": 128},
    {"column": "Entity",      "task": "entity_tracking",   "data_path": str(ENTITY_ROOT.resolve()),   "batch_size": 128},
    {"column": "COMPS",       "task": "comps",             "data_path": str(COMPS_ROOT.resolve()),    "batch_size": 128},
    {"column": "GlobalPIQA_parallel",   "task": "global_piqa_parallel",    "data_path": str((GLOBALPIQA_FULL / "global_piqa_parallel").resolve()),    "batch_size": 128},
    {"column": "GlobalPIQA_nonparallel","task": "global_piqa_nonparallel", "data_path": str((GLOBALPIQA_FULL / "global_piqa_nonparallel").resolve()), "batch_size": 128},
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run_zero_shot(model_dir, task_info, out_base, gpu):
    task_out = out_base / "zero_shot" / task_info["task"]
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ZERO_SHOT.resolve()),
        "--model_path_or_name", str(pathlib.Path(model_dir).resolve()),
        "--task", task_info["task"],
        "--task_path", task_info["data_path"],
        "--backend", "causal",
        "--batch_size", str(task_info["batch_size"]),
        "--output_dir", str(task_out.resolve()),
        "--save_predictions",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HOME"] = str(out_base / ".hf_cache")
    env["TRANSFORMERS_CACHE"] = str(out_base / ".hf_cache")
    env["HF_MODULES_CACHE"] = str(out_base / ".hf_modules")
    print(f"  Running {task_info['task']}...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  ERROR in {task_info['task']}: {proc.stderr[-500:]}", flush=True)
        return None
    # Parse output for accuracy
    return task_out


def run_reading(model_dir, out_base, gpu):
    task_out = out_base / "reading"
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(READING.resolve()),
        "--model_path_or_name", str(pathlib.Path(model_dir).resolve()),
        "--test_path", str(READING_DATA.resolve()),
        "--output_dir", str(task_out.resolve()),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HOME"] = str(out_base / ".hf_cache")
    env["TRANSFORMERS_CACHE"] = str(out_base / ".hf_cache")
    env["HF_MODULES_CACHE"] = str(out_base / ".hf_modules")
    print(f"  Running reading...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"  ERROR in reading: {proc.stderr[-500:]}", flush=True)
        return None
    return task_out


def collect_scores(out_base, model_dir):
    """Compute scores from prediction files using official scoring logic.
    
    Reimplements the core of calculate_results_from_pred.py score calculation
    for the 7 cheap columns without requiring the full collation pipeline.
    """
    scores = {}
    eval_data = EVAL_REPO / "full_eval"

    # --- BLiMP ---
    pred_path = out_base / "zero_shot" / "blimp"
    blimp_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if blimp_pred_files:
        try:
            preds = json.loads(blimp_pred_files[0].read_text())
            subtask_scores = []
            data_dir = eval_data / "blimp"
            for subtask, results in preds.items():
                items = results.get("predictions", results) if isinstance(results, dict) else results
                data_file = (data_dir / subtask).with_suffix(".jsonl")
                if not data_file.exists():
                    continue
                correct, total = 0, 0
                with open(data_file) as df:
                    for res, line in zip(items, df):
                        d = json.loads(line)
                        total += 1
                        if res.get("pred", "").strip() == d.get("sentence_good", "").strip():
                            correct += 1
                if total > 0:
                    subtask_scores.append(correct / total * 100)
            if subtask_scores:
                scores["BLiMP"] = sum(subtask_scores) / len(subtask_scores)
        except Exception as e:
            print(f"  BLiMP scoring error: {e}", flush=True)

    # --- Supplement ---
    pred_path = out_base / "zero_shot" / "supplement"
    supp_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if supp_pred_files:
        try:
            preds = json.loads(supp_pred_files[0].read_text())
            subtask_scores = []
            data_dir = eval_data / "blimp_supplement"
            for subtask, results in preds.items():
                items = results.get("predictions", results) if isinstance(results, dict) else results
                data_file = (data_dir / subtask).with_suffix(".jsonl")
                if not data_file.exists():
                    continue
                correct, total = 0, 0
                with open(data_file) as df:
                    for res, line in zip(items, df):
                        d = json.loads(line)
                        total += 1
                        if res.get("pred", "").strip() == d.get("sentence_good", "").strip():
                            correct += 1
                if total > 0:
                    subtask_scores.append(correct / total * 100)
            if subtask_scores:
                scores["Supplement"] = sum(subtask_scores) / len(subtask_scores)
        except Exception as e:
            print(f"  Supplement scoring error: {e}", flush=True)

    # --- EWoK ---
    pred_path = out_base / "zero_shot" / "ewok"
    ewok_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if ewok_pred_files:
        try:
            preds = json.loads(ewok_pred_files[0].read_text())
            subtask_scores = []
            data_dir = eval_data / "ewok"
            for subtask, results in preds.items():
                items = results.get("predictions", results) if isinstance(results, dict) else results
                data_file = (data_dir / subtask).with_suffix(".jsonl")
                if not data_file.exists():
                    continue
                correct, total = 0, 0
                with open(data_file) as df:
                    for res, line in zip(items, df):
                        d = json.loads(line)
                        total += 1
                        gold = " ".join([d.get("Context1", ""), d.get("Target1", "")])
                        if res.get("pred", "").strip() == gold.strip():
                            correct += 1
                if total > 0:
                    subtask_scores.append(correct / total * 100)
            if subtask_scores:
                scores["EWoK"] = sum(subtask_scores) / len(subtask_scores)
        except Exception as e:
            print(f"  EWoK scoring error: {e}", flush=True)

    # --- Entity Tracking ---
    pred_path = out_base / "zero_shot" / "entity_tracking"
    et_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if et_pred_files:
        try:
            preds = json.loads(et_pred_files[0].read_text())
            subtask_scores = []
            data_dir = eval_data / "entity_tracking"
            for data_file in sorted(data_dir.glob("*.jsonl")):
                num_ops = 0
                idx = 0
                correct, total = 0, 0
                subsubtask = f"{data_file.stem}_{num_ops}_ops"
                with open(data_file) as df:
                    for line in df:
                        d = json.loads(line)
                        if d.get("numops") != num_ops:
                            if total > 0:
                                subtask_scores.append(correct / total * 100)
                            correct, total, idx = 0, 0, 0
                            num_ops = d["numops"]
                            subsubtask = f"{data_file.stem}_{num_ops}_ops"
                        if subsubtask in preds:
                            items = preds[subsubtask].get("predictions", [])
                            if idx < len(items):
                                total += 1
                                if items[idx].get("pred", "").strip() == d["options"][0].strip():
                                    correct += 1
                        idx += 1
                    if total > 0:
                        subtask_scores.append(correct / total * 100)
            if subtask_scores:
                scores["Entity"] = sum(subtask_scores) / len(subtask_scores)
        except Exception as e:
            print(f"  Entity scoring error: {e}", flush=True)

    # --- COMPS ---
    pred_path = out_base / "zero_shot" / "comps"
    comps_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if comps_pred_files:
        try:
            preds = json.loads(comps_pred_files[0].read_text())
            subtask_scores = []
            file_map = {"base": "comps_base", "wugs_dist_before": "comps_wugs_dist-before",
                        "wugs_dist_in_between": "comps_wugs_dist-in-between", "wugs": "comps_wugs"}
            data_dir = eval_data / "comps"
            for subtask, results in preds.items():
                items = results.get("predictions", results) if isinstance(results, dict) else results
                fname = file_map.get(subtask, subtask)
                data_file = (data_dir / fname).with_suffix(".jsonl")
                if not data_file.exists():
                    continue
                correct, total = 0, 0
                with open(data_file) as df:
                    for res, line in zip(items, df):
                        d = json.loads(line)
                        total += 1
                        gold = " ".join([d.get("prefix_acceptable", ""), d.get("property_phrase", "")])
                        if res.get("pred", "").strip() == gold.strip():
                            correct += 1
                if total > 0:
                    subtask_scores.append(correct / total * 100)
            if subtask_scores:
                scores["COMPS"] = sum(subtask_scores) / len(subtask_scores)
        except Exception as e:
            print(f"  COMPS scoring error: {e}", flush=True)

    # --- GlobalPIQA ---
    for gp_task in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        pred_path = out_base / "zero_shot" / gp_task
        gp_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
        if gp_files:
            try:
                preds = json.loads(gp_files[0].read_text())
                correct, total = 0, 0
                data_dir = eval_data / "global_piqa" / gp_task
                for data_file in sorted(data_dir.glob("*.jsonl")):
                    with open(data_file) as df:
                        for line in df:
                            d = json.loads(line)
                            eid = d.get("example_id", "")
                            if eid in preds:
                                total += 1
                                items = preds[eid].get("predictions", [])
                                if items and items[0].get("pred") == d.get("label"):
                                    correct += 1
                if total > 0:
                    col = "GlobalPIQA_parallel" if "parallel" in gp_task and "non" not in gp_task else "GlobalPIQA_nonparallel"
                    scores[col] = correct / total * 100
            except Exception as e:
                print(f"  {gp_task} scoring error: {e}", flush=True)

    # --- Reading ---
    pred_path = out_base / "reading"
    reading_pred_files = sorted(pred_path.glob("**/predictions.json")) if pred_path.exists() else []
    if reading_pred_files:
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
            preds_data = json.loads(reading_pred_files[0].read_text())
            reading_items = preds_data.get("reading", {}).get("predictions", [])
            data = pd.read_csv(str(READING_DATA), dtype={"item": str})
            data["pred"] = [it["pred"] for it in reading_items]
            data["prev_pred"] = [it["prev_pred"] for it in reading_items]
            # Self-paced reading R² increment
            et_vars = ["RTfirstfix", "RTfirstpass", "RTgopast", "RTrightbound"]
            et_results = []
            for dv in et_vars:
                temp = data[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
                base = smf.ols(f"{dv} ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length", data=temp).fit()
                temp2 = data[[dv, "Subtlex_log10", "length", "context_length", "pred"]].dropna()
                full = smf.ols(f"{dv} ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred", data=temp2).fit()
                et_results.append(((full.rsquared - base.rsquared) / (1 - base.rsquared)) * 100)
            # SPR
            temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
            base = smf.ols("self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred", data=temp).fit()
            temp2 = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred", "pred"]].dropna()
            full = smf.ols("self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred", data=temp2).fit()
            spr_result = ((full.rsquared - base.rsquared) / (1 - base.rsquared)) * 100
            scores["Reading"] = (spr_result + sum(et_results) / len(et_results)) / 2
        except Exception as e:
            print(f"  Reading scoring error: {e}", flush=True)

    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF model directory")
    ap.add_argument("--output-dir", required=True, help="Evaluation output directory")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    out_base = pathlib.Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating: {args.model_dir}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"GPU: {args.gpu}", flush=True)

    # Run all zero-shot tasks
    for task_info in ZERO_SHOT_TASKS:
        run_zero_shot(args.model_dir, task_info, out_base, args.gpu)

    # Run reading
    run_reading(args.model_dir, out_base, args.gpu)

    # Collect and compute cheap7
    scores = collect_scores(out_base, args.model_dir)
    print(f"\n  Raw scores: {json.dumps(scores, indent=2)}", flush=True)

    # Compute GlobalPIQA as mean of parallel and nonparallel
    if "GlobalPIQA_parallel" in scores and "GlobalPIQA_nonparallel" in scores:
        scores["GlobalPIQA"] = (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2

    # Cheap7 columns
    cheap7_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    cheap7_vals = [scores.get(c) for c in cheap7_cols]
    n_present = sum(1 for v in cheap7_vals if v is not None)
    cheap7 = sum(v for v in cheap7_vals if v is not None) / max(n_present, 1) if n_present > 0 else None

    summary = {
        "status": "CAUSAL_CHEAP7_EVAL",
        "model_dir": args.model_dir,
        "gpu": args.gpu,
        "elapsed_sec": round(time.time() - t0, 1),
        "scores": scores,
        "cheap7": cheap7,
        "cheap7_columns_present": n_present,
        "cheap7_columns": {c: scores.get(c) for c in cheap7_cols},
    }

    summary_path = out_base / "cheap7_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"cheap7: {cheap7}")
    print(f"Columns: {n_present}/7 present")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
