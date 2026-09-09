#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, subprocess, sys

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
MODEL = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M").resolve()
DATA = STRICT / "evaluation_data/full_eval/glue_filtered"
OUT_BASE = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_superglue").resolve()
RESULTS_DIR = OUT_BASE / "results"
MODELS_DIR = OUT_BASE / "models"
OUT_JSON = ROOT / "data/debertav2_b256_superglue_result.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_superglue_result.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_superglue_eval.log')
TASKS = [
    {"task":"boolq", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"multirc", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"rte", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"wsc", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":30},
    {"task":"mrpc", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"qqp", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"mnli", "num_labels":3, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy"], "epochs":10},
]


def setup_env():
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_task(spec, env, logf):
    t = spec["task"]
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(MODEL),
        "--train_data", str((DATA / f"{t}.train.jsonl").resolve()),
        "--valid_data", str((DATA / f"{t}.valid.jsonl").resolve()),
        "--predict_data", str((DATA / f"{t}.valid.jsonl").resolve()),
        "--task", t,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", "3e-5",
        "--num_epochs", str(spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str(RESULTS_DIR),
        "--save",
        "--save_dir", str(MODELS_DIR),
        "--metrics", *spec["metrics"],
        "--metric_for_valid", spec["metric_for_valid"],
        "--seed", "42",
        "--verbose",
        "--padding_side", "left",
        "--take_final",
    ]
    logf.write("\n$ " + " ".join(cmd) + "\n"); logf.flush()
    p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-5000:])
    logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"{t} failed with {p.returncode}")
    pred = RESULTS_DIR / "chck_100M" / "main" / "finetune" / t / "predictions.json"
    if not pred.exists():
        hits = list(RESULTS_DIR.rglob(f"finetune/{t}/predictions.json"))
        if not hits:
            raise FileNotFoundError(f"missing predictions for {t} under {RESULTS_DIR}")
        pred = hits[0]
    return {"task": t, "predictions": str(pred), "num_predictions": len(json.loads(pred.read_text())[t]["predictions"])}


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = setup_env()
    rows=[]
    with LOG.open("a", encoding="utf-8") as logf:
        for spec in TASKS:
            rows.append(run_task(spec, env, logf))
    payload={"model_path":str(MODEL),"results_dir":str(RESULTS_DIR),"models_dir":str(MODELS_DIR),"tasks":rows,"note":"Predictions generated for official glue_filtered validation sets; aggregate score parsed in a later lightweight step."}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    OUT_NOTE.write_text("# research DeBERTa-v2 b256 (Super)GLUE\n\nPredictions generated for tasks: "+", ".join(r['task'] for r in rows)+f"\n\nEvidence: `{OUT_JSON}`\n", encoding="utf-8")
    print(json.dumps({"status":"SUPERGLUE_DONE","out":str(OUT_JSON),"tasks":rows}, indent=2))

if __name__ == "__main__":
    main()