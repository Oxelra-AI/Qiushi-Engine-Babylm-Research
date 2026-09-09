#!/usr/bin/env python3
"""research: run/aggregate S1 and S2 (Super)GLUE for coordinate closure.

This is adapted from the trusted research protected-model runner but parameterized for
S1/S2 chck_100M. It resumes tasks whose predictions already exist and computes the
mean validation accuracy directly against full_eval/glue_filtered/*.valid.jsonl.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
DATA = STRICT / "evaluation_data/full_eval/glue_filtered"
OUT_JSON = ROOT / "data/s1_s2_100m_superglue_results.json"
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_s2_superglue_eval.log')

MODEL_SPECS = {
    "s1": {
        "model": ROOT / "training/runs/babylm_leadershape_s1_100M_aligned_micro128/hf_model/chck_100M",
        "out_base": ROOT / "training/runs/babylm_leadershape_s1_100M_aligned_micro128/eval_results_superglue",
    },
    "s2": {
        "model": ROOT / "training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/hf_model/chck_100M",
        "out_base": ROOT / "training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/eval_results_superglue",
    },
}
TASKS = [
    {"task":"boolq", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"multirc", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"rte", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"wsc", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":30},
    {"task":"mrpc", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"qqp", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"mnli", "num_labels":3, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy"], "epochs":10},
]


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def load_payload() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {"status": "S1_S2_SUPERGLUE_IN_PROGRESS", "models": {}}


def pred_path(results_dir: pathlib.Path, task: str) -> pathlib.Path:
    return results_dir / "chck_100M" / "main" / "finetune" / task / "predictions.json"


def labels_for(task: str) -> list[int]:
    p = DATA / f"{task}.valid.jsonl"
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(int(json.loads(line)["label"]))
    return labels


def score_predictions(task: str, pp: pathlib.Path) -> dict:
    data = json.loads(pp.read_text(encoding="utf-8"))
    preds = data[task]["predictions"]
    labels = labels_for(task)
    if len(preds) != len(labels):
        raise RuntimeError(f"{task}: predictions={len(preds)} labels={len(labels)}")
    correct = sum(1 for r, y in zip(preds, labels) if int(r["pred"]) == int(y))
    return {
        "task": task,
        "predictions": str(pp),
        "num_examples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels) * 100.0,
    }


def run_task(model_name: str, spec: dict, env: dict[str, str], logf) -> dict:
    ms = MODEL_SPECS[model_name]
    model = ms["model"].resolve()
    out_base = ms["out_base"].resolve()
    results_dir = out_base / "results"
    models_dir = out_base / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    t = spec["task"]
    pp = pred_path(results_dir, t)
    if pp.exists():
        rec = score_predictions(t, pp)
        rec["resumed_existing"] = True
        return rec
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(model),
        "--train_data", str((DATA / f"{t}.train.jsonl").resolve()),
        "--valid_data", str((DATA / f"{t}.valid.jsonl").resolve()),
        "--predict_data", str((DATA / f"{t}.valid.jsonl").resolve()),
        "--task", t,
        "--num_labels", str(spec["num_labels"]),
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", "3e-5",
        "--num_epochs", str(spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str(results_dir),
        "--save",
        "--save_dir", str(models_dir),
        "--metrics", *spec["metrics"],
        "--metric_for_valid", spec["metric_for_valid"],
        "--seed", "42",
        "--verbose",
        "--padding_side", "left",
        "--take_final",
    ]
    line = "$ " + " ".join(cmd)
    print(f"[{model_name}] {line}", flush=True)
    logf.write("\n" + f"[{model_name}] " + line + "\n")
    logf.flush()
    p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True)
    logf.write(p.stdout + f"\n[returncode={p.returncode}]\n")
    logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"{model_name}:{t} failed with {p.returncode}\n{p.stdout[-6000:]}")
    if not pp.exists():
        hits = sorted(results_dir.rglob(f"finetune/{t}/predictions.json"), key=lambda x: str(x))
        if not hits:
            raise FileNotFoundError(f"missing predictions for {model_name}:{t} under {results_dir}")
        pp = hits[-1]
    rec = score_predictions(t, pp)
    rec["resumed_existing"] = False
    return rec


def run_model(model_name: str, env: dict[str, str], payload: dict, logf) -> None:
    ms = MODEL_SPECS[model_name]
    if not ms["model"].exists():
        raise FileNotFoundError(f"missing {model_name}: {ms['model']}")
    rec = payload["models"].setdefault(model_name, {
        "model_path": str(ms["model"]),
        "results_dir": str((ms["out_base"] / "results")),
        "models_dir": str((ms["out_base"] / "models")),
        "tasks": [],
    })
    done_by_task = {r["task"]: r for r in rec.get("tasks", []) if "accuracy" in r}
    rows = []
    for spec in TASKS:
        t = spec["task"]
        if t in done_by_task and pathlib.Path(done_by_task[t]["predictions"]).exists():
            row = score_predictions(t, pathlib.Path(done_by_task[t]["predictions"]))
            row["resumed_existing"] = True
        else:
            row = run_task(model_name, spec, env, logf)
        rows.append(row)
        rec["tasks"] = rows
        rec["superglue"] = sum(r["accuracy"] for r in rows) / len(rows)
        rec["num_completed_tasks"] = len(rows)
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rec["superglue"] = sum(r["accuracy"] for r in rows) / len(rows)
    rec["num_completed_tasks"] = len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["s1", "s2"], choices=sorted(MODEL_SPECS))
    args = ap.parse_args()
    t0 = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = setup_env()
    payload = load_payload()
    payload["status"] = "S1_S2_SUPERGLUE_IN_PROGRESS"
    payload["glue_data"] = str(DATA)
    with LOG.open("a", encoding="utf-8") as logf:
        for m in args.models:
            run_model(m, env, payload, logf)
    payload["elapsed_sec"] = round(time.time() - t0, 1)
    if all(payload["models"].get(m, {}).get("num_completed_tasks") == len(TASKS) for m in args.models):
        payload["status"] = "S1_S2_SUPERGLUE_DONE"
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "models": {m: payload["models"][m].get("superglue") for m in args.models}, "elapsed_sec": payload["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
