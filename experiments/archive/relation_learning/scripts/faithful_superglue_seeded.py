#!/usr/bin/env python3
"""research: faithful AutoModel SuperGLUE with an explicit fine-tuning seed.

The standard research/research evaluator hardcodes SuperGLUE fine-tuning seed 42.
That is the historical coordinate, but after the AutoModel bridge repair the final
candidate should also be tested under at least one additional downstream seed.  This
script evaluates a repaired AutoModel checkpoint directly through the official
finetune runner while exposing the seed as an argument.

It assumes the checkpoint directory has already been repaired and validated so that
AutoModel hidden states match the trusted AutoModelForMaskedLM encoder.  Use
`automodel_bridge_and_validate.py` before using this script on a new
adapter/private endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
INITIAL_MODEL_STUDIES = ROOT / "experiments/archive/initial_model_studies"
STRICT = INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
DEFAULT_OUT = ROOT / "experiments/archive/relation_learning/data/faithful_superglue_seeded"
SUPERGLUE_TASKS = [
    {"task": "boolq", "num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "multirc", "num_labels": 2, "batch_size": 16, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "rte", "num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "wsc", "num_labels": 2, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy", "f1", "mcc"], "epochs": 30},
    {"task": "mrpc", "num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "qqp", "num_labels": 2, "batch_size": 32, "metric_for_valid": "f1", "metrics": ["accuracy", "f1", "mcc"], "epochs": 10},
    {"task": "mnli", "num_labels": 3, "batch_size": 32, "metric_for_valid": "accuracy", "metrics": ["accuracy"], "epochs": 10},
]
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    if not root.exists():
        return None
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def parse_results_txt(path: pathlib.Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, val = line.partition(":")
        if sep:
            try:
                out[key.strip()] = float(val.strip()) * 100.0
            except ValueError:
                pass
    return out


def cheap_scores_from(path: pathlib.Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    data = read_json(path)
    scores = None
    if isinstance(data.get("score_arithmetic"), dict):
        scores = data["score_arithmetic"].get("scores")
    if scores is None and isinstance(data.get("score_arithmetic_candidate_native"), dict):
        scores = data["score_arithmetic_candidate_native"].get("scores")
    if scores is None and isinstance(data.get("official_overall"), dict):
        scores = data["official_overall"].get("scores")
    if scores is None and isinstance(data.get("scores"), dict):
        scores = data.get("scores")
    if scores is None and isinstance(data.get("cheap_scores"), dict):
        scores = data.get("cheap_scores")
    if not isinstance(scores, dict) or not all(c in scores for c in CHEAP_COLUMNS):
        return None
    return {c: float(scores[c]) for c in CHEAP_COLUMNS}


def aoa_from(path: pathlib.Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    data = read_json(path)
    if data.get("aoa_leaderboard_score") is not None:
        return float(data["aoa_leaderboard_score"])
    if data.get("aoa") is not None:
        return float(data["aoa"])
    if isinstance(data.get("score_arithmetic"), dict):
        scores = data["score_arithmetic"].get("scores") or {}
        if scores.get("AoA") is not None:
            return float(scores["AoA"])
    return None


def setup_env(out_dir: pathlib.Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache = out_dir / "runtime_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "NLTK_DATA": INITIAL_MODEL_STUDIES / "data/nltk_data",
        "TMPDIR": cache / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    return env


def wait_for_path(path: pathlib.Path, timeout: float, interval: float) -> None:
    if timeout <= 0:
        return
    start = time.time()
    while not path.exists():
        if time.time() - start >= timeout:
            raise TimeoutError(f"waited {timeout}s for {path}")
        print(json.dumps({"event": "waiting_for_dependency", "path": rel(path), "elapsed_sec": round(time.time() - start, 1), "utc": now()}), flush=True)
        time.sleep(interval)
    print(json.dumps({"event": "dependency_available", "path": rel(path), "waited_sec": round(time.time() - start, 1), "utc": now()}), flush=True)


def maybe_stage_checkpoint(src: pathlib.Path, dst: pathlib.Path, copy_model: bool) -> pathlib.Path:
    if not copy_model:
        return src
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir(), key=lambda p: p.name):
        target = dst / item.name
        if item.is_file():
            if item.name == "model.safetensors":
                os.symlink(item.resolve(), target)
            else:
                shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, symlinks=True)
    return dst


def command_for_task(args: argparse.Namespace, model_path: pathlib.Path, out_dir: pathlib.Path, task_spec: dict[str, Any]) -> list[str]:
    task = task_spec["task"]
    return [
        sys.executable, "-B", "-m", "evaluation_pipeline.finetune.run",
        "--model_name_or_path", str(model_path.resolve()),
        "--train_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.train.jsonl").resolve()),
        "--valid_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.valid.jsonl").resolve()),
        "--predict_data", str((STRICT / "evaluation_data/full_eval/glue_filtered" / f"{task}.valid.jsonl").resolve()),
        "--task", task,
        "--num_labels", str(task_spec["num_labels"]),
        "--batch_size", str(task_spec["batch_size"]),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(task_spec["epochs"]),
        "--sequence_length", "512",
        "--results_dir", str((out_dir / "superglue_results").resolve()),
        "--save",
        "--save_dir", str((out_dir / "superglue_models").resolve()),
        "--metrics", *task_spec["metrics"],
        "--metric_for_valid", task_spec["metric_for_valid"],
        "--seed", str(args.finetune_seed),
        "--verbose",
        "--padding_side", "left",
        "--take_final",
        "--revision_name", args.revision_name,
    ]


def run_task(args: argparse.Namespace, model_path: pathlib.Path, out_dir: pathlib.Path, env: dict[str, str], task_spec: dict[str, Any]) -> dict[str, Any]:
    task = task_spec["task"]
    cmd = command_for_task(args, model_path, out_dir, task_spec)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"superglue_{task}_seed{args.finetune_seed}.log"
    result_root = out_dir / "superglue_results" / model_path.stem / args.revision_name / "finetune" / task
    results_txt = result_root / "results.txt"
    predictions_json = result_root / "predictions.json"
    if results_txt.exists() and predictions_json.exists() and not args.force:
        metrics = parse_results_txt(results_txt)
        primary = metrics.get(PRIMARY[task])
        return {"task": task, "skipped_existing": True, "results_txt": rel(results_txt), "predictions": rel(predictions_json), "primary_metric": PRIMARY[task], "primary_score": primary, "all_metrics": metrics}
    print(json.dumps({"event": "seeded_superglue_task_start", "label": args.label, "task": task, "seed": args.finetune_seed, "gpu": args.gpu, "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=f, stderr=subprocess.STDOUT, timeout=args.task_timeout)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        print(json.dumps({"event": "seeded_superglue_task_failed", "task": task, "returncode": proc.returncode, "log_tail": tail, "utc": now()}), flush=True)
        raise SystemExit(proc.returncode)
    if not results_txt.exists():
        raise FileNotFoundError(results_txt)
    metrics = parse_results_txt(results_txt)
    primary = metrics.get(PRIMARY[task])
    if primary is None:
        raise RuntimeError(f"primary metric {PRIMARY[task]} absent in {results_txt}: {metrics}")
    rec = {
        "task": task,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
        "results_txt": rel(results_txt),
        "predictions": rel(predictions_json) if predictions_json.exists() else None,
        "primary_metric": PRIMARY[task],
        "primary_score": primary,
        "all_metrics": metrics,
        "log": rel(log_path),
    }
    print(json.dumps({"event": "seeded_superglue_task_done", "label": args.label, "task": task, "primary_score": primary, "elapsed_sec": rec["elapsed_sec"], "utc": now()}), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", required=True, help="Validated repaired AutoModel checkpoint directory")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--finetune-seed", type=int, default=43)
    ap.add_argument("--revision-name", default=None)
    ap.add_argument("--tasks", nargs="+", default=[x["task"] for x in SUPERGLUE_TASKS], choices=[x["task"] for x in SUPERGLUE_TASKS])
    ap.add_argument("--cheap-summary", default="")
    ap.add_argument("--aoa-summary", default="")
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--task-timeout", type=float, default=14400.0)
    ap.add_argument("--copy-checkpoint", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-for", default="", help="Optional dependency file to wait for before starting GPU work")
    ap.add_argument("--wait-timeout", type=float, default=0.0)
    ap.add_argument("--wait-interval", type=float, default=120.0)
    args = ap.parse_args()
    if args.revision_name is None:
        args.revision_name = f"seeded_sg_{args.label}_ftseed{args.finetune_seed}"
    if args.wait_for:
        dep = pathlib.Path(args.wait_for)
        if not dep.is_absolute():
            dep = ROOT / dep
        wait_for_path(dep, float(args.wait_timeout), float(args.wait_interval))
    checkpoint = pathlib.Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    out_root = pathlib.Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_dir = out_root / f"{args.label}_ftseed{args.finetune_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = maybe_stage_checkpoint(checkpoint, out_dir / "model", args.copy_checkpoint)
    selected = [spec for spec in SUPERGLUE_TASKS if spec["task"] in set(args.tasks)]
    env = setup_env(out_dir, int(args.gpu))
    dry_commands = [command_for_task(args, model_path, out_dir, spec) for spec in selected]
    if args.dry_run:
        payload = {"status": "FAITHFUL_SUPERGLUE_SEEDED_DRY_RUN", "label": args.label, "checkpoint": rel(checkpoint), "model_path": rel(model_path), "finetune_seed": args.finetune_seed, "tasks": args.tasks, "commands": dry_commands, "created_utc": now()}
        write_json(out_dir / "dry_run.json", payload)
        print(json.dumps({"status": payload["status"], "out_json": rel(out_dir / "dry_run.json"), "task_count": len(dry_commands)}, indent=2), flush=True)
        return
    task_records = [run_task(args, model_path, out_dir, env, spec) for spec in selected]
    primary_scores = [float(r["primary_score"]) for r in task_records if r.get("primary_score") is not None]
    sg = sum(primary_scores) / len(primary_scores) if primary_scores else None
    cheap_path = pathlib.Path(args.cheap_summary) if args.cheap_summary else None
    if cheap_path is not None and not cheap_path.is_absolute():
        cheap_path = ROOT / cheap_path
    aoa_path = pathlib.Path(args.aoa_summary) if args.aoa_summary else None
    if aoa_path is not None and not aoa_path.is_absolute():
        aoa_path = ROOT / aoa_path
    cheap = cheap_scores_from(cheap_path) if cheap_path else None
    aoa = aoa_from(aoa_path) if aoa_path else None
    overall = None
    if cheap is not None and sg is not None and aoa is not None and len(task_records) == len(SUPERGLUE_TASKS):
        overall = mean([cheap[c] for c in CHEAP_COLUMNS] + [sg, aoa])
    summary = {
        "status": "FAITHFUL_SUPERGLUE_SEEDED_DONE",
        "label": args.label,
        "checkpoint": rel(checkpoint),
        "model_path": rel(model_path),
        "finetune_seed": args.finetune_seed,
        "revision_name": args.revision_name,
        "tasks": args.tasks,
        "task_records": task_records,
        "superglue_primary_metric_mean": sg,
        "cheap_summary": rel(cheap_path) if cheap_path else None,
        "cheap_scores": cheap,
        "aoa_summary": rel(aoa_path) if aoa_path else None,
        "aoa": aoa,
        "overall_if_all_columns_available": overall,
        "created_utc": now(),
        "interpretation": "Use this as a downstream fine-tuning seed repeat for a checkpoint whose AutoModel bridge was validated before evaluation.",
    }
    write_json(out_dir / "faithful_superglue_seeded_summary.json", summary)
    lines = [f"# research faithful SuperGLUE seeded — {args.label}\n\n"]
    lines.append(f"Fine-tuning seed: `{args.finetune_seed}`. SuperGLUE primary-metric mean: `{sg}`.\n\n")
    for rec in task_records:
        lines.append(f"- `{rec['task']}` {rec['primary_metric']}: `{rec['primary_score']}` (`{rec.get('results_txt')}`)\n")
    if overall is not None:
        lines.append(f"\nOverall with supplied cheap/AoA: `{overall}`.\n")
    lines.append(f"\nJSON: `{rel(out_dir / 'faithful_superglue_seeded_summary.json')}`\n")
    (out_dir / "faithful_superglue_seeded_summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "label": args.label, "seed": args.finetune_seed, "superglue": sg, "overall": overall, "out_json": rel(out_dir / "faithful_superglue_seeded_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
