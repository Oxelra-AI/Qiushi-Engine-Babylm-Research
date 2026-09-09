#!/usr/bin/env python3
"""research: official-compatible cheap-column evaluator for causal GPT checkpoints.

This is a hardened replacement for the first research causal eval wrapper.  It uses
exact data directories from the current pristine Strict coordinate where available,
runs the official BabyLM zero-shot/reading programs with --backend causal, and
parses the official reports rather than reimplementing task scoring from possibly
wrong paths.  It also records prediction/report paths so later item-level analysis
can compare compact vs repeat decisions.

This script evaluates the seven cheap columns only: BLiMP, Supplement, EWoK,
Entity Tracking, COMPS, GlobalPIQA, and Reading.  SuperGLUE/AoA are endpoint work,
not needed for the causal-transfer mechanism screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
EVAL_REPO = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline"
STRICT_ROOT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
CURRENT_FULL = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
LEGACY_FULL = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
ZERO_SHOT = EVAL_REPO / "sentence_zero_shot/run.py"
READING_RUN = EVAL_REPO / "reading/run.py"
READING_DATA = CURRENT_FULL / "reading/reading_data.csv"

ZERO_SHOT_TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": CURRENT_FULL / "blimp_filtered", "batch_size": 128},
    {"column": "Supplement", "task": "blimp", "data_path": CURRENT_FULL / "supplement_filtered", "batch_size": 128},
    {"column": "EWoK", "task": "ewok", "data_path": CURRENT_FULL / "ewok_filtered", "batch_size": 64},
    {"column": "Entity", "task": "entity_tracking", "data_path": CURRENT_FULL / "entity_tracking", "batch_size": 128},
    {"column": "COMPS", "task": "comps", "data_path": CURRENT_FULL / "comps", "batch_size": 128},
    # The current pristine coordinate stores GlobalPIQA elsewhere; research uses the INITIAL_MODEL_STUDIES full_eval copy.
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": LEGACY_FULL / "global_piqa_parallel", "batch_size": 128},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": LEGACY_FULL / "global_piqa_nonparallel", "batch_size": 128},
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_sentence_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -5.0 <= val <= 105.0:
                return val
            return None
    return None


def parse_reading_scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def setup_env(out_base: pathlib.Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    cache = out_base / "runtime_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["NLTK_DATA"] = str((USER_ROOT / "experiments/archive/initial_model_studies/data/nltk_data").resolve())
    # Ensure imports like evaluation_pipeline.* resolve exactly as in the official scripts.
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(EVAL_REPO.parent.resolve()) + ((":" + existing_py) if existing_py else "")
    return env


def run_cmd(cmd: list[str], cwd: pathlib.Path, env: dict[str, str], log: pathlib.Path, timeout: int) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd.resolve()), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        f.write(f"\n[returncode={proc.returncode} elapsed_sec={time.time() - t0:.2f}]\n")
    return {"returncode": proc.returncode, "elapsed_sec": round(time.time() - t0, 3), "log": str(log)}


def run_zero(model_dir: pathlib.Path, revision: str, spec: dict[str, Any], out_base: pathlib.Path, gpu: int, force: bool) -> dict[str, Any]:
    column = spec["column"]
    task_out = out_base / "official_outputs" / column
    log = out_base / "logs" / f"{column}.log"
    pred_existing = latest_file(task_out, "predictions.json")
    report_existing = latest_file(task_out, "best_temperature_report.txt")
    if pred_existing and report_existing and not force:
        score = parse_sentence_score(report_existing.read_text(encoding="utf-8", errors="replace"))
        return {"status": "skipped_existing", "column": column, "score": score, "predictions": str(pred_existing), "report": str(report_existing), "output_dir": str(task_out)}

    env = setup_env(out_base, gpu)
    cmd = [
        sys.executable, str(ZERO_SHOT.resolve()),
        "--model_path_or_name", str(model_dir.resolve()),
        "--task", spec["task"],
        "--data_path", str(pathlib.Path(spec["data_path"]).resolve()),
        "--backend", "causal",
        "--batch_size", str(spec["batch_size"]),
        "--output_dir", str(task_out.resolve()),
        "--revision_name", revision,
        "--save_predictions",
    ]
    rec = {"column": column, "task": spec["task"], "data_path": str(spec["data_path"]), "output_dir": str(task_out), "revision": revision}
    rr = run_cmd(cmd, STRICT_ROOT, env, log, timeout=7200)
    rec.update(rr)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    rec["predictions"] = str(pred) if pred else None
    rec["predictions_sha256"] = sha256_file(pred) if pred else None
    rec["report"] = str(report) if report else None
    score = parse_sentence_score(report.read_text(encoding="utf-8", errors="replace")) if report else None
    rec["score"] = score
    if rec["returncode"] != 0 or score is None or pred is None:
        rec["status"] = "failed"
        raise RuntimeError(json.dumps(rec, indent=2))
    rec["status"] = "done"
    return rec


def run_reading(model_dir: pathlib.Path, revision: str, out_base: pathlib.Path, gpu: int, force: bool) -> dict[str, Any]:
    column = "Reading"
    task_out = out_base / "official_outputs" / column
    log = out_base / "logs" / f"{column}.log"
    pred_existing = latest_file(task_out, "predictions.json")
    report_existing = latest_file(task_out, "report.txt")
    if pred_existing and report_existing and not force:
        scores = parse_reading_scores(report_existing.read_text(encoding="utf-8", errors="replace"))
        return {"status": "skipped_existing", "column": column, "scores": scores, "predictions": str(pred_existing), "report": str(report_existing), "output_dir": str(task_out)}

    env = setup_env(out_base, gpu)
    cmd = [
        sys.executable, str(READING_RUN.resolve()),
        "--model_path_or_name", str(model_dir.resolve()),
        "--data_path", str(READING_DATA.resolve()),
        "--backend", "causal",
        "--output_dir", str(task_out.resolve()),
        "--revision_name", revision,
    ]
    rec: dict[str, Any] = {"column": column, "data_path": str(READING_DATA), "output_dir": str(task_out), "revision": revision}
    rr = run_cmd(cmd, STRICT_ROOT, env, log, timeout=7200)
    rec.update(rr)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "report.txt")
    rec["predictions"] = str(pred) if pred else None
    rec["predictions_sha256"] = sha256_file(pred) if pred else None
    rec["report"] = str(report) if report else None
    scores = parse_reading_scores(report.read_text(encoding="utf-8", errors="replace")) if report else {}
    rec["scores"] = scores
    rec["score"] = scores.get("Reading")
    if rec["returncode"] != 0 or rec["score"] is None or pred is None:
        rec["status"] = "failed"
        raise RuntimeError(json.dumps(rec, indent=2))
    rec["status"] = "done"
    return rec


def model_meta(model_dir: pathlib.Path) -> dict[str, Any]:
    out: dict[str, Any] = {"model_dir": str(model_dir), "exists": model_dir.exists()}
    if (model_dir / "config.json").exists():
        out["config_sha256"] = sha256_file(model_dir / "config.json")
        try:
            cfg = json.loads((model_dir / "config.json").read_text())
            out["model_type"] = cfg.get("model_type")
            out["architectures"] = cfg.get("architectures")
            out["n_layer"] = cfg.get("n_layer")
            out["n_embd"] = cfg.get("n_embd")
            out["n_head"] = cfg.get("n_head")
        except Exception as e:
            out["config_error"] = repr(e)
    sf = model_dir / "model.safetensors"
    if sf.exists():
        out["model_safetensors_sha256"] = sha256_file(sf)
        out["model_safetensors_size"] = sf.stat().st_size
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--columns", default="BLiMP,Supplement,EWoK,Entity,COMPS,GlobalPIQA,Reading", help="Comma-separated cheap columns; GlobalPIQA expands to parallel+nonparallel")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model_dir = pathlib.Path(args.model_dir)
    out_base = pathlib.Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    requested = {x.strip() for x in args.columns.split(",") if x.strip()}
    needed_paths = {
        "zero_shot_run": ZERO_SHOT,
        "reading_run": READING_RUN,
        "strict_root": STRICT_ROOT,
        "current_full": CURRENT_FULL,
        "legacy_full": LEGACY_FULL,
        "reading_data": READING_DATA,
        "model_dir": model_dir,
    }
    preflight = {
        "status": "CAUSAL_CHEAP7_PREFLIGHT",
        "created_utc": now_utc(),
        "model": model_meta(model_dir),
        "gpu": args.gpu,
        "requested_columns": sorted(requested),
        "paths": {k: {"path": str(v), "exists": pathlib.Path(v).exists()} for k, v in needed_paths.items()},
        "zero_shot_tasks": [{**s, "data_path": str(s["data_path"])} for s in ZERO_SHOT_TASKS],
    }
    (out_base / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2), flush=True)
    missing = [k for k, v in preflight["paths"].items() if not v["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    if args.preflight_only:
        return

    tasks: dict[str, Any] = {}
    for spec in ZERO_SHOT_TASKS:
        if spec["column"] in requested or ("GlobalPIQA" in requested and spec["column"].startswith("GlobalPIQA")):
            print(json.dumps({"event": "run_column", "column": spec["column"], "model_dir": str(model_dir), "gpu": args.gpu}), flush=True)
            tasks[spec["column"]] = run_zero(model_dir, args.revision, spec, out_base, args.gpu, args.force)
    if "Reading" in requested:
        print(json.dumps({"event": "run_column", "column": "Reading", "model_dir": str(model_dir), "gpu": args.gpu}), flush=True)
        tasks["Reading"] = run_reading(model_dir, args.revision, out_base, args.gpu, args.force)

    scores: dict[str, float] = {}
    for k, rec in tasks.items():
        if k.startswith("GlobalPIQA_"):
            scores[k] = float(rec["score"])
        elif k == "Reading":
            scores[k] = float(rec["score"])
        else:
            scores[k] = float(rec["score"])
    if "GlobalPIQA_parallel" in scores and "GlobalPIQA_nonparallel" in scores:
        scores["GlobalPIQA"] = (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2.0

    cheap_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    present = [scores[c] for c in cheap_cols if c in scores]
    summary = {
        "status": "CAUSAL_CHEAP7_DONE",
        "created_utc": now_utc(),
        "elapsed_sec": round(time.time() - t0, 1),
        "model_dir": str(model_dir),
        "model": model_meta(model_dir),
        "gpu": args.gpu,
        "scores": scores,
        "cheap7_columns": {c: scores.get(c) for c in cheap_cols},
        "cheap7_columns_present": len(present),
        "cheap7": sum(present) / len(present) if present else None,
        "tasks": tasks,
        "coordinate": "current pristine full_eval for BLiMP/Supplement/EWoK/Entity/COMPS/Reading plus legacy GlobalPIQA full_eval, backend=causal; report-parsed official outputs",
    }
    (out_base / "cheap7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        f"# Causal cheap7 eval: {model_dir.name}",
        "",
        f"Status: {summary['status']}",
        f"cheap7: {summary['cheap7']}",
        "",
        "| column | score |",
        "|---|---:|",
    ]
    for c in cheap_cols:
        md.append(f"| {c} | {scores.get(c)} |")
    md.append("")
    md.append(f"JSON: `{out_base / 'cheap7_summary.json'}`")
    (out_base / "cheap7_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
