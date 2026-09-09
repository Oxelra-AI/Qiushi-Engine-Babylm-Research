#!/usr/bin/env python3
"""research: full-eval zero-shot/Reading trajectory for existing checkpoints.

This does not run SuperGLUE or AoA. It evaluates full official zero-shot columns
and Reading for every named checkpoint under one hf_model root, saving predictions
and a trajectory summary. Its purpose is to select the best existing endpoint(s)
for expensive SuperGLUE and to test whether the 100M endpoint is masking a better
80M/90M point as in initial_model_studies research.
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
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

WORKSPACE = _public_path('experiments/archive/compact_experience')
ROOT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies')
# Resolve the initial-reference evaluator from the project layout.
ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/checkpoint_trajectory')

TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": "evaluation_data/full_eval/blimp_filtered", "batch_size": 128},
    {"column": "Supplement", "task": "blimp", "data_path": "evaluation_data/full_eval/supplement_filtered", "batch_size": 128},
    {"column": "EWoK", "task": "ewok", "data_path": "evaluation_data/full_eval/ewok_filtered", "batch_size": 64},
    {"column": "Entity", "task": "entity_tracking", "data_path": "evaluation_data/full_eval/entity_tracking", "batch_size": 128},
    {"column": "COMPS", "task": "comps", "data_path": "evaluation_data/full_eval/comps", "batch_size": 128},
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": "evaluation_data/full_eval/global_piqa_parallel", "batch_size": 128},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": "evaluation_data/full_eval/global_piqa_nonparallel", "batch_size": 128},
]
TASK_BY_COL = {x["column"]: x for x in TASKS}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_env(out_root: pathlib.Path, target: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / target
    tmp = out_root / "tmp" / target
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TMPDIR"] = str(tmp.resolve())
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def append_log(path: pathlib.Path, txt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(txt)
        if not txt.endswith("\n"):
            f.write("\n")


def run_cmd(cmd: List[str], env: Dict[str, str], log: pathlib.Path, timeout: int = 7200) -> Dict[str, Any]:
    append_log(log, "\n[{}] $ {}".format(now(), " ".join(cmd)))
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.time() - t0
    append_log(log, f"[returncode={p.returncode} elapsed_sec={elapsed:.2f}]")
    return {"returncode": p.returncode, "elapsed_sec": round(elapsed, 3), "log": str(log)}


def parse_report_score(text: str) -> Optional[float]:
    m = re.search(r"### AVERAGE [A-Z_ ]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    return None


def latest(root: pathlib.Path, pattern: str) -> Optional[pathlib.Path]:
    xs = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return xs[-1] if xs else None


def read_score(out_dir: pathlib.Path) -> Optional[float]:
    p = latest(out_dir, "best_temperature_report.txt")
    if not p:
        return None
    return parse_report_score(p.read_text(encoding="utf-8", errors="replace"))


def parse_reading(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_reading(out_dir: pathlib.Path) -> Dict[str, float]:
    p = latest(out_dir, "report.txt")
    return parse_reading(p.read_text(encoding="utf-8", errors="replace")) if p else {}


def eval_column(target: str, ckpt: str, model_path: pathlib.Path, col: str, out_root: pathlib.Path, env: Dict[str, str], force: bool) -> Dict[str, Any]:
    spec = TASK_BY_COL[col]
    out_dir = out_root / "official_outputs" / target / ckpt / col
    log = out_root / "logs" / target / ckpt / f"{col}.log"
    revision = f"step026_{target}_{ckpt}_{col}"
    rec_path = out_root / "per_checkpoint" / target / ckpt / f"{col}.json"
    if rec_path.exists() and not force:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("returncode") == 0 and rec.get("score") is not None:
            return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
           "--model_path_or_name", str(model_path.resolve()), "--backend", "mlm",
           "--task", spec["task"], "--data_path", spec["data_path"],
           "--revision_name", revision, "--save_predictions", "--batch_size", str(spec["batch_size"]),
           "--non_causal_batch_size", "64", "--output_dir", str(out_dir.resolve())]
    rec = {"target": target, "checkpoint": ckpt, "column": col, "task": spec["task"], "data_path": spec["data_path"], "revision_name": revision, "output_dir": str(out_dir)}
    rr = run_cmd(cmd, env, log)
    rec.update(rr)
    rec["score"] = read_score(out_dir)
    pred = latest(out_dir, "predictions.json")
    report = latest(out_dir, "best_temperature_report.txt")
    if pred:
        rec["predictions"] = str(pred)
    if report:
        rec["report"] = str(report)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if rec["returncode"] != 0 or rec["score"] is None:
        raise RuntimeError(f"{target} {ckpt} {col} failed: {rec}")
    print(json.dumps({"event": "column_done", "target": target, "checkpoint": ckpt, "column": col, "score": rec["score"]}), flush=True)
    return rec


def eval_reading(target: str, ckpt: str, model_path: pathlib.Path, out_root: pathlib.Path, env: Dict[str, str], force: bool) -> Dict[str, Any]:
    col = "Reading"
    out_dir = out_root / "official_outputs" / target / ckpt / col
    log = out_root / "logs" / target / ckpt / f"{col}.log"
    revision = f"step026_{target}_{ckpt}_{col}"
    rec_path = out_root / "per_checkpoint" / target / ckpt / f"{col}.json"
    if rec_path.exists() and not force:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("returncode") == 0 and rec.get("scores", {}).get("Reading") is not None:
            return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "evaluation_pipeline.reading.run",
           "--model_path_or_name", str(model_path.resolve()), "--backend", "mlm",
           "--data_path", "evaluation_data/full_eval/reading/reading_data.csv",
           "--revision_name", revision, "--output_dir", str(out_dir.resolve())]
    rec: Dict[str, Any] = {"target": target, "checkpoint": ckpt, "column": col, "data_path": "evaluation_data/full_eval/reading/reading_data.csv", "revision_name": revision, "output_dir": str(out_dir)}
    rr = run_cmd(cmd, env, log)
    rec.update(rr)
    rec["scores"] = read_reading(out_dir)
    pred = latest(out_dir, "predictions.json")
    report = latest(out_dir, "report.txt")
    if pred:
        rec["predictions"] = str(pred)
    if report:
        rec["report"] = str(report)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if rec["returncode"] != 0 or rec.get("scores", {}).get("Reading") is None:
        raise RuntimeError(f"{target} {ckpt} Reading failed: {rec}")
    print(json.dumps({"event": "reading_done", "target": target, "checkpoint": ckpt, "scores": rec["scores"]}), flush=True)
    return rec


def summarize(target: str, out_root: pathlib.Path, checkpoints: List[str], model_root: pathlib.Path) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Any]] = {}
    for ckpt in checkpoints:
        ckpt_dir = out_root / "per_checkpoint" / target / ckpt
        row: Dict[str, Any] = {}
        for col in [x["column"] for x in TASKS]:
            p = ckpt_dir / f"{col}.json"
            if p.exists():
                row[col] = json.loads(p.read_text(encoding="utf-8")).get("score")
        p = ckpt_dir / "Reading.json"
        if p.exists():
            scores = json.loads(p.read_text(encoding="utf-8")).get("scores", {})
            row.update(scores)
        if row.get("GlobalPIQA_parallel") is not None and row.get("GlobalPIQA_nonparallel") is not None:
            row["GlobalPIQA"] = (float(row["GlobalPIQA_parallel"]) + float(row["GlobalPIQA_nonparallel"])) / 2.0
        keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
        if all(row.get(k) is not None for k in keys):
            row["equal7_full_eval"] = sum(float(row[k]) for k in keys) / len(keys)
        table[ckpt] = row
    best = None
    valid = {k: v for k, v in table.items() if v.get("equal7_full_eval") is not None}
    if valid:
        best = max(valid.items(), key=lambda kv: float(kv[1]["equal7_full_eval"]))
    payload = {"status": "FULL_ZEROSHOT_READING_TRAJECTORY", "target": target, "model_root": str(model_root), "checkpoints": checkpoints, "table": table, "best_by_equal7_full_eval": {"checkpoint": best[0], "row": best[1]} if best else None, "finished_utc": now()}
    out_path = out_root / f"{target}_trajectory_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "trajectory_summary", "target": target, "out": str(out_path), "best": payload["best_by_equal7_full_eval"]}), flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--model_root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--checkpoints", nargs="*", default=[f"chck_{i}M" for i in range(10, 101, 10)])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_root = pathlib.Path(args.model_root)
    env = setup_env(out_root, args.target, args.gpu)
    print(json.dumps({"event": "trajectory_start", "target": args.target, "model_root": str(model_root), "gpu": args.gpu, "checkpoints": args.checkpoints, "started_utc": now()}), flush=True)
    for ckpt in args.checkpoints:
        model_path = model_root / ckpt
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        for col in [x["column"] for x in TASKS]:
            eval_column(args.target, ckpt, model_path, col, out_root, env, args.force)
        eval_reading(args.target, ckpt, model_path, out_root, env, args.force)
        summarize(args.target, out_root, args.checkpoints, model_root)
    summarize(args.target, out_root, args.checkpoints, model_root)


if __name__ == "__main__":
    main()
