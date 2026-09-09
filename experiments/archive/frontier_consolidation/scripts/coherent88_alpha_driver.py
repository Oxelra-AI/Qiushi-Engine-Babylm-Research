#!/usr/bin/env python3
"""research: materialize and evaluate coherent88 private-alpha variants.

Uses the validated research private-scale materializer and research official-compatible
evaluator. This script never submits to the leaderboard. It writes run-like endpoints
under training/runs/coherent88_alpha*/ so research can evaluate them.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SOURCE = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/hf_model/final')
SOURCE_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen84_coherent88_seed43022/scientific_metrics.json')
MATERIALIZER = _public_path('experiments/archive/frontier_consolidation/scripts/materialize_private_scale.py')
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/coherent88_alpha_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/coherent88_alpha_collate')
RUN_ROOT = _public_path('experiments/archive/frontier_consolidation/training/runs')

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHCK84_SCORES = {
    "BLiMP": 68.25,
    "Supplement": 63.48,
    "EWoK": 50.07,
    "Entity": 28.58,
    "COMPS": 52.21,
    "GlobalPIQA": 38.12,
    "Reading": 8.155,
}
COHERENT86_ALPHA075 = {
    "cheap7": 44.18142857142857,
    "scores": {
        "BLiMP": 68.51,
        "Supplement": 63.64,
        "EWoK": 50.02,
        "Entity": 28.32,
        "COMPS": 52.05,
        "GlobalPIQA": 38.565,
        "Reading": 8.165,
    },
    "projected_overall_aoa0": 42.1210247099666,
    "superglue": 69.81922238969935,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def label(scale: float) -> str:
    s = ("%.4f" % scale).rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {c: None for c in CHEAP_COLUMNS}
    official = payload.get("official_overall", {}).get("scores", {})
    for c in CHEAP_COLUMNS:
        if official.get(c) is not None:
            scores[c] = float(official[c])
    tasks = payload.get("tasks", {})
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if scores[c] is None:
            r = tasks.get(c, {})
            if isinstance(r, dict) and r.get("score") is not None:
                scores[c] = float(r["score"])
    if scores["GlobalPIQA"] is None:
        vals = []
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            r = tasks.get(c, {})
            if isinstance(r, dict) and r.get("score") is not None:
                vals.append(float(r["score"]))
        if len(vals) == 2:
            scores["GlobalPIQA"] = float(mean(vals))
    if scores["Reading"] is None:
        r = tasks.get("Reading", {})
        if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
            scores["Reading"] = float(r["scores"]["Reading"])
        elif isinstance(r, dict) and r.get("score") is not None:
            scores["Reading"] = float(r["score"])
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def stable(scores: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    vals6 = [scores.get(c) for c in CHEAP_COLUMNS if c != "GlobalPIQA"]
    vals5 = [scores.get(c) for c in CHEAP_COLUMNS if c not in {"GlobalPIQA", "Reading"}]
    out["cheap6_no_GlobalPIQA"] = None if any(v is None for v in vals6) else float(mean(float(v) for v in vals6))
    out["cheap5_no_GlobalPIQA_Reading"] = None if any(v is None for v in vals5) else float(mean(float(v) for v in vals5))
    out["EWoK_plus_Entity"] = None if scores.get("EWoK") is None or scores.get("Entity") is None else float((float(scores["EWoK"]) + float(scores["Entity"])) / 2.0)
    out["Supplement"] = scores.get("Supplement")
    out["Entity"] = scores.get("Entity")
    out["COMPS"] = scores.get("COMPS")
    return out


def materialize(scale: float, run_dir: Path, force: bool) -> dict[str, Any]:
    endpoint = run_dir / "hf_model" / "final"
    cmd = [sys.executable, "-B", str(MATERIALIZER), "--source", str(SOURCE), "--output", str(endpoint), "--scale", str(scale), "--force"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"materialize failed scale={scale}\nSTDOUT={proc.stdout[-2000:]}\nSTDERR={proc.stderr[-3000:]}")
    metrics = read_json(SOURCE_METRICS)
    metrics.update({
        "status": "COHERENT88_ALPHA_ENDPOINT",
        "source_metrics_path": rel(SOURCE_METRICS),
        "private_adapter_scale_materialized": scale,
        "created_utc": now(),
        "note": "Inference-time materialization only; no additional training beyond research coherent88 private replay.",
    })
    write_json(run_dir / "scientific_metrics.json", metrics)
    return {"stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:], "endpoint": rel(endpoint)}


def evaluate(scale: float, gpu: int, force: bool) -> dict[str, Any]:
    lab = label(scale)
    target = f"coherent88_alpha{lab}"
    run_dir = RUN_ROOT / target
    run_dir.mkdir(parents=True, exist_ok=True)
    mat = materialize(scale, run_dir, force=force)
    out_root = OUT_ROOT / target
    collate_root = COLLATE_ROOT / target
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache = out_root / "runtime_cache"
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", "final",
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    log_dir = out_root / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"event": "eval_start", "target": target, "scale": scale, "gpu": gpu, "cmd": [str(x) for x in cmd], "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=9000)
    elapsed = time.time() - t0
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        rec = {
            "status": "EVAL_FAILED", "target": target, "scale": scale, "gpu": gpu,
            "returncode": proc.returncode, "elapsed_sec": elapsed,
            "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-4000:],
            "materialize": mat, "stdout_log": rel(log_dir / "stdout.log"), "stderr_log": rel(log_dir / "stderr.log"),
        }
        write_json(out_root / f"{target}_summary.json", rec)
        raise RuntimeError(json.dumps(rec, indent=2))
    payload_path = out_root / "per_target" / f"{target}.json"
    payload = read_json(payload_path)
    scores = extract_scores(payload)
    c7 = cheap7(scores)
    st = stable(scores)
    chck84_c7 = mean(CHCK84_SCORES[c] for c in CHEAP_COLUMNS)
    rec = {
        "status": "COHERENT88_ALPHA_CHEAP7_EVAL",
        "created_utc": now(),
        "target": target,
        "scale": scale,
        "run_dir": rel(run_dir),
        "endpoint": "final",
        "model_path": rel(run_dir / "hf_model" / "final"),
        "source_model": rel(SOURCE),
        "materialize": mat,
        "scores": scores,
        "cheap7": c7,
        "stable_readouts": st,
        "deltas_vs_chck84": {c: (None if scores.get(c) is None else float(float(scores[c]) - CHCK84_SCORES[c])) for c in CHEAP_COLUMNS},
        "cheap7_delta_vs_chck84": None if c7 is None else float(c7 - chck84_c7),
        "stable_delta_vs_chck84": {
            k: (None if v is None else float(v - stable({c: float(vv) for c, vv in CHCK84_SCORES.items()})[k])) for k, v in st.items()
        },
        "coherent86_alpha075_reference": COHERENT86_ALPHA075,
        "cheap7_delta_vs_coherent86_alpha075": None if c7 is None else float(c7 - COHERENT86_ALPHA075["cheap7"]),
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / "stdout.log"),
        "stderr_log": rel(log_dir / "stderr.log"),
        "elapsed_sec": round(elapsed, 1),
        "returncode": proc.returncode,
    }
    write_json(out_root / f"{target}_summary.json", rec)
    print(json.dumps({"status": rec["status"], "target": target, "scale": scale, "cheap7": c7, "scores": scores, "stable_readouts": st, "summary": rel(out_root / f"{target}_summary.json")}, indent=2, ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rec = evaluate(args.scale, args.gpu, force=args.force)
    print(json.dumps({"status": "DONE", "summary": rec}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
