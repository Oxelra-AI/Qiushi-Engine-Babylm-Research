#!/usr/bin/env python3
"""Materialize and evaluate a minimal private-scale sentinel for frozen-tail models.

This is not a training script.  It creates a run-like HF endpoint with a changed
`private_adapter_scale` and evaluates only selected official-compatible columns to test
whether full-scale private-branch damage is amplitude-controlled.  The default column
set targets the observed aligned-tail tradeoff: Supplement loss vs EWoK/Reading and
GlobalPIQA movement.
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
import subprocess
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
MATERIALIZER = _public_path('experiments/archive/frontier_consolidation/scripts/materialize_private_scale.py')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
DEFAULT_SOURCE = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_aligned_seed43022/hf_model/final')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval')
DEFAULT_COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_collate')
DEFAULT_RUN_ROOT = _public_path('experiments/archive/frontier_consolidation/training/runs')

COLUMN_MAP = {
    "BLiMP": ["BLiMP"],
    "Supplement": ["Supplement"],
    "EWoK": ["EWoK"],
    "Entity": ["Entity"],
    "COMPS": ["COMPS"],
    "GlobalPIQA": ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"],
    "Reading": ["Reading"],
}
DEFAULT_SENTINEL_COLUMNS = ["Supplement", "EWoK", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_scale_label(scale: float) -> str:
    s = ("%.4f" % scale).rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def ref_scores() -> dict[str, Any]:
    j = read_json(CHCK82_VERIFY)
    scores = j["score_arithmetic"]["scores"]
    return {k: float(v) for k, v in scores.items() if v is not None}


def extract_scores(payload: dict[str, Any], logical_columns: list[str]) -> dict[str, float | None]:
    official = payload.get("official_overall", {}).get("scores", {})
    tasks = payload.get("tasks", {})
    out = {}
    for col in logical_columns:
        if col == "GlobalPIQA":
            vals = []
            for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
                if sub in tasks and tasks[sub].get("score") is not None:
                    vals.append(float(tasks[sub]["score"]))
            if len(vals) == 2:
                out[col] = float(mean(vals))
            elif official.get("GlobalPIQA") is not None:
                out[col] = float(official["GlobalPIQA"])
            else:
                out[col] = None
        elif col == "Reading":
            r = tasks.get("Reading", {})
            if isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
                out[col] = float(r["scores"]["Reading"])
            elif official.get("Reading") is not None:
                out[col] = float(official["Reading"])
            else:
                out[col] = None
        else:
            if col in tasks and tasks[col].get("score") is not None:
                out[col] = float(tasks[col]["score"])
            elif official.get(col) is not None:
                out[col] = float(official[col])
            else:
                out[col] = None
    return out


def write_training_metrics(run_dir: pathlib.Path, source_metrics: pathlib.Path | None, scale: float) -> None:
    base = {}
    if source_metrics is not None and source_metrics.exists():
        base = read_json(source_metrics)
    base.update({
        "status": "PRIVATE_SCALE_SENTINEL_ENDPOINT",
        "source_metrics_path": None if source_metrics is None else rel(source_metrics),
        "private_adapter_scale_materialized": scale,
        "created_utc": now(),
        "note": "Inference-time materialization only; no additional training beyond source frozen-tail run.",
    })
    (run_dir / "scientific_metrics.json").write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--target-prefix", default="tail4M_aligned_private_scale")
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--columns", nargs="+", default=DEFAULT_SENTINEL_COLUMNS, choices=list(COLUMN_MAP))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    scale_label = safe_scale_label(args.scale)
    target = f"{args.target_prefix}{scale_label}"
    run_dir = pathlib.Path(args.run_root) / target
    endpoint_dir = run_dir / "hf_model" / "final"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(args.source)
    source_metrics = source.parents[1] / "scientific_metrics.json" if len(source.parents) >= 2 else None

    mat_cmd = [sys.executable, "-B", str(MATERIALIZER), "--source", str(source), "--output", str(endpoint_dir), "--scale", str(args.scale), "--force"]
    print(json.dumps({"event": "materialize_scale", "target": target, "cmd": mat_cmd, "utc": now()}), flush=True)
    mat = subprocess.run(mat_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    print(json.dumps({"event": "materialize_done", "returncode": mat.returncode, "stdout_tail": mat.stdout[-1000:], "stderr_tail": mat.stderr[-1000:]}), flush=True)
    if mat.returncode != 0:
        raise SystemExit(mat.returncode)
    write_training_metrics(run_dir, source_metrics, args.scale)

    eval_cols = []
    for c in args.columns:
        eval_cols.extend(COLUMN_MAP[c])
    out_root = pathlib.Path(args.out_root) / target
    collate_root = pathlib.Path(args.collate_root) / target
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

    eval_cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", "final",
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(args.gpu),
        "--columns", *eval_cols,
    ]
    if args.force:
        eval_cmd.append("--force")
    print(json.dumps({"event": "launch_scale_sentinel_eval", "target": target, "scale": args.scale, "columns": eval_cols, "gpu": args.gpu, "utc": now()}), flush=True)
    t0 = time.time()
    proc = subprocess.run(eval_cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=7200)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({"event": "scale_sentinel_eval_done", "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload_path = out_root / "per_target" / f"{target}.json"
    payload = read_json(payload_path)
    scores = extract_scores(payload, args.columns)
    refs = ref_scores()
    deltas = {c: (None if scores.get(c) is None else float(scores[c] - refs[c])) for c in args.columns}
    summary = {
        "status": "PRIVATE_SCALE_SENTINEL_EVAL",
        "created_utc": now(),
        "target": target,
        "scale": args.scale,
        "source": rel(source),
        "run_dir": rel(run_dir),
        "endpoint": "final",
        "logical_columns": args.columns,
        "eval_columns": eval_cols,
        "scores": scores,
        "deltas_vs_chck82": deltas,
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / "stdout.log"),
        "stderr_log": rel(log_dir / "stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
        "scientific_reading": "This inference-only sentinel asks whether private-tail score movement is amplitude-controlled. It does not replace aligned-vs-shuffled or source-free probe evidence.",
    }
    summary_path = out_root / f"{target}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "target": target, "scale": args.scale, "scores": scores, "deltas_vs_chck82": deltas, "summary_path": rel(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
