#!/usr/bin/env python3
"""Evaluate arbitrary model checkpoints through the research official-compatible cheap-column harness."""
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

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_scores(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    out = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp.append(float(rec["score"]))
    out["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_base = Path(args.out_base)
    out_root = out_base / "eval"
    collate_root = out_base / "collate"
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    model_file = run_dir / "hf_model" / args.endpoint / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", args.target,
        "--endpoint", args.endpoint,
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(args.gpu),
        "--columns", *COLUMNS,
    ]
    if args.force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = out_base / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{args.target}_stdout.log"
    stderr_log = log_dir / f"{args.target}_stderr.log"
    print(json.dumps({"event": "custom_eval_start", "target": args.target, "endpoint": args.endpoint, "gpu": args.gpu, "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=2400)
    elapsed = time.time() - t0
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    rec = {"target": args.target, "run_dir": str(run_dir), "endpoint": args.endpoint, "returncode": proc.returncode, "elapsed_sec": elapsed, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-2000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
    else:
        per_target = out_root / "per_target" / f"{args.target}.json"
        payload = json.loads(per_target.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        rec["scores"] = scores
        rec["cheap7"] = cheap7(scores)
        rec["per_target"] = str(per_target)
    out_json = out_base / f"{args.target}_summary.json"
    out_md = out_base / f"{args.target}_summary.md"
    out_json.write_text(json.dumps({"status": "CUSTOM_EVAL", "record": rec}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# {args.target}", "", f"returncode: {proc.returncode}", f"elapsed_sec: {elapsed:.1f}"]
    if rec.get("scores"):
        sc = rec["scores"]
        lines += ["", f"cheap7: {rec['cheap7']:.4f}", "", "| BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |"]
    else:
        lines += ["", rec.get("stderr_tail", "")[-1000:]]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CUSTOM_EVAL", "out_json": str(out_json), "out_md": str(out_md), "cheap7": rec.get("cheap7"), "returncode": proc.returncode}, indent=2), flush=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
