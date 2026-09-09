#!/usr/bin/env python3
"""Evaluate research train-time adapter-scale 20M arms."""
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
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_collate')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
research = {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67}
LIVE = {"BLiMP": 60.38, "Supplement": 56.23, "EWoK": 49.49, "Entity": 18.65, "COMPS": 50.44, "GlobalPIQA": 32.225, "Reading": 8.31}
SCALE_MAP = {
    "infer_scale1p75": {"BLiMP": 60.56, "Supplement": 57.12, "EWoK": 50.74, "Entity": 18.47, "COMPS": 50.46, "GlobalPIQA": 32.71, "Reading": 8.225},
    "infer_scale2p00": {"BLiMP": 60.61, "Supplement": 57.35, "EWoK": 51.08, "Entity": 18.46, "COMPS": 50.33, "GlobalPIQA": 32.71, "Reading": 8.195},
}
ARMS = {
    "scale1p75": {"scale": 1.75, "label": "train-time adapter scale 1.75", "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M20M_seed43022'), "target": "adapter128_scale1p75_h100M20M_seed43022"},
    "scale2p00": {"scale": 2.0, "label": "train-time adapter scale 2.00", "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale2p00_h100M20M_seed43022'), "target": "adapter128_scale2p00_h100M20M_seed43022"},
}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cheap7(scores):
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(vals))


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


def eval_one(key: str, gpu: int, force: bool = False):
    arm = ARMS[key]
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model/chck_20M/model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    target = arm["target"]
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", "chck_20M",
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval/logs_eval_driver')
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{target}_stdout.log"
    stderr_log = log_dir / f"{target}_stderr.log"
    print(json.dumps({"event": "scaled_train_eval_start", "arm": key, "gpu": gpu, "cmd": cmd, "utc": now()}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=2400)
    elapsed = time.time() - t0
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    rec = {"arm": key, **arm, "run_dir": str(run_dir), "returncode": proc.returncode, "elapsed_sec": elapsed, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}
    per_target = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval/per_target') / f"{target}.json"
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-2000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
        print(json.dumps({"event": "scaled_train_eval_failed", **rec}, ensure_ascii=False), flush=True)
        return rec
    payload = json.loads(per_target.read_text(encoding="utf-8"))
    scores = extract_scores(payload)
    rec["scores"] = scores
    rec["cheap7"] = cheap7(scores)
    rec["delta_vs_step35"] = {c: scores[c] - research[c] for c in CHEAP}
    rec["delta_vs_step35"]["cheap7"] = rec["cheap7"] - cheap7(research)
    rec["delta_vs_step103_live_scale1"] = {c: scores[c] - LIVE[c] for c in CHEAP}
    rec["delta_vs_step103_live_scale1"]["cheap7"] = rec["cheap7"] - cheap7(LIVE)
    inf_key = "infer_scale1p75" if key == "scale1p75" else "infer_scale2p00"
    rec["delta_vs_inference_scale_map"] = {c: scores[c] - SCALE_MAP[inf_key][c] for c in CHEAP}
    rec["delta_vs_inference_scale_map"]["cheap7"] = rec["cheap7"] - cheap7(SCALE_MAP[inf_key])
    rec["per_target"] = str(per_target)
    print(json.dumps({"event": "scaled_train_eval_done", "arm": key, "cheap7": rec["cheap7"], "delta_step35": rec["delta_vs_step35"]["cheap7"], "elapsed_sec": elapsed}, ensure_ascii=False), flush=True)
    return rec


def summarize(records):
    rows = {
        "reference_20M": {"label": "research legal 20M", "scores": research, "cheap7": cheap7(research)},
        "live_scale1": {"label": "research trained scale1, evaluated scale1", "scores": LIVE, "cheap7": cheap7(LIVE)},
        "Inference_scale1p75": {"label": "research live20M evaluated scale1.75", "scores": SCALE_MAP["infer_scale1p75"], "cheap7": cheap7(SCALE_MAP["infer_scale1p75"])},
        "Inference_scale2p00": {"label": "research live20M evaluated scale2.00", "scores": SCALE_MAP["infer_scale2p00"], "cheap7": cheap7(SCALE_MAP["infer_scale2p00"])},
    }
    for r in records:
        if r.get("scores"):
            rows[r["arm"]] = {"label": r["label"], "scores": r["scores"], "cheap7": r["cheap7"], "delta_vs_step35": r.get("delta_vs_step35"), "delta_vs_inference_scale_map": r.get("delta_vs_inference_scale_map")}
    out = {"status": "SCALED_TRAIN_20M_EVAL", "records": records, "rows": rows}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval/scaled_train_20M_summary.json')
    out_md = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval/scaled_train_20M_summary.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research train-time adapter scale 20M evaluation", "", "| row | cheap7 | Δ vs research | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, row in rows.items():
        sc = row["scores"]
        c7 = row["cheap7"]
        d = c7 - cheap7(research)
        lines.append(f"| {row['label']} | {c7:.4f} | {d:+.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_json, out_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--arms", nargs="+", choices=sorted(ARMS), required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    records = [eval_one(a, args.gpu, args.force) for a in args.arms]
    out_json, out_md = summarize(records)
    print(json.dumps({"status": "SCALED_TRAIN_20M_EVAL", "out_json": str(out_json), "out_md": str(out_md), "n": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
