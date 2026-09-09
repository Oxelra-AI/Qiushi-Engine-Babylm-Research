#!/usr/bin/env python3
"""research: parallel-safe cheap-column evaluation for one LAMB 20M arm.

Each arm writes to a distinct out/collate root so lr005 and lr007 can run on two
H100s concurrently without file-lock conflicts. This is a bounded screen only:
BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading. No SuperGLUE/AoA
and no 80M continuation are launched here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')

ARMS = {
    "lr005": {
        "label": "LAMB lr=0.005",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M'),
        "target": "lamb_lr005_seed43022_20M",
    },
    "lr007": {
        "label": "LAMB lr=0.007",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr007_seed43022_20M'),
        "target": "lamb_lr007_seed43022_20M",
    },
}

EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ROOT_OUT = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval_parts')


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
    out["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--endpoint", default="chck_20M")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    run_dir = arm["run_dir"]
    endpoint = args.endpoint
    if args.arm == "lr005" and endpoint == "chck_14M":
        run_dir = _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow')
    model_path = run_dir / "hf_model" / endpoint
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"missing model checkpoint: {model_path}/model.safetensors")

    out_root = ROOT_OUT / args.arm / "eval"
    collate_root = ROOT_OUT / args.arm / "collate"
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    target = arm["target"] if endpoint == "chck_20M" else f"{arm['target']}_{endpoint}"
    per_target = out_root / "per_target" / f"{target}.json"

    if not per_target.exists():
        cmd = [sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
               "--run-dir", str(run_dir), "--target", target, "--endpoint", endpoint,
               "--out-root", str(out_root), "--collate-root", str(collate_root),
               "--gpu", str(args.gpu), "--columns", *EVAL_COLUMNS]
        print(json.dumps({"event": "lamb_arm_eval_start", "arm": args.arm, "gpu": args.gpu,
                          "run_dir": str(run_dir), "endpoint": endpoint, "target": target}), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if proc.returncode != 0:
            print(proc.stdout[-4000:], file=sys.stdout)
            print(proc.stderr[-6000:], file=sys.stderr)
            raise RuntimeError(f"evaluator failed for {args.arm} with code {proc.returncode}")
    payload = json.loads(per_target.read_text())
    scores = extract_scores(payload)
    result = {
        "status": "LAMB_ARM_20M_EVAL",
        "arm": args.arm,
        "label": arm["label"],
        "target": target,
        "run_dir": str(run_dir),
        "endpoint": endpoint,
        "per_target_json": str(per_target),
        "scores": scores,
        "cheap7": cheap7(scores),
    }
    (ROOT_OUT / args.arm / "lamb_arm_20M_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
