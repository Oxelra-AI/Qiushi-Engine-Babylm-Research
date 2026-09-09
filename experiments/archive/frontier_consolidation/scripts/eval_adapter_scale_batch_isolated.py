#!/usr/bin/env python3
"""research: isolated-output adapter scale batch evaluator.

Reads scale shadow checkpoints from the main research scale-sweep directory, but
writes all evaluation outputs under a caller-specified output root.  This avoids
background write-lock conflicts between GPU batches.
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

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
MAIN_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep')
SHADOW_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep/shadow_runs')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
research = {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tag(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p").replace("-", "m")


def run_dir_for(scale: float) -> Path:
    return SHADOW_ROOT / f"adapter128_scale_{tag(scale)}_from_live20M"


def target_for(scale: float) -> str:
    return f"adapter128_scale_{tag(scale)}_from_live20M"


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


def eval_one(scale: float, gpu: int, out_root: Path, collate_root: Path, force: bool) -> dict:
    rd = run_dir_for(scale)
    target = target_for(scale)
    ckpt = rd / "hf_model/chck_20M/model.safetensors"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(rd),
        "--target", target,
        "--endpoint", "chck_20M",
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(gpu),
        "--columns", *COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = out_root / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{target}_stdout.log"
    stderr_log = log_dir / f"{target}_stderr.log"
    print(json.dumps({"event": "eval_scale_start", "scale": scale, "target": target, "gpu": gpu, "utc": now(), "out_root": str(out_root)}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=3600)
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    elapsed = time.time() - t0
    rec = {"scale": scale, "tag": tag(scale), "target": target, "run_dir": str(rd), "returncode": proc.returncode, "elapsed_sec": elapsed, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log), "out_root": str(out_root)}
    per_target = out_root / "per_target" / f"{target}.json"
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-2000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
        print(json.dumps({"event": "eval_scale_failed", **rec}, ensure_ascii=False), flush=True)
        return rec
    if not per_target.exists():
        rec["error"] = f"missing per_target {per_target}"
        print(json.dumps({"event": "eval_scale_missing_json", **rec}, ensure_ascii=False), flush=True)
        return rec
    payload = json.loads(per_target.read_text(encoding="utf-8"))
    scores = extract_scores(payload)
    c7 = cheap7(scores)
    rec["scores"] = scores
    rec["cheap7"] = c7
    rec["delta_vs_step35"] = {c: (scores[c] - research[c] if scores.get(c) is not None else None) for c in CHEAP}
    rec["delta_vs_step35"]["cheap7"] = None if c7 is None else c7 - cheap7(research)
    rec["per_target"] = str(per_target)
    print(json.dumps({"event": "eval_scale_done", "scale": scale, "cheap7": c7, "delta_cheap7": rec["delta_vs_step35"]["cheap7"], "elapsed_sec": elapsed}, ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--scales", nargs="+", type=float, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_base = Path(args.out_base)
    out_root = out_base / f"eval_{args.split}"
    collate_root = out_base / f"collate_{args.split}"
    records = [eval_one(float(s), int(args.gpu), out_root, collate_root, args.force) for s in args.scales]
    summary = {"status": "SCALE_BATCH_DONE", "gpu": args.gpu, "split": args.split, "scales": args.scales, "records": records, "main_shadow_root": str(SHADOW_ROOT)}
    out_base.mkdir(parents=True, exist_ok=True)
    out_json = out_base / f"scale_batch_{args.split}.json"
    out_md = out_base / f"scale_batch_{args.split}.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# research adapter scale batch {args.split}", "", "| scale | cheap7 | Δcheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        scores = r.get("scores") or {}
        c7 = r.get("cheap7")
        d = (r.get("delta_vs_step35") or {}).get("cheap7")
        def f(x):
            v = scores.get(x)
            return "" if v is None else f"{v:.3f}"
        lines.append(f"| {r['scale']:.2f} | {'' if c7 is None else f'{c7:.4f}'} | {'' if d is None else f'{d:+.4f}'} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SCALE_BATCH_DONE", "out_json": str(out_json), "out_md": str(out_md), "n": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
