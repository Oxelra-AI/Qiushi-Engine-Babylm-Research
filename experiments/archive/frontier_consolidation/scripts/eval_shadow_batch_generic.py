#!/usr/bin/env python3
"""Evaluate generic adapter-scale shadow checkpoints on cheap official-compatible columns."""
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
from typing import Dict, Any

USER_ROOT = _public_path('.')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tag(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p").replace("-", "m")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(path)


def extract_scores(payload: Dict[str, Any]) -> Dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: Dict[str, float | None] = {}
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


def cheap7(scores: Dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def eval_one(shadow_root: Path, prefix: str, endpoint: str, scale: float, gpu: int, out_base: Path, force: bool) -> Dict[str, Any]:
    run_dir = shadow_root / f"{prefix}_scale_{tag(scale)}_from_{endpoint}"
    target = f"{prefix}_scale_{tag(scale)}_{endpoint}"
    model_file = run_dir / "hf_model" / endpoint / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    out_root = out_base / f"eval_{prefix}_{endpoint}"
    collate_root = out_base / f"collate_{prefix}_{endpoint}"
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", rel(run_dir),
        "--target", target,
        "--endpoint", endpoint,
        "--out-root", rel(out_root),
        "--collate-root", rel(collate_root),
        "--gpu", str(gpu),
        "--columns", *COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = out_base / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{target}_stdout.log"
    stderr_log = log_dir / f"{target}_stderr.log"
    print(json.dumps({"event": "shadow_eval_start", "scale": scale, "target": target, "endpoint": endpoint, "gpu": gpu, "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    rec: Dict[str, Any] = {"scale": scale, "tag": tag(scale), "target": target, "run_dir": rel(run_dir), "endpoint": endpoint, "returncode": proc.returncode, "elapsed_sec": elapsed, "stdout_log": rel(stdout_log), "stderr_log": rel(stderr_log)}
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-2000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
    else:
        per_target = out_root / "per_target" / f"{target}.json"
        payload = json.loads(per_target.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        rec["scores"] = scores
        rec["cheap7"] = cheap7(scores)
        rec["per_target"] = rel(per_target)
    print(json.dumps({"event": "shadow_eval_done", "scale": scale, "target": target, "returncode": proc.returncode, "cheap7": rec.get("cheap7"), "elapsed_sec": elapsed}), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"shadow eval failed for scale {scale} rc={proc.returncode}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--scales", nargs="+", type=float, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    shadow_root = USER_ROOT / args.shadow_root if not Path(args.shadow_root).is_absolute() else Path(args.shadow_root)
    out_base = USER_ROOT / args.out_base if not Path(args.out_base).is_absolute() else Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    records = [eval_one(shadow_root, args.prefix, args.endpoint, float(s), int(args.gpu), out_base, args.force) for s in args.scales]
    out_json = out_base / f"{args.prefix}_{args.endpoint}_shadow_eval_batch.json"
    out_md = out_base / f"{args.prefix}_{args.endpoint}_shadow_eval_batch.md"
    out_json.write_text(json.dumps({"status": "SHADOW_EVAL_BATCH", "records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# research shadow eval batch {args.prefix} {args.endpoint}", "", "| scale | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        sc = r.get("scores") or {}
        def f(c: str) -> str:
            v = sc.get(c)
            return "" if v is None else f"{float(v):.3f}"
        c7 = r.get("cheap7")
        lines.append(f"| {float(r['scale']):.2f} | {'' if c7 is None else f'{float(c7):.4f}'} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SHADOW_EVAL_BATCH", "out_json": rel(out_json), "out_md": rel(out_md), "n": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
