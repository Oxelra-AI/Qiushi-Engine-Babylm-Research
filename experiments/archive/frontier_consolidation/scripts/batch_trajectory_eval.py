#!/usr/bin/env python3
"""research: batch cheap7 trajectory evaluator for dense checkpoint ladders.

Given a training run directory with dense checkpoints (chck_2M, chck_4M, ...),
evaluates each on the official zero-shot tasks to produce a trajectory CSV.
Processes checkpoints sequentially on one GPU (each evaluation uses the full GPU).

Scientific purpose: map the cheap7 trajectory at fine resolution to characterize
peak position, width, and robustness across seeds and adapter scales.

Usage:
  python -B batch_trajectory_eval.py \
    --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M \
    --gpu 0 --min-words 60000000 --max-words 100000000 \
    --out-dir experiments/archive/frontier_consolidation/data/trajectory_eval_scale1p75_seed43122
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
from statistics import mean
from typing import Any

HERE = _public_path('experiments/archive/frontier_consolidation/scripts')


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
EVALUATOR = STUDY / "scripts/evaluate_compliant_endpoint.py"

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]


def find_checkpoints(run_dir: pathlib.Path, min_words: int, max_words: int) -> list[tuple[int, pathlib.Path]]:
    """Find chck_XM directories within word range."""
    hf_model = run_dir / "hf_model"
    if not hf_model.exists():
        return []
    ckpts = []
    for d in sorted(hf_model.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"chck_(\d+)M", d.name)
        if not m:
            continue
        words = int(m.group(1)) * 1_000_000
        if min_words <= words <= max_words:
            if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
                ckpts.append((words, d))
    return sorted(ckpts)


def extract_scores(payload: dict) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp_vals = []
    for c in GP_COLS:
        rec = tasks.get(c, {})
        if "score" in rec and rec["score"] is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def eval_one_checkpoint(run_dir: pathlib.Path, endpoint: str, target: str,
                        out_root: pathlib.Path, collate_root: pathlib.Path,
                        gpu: int) -> dict[str, Any] | None:
    """Run cheap7 evaluation on one checkpoint. Returns scores or None."""
    per_target = out_root / "per_target" / f"{target}.json"
    if per_target.exists():
        payload = json.loads(per_target.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        c7 = cheap7(scores)
        return {"scores": scores, "cheap7": c7, "cached": True}

    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", endpoint,
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:] if proc.stderr else "unknown", "elapsed": elapsed}

    if per_target.exists():
        payload = json.loads(per_target.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        c7 = cheap7(scores)
        return {"scores": scores, "cheap7": c7, "elapsed": elapsed, "cached": False}
    return {"error": "per_target not created", "elapsed": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=100_000_000)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = args.out_dir / "eval"
    collate_root = args.out_dir / "collate"

    ckpts = find_checkpoints(args.run_dir, args.min_words, args.max_words)
    print(f"Found {len(ckpts)} checkpoints in [{args.min_words/1e6:.0f}M, {args.max_words/1e6:.0f}M]", flush=True)

    trajectory = []
    for words, ckpt_path in ckpts:
        endpoint = ckpt_path.name
        target = f"{args.label}_{endpoint}" if args.label else f"traj_{endpoint}"
        print(f"\n{'='*60}\nEvaluating {endpoint} ({words/1e6:.0f}M words)...", flush=True)

        result = eval_one_checkpoint(args.run_dir, endpoint, target, eval_root, collate_root, args.gpu)
        if result and "error" not in result:
            row = {"words": words, "endpoint": endpoint, **result["scores"], "cheap7": result["cheap7"]}
            trajectory.append(row)
            print(f"  cheap7={result['cheap7']:.4f}  BLiMP={result['scores'].get('BLiMP', 'N/A')}"
                  f"  EWoK={result['scores'].get('EWoK', 'N/A')}  Entity={result['scores'].get('Entity', 'N/A')}"
                  f"  {'(cached)' if result.get('cached') else ''}", flush=True)
        else:
            print(f"  ERROR: {result.get('error', 'unknown')[:200] if result else 'null'}", flush=True)
            trajectory.append({"words": words, "endpoint": endpoint, "error": str(result)})

    # Save trajectory
    traj_json = args.out_dir / "trajectory.json"
    traj_json.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save CSV
    csv_path = args.out_dir / "trajectory.csv"
    cols = ["words", "endpoint"] + CHEAP_COLUMNS + ["cheap7"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in trajectory:
            if "error" in row and row.get("cheap7") is None:
                continue
            vals = [str(row.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")

    # Summary
    valid = [r for r in trajectory if r.get("cheap7") is not None]
    if valid:
        best = max(valid, key=lambda r: r["cheap7"])
        summary = {
            "status": "TRAJECTORY_EVAL_COMPLETE",
            "run_dir": str(args.run_dir),
            "label": args.label,
            "checkpoints_evaluated": len(valid),
            "checkpoints_failed": len(trajectory) - len(valid),
            "best_endpoint": best["endpoint"],
            "best_words": best["words"],
            "best_cheap7": best["cheap7"],
            "best_scores": {c: best.get(c) for c in CHEAP_COLUMNS},
            "trajectory_json": str(traj_json),
            "trajectory_csv": str(csv_path),
        }
    else:
        summary = {"status": "TRAJECTORY_EVAL_NO_VALID_RESULTS", "run_dir": str(args.run_dir)}

    summary_path = args.out_dir / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown
    md_path = args.out_dir / "trajectory_summary.md"
    md = [f"# research trajectory evaluation: {args.label or args.run_dir.name}\n\n"]
    if valid:
        md.append(f"**Best**: {best['endpoint']} ({best['words']/1e6:.0f}M) cheap7={best['cheap7']:.4f}\n\n")
        md.append("| Checkpoint | Words(M) | " + " | ".join(CHEAP_COLUMNS) + " | cheap7 |\n")
        md.append("|---|---:|" + "|---:" * (len(CHEAP_COLUMNS) + 1) + "|\n")
        for row in sorted(valid, key=lambda r: r["words"]):
            vals = [f"{row.get(c, 0):.2f}" if row.get(c) is not None else "—" for c in CHEAP_COLUMNS]
            md.append(f"| {row['endpoint']} | {row['words']/1e6:.0f} | " + " | ".join(vals)
                      + f" | {row['cheap7']:.4f} |\n")
    md_path.write_text("".join(md), encoding="utf-8")

    print(f"\n{'='*60}\nDone. {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
