#!/usr/bin/env python3
"""research: selected checkpoint trajectory evaluator for causal GPT compact-vs-repeat arms.

Runs the hardened research official-compatible causal cheap7 evaluator on an explicit
list of checkpoints. This prevents spending GPU time on all 50 checkpoints before
we know whether the cross-architecture compact-view signal exists.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys
import time
from statistics import mean
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
EVAL = STUDY / "scripts/causal_eval_officialish.py"
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def endpoint_words(endpoint: str) -> int:
    m = re.fullmatch(r"chck_(\d+)M", endpoint)
    if not m:
        raise ValueError(f"Endpoint must look like chck_20M: {endpoint}")
    return int(m.group(1)) * 1_000_000


def read_summary(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--columns", default="BLiMP,Supplement,EWoK,Entity,COMPS,GlobalPIQA,Reading")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hf_model = args.run_dir / "hf_model"
    found = {p.name: p for p in hf_model.iterdir() if p.is_dir()} if hf_model.exists() else {}
    missing = [e for e in args.endpoints if e not in found]
    plan = {
        "status": "DRY_RUN" if args.dry_run else "RUNNING_SELECTED_CAUSAL_EVAL",
        "run_dir": str(args.run_dir),
        "label": args.label,
        "endpoints_requested": args.endpoints,
        "missing_endpoints": missing,
        "gpu": args.gpu,
        "columns": args.columns,
        "out_dir": str(args.out_dir),
        "evaluator": str(EVAL),
        "per_endpoint_output_dirs": {e: str(args.out_dir / "per_endpoint" / e) for e in args.endpoints},
    }
    (args.out_dir / "selected_causal_eval_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    if missing:
        raise FileNotFoundError(f"Missing causal checkpoints: {missing}")

    rows: list[dict[str, Any]] = []
    for endpoint in args.endpoints:
        model_dir = hf_model / endpoint
        ep_out = args.out_dir / "per_endpoint" / endpoint
        summary_path = ep_out / "cheap7_summary.json"
        summary = None if args.force else read_summary(summary_path)
        if summary is None:
            ep_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-B", str(EVAL),
                "--model-dir", str(model_dir),
                "--output-dir", str(ep_out),
                "--gpu", str(args.gpu),
                "--columns", args.columns,
            ]
            if args.force:
                cmd.append("--force")
            print(json.dumps({"event": "run_causal_endpoint", "endpoint": endpoint, "cmd": cmd}), flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=14400)
            elapsed = time.time() - t0
            (ep_out / "wrapper_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (ep_out / "wrapper_stderr.log").write_text(proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                rows.append({"endpoint": endpoint, "words": endpoint_words(endpoint), "error": proc.stderr[-2000:] or proc.stdout[-2000:], "elapsed_sec": elapsed})
                continue
            summary = read_summary(summary_path)
        if summary is None:
            rows.append({"endpoint": endpoint, "words": endpoint_words(endpoint), "error": "summary missing"})
            continue
        score_vec = summary.get("cheap7_columns") or summary.get("scores") or {}
        row = {"endpoint": endpoint, "words": endpoint_words(endpoint), "cheap7": summary.get("cheap7")}
        for c in CHEAP_COLUMNS:
            row[c] = score_vec.get(c)
        row["summary_path"] = str(summary_path)
        rows.append(row)
        print(json.dumps({"endpoint": endpoint, "cheap7": row["cheap7"]}, ensure_ascii=False), flush=True)

    traj_json = args.out_dir / "selected_causal_trajectory.json"
    traj_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.out_dir / "selected_causal_trajectory.csv"
    cols = ["words", "endpoint"] + CHEAP_COLUMNS + ["cheap7", "summary_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            if row.get("cheap7") is not None:
                wr.writerow({c: row.get(c, "") for c in cols})
    valid = [r for r in rows if r.get("cheap7") is not None]
    best = max(valid, key=lambda r: r["cheap7"]) if valid else None
    result = {
        "status": "SELECTED_CAUSAL_CHECKPOINT_EVAL_COMPLETE" if best else "SELECTED_CAUSAL_CHECKPOINT_EVAL_NO_VALID_RESULTS",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "endpoints_requested": args.endpoints,
        "n_valid": len(valid),
        "n_failed": len(rows)-len(valid),
        "best_endpoint": best.get("endpoint") if best else None,
        "best_words": best.get("words") if best else None,
        "best_cheap7": best.get("cheap7") if best else None,
        "trajectory_json": str(traj_json),
        "trajectory_csv": str(csv_path),
    }
    summary_out = args.out_dir / "selected_causal_trajectory_summary.json"
    summary_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [f"# Selected causal checkpoint eval: {args.label}\n\n"]
    if best:
        md.append(f"Best selected endpoint: **{best['endpoint']}** cheap7={best['cheap7']}.\n\n")
        md.append("| Endpoint | Words(M) | " + " | ".join(CHEAP_COLUMNS) + " | cheap7 |\n")
        md.append("|---|---:|" + "|---:" * (len(CHEAP_COLUMNS)+1) + "|\n")
        for row in valid:
            vals = [f"{row.get(c):.3f}" if row.get(c) is not None else "" for c in CHEAP_COLUMNS]
            md.append(f"| {row['endpoint']} | {row['words']/1e6:.0f} | " + " | ".join(vals) + f" | {row['cheap7']:.6f} |\n")
    md.append(f"\nJSON: `{summary_out}`\n")
    (args.out_dir / "selected_causal_trajectory_summary.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
