#!/usr/bin/env python3
"""research: selected official-compatible cheap7 evaluator for MLM checkpoint ladders.

The research dense runs saved 2M-spaced checkpoints. Full dense scoring is not the
lowest reliable action while the causal GPT arms use both H100s, so this wrapper
scores only predeclared endpoints needed to test peak timing/width. It reuses the
repaired research evaluator functions, but accepts an explicit endpoint list.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import pathlib
import sys
from statistics import mean
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
SCRIPT_DIR = STUDY / "scripts"
research = SCRIPT_DIR / "batch_trajectory_eval.py"
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load_step168():
    spec = importlib.util.spec_from_file_location("batch_trajectory_eval", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def endpoint_words(endpoint: str) -> int:
    if not endpoint.startswith("chck_") or not endpoint.endswith("M"):
        raise ValueError(f"Endpoint must look like chck_82M: {endpoint}")
    return int(endpoint[len("chck_"):-1]) * 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_root = args.out_dir / "eval"
    collate_root = args.out_dir / "collate"
    mod = load_step168()

    found = {p.name: p for p in (args.run_dir / "hf_model").iterdir() if p.is_dir()} if (args.run_dir / "hf_model").exists() else {}
    missing = [e for e in args.endpoints if e not in found]
    plan = {
        "status": "DRY_RUN" if args.dry_run else "RUNNING_SELECTED_EVAL",
        "run_dir": str(args.run_dir),
        "label": args.label,
        "endpoints_requested": args.endpoints,
        "missing_endpoints": missing,
        "gpu": args.gpu,
        "out_dir": str(args.out_dir),
        "eval_root": str(eval_root),
        "collate_root": str(collate_root),
    }
    (args.out_dir / "selected_eval_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    if missing:
        raise FileNotFoundError(f"Missing endpoints under {args.run_dir / 'hf_model'}: {missing}")

    trajectory: list[dict[str, Any]] = []
    for endpoint in args.endpoints:
        target = f"{args.label}_{endpoint}"
        print(f"Evaluating {endpoint} as {target}", flush=True)
        result = mod.eval_one_checkpoint(args.run_dir, endpoint, target, eval_root, collate_root, args.gpu)
        if result and "error" not in result:
            row = {"words": endpoint_words(endpoint), "endpoint": endpoint, **result["scores"], "cheap7": result["cheap7"], "cached": bool(result.get("cached", False))}
            trajectory.append(row)
            print(json.dumps({"endpoint": endpoint, "cheap7": result["cheap7"], "scores": result["scores"], "cached": result.get("cached", False)}, ensure_ascii=False), flush=True)
        else:
            row = {"words": endpoint_words(endpoint), "endpoint": endpoint, "error": result.get("error", "unknown") if result else "null"}
            trajectory.append(row)
            print(json.dumps({"endpoint": endpoint, "error": row["error"]}, ensure_ascii=False), flush=True)

    traj_json = args.out_dir / "selected_trajectory.json"
    traj_json.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.out_dir / "selected_trajectory.csv"
    cols = ["words", "endpoint"] + CHEAP_COLUMNS + ["cheap7", "cached"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in trajectory:
            if row.get("cheap7") is None:
                continue
            wr.writerow({c: row.get(c, "") for c in cols})

    valid = [r for r in trajectory if r.get("cheap7") is not None]
    best = max(valid, key=lambda r: r["cheap7"]) if valid else None
    summary = {
        "status": "SELECTED_MLM_CHECKPOINT_EVAL_COMPLETE" if best else "SELECTED_MLM_CHECKPOINT_EVAL_NO_VALID_RESULTS",
        "run_dir": str(args.run_dir),
        "label": args.label,
        "endpoints_requested": args.endpoints,
        "n_valid": len(valid),
        "n_failed": len(trajectory) - len(valid),
        "best_endpoint": best.get("endpoint") if best else None,
        "best_words": best.get("words") if best else None,
        "best_cheap7": best.get("cheap7") if best else None,
        "best_scores": {c: best.get(c) for c in CHEAP_COLUMNS} if best else None,
        "mean_selected_cheap7": mean([r["cheap7"] for r in valid]) if valid else None,
        "trajectory_json": str(traj_json),
        "trajectory_csv": str(csv_path),
    }
    summary_path = args.out_dir / "selected_trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = args.out_dir / "selected_trajectory_summary.md"
    md = [f"# Selected MLM trajectory eval: {args.label}\n\n"]
    if best:
        md.append(f"Best selected endpoint: **{best['endpoint']}** cheap7={best['cheap7']:.6f}.\n\n")
        md.append("| Endpoint | Words(M) | " + " | ".join(CHEAP_COLUMNS) + " | cheap7 |\n")
        md.append("|---|---:|" + "|---:" * (len(CHEAP_COLUMNS) + 1) + "|\n")
        for row in valid:
            vals = [f"{row.get(c):.3f}" if row.get(c) is not None else "" for c in CHEAP_COLUMNS]
            md.append(f"| {row['endpoint']} | {row['words']/1e6:.0f} | " + " | ".join(vals) + f" | {row['cheap7']:.6f} |\n")
    md.append(f"\nJSON: `{summary_path}`\n")
    md_path.write_text("".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
