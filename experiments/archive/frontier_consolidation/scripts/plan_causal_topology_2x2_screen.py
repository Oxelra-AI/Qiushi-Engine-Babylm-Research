#!/usr/bin/env python3
"""research: dry-run plan for a possible causal topology 2x2 short screen.

This script only writes command plans. It does not launch training/evaluation.
It exists so that a future authorization can use exact, matched commands without
inventing new settings. The default screen is 20M (two legal epochs) and selected
eval endpoints chck_10M/chck_20M.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
SCAFFOLD = ROOT / "data/causal_topology_2x2_scaffold/manifest.json"
TRAINER = ROOT / "scripts/causal_gpt_trainer.py"
EVAL = ROOT / "scripts/selected_causal_checkpoint_eval.py"
COMPARE = ROOT / "scripts/compare_causal_topology_2x2.py"
TOKENIZER = ROOT / "data/causal_transfer_scaffold/neutral_tokenizer"
OUT_DEFAULT = ROOT / "data/causal_topology_2x2_screen_plan"

ARM_ORDER = ["compact_oneway", "repeat_oneway", "compact_reciprocal", "repeat_reciprocal"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold", type=pathlib.Path, default=SCAFFOLD)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--total-words", type=int, default=20_000_000)
    ap.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-endpoints", nargs="+", default=["chck_10M", "chck_20M"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.scaffold.read_text(encoding="utf-8"))

    train_runs: dict[str, Any] = {}
    eval_runs: dict[str, Any] = {}
    for i, arm in enumerate(ARM_ORDER):
        pool = manifest["arms"][arm]["path"]
        run_dir = ROOT / f"training/runs/topology2x2_{arm}_seed{args.seed}_{args.total_words//1_000_000}M"
        out_eval = ROOT / f"data/topology2x2_selected_eval_{arm}_{args.total_words//1_000_000}M"
        gpu_hint = i % 2
        train_cmd = [
            "PYTHONDONTWRITEBYTECODE=1", f"CUDA_VISIBLE_DEVICES={gpu_hint}", "python", "-B", str(TRAINER),
            "--pool", pool,
            "--tokenizer", str(TOKENIZER),
            "--run-dir", str(run_dir),
            "--gpu", "0",
            "--seed", str(args.seed),
            "--total-words", str(args.total_words),
            "--checkpoint-interval", str(args.checkpoint_interval),
            "--batch-size", str(args.batch_size),
        ]
        eval_cmd = [
            "PYTHONDONTWRITEBYTECODE=1", f"CUDA_VISIBLE_DEVICES={gpu_hint}", "python", "-B", str(EVAL),
            "--run-dir", str(run_dir),
            "--label", f"topology2x2_{arm}",
            "--gpu", "0",
            "--out-dir", str(out_eval),
            "--endpoints", *args.eval_endpoints,
        ]
        train_runs[arm] = {
            "arm": arm,
            "pool": pool,
            "pool_sha256": manifest["arms"][arm]["sha256"],
            "run_dir": str(run_dir),
            "gpu_hint": gpu_hint,
            "command": " ".join(train_cmd),
            "write_path": str(run_dir),
        }
        eval_runs[arm] = {
            "arm": arm,
            "run_dir": str(run_dir),
            "out_dir": str(out_eval),
            "gpu_hint": gpu_hint,
            "command": " ".join(eval_cmd),
            "write_path": str(out_eval),
            "trajectory_json": str(out_eval / "selected_causal_trajectory.json"),
        }
    compare_cmd = [
        "PYTHONDONTWRITEBYTECODE=1", "python", "-B", str(COMPARE),
        "--compact-oneway", eval_runs["compact_oneway"]["trajectory_json"],
        "--repeat-oneway", eval_runs["repeat_oneway"]["trajectory_json"],
        "--compact-reciprocal", eval_runs["compact_reciprocal"]["trajectory_json"],
        "--repeat-reciprocal", eval_runs["repeat_reciprocal"]["trajectory_json"],
        "--scaffold-manifest", str(args.scaffold),
        "--out-dir", str(ROOT / f"data/topology2x2_interaction_compare_{args.total_words//1_000_000}M"),
    ]
    plan = {
        "status": "CAUSAL_TOPOLOGY_2X2_SCREEN_PLAN_ONLY",
        "training_launched": False,
        "evaluation_launched": False,
        "scaffold_manifest": str(args.scaffold),
        "seed": args.seed,
        "total_words": args.total_words,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": args.batch_size,
        "eval_endpoints": args.eval_endpoints,
        "scientific_precondition": "Do not run unless A01 triangle or other direct evidence revives topology interaction despite research copied-token warning and after pending DeBERTa seed/scale grids are interpreted.",
        "stop_after_screen_unless": "semantic-by-topology interaction is positive across cheap7, cheap6(no GlobalPIQA), cheap5(no GlobalPIQA/Reading), and >=4 official columns at multiple selected endpoints.",
        "train_runs": train_runs,
        "eval_runs": eval_runs,
        "compare_command": " ".join(compare_cmd),
        "write_path_for_compare": str(ROOT / f"data/topology2x2_interaction_compare_{args.total_words//1_000_000}M"),
    }
    out_json = args.out_dir / "causal_topology_2x2_screen_plan.json"
    out_md = args.out_dir / "causal_topology_2x2_screen_plan.md"
    out_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research causal topology 2×2 short-screen plan\n\n"]
    md.append("This is a dry command plan only. No training or evaluation was launched.\n\n")
    md.append(f"Precondition: {plan['scientific_precondition']}\n\n")
    md.append(f"Settings: seed {args.seed}, total_words {args.total_words:,}, checkpoint_interval {args.checkpoint_interval:,}, batch_size {args.batch_size}, eval endpoints {args.eval_endpoints}.\n\n")
    md.append("## Training commands\n\n")
    for arm in ARM_ORDER:
        md.append(f"### {arm}\n\n```bash\n{train_runs[arm]['command']}\n```\n\n")
    md.append("## Evaluation commands\n\n")
    for arm in ARM_ORDER:
        md.append(f"### {arm}\n\n```bash\n{eval_runs[arm]['command']}\n```\n\n")
    md.append("## Comparison command\n\n")
    md.append(f"```bash\n{plan['compare_command']}\n```\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
