#!/usr/bin/env python3
"""Merge parallel research fast probe condition outputs and run central checks.

The full research learned run is split into two background jobs:
- fast_balanced_probe_v2_k00
- fast_balanced_probe_v2_k16

This script merges all_results and per-row files into one directory once both
evaluations finish.  It then can be followed by:

python3 -B experiments/archive/representation_and_objectives/scripts/analyze_balanced_probe_results.py \
  --run-out experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_merged \
  --analysis-out experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_analysis

python3 -B experiments/archive/representation_and_objectives/scripts/exact_choice_metrics.py \
  --include-learned \
  --learned-per-row experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_merged/per_row_eval_predictions.jsonl \
  --out experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_exact_choice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_INPUTS = [
    PROJECT / "data/fast_balanced_probe_v2_k00",
    PROJECT / "data/fast_balanced_probe_v2_k16",
]
DEFAULT_OUT = PROJECT / "data/fast_balanced_probe_v2_merged"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def append_file(dst: Path, src: Path) -> int:
    if not src.exists():
        return 0
    text = read_text(src)
    with dst.open("a", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
    return len([line for line in text.splitlines() if line.strip()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    needed = ["all_results.json", "per_row_eval_predictions.jsonl", "fast_balanced_probe_summary.md"]
    status: Dict[str, Any] = {str(inp): {name: (inp / name).exists() for name in needed} for inp in args.inputs}
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY", "inputs": status}, indent=2, sort_keys=True), flush=True)
        return
    missing = [(inp, name) for inp in args.inputs for name in ["all_results.json", "per_row_eval_predictions.jsonl"] if not (inp / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {[(str(a), b) for a, b in missing]}")
    args.out.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    for inp in args.inputs:
        all_results.extend(load_json(inp / "all_results.json"))
    (args.out / "all_results.json").write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {}
    for fname in ["per_row_eval_predictions.jsonl", "per_row_train_predictions.jsonl"]:
        dst = args.out / fname
        if dst.exists():
            dst.unlink()
        counts[fname] = sum(append_file(dst, inp / fname) for inp in args.inputs)
    manifest = {
        "status": "PARALLEL_FAST_PROBE_MERGED",
        "inputs": [str(x) for x in args.inputs],
        "out": str(args.out),
        "n_all_results": len(all_results),
        "jsonl_line_counts": counts,
        "source_status": status,
    }
    (args.out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
