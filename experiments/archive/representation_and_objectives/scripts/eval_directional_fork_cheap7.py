#!/usr/bin/env python3
"""Evaluate research directional fork final checkpoints with repaired causal cheap7 harness.

This wraps frontier_consolidation's research official-compatible causal evaluator for the four
final arms FF/FR/RR/RF and then runs the research directional-interaction readout.
It deliberately evaluates endpoint final checkpoints only; full SuperGLUE/AoA are
not needed for this mechanism screen.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

EVAL = pathlib.Path("experiments/archive/frontier_consolidation/scripts/causal_eval_officialish.py")
READOUT = pathlib.Path("experiments/archive/representation_and_objectives/scripts/directional_interaction_readout.py")
ARMS = {"ff": "forward_branch/ff", "fr": "forward_branch/fr", "rr": "reverse_branch/rr", "rf": "reverse_branch/rf"}


def run_logged(cmd: List[str], log_path: pathlib.Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    rec = {"returncode": rc, "elapsed_sec": time.time() - t0, "cmd": cmd, "log": str(log_path)}
    if rc != 0 and log_path.exists():
        rec["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_type", default="compact")
    ap.add_argument("--columns", default="BLiMP,Supplement,EWoK,Entity,COMPS,GlobalPIQA,Reading")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_root = pathlib.Path(args.run_root)
    eval_root = pathlib.Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for arm, rel in ARMS.items():
        model = run_root / rel / "hf_model" / "final"
        if not (model / "config.json").exists():
            summary = {"status": "EVAL_MISSING_MODEL", "arm": arm, "model": str(model)}
            (eval_root / "eval_launcher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise SystemExit(json.dumps(summary))
        out = eval_root / arm
        cmd = [
            sys.executable,
            "-B",
            str(EVAL),
            "--model-dir",
            str(model),
            "--output-dir",
            str(out),
            "--gpu",
            str(args.gpu),
            "--revision",
            "final",
            "--columns",
            args.columns,
        ]
        if args.force:
            cmd.append("--force")
        results[arm] = run_logged(cmd, eval_root / "logs" / f"{arm}.log")
        if results[arm]["returncode"] != 0:
            summary = {"status": f"EVAL_FAIL_{arm.upper()}", "results": results, "run_root": str(run_root), "eval_root": str(eval_root)}
            (eval_root / "eval_launcher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise SystemExit(1)
    readout_cmd = [
        sys.executable,
        "-B",
        str(READOUT),
        "--run_root",
        str(run_root),
        "--eval_root",
        str(eval_root),
        "--data_type",
        args.data_type,
        "--output",
        str(eval_root / "directional_interaction_readout.json"),
    ]
    results["readout"] = run_logged(readout_cmd, eval_root / "logs" / "readout.log")
    status = "DIRECTIONAL_CHEAP7_EVAL_COMPLETE" if results["readout"]["returncode"] == 0 else "DIRECTIONAL_CHEAP7_EVAL_READOUT_FAIL"
    summary = {"status": status, "run_root": str(run_root), "eval_root": str(eval_root), "results": results}
    (eval_root / "eval_launcher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not status.endswith("COMPLETE"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
