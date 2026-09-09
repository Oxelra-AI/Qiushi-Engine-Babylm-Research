#!/usr/bin/env python3
"""research: evaluate research directional fork final checkpoints in parallel.

Runs the repaired official-compatible causal cheap7 evaluator for FF/FR/RR/RF
with two GPUs and then executes the directional interaction readout. Endpoint
scores are used only as a transfer signal for the compact reciprocal mechanism
screen, not as a BabyLM submission result.
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
ARMS = {
    "ff": "forward_branch/ff",
    "fr": "forward_branch/fr",
    "rr": "reverse_branch/rr",
    "rf": "reverse_branch/rf",
}


def make_cmd(model: pathlib.Path, out: pathlib.Path, gpu: int, columns: str, force: bool) -> List[str]:
    cmd = [
        sys.executable,
        "-B",
        str(EVAL),
        "--model-dir",
        str(model),
        "--output-dir",
        str(out),
        "--gpu",
        str(gpu),
        "--revision",
        "final",
        "--columns",
        columns,
    ]
    if force:
        cmd.append("--force")
    return cmd


def run_group(group: list[tuple[str, int, List[str], pathlib.Path]]) -> Dict[str, Any]:
    procs = {}
    t0 = time.time()
    for arm, gpu, cmd, log_path in group:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = log_path.open("w", encoding="utf-8")
        print("RUN", arm, "gpu", gpu, " ".join(cmd), flush=True)
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        procs[arm] = {"process": p, "file": f, "cmd": cmd, "log": str(log_path), "gpu": gpu, "start": time.time()}
    out = {}
    for arm, rec in procs.items():
        rc = rec["process"].wait()
        rec["file"].close()
        cur = {"returncode": rc, "elapsed_sec": time.time() - rec["start"], "cmd": rec["cmd"], "log": rec["log"], "gpu": rec["gpu"]}
        if rc != 0:
            lp = pathlib.Path(rec["log"])
            cur["log_tail"] = lp.read_text(encoding="utf-8", errors="replace")[-6000:] if lp.exists() else ""
        out[arm] = cur
    out["group_elapsed_sec"] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--data_type", default="compact")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--columns", default="BLiMP,Supplement,EWoK,Entity,COMPS,GlobalPIQA,Reading")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_root = pathlib.Path(args.run_root)
    eval_root = pathlib.Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) < 2:
        gpu_ids = gpu_ids * 2

    specs = []
    assignment = {"ff": gpu_ids[0], "fr": gpu_ids[1], "rr": gpu_ids[0], "rf": gpu_ids[1]}
    for arm, rel in ARMS.items():
        model = run_root / rel / "hf_model" / "final"
        if not (model / "config.json").exists():
            summary = {"status": "PARALLEL_EVAL_MISSING_MODEL", "arm": arm, "model": str(model)}
            (eval_root / "parallel_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise SystemExit(json.dumps(summary))
        specs.append((arm, assignment[arm], make_cmd(model, eval_root / arm, assignment[arm], args.columns, args.force), eval_root / "logs" / f"{arm}.log"))

    # Run in two waves to keep one evaluation per GPU while using both GPUs.
    waves = [[specs[0], specs[1]], [specs[2], specs[3]]]
    results: Dict[str, Any] = {}
    t0 = time.time()
    for idx, wave in enumerate(waves, start=1):
        print(f"WAVE {idx}", flush=True)
        wave_res = run_group(wave)
        results[f"wave_{idx}"] = wave_res
        for arm, rec in wave_res.items():
            if arm.startswith("group_"):
                continue
            results[arm] = rec
            if rec["returncode"] != 0:
                summary = {"status": f"PARALLEL_EVAL_FAIL_{arm.upper()}", "run_root": str(run_root), "eval_root": str(eval_root), "results": results}
                (eval_root / "parallel_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2), flush=True)
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
    print("RUN readout", " ".join(readout_cmd), flush=True)
    rt0 = time.time()
    rlog = eval_root / "logs" / "readout.log"
    with rlog.open("w", encoding="utf-8") as f:
        rp = subprocess.Popen(readout_cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        rrc = rp.wait()
    results["readout"] = {"returncode": rrc, "elapsed_sec": time.time() - rt0, "cmd": readout_cmd, "log": str(rlog)}
    if rrc != 0:
        results["readout"]["log_tail"] = rlog.read_text(encoding="utf-8", errors="replace")[-6000:]
    status = "DIRECTIONAL_PARALLEL_CHEAP7_EVAL_COMPLETE" if rrc == 0 else "DIRECTIONAL_PARALLEL_CHEAP7_READOUT_FAIL"
    summary = {"status": status, "run_root": str(run_root), "eval_root": str(eval_root), "results": results, "elapsed_sec": time.time() - t0}
    (eval_root / "parallel_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if status.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
