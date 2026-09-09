#!/usr/bin/env python3
"""Evaluate multiple existing checkpoints with the research custom evaluator.

Used in research to compare the 30M/40M trajectory of the adapter128 scale1.75
run against matched research without launching new training exposure.
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

USER_ROOT = _public_path('.')
EVAL = _public_path('experiments/archive/frontier_consolidation/scripts/eval_custom_checkpoint.py')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-prefix", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    args = ap.parse_args()
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    records = []
    for ep in args.endpoints:
        target = f"{args.target_prefix}_{ep}"
        cmd = [
            sys.executable, "-B", str(EVAL),
            "--gpu", str(args.gpu),
            "--run-dir", args.run_dir,
            "--endpoint", ep,
            "--target", target,
            "--out-base", str(out_base / ep),
        ]
        print(json.dumps({"event": "trajectory_eval_start", "target": target, "endpoint": ep, "gpu": args.gpu, "cmd": cmd}), flush=True)
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=2600)
        (out_base / f"{target}_driver_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (out_base / f"{target}_driver_stderr.log").write_text(proc.stderr, encoding="utf-8")
        rec = {"endpoint": ep, "target": target, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        if proc.returncode == 0:
            summary = out_base / ep / f"{target}_summary.json"
            if summary.exists():
                payload = json.loads(summary.read_text(encoding="utf-8"))
                rec["summary"] = str(summary)
                rec["cheap7"] = payload.get("record", {}).get("cheap7")
                rec["scores"] = payload.get("record", {}).get("scores")
        records.append(rec)
        print(json.dumps({"event": "trajectory_eval_done", **rec}), flush=True)
        if proc.returncode != 0:
            break
    out = {"status": "TRAJECTORY_CHECKPOINT_EVAL", "run_dir": args.run_dir, "records": records}
    out_json = out_base / f"{args.target_prefix}_trajectory_eval_summary.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(r["returncode"] != 0 for r in records):
        sys.exit(1)
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "n": len(records), "cheap7": {r['endpoint']: r.get('cheap7') for r in records}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
