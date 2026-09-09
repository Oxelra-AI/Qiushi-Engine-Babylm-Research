#!/usr/bin/env python3
"""Evaluate adapter128 scale1.75 70M/80M cheap official-compatible columns.

Uses the research custom checkpoint evaluator for the new adapter run only; research
70M/80M references already exist in research and are merged separately.
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
RUN = "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out-base", default="experiments/archive/frontier_consolidation/data/scale1p75_70_80_eval")
    ap.add_argument("--endpoints", nargs="+", default=["chck_70M", "chck_80M"])
    args = ap.parse_args()
    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    records = []
    for ep in args.endpoints:
        model_file = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model') / ep / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(model_file)
        target = f"adapter128_scale1p75_{ep}"
        cmd = [sys.executable, "-B", str(EVAL), "--gpu", str(args.gpu), "--run-dir", RUN, "--endpoint", ep, "--target", target, "--out-base", str(out_base / ep)]
        print(json.dumps({"event": "mature_eval_start", "target": target, "endpoint": ep, "cmd": cmd}), flush=True)
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), capture_output=True, text=True, timeout=3000)
        (out_base / f"{target}_driver_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (out_base / f"{target}_driver_stderr.log").write_text(proc.stderr, encoding="utf-8")
        rec = {"target": target, "endpoint": ep, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
        if proc.returncode == 0:
            summ = out_base / ep / f"{target}_summary.json"
            payload = json.loads(summ.read_text(encoding="utf-8"))
            rec.update({"summary": str(summ), "cheap7": payload.get("record", {}).get("cheap7"), "scores": payload.get("record", {}).get("scores")})
        records.append(rec)
        print(json.dumps({"event": "mature_eval_done", **rec}), flush=True)
        if proc.returncode != 0:
            break
    out = {"status": "SCALE1P75_70_80_EVAL", "records": records}
    out_json = out_base / "scale1p75_70_80_eval_summary.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(r["returncode"] != 0 for r in records):
        sys.exit(1)
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "cheap7": {r['endpoint']: r.get('cheap7') for r in records}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
