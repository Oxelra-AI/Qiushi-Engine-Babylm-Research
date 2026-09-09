#!/usr/bin/env python3
"""Wait for corrected research format-eval payloads and run the paired item table.

This turns the format screen into the same decision object used for coherent86: per-
item gains/losses against the shared chck_82M trunk.  It is safe to launch before
GPU evaluation has finished; it waits for the expected payloads and then calls the
research table builder.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = _public_path('.')
PAIR = _public_path('experiments/archive/relation_learning/scripts/pairwise_item_flip_table.py')
BASE = _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected')
OUT = _public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips')
ARMS = ["isolated_all", "half_coherent_half_isolated", "coherent_unsplit_special"]
SEEDS = [98097, 98098]


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def expected_payloads() -> dict[str, pathlib.Path]:
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            label = f"step098_{arm}_seed{seed}_alpha0p75"
            out[label] = BASE / arm / "eval" / label / "per_target" / f"{label}.json"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-timeout", type=float, default=21600.0)
    ap.add_argument("--wait-interval", type=float, default=120.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = expected_payloads()
    start = time.time()
    while True:
        missing = {k: rel(v) for k, v in payloads.items() if not v.exists()}
        if not missing:
            break
        if time.time() - start >= args.wait_timeout:
            state = {"status": "FORMAT_PAIRWISE_WAIT_TIMEOUT", "missing": missing, "elapsed_sec": round(time.time()-start, 1), "created_utc": now()}
            (_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/wait_timeout.json')).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
            raise SystemExit(2)
        print(json.dumps({"event": "waiting_for_format_payloads", "missing_count": len(missing), "elapsed_sec": round(time.time()-start, 1), "utc": now()}), flush=True)
        time.sleep(float(args.wait_interval))
    cmd = [sys.executable, "-B", str(PAIR), "--out-dir", str(OUT)]
    for label, path in payloads.items():
        cmd += ["--payload", f"{label}={path}"]
    print(json.dumps({"event": "pairwise_start", "cmd": cmd, "utc": now()}), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=1800)
    (_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/launcher_stdout.log')).write_text(p.stdout, encoding="utf-8")
    (_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/launcher_stderr.log')).write_text(p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-3000:], flush=True)
        print(p.stderr[-3000:], flush=True)
        raise SystemExit(p.returncode)
    summary = {"status": "FORMAT_PAIRWISE_DONE", "created_utc": now(), "payloads": {k: rel(v) for k, v in payloads.items()}, "out_json": rel(_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/pairwise_item_flips.json')), "out_md": rel(_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/pairwise_item_flips.md')), "stdout_tail": p.stdout[-3000:]}
    (_public_path('experiments/archive/relation_learning/data/format_pairwise_item_flips/postprocess_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
