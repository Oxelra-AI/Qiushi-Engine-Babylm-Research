#!/usr/bin/env python3
"""research: memory-safe cheap-column evaluation of mature reinvest checkpoints.

The matched clean-control trajectory launched in research will retain 20M, 70M,
and 80M checkpoints.  The reinvest side already exists through 100M under the
same research legal tokenizer.  This script evaluates only the cheap official-
compatible columns for reinvest chck_70M and chck_80M, with isolated target names,
so a later clean-control result can be compared immediately in the mature regime.

It does not run SuperGLUE or AoA and does not collate a final endpoint score; the
outputs are research-facing matched-reference evidence for the legal-tokenizer
mechanism question.
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
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
RUN_DIR = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2"
OUT_ROOT = WORKSPACE / "data/legal_mature_treatment_effect_eval"
COLLATE_ROOT = WORKSPACE / "data/legal_mature_treatment_effect_collate"
LOG_ROOT = WORKSPACE / "data/legal_mature_treatment_effect_eval/logs_step050_wrapper"
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
DEFAULT_EXPOSURES = [70, 80]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def smi_rows() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {p.stderr}")
    rows: list[dict[str, int]] = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total_mib": total, "used_mib": used, "free_mib": free, "util_gpu_pct": util})
    return rows


def run_cmd(cmd: list[str], *, log_path: pathlib.Path, timeout: int, dry_run: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"cmd": cmd, "dry_run": dry_run, "timeout": timeout, "started_utc": now_utc()}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "command_start", **rec}, ensure_ascii=False) + "\n")
    if dry_run:
        rec.update({"returncode": None, "stdout_tail": "", "stderr_tail": ""})
        return rec
    p = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=timeout)
    rec.update({
        "returncode": p.returncode,
        "finished_utc": now_utc(),
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-6000:],
    })
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "command_finish", **rec}, ensure_ascii=False) + "\n")
    if p.returncode != 0:
        raise RuntimeError(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


def eval_command(endpoint_m: int, gpu: int, preflight: bool, force: bool) -> list[str]:
    target = f"complianttok_reinvest_seed43022_{endpoint_m}M"
    endpoint = f"chck_{endpoint_m}M"
    cmd = [
        sys.executable,
        str(EVAL),
        "--arm", "reinvest",
        "--target", target,
        "--run-dir", str(RUN_DIR),
        "--endpoint", endpoint,
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *CHEAP_COLUMNS,
    ]
    if preflight:
        cmd.append("--preflight-only")
    if force:
        cmd.append("--force")
    return cmd


def wait_for_memory(gpu: int, min_free_mib: int, timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    last: dict[str, Any] | None = None
    while True:
        rows = smi_rows()
        row = next((r for r in rows if r["index"] == gpu), None)
        if row is None:
            raise RuntimeError(f"GPU {gpu} not found in {rows}")
        rec = {"event": "gpu_wait_sample", "time_utc": now_utc(), "gpu": gpu, "row": row, "min_free_mib": min_free_mib}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec), flush=True)
        last = rec
        if row["free_mib"] >= min_free_mib:
            return {"waited_sec": round(time.time() - start, 1), "last_sample": last}
        if time.time() - start > timeout_sec:
            out = {"status": "REINVEST_MATURE_REF_WAIT_TIMEOUT", "gpu": gpu, "waited_sec": round(time.time() - start, 1), "last_sample": last}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
            raise SystemExit(2)
        time.sleep(poll_sec)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--exposures", type=int, nargs="*", default=DEFAULT_EXPOSURES, help="Million-word checkpoint endpoints to evaluate, e.g. 70 80")
    ap.add_argument("--min-free-mib", type=int, default=60_000)
    ap.add_argument("--wait-timeout-sec", type=int, default=18_000)
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exposures = sorted(set(args.exposures))
    if not exposures or any(x <= 0 or x > 100 for x in exposures):
        raise RuntimeError(f"Invalid exposures: {args.exposures}")
    if any(x < 70 for x in exposures):
        raise RuntimeError(f"This mature-reference wrapper should not duplicate the research 20M evaluation: {exposures}")
    if not EVAL.exists():
        raise FileNotFoundError(EVAL)
    missing = [x for x in exposures if not (RUN_DIR / "hf_model" / f"chck_{x}M").exists()]
    if missing:
        raise FileNotFoundError(f"Missing reinvest checkpoints: {missing}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"reinvest_mature_refs_{'_'.join(map(str, exposures))}M_gpu{args.gpu}.jsonl"
    events: list[dict[str, Any]] = []
    header = {
        "status": "REINVEST_MATURE_REF_PREFLIGHT" if args.dry_run else "REINVEST_MATURE_REF_START",
        "time_utc": now_utc(),
        "scientific_decision": "provide same-tokenizer reinvest reference scores at mature checkpoints for the clean-control survival test",
        "gpu": args.gpu,
        "exposures_m": exposures,
        "run_dir": str(RUN_DIR),
        "out_root": str(OUT_ROOT),
        "collate_root": str(COLLATE_ROOT),
        "cheap_columns": CHEAP_COLUMNS,
        "min_free_mib": args.min_free_mib,
        "wait_timeout_sec": args.wait_timeout_sec,
        "log_path": str(log_path),
        "initial_gpu_rows": smi_rows(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "header", **header}, ensure_ascii=False) + "\n")
    print(json.dumps(header, indent=2, ensure_ascii=False), flush=True)

    # CPU/lightweight preflight for exact paths before any GPU wait.
    for x in exposures:
        rec = run_cmd(eval_command(x, args.gpu, preflight=True, force=False), log_path=log_path, timeout=600, dry_run=args.dry_run)
        events.append({"stage": "preflight", "exposure_m": x, **rec})

    wait_rec = wait_for_memory(args.gpu, args.min_free_mib, args.wait_timeout_sec, args.poll_sec, log_path) if not args.dry_run else {"waited_sec": 0, "last_sample": None}
    events.append({"stage": "gpu_wait", **wait_rec})

    for x in exposures:
        rec = run_cmd(eval_command(x, args.gpu, preflight=False, force=args.force), log_path=log_path, timeout=8 * 3600, dry_run=args.dry_run)
        events.append({"stage": "cheap_eval", "exposure_m": x, **rec})

    summary = {
        "status": "REINVEST_MATURE_REF_DONE" if not args.dry_run else "REINVEST_MATURE_REF_DRY_RUN_DONE",
        "time_utc": now_utc(),
        "gpu": args.gpu,
        "exposures_m": exposures,
        "out_root": str(OUT_ROOT),
        "per_target_jsons": [str(OUT_ROOT / "per_target" / f"complianttok_reinvest_seed43022_{x}M.json") for x in exposures],
        "log_path": str(log_path),
        "events": events,
    }
    out = OUT_ROOT / f"reinvest_mature_refs_{'_'.join(map(str, exposures))}M_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_json": str(out), "per_target_jsons": summary["per_target_jsons"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
