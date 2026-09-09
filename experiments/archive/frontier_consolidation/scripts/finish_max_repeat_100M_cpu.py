#!/usr/bin/env python3
"""research: targeted CPU-only finisher for MAX-repeat 100M stable rows.

The old serialized evaluator was cancelled after writing only BLiMP and
Supplement for `dose_max_repeat_chck_100M`.  This script runs exactly
the missing stable columns for that target under the same official-compatible
research harness, but hides CUDA (`CUDA_VISIBLE_DEVICES=-1`) so the H100s remain
reserved for training.  It updates the original per-target JSON in-place and
then verifies all six stable columns are finite.

No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
EVALUATOR = WS / "scripts/evaluate_compliant_endpoint.py"
OUT_ROOT = WS / "data/dose_ladder_stable_eval/eval"
COLLATE_ROOT = WS / "data/dose_ladder_stable_eval/collate"
PER_TARGET = OUT_ROOT / "per_target/dose_max_repeat_chck_100M.json"
RUN_DIR = WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022"
STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
MISSING_DEFAULT = ["EWoK", "Entity", "COMPS", "Reading"]
TARGET = "dose_max_repeat_chck_100M"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def task_score(rec: dict[str, Any], col: str) -> Any:
    if col == "Reading":
        s = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        return s.get("Reading") if s else rec.get("score")
    return rec.get("score")


def ready_cols(payload: dict[str, Any]) -> list[str]:
    out = []
    for c in STABLE:
        rec = (payload.get("tasks") or {}).get(c)
        if isinstance(rec, dict) and rec.get("returncode") == 0 and finite(task_score(rec, c)):
            out.append(c)
    return out


def run_eval(columns: list[str], force: bool, plan_only: bool) -> dict[str, Any]:
    if not EVALUATOR.exists():
        raise FileNotFoundError(EVALUATOR)
    model_path = RUN_DIR / "hf_model/chck_100M"
    metrics = RUN_DIR / "scientific_metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metrics.exists():
        raise FileNotFoundError(metrics)
    before = read_json(PER_TARGET) if PER_TARGET.exists() else {}
    before_ready = ready_cols(before)
    to_run = [c for c in columns if force or c not in before_ready]
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--target", TARGET,
        "--run-dir", str(RUN_DIR),
        "--endpoint", "chck_100M",
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", "0",
        "--columns", *to_run,
    ]
    if force:
        cmd.append("--force")
    preflight = {
        "status": "MAX_REPEAT_100M_CPU_FINISHER_PLAN",
        "created_utc": now(),
        "target": TARGET,
        "run_dir": rel(RUN_DIR),
        "model_path": rel(model_path),
        "per_target": rel(PER_TARGET),
        "before_ready_cols": before_ready,
        "requested_columns": columns,
        "to_run": to_run,
        "command": cmd,
        "cuda_visible_devices": "-1",
        "scientific_role": "Complete the late MAX-repeat 100M stable-family point so the fixed-budget identity and turnover near full exposure are measured rather than inferred.",
        "lowest_cost_method": "Only the four missing stable columns are run; CUDA is hidden so H100 training is not displaced; no GlobalPIQA/SuperGLUE/AoA is run.",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    if plan_only or not to_run:
        return {**preflight, "plan_only": plan_only, "no_new_eval_needed": not to_run}

    log = OUT_ROOT / "logs" / TARGET / "cpu_finish_stdout.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str((OUT_ROOT / "tmp" / f"{TARGET}_step265_cpu").resolve())
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # Ensure HF/module caches are writable and separate from old GPU jobs.
    cache = OUT_ROOT / "hf_cache" / f"{TARGET}_step265_cpu"
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "cpu_finish_start", "utc": now(), "cmd": cmd, "to_run": to_run}) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=28800)
        fh.write(json.dumps({"event": "cpu_finish_end", "utc": now(), "returncode": proc.returncode, "elapsed_sec": round(time.time()-t0, 2)}) + "\n")
    after = read_json(PER_TARGET) if PER_TARGET.exists() else {}
    after_ready = ready_cols(after)
    missing = [c for c in STABLE if c not in after_ready]
    result = {
        **preflight,
        "status": "MAX_REPEAT_100M_CPU_FINISHER_DONE" if proc.returncode == 0 and not missing else "MAX_REPEAT_100M_CPU_FINISHER_INCOMPLETE",
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 2),
        "log": rel(log),
        "after_ready_cols": after_ready,
        "missing_after": missing,
        "per_target_size": PER_TARGET.stat().st_size if PER_TARGET.exists() else 0,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", nargs="*", default=MISSING_DEFAULT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    result = run_eval(args.columns, args.force, args.plan_only)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result.get("status") == "MAX_REPEAT_100M_CPU_FINISHER_INCOMPLETE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
