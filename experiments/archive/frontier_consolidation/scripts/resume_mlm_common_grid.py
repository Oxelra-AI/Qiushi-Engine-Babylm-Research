#!/usr/bin/env python3
"""research: resume missing/incomplete selected DeBERTa common-grid cheap7 endpoints.

The research bg scorings timed out after producing many per-target payloads.  This
script resumes only missing cheap-task columns/checkpoints under the same output
roots, relying on research's existing-task skip logic and explicitly requesting only
columns whose cheap7 components are absent.

No SuperGLUE/AoA or leaderboard submission is performed.
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
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EVAL_COLUMNS_ALL = ZERO_COLUMNS + GP_COLS + ["Reading"]
EXPECTED_COMMON2M = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M",
    "chck_94M", "chck_96M", "chck_98M", "chck_100M",
]


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
research = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
RECONSTRUCT = WORKSPACE / "scripts/reconstruct_selected_mlm_trajectory.py"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        out[c] = fnum(rec.get("score")) if isinstance(rec, dict) else None
    gp_vals: list[float] = []
    for c in GP_COLS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        v = fnum(rec.get("score")) if isinstance(rec, dict) else None
        if v is not None:
            gp_vals.append(v)
    out["GlobalPIQA"] = float(mean(gp_vals)) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {}) if isinstance(tasks, dict) else {}
    if isinstance(rd, dict) and isinstance(rd.get("scores"), dict):
        out["Reading"] = fnum(rd["scores"].get("Reading"))
    elif isinstance(rd, dict):
        out["Reading"] = fnum(rd.get("score"))
    else:
        out["Reading"] = None
    return out


def missing_underlying_columns(payload_path: pathlib.Path) -> list[str]:
    if not payload_path.exists():
        return list(EVAL_COLUMNS_ALL)
    payload = read_json(payload_path)
    scores = extract_scores(payload)
    missing: list[str] = []
    for c in ZERO_COLUMNS:
        if scores.get(c) is None:
            missing.append(c)
    if scores.get("GlobalPIQA") is None:
        # Request both split columns; research will skip any split already done.
        missing.extend(GP_COLS)
    if scores.get("Reading") is None:
        missing.append("Reading")
    return missing


def parse_job(spec: str) -> tuple[str, pathlib.Path, pathlib.Path]:
    # Prefer comma in shell commands; retain pipe for older notes when quoted.
    if "," in spec:
        parts = spec.split(",", 2)
    else:
        parts = spec.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use LABEL,RUN_DIR,OUT_DIR (or quoted LABEL|RUN_DIR|OUT_DIR)")
    return parts[0], pathlib.Path(parts[1]), pathlib.Path(parts[2])


def run_eval(label: str, run_dir: pathlib.Path, out_dir: pathlib.Path, endpoint: str, cols: list[str], gpu: int) -> dict[str, Any]:
    target = f"{label}_{endpoint}"
    eval_root = out_dir / "eval"
    collate_root = out_dir / "collate"
    log_dir = out_dir / "resume_logs" / target
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "runtime_cache_resume" / target
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k, p in {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TMPDIR": cache_root / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    cmd = [
        sys.executable, "-B", str(research),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", target,
        "--endpoint", endpoint,
        "--out-root", str(eval_root),
        "--collate-root", str(collate_root),
        "--gpu", str(gpu),
        "--columns", *cols,
    ]
    t0 = time.time()
    print(json.dumps({"event": "resume_eval_start", "target": target, "endpoint": endpoint, "columns": cols, "gpu": gpu, "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=5400)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    rec = {
        "target": target,
        "endpoint": endpoint,
        "columns_requested": cols,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
        "stdout_log": str(log_dir / "stdout.log"),
        "stderr_log": str(log_dir / "stderr.log"),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    print(json.dumps({"event": "resume_eval_done", **{k: rec[k] for k in ["target", "endpoint", "returncode", "elapsed_sec"]}, "stdout_tail": rec["stdout_tail"][-500:], "stderr_tail": rec["stderr_tail"][-500:]}, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(rec)
    return rec


def reconstruct(label: str, out_dir: pathlib.Path, endpoints: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-B", str(RECONSTRUCT), "--out-dir", str(out_dir), "--label", label, "--endpoints", *endpoints, "--write"]
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), capture_output=True, text=True, timeout=120)
    rec = {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    print(json.dumps({"event": "reconstruct_after_resume", "label": label, "returncode": proc.returncode, "stdout_tail": rec["stdout"][-1000:], "stderr_tail": rec["stderr"][-1000:]}, ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", type=parse_job, required=True, help="LABEL|RUN_DIR|OUT_DIR")
    ap.add_argument("--endpoints", nargs="+", default=EXPECTED_COMMON2M)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    all_records: list[dict[str, Any]] = []
    for label, run_dir, out_dir in args.job:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in args.endpoints:
            target = f"{label}_{ep}"
            payload_path = out_dir / "eval" / "per_target" / f"{target}.json"
            missing = missing_underlying_columns(payload_path)
            if not missing:
                all_records.append({"label": label, "endpoint": ep, "target": target, "action": "skip_complete"})
                continue
            rec = run_eval(label, run_dir, out_dir, ep, missing, args.gpu)
            all_records.append({"label": label, "endpoint": ep, "target": target, "action": "evaluated", **rec})
        all_records.append({"label": label, "action": "reconstruct", "reconstruct": reconstruct(label, out_dir, args.endpoints)})
    summary = {"status": "RESUME_MLM_COMMON_GRID_COMPLETE", "gpu": args.gpu, "jobs": [j[0] for j in args.job], "endpoints": args.endpoints, "records": all_records}
    first_out = args.job[0][2]
    out_json = first_out / "resume_common_grid_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "n_records": len(all_records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
