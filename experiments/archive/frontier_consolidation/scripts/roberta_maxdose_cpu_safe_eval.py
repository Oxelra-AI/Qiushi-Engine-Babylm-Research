#!/usr/bin/env python3
"""research: CPU-safe RoBERTa MAX-dose stable-family evaluation driver.

This driver wraps the verified research CPU-safe scoring worker for RoBERTa MAX
arms.  Its immediate purpose is to score the just-launched MAX repeat RoBERTa
checkpoints after training finishes, so the existing MAX view ladder can be
compared with a same-architecture repeat counterfactual.  It can also resume
missing clean/view columns if needed, but it performs no GPU work and no
GlobalPIQA/SuperGLUE/AoA/upload/leaderboard action.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path.cwd()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
WORKER = WS / "scripts" / "cpu_safe_scoring_worker.py"
DEFAULT_OUT_ROOT = WS / "data" / "roberta_maxdose_repeat_eval" / "eval"
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "max_view": {
        "run_dir": WS / "training/runs/roberta_view_dose2p64x_matched_rowholdout_100M_seed43022",
        "target_prefix": "roberta_max_view_seed43022",
        "description": "RoBERTa MAX 2.64x compact-view arm, fixed research tokenizer and research view stream; already trained in research.",
        "family": "roberta_max_view_seed43022",
        "data_arm": "view",
        "dose_name": "dose2p64",
        "seed": 43022,
    },
    "max_repeat": {
        "run_dir": WS / "training/runs/roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022",
        "target_prefix": "roberta_max_repeat_seed43022",
        "description": "RoBERTa MAX 2.64x repeat arm, fixed research tokenizer and research repeat stream; counterfactual for MAX semantic V-R transfer.",
        "family": "roberta_max_repeat_seed43022",
        "data_arm": "repeat",
        "dose_name": "dose2p64",
        "seed": 43022,
    },
    "max_clean": {
        "run_dir": WS / "training/runs/roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022",
        "target_prefix": "roberta_max_clean_seed43022",
        "description": "RoBERTa MAX clean fixed-budget arm; total-effect V-C comparison against MAX view when clean scoring is complete.",
        "family": "roberta_max_clean_seed43022",
        "data_arm": "clean",
        "dose_name": "dose2p64",
        "seed": 43022,
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_score(value: Any) -> bool:
    try:
        x = float(value)
        return x == x
    except Exception:
        return False


def ready_cols(payload: dict[str, Any]) -> list[str]:
    tasks = payload.get("tasks") or {}
    ready: list[str] = []
    stable = payload.get("stable_family_scores") or {}
    for col in STABLE_COLUMNS:
        if finite_score(stable.get(col)):
            ready.append(col)
            continue
        rec = tasks.get(col) or {}
        if not isinstance(rec, dict) or rec.get("returncode") != 0:
            continue
        if col == "Reading":
            val = (rec.get("scores") or {}).get("Reading", rec.get("score"))
        else:
            val = rec.get("score")
        if finite_score(val):
            ready.append(col)
    return ready


def target_name(cfg: dict[str, Any], ck: str) -> str:
    return f"{cfg['target_prefix']}_{ck}"


def payload_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / "per_target" / f"{target}.json"


def model_exists(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return d.exists() and ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists())


def inspect_arm(cfg: dict[str, Any], out_root: pathlib.Path, checkpoints: list[str], columns: list[str], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_dir = pathlib.Path(cfg["run_dir"])
    metrics = run_dir / "scientific_metrics.json"
    for ck in checkpoints:
        target = target_name(cfg, ck)
        p = payload_path(out_root, target)
        existing = read_json(p) if p.exists() else {"tasks": {}}
        ready = ready_cols(existing)
        to_run = columns if force else [c for c in columns if c not in ready]
        rows.append({
            "arm": cfg["data_arm"],
            "family": cfg["family"],
            "dose_name": cfg["dose_name"],
            "seed": cfg["seed"],
            "checkpoint": ck,
            "target": target,
            "run_dir": rel(run_dir),
            "model_path": rel(run_dir / "hf_model" / ck),
            "model_exists": model_exists(run_dir, ck),
            "metrics_exists": metrics.exists(),
            "per_target": rel(p),
            "existing_ready_cols": ready,
            "requested_columns": columns,
            "to_run": to_run,
        })
    return rows


def run_worker(row: dict[str, Any], cfg: dict[str, Any], out_root: pathlib.Path, timeout_sec: int, force: bool) -> dict[str, Any]:
    if not row["model_exists"]:
        return {"status": "skip_missing_model", **row}
    if not row["metrics_exists"]:
        return {"status": "skip_missing_metrics", **row}
    if not row["to_run"] and not force:
        return {"status": "skip_complete", **row}
    cmd = [
        sys.executable, "-B", str(WORKER), "official",
        "--target", row["target"],
        "--run-dir", str(pathlib.Path(cfg["run_dir"])),
        "--endpoint", row["checkpoint"],
        "--out-root", str(out_root),
        "--description", cfg["description"],
        "--family", cfg["family"],
        "--timeout-sec", str(timeout_sec),
        "--columns", *row["to_run"],
    ]
    if force:
        cmd.append("--force")
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "status": "worker_finished" if proc.returncode == 0 else "worker_failed",
        **row,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - start, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--columns", nargs="*", default=None)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--timeout-sec", type=int, default=7200)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    checkpoints = args.checkpoints or CHECKPOINTS
    columns = args.columns or STABLE_COLUMNS
    cfg = ARM_CONFIGS[args.arm]
    rows = inspect_arm(cfg, out_root, checkpoints, columns, args.force)
    plan = {
        "status": "ROBERTA_MAXDOSE_CPU_SAFE_EVAL_PLAN" if args.plan_only else "ROBERTA_MAXDOSE_CPU_SAFE_EVAL_START",
        "created_utc": now(),
        "arm_key": args.arm,
        "arm_config": {k: rel(v) if isinstance(v, pathlib.Path) else v for k, v in cfg.items()},
        "out_root": rel(out_root),
        "columns": columns,
        "checkpoints": checkpoints,
        "rows": rows,
        "ready_to_score_count": sum(1 for r in rows if r["model_exists"] and r["metrics_exists"] and (args.force or r["to_run"])),
        "cpu_safety_source": rel(WORKER),
        "scientific_role": "Score RoBERTa MAX-dose arms in the same stable-family coordinate so MAX view-minus-repeat can test cross-architecture transfer of the Entity/state carrier.",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    summary_path = out_root / "cpu_safe_worker_results" / f"{args.arm}_ladder_plan_or_result.json"
    write_json(summary_path, plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    results: list[dict[str, Any]] = []
    for row in rows:
        res = run_worker(row, cfg, out_root, args.timeout_sec, args.force)
        results.append(res)
        print(json.dumps({k: res.get(k) for k in ["status", "target", "checkpoint", "to_run", "returncode", "elapsed_sec"]}, ensure_ascii=False), flush=True)
        if res.get("status") == "worker_failed":
            break
    final = {**plan, "status": "ROBERTA_MAXDOSE_CPU_SAFE_EVAL_DONE", "finished_utc": now(), "results": results}
    write_json(summary_path, final)
    print(json.dumps({"status": final["status"], "summary": rel(summary_path), "result_count": len(results), "failed_count": sum(1 for r in results if r.get("status") == "worker_failed")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
