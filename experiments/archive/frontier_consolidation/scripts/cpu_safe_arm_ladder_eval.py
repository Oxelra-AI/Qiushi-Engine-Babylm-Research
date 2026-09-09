#!/usr/bin/env python3
"""research: CPU-safe stable-family ladder evaluation for ready mechanism arms.

This is a small driver around the verified research CPU-safe scoring worker.  It
runs no GPU code.  Its purpose is to evaluate newly ready second-basin or breadth
checkpoints with the same official-compatible stable-family coordinate while
preserving one per-target JSON per checkpoint.

Default use in research is the already-finished second-basin MAX repeat arm.  The
view arm and breadth arm should be evaluated only after their training tasks have
finished and their checkpoint directories exist.
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
OUT_ROOT = WS / "data" / "second_basin_and_breadth_stable_eval" / "eval"

CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "second_basin_repeat": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "second_basin_max_repeat_seed43122",
        "description": "Second-basin MAX 2.64x repeat arm; stable-family counterfactual for fresh-basin compact view-minus-repeat test.",
        "family": "second_basin_max_repeat_seed43122",
        "data_arm": "repeat",
        "dose_name": "dose2p64",
        "seed": 43122,
    },
    "second_basin_view": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "second_basin_max_view_seed43122",
        "description": "Second-basin MAX 2.64x compact-view arm; stable-family half of the fresh-basin V-R mechanism test.",
        "family": "second_basin_max_view_seed43122",
        "data_arm": "view",
        "dose_name": "dose2p64",
        "seed": 43122,
    },
    "max_breadth": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "max_breadth_seed43022",
        "description": "MAX 2.64x source-preserving breadth arm; tests compact re-expression against additional same-population sentence experience.",
        "family": "max_breadth_seed43022",
        "data_arm": "breadth",
        "dose_name": "dose2p64",
        "seed": 43022,
    },
    "max_permuted": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_permuted_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "max_permuted_seed43022",
        "description": "MAX 2.64x correspondence-broken compact companion arm; same source and compact-rewrite multisets as MAX view but no source-to-own-view pairing.",
        "family": "max_permuted_seed43022",
        "data_arm": "permuted_companion",
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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def ready_cols(payload: dict[str, Any]) -> list[str]:
    tasks = payload.get("tasks") or {}
    ready: list[str] = []
    for col in STABLE_COLUMNS:
        rec = tasks.get(col) or {}
        if not isinstance(rec, dict) or rec.get("returncode") != 0:
            continue
        score = None
        if col == "Reading":
            score = (rec.get("scores") or {}).get("Reading", rec.get("score"))
        else:
            score = rec.get("score")
        try:
            if score is not None and float(score) == float(score):
                ready.append(col)
        except Exception:
            pass
    return ready


def target_name(cfg: dict[str, Any], ck: str) -> str:
    return f"{cfg['target_prefix']}_{ck}"


def payload_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / "per_target" / f"{target}.json"


def inspect_arm(cfg: dict[str, Any], out_root: pathlib.Path, checkpoints: list[str], columns: list[str], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_dir = pathlib.Path(cfg["run_dir"])
    metrics = run_dir / "scientific_metrics.json"
    for ck in checkpoints:
        model_path = run_dir / "hf_model" / ck
        target = target_name(cfg, ck)
        p = payload_path(out_root, target)
        existing = read_json(p) if p.exists() else {"tasks": {}}
        ready = ready_cols(existing)
        requested = columns or STABLE_COLUMNS
        to_run = requested if force else [c for c in requested if c not in ready]
        rows.append({
            "arm": cfg.get("data_arm"),
            "family": cfg.get("family"),
            "dose_name": cfg.get("dose_name"),
            "seed": cfg.get("seed"),
            "checkpoint": ck,
            "target": target,
            "run_dir": rel(run_dir),
            "model_path": rel(model_path),
            "model_exists": model_path.exists(),
            "metrics_exists": metrics.exists(),
            "per_target": rel(p),
            "existing_ready_cols": ready,
            "requested_columns": requested,
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
    elapsed = round(time.time() - start, 3)
    res = {"status": "worker_finished" if proc.returncode == 0 else "worker_failed", **row, "returncode": proc.returncode, "elapsed_sec": elapsed, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--columns", nargs="*", default=None)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
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
        "status": "CPU_SAFE_ARM_LADDER_PLAN" if args.plan_only else "CPU_SAFE_ARM_LADDER_START",
        "created_utc": now(),
        "arm_key": args.arm,
        "arm_config": {k: rel(v) if isinstance(v, pathlib.Path) else v for k, v in cfg.items()},
        "out_root": rel(out_root),
        "columns": columns,
        "checkpoints": checkpoints,
        "rows": rows,
        "ready_to_score_count": sum(1 for r in rows if r["model_exists"] and r["metrics_exists"] and (args.force or r["to_run"])),
        "cpu_safety_source": rel(WORKER),
        "scientific_role": "Evaluate ready mechanism arms in the stable-family coordinate without GPU visibility; no GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
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
        if res.get("returncode") not in (None, 0) and res.get("status") == "worker_failed":
            break
    final = {**plan, "status": "CPU_SAFE_ARM_LADDER_DONE", "finished_utc": now(), "results": results}
    write_json(summary_path, final)
    print(json.dumps({"status": final["status"], "summary": rel(summary_path), "result_count": len(results), "failed_count": sum(1 for r in results if r.get("status") == "worker_failed")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
