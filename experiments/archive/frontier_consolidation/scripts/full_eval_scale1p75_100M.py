#!/usr/bin/env python3
"""research: Full nine-column evaluation and Overall computation for scale1.75 100M endpoint.

Waits for training completion, verifies the AoA checkpoint ladder, runs all nine
columns across both GPUs, computes pristine-collated Overall, and compares with 41.8.

Execution plan:
  GPU0: cheap columns (BLiMP, Supplement, EWoK, Entity, COMPS, GP_par, GP_nonpar, Reading)
  GPU1: SuperGLUE fine-tuning + AoA  (or reverse if SuperGLUE is faster)
  CPU:  pristine collation after all columns complete

The adapter model requires trust_remote_code=True, which the official evaluation pipeline
and the research evaluator already support.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
EVAL_SCRIPT = str(WORKSPACE / "scripts/evaluate_compliant_endpoint.py")
COLLATE_SCRIPT = str(USER_ROOT / "experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py")

# Default paths for the scale1.75 100M endpoint
DEFAULT_RUN_DIR = str(WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder")
DEFAULT_OUT_ROOT = str(WORKSPACE / "data/scale1p75_100M_full_eval")
DEFAULT_COLLATE_ROOT = str(WORKSPACE / "data/scale1p75_100M_collate")
DEFAULT_SUMMARY_DIR = str(WORKSPACE / "data/scale1p75_100M_summary")
DEFAULT_TARGET = "scale1p75_100M_seed43022"
DEFAULT_ENDPOINT = "chck_100M"

# AoA steps required for strict-small
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]

# research legal 100M reference (exact pristine collation values)
reference_100M_REF = {
    "Overall": 41.257770896404615,
    "BLiMP": 65.8707181799453, "Supplement": 61.16566092036889,
    "EWoK": 50.39323748109589, "Entity": 27.400833994026197,
    "COMPS": 52.00834536316919, "SuperGLUE": 70.27986764740969,
    "GlobalPIQA": 36.0631067961165, "Reading": 8.13816768550987,
    "AoA": 0.0,
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS_GPU0 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                      "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
EVAL_COLUMNS_GPU1 = ["SuperGLUE", "AoA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def verify_checkpoint_ladder(run_dir: pathlib.Path) -> dict[str, Any]:
    """Verify that all 19 AoA checkpoints exist."""
    hf_model = run_dir / "hf_model"
    present = []
    missing = []
    for step in AOA_STEPS:
        p = hf_model / step / "model.safetensors"
        if p.exists():
            present.append(step)
        else:
            missing.append(step)
    return {"present": present, "missing": missing, "complete": len(missing) == 0}


def run_eval_column_set(
    run_dir: str, target: str, endpoint: str,
    out_root: str, collate_root: str, gpu: int,
    columns: list[str], force: bool = False,
) -> dict[str, Any]:
    """Run research evaluator for a set of columns on one GPU."""
    cmd = [
        sys.executable, "-B", EVAL_SCRIPT,
        "--arm", "reinvest",
        "--run-dir", run_dir,
        "--target", target,
        "--endpoint", endpoint,
        "--out-root", out_root,
        "--collate-root", collate_root,
        "--gpu", str(gpu),
        "--columns", *columns,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    print(json.dumps({"event": "eval_set_start", "gpu": gpu, "columns": columns, "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=14400)
    elapsed = time.time() - t0
    rec = {
        "gpu": gpu, "columns": columns, "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:],
    }
    print(json.dumps({"event": "eval_set_done", **{k: v for k, v in rec.items() if k not in ("stdout_tail", "stderr_tail")}}), flush=True)
    if proc.returncode != 0:
        rec["error"] = True
    return rec


def run_pristine_collation(
    run_dir: str, target: str, endpoint: str,
    out_root: str, collate_root: str, summary_dir: str,
) -> dict[str, Any]:
    """Run pristine collation to compute the official Overall."""
    hf_model = pathlib.Path(run_dir) / "hf_model"
    # Find EWoK predictions
    ewok_pred = None
    ewok_search = pathlib.Path(out_root) / "official_outputs" / target / "EWoK"
    if ewok_search.exists():
        for p in ewok_search.rglob("predictions.json"):
            ewok_pred = str(p)
            break
    aoa_dir = str(pathlib.Path(out_root) / "aoa_outputs" / target)
    collate_out = str(pathlib.Path(collate_root) / target)

    cmd = [
        sys.executable, "-B", COLLATE_SCRIPT,
        "--full-root", out_root,
        "--target", target,
        "--model-root", str(hf_model),
        "--out-dir", collate_out,
        "--aoa-dir", aoa_dir,
        "--endpoint", endpoint,
        "--tag", target,
    ]
    if ewok_pred:
        cmd += ["--pristine-ewok-predictions", ewok_pred]

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    t0 = time.time()
    print(json.dumps({"event": "collation_start", "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    rec = {
        "returncode": proc.returncode, "elapsed_sec": round(elapsed, 1),
        "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:],
    }
    print(json.dumps({"event": "collation_done", **{k: v for k, v in rec.items() if k not in ("stdout_tail", "stderr_tail")}}), flush=True)

    # Try to extract Overall from collation output
    overall_json = pathlib.Path(collate_out) / "overall_summary.json"
    if overall_json.exists():
        rec["overall_data"] = json.loads(overall_json.read_text(encoding="utf-8"))
    else:
        # Try to find it in stdout
        for line in proc.stdout.splitlines():
            if "Overall" in line:
                try:
                    d = json.loads(line)
                    if "Overall" in d:
                        rec["overall_data"] = d
                        break
                except Exception:
                    pass
    return rec


def extract_scores(payload: dict) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        r = tasks.get(c, {})
        scores[c] = float(r["score"]) if isinstance(r, dict) and r.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = tasks.get(c, {})
        if isinstance(r, dict) and r.get("score") is not None:
            gp.append(float(r["score"]))
    scores["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    r = tasks.get("Reading", {})
    if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
        scores["Reading"] = float(r["scores"]["Reading"])
    elif isinstance(r, dict) and r.get("score") is not None:
        scores["Reading"] = float(r["score"])
    else:
        scores["Reading"] = None
    # SuperGLUE
    sg = tasks.get("SuperGLUE", {})
    scores["SuperGLUE"] = float(sg.get("superglue_mean")) if isinstance(sg, dict) and sg.get("superglue_mean") is not None else None
    # AoA
    aoa = tasks.get("AoA", {})
    scores["AoA"] = float(aoa.get("aoa_leaderboard_score", 0.0)) if isinstance(aoa, dict) else None
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description="Full nine-column evaluation for scale1.75 100M endpoint")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--collate-root", default=DEFAULT_COLLATE_ROOT)
    p.add_argument("--summary-dir", default=DEFAULT_SUMMARY_DIR)
    p.add_argument("--skip-wait", action="store_true", help="Skip waiting for training completion")
    p.add_argument("--cheap-only", action="store_true", help="Run only cheap7 columns")
    p.add_argument("--skip-collation", action="store_true", help="Skip pristine collation")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    run_dir = pathlib.Path(args.run_dir)
    summary_dir = pathlib.Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Wait for training completion
    if not args.skip_wait:
        print(json.dumps({"event": "waiting_for_training", "run": str(run_dir), "utc": now()}), flush=True)
        while True:
            sm = run_dir / "scientific_metrics.json"
            cp = run_dir / "hf_model" / args.endpoint / "model.safetensors"
            if sm.exists() and cp.exists():
                break
            time.sleep(30)
            if time.time() - t0 > 14400:
                raise TimeoutError("Training did not complete within 4 hours")
    print(json.dumps({"event": "training_verified", "utc": now()}), flush=True)

    # Verify AoA ladder
    ladder = verify_checkpoint_ladder(run_dir)
    print(json.dumps({"event": "ladder_check", "complete": ladder["complete"],
                       "present": len(ladder["present"]), "missing": ladder["missing"][:5]}), flush=True)
    if not ladder["complete"]:
        print(json.dumps({"warning": "Incomplete AoA ladder", "missing": ladder["missing"]}), flush=True)

    # Run evaluation in parallel across GPUs
    pathlib.Path(args.out_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.collate_root).mkdir(parents=True, exist_ok=True)

    if args.cheap_only:
        # Only cheap columns on GPU0
        results = [run_eval_column_set(
            args.run_dir, args.target, args.endpoint,
            args.out_root, args.collate_root, 0,
            EVAL_COLUMNS_GPU0, args.force,
        )]
    else:
        # Parallel: cheap columns on GPU0, SuperGLUE+AoA on GPU1
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(run_eval_column_set,
                          args.run_dir, args.target, args.endpoint,
                          args.out_root, args.collate_root, 0,
                          EVAL_COLUMNS_GPU0, args.force),
                ex.submit(run_eval_column_set,
                          args.run_dir, args.target, args.endpoint,
                          args.out_root, args.collate_root, 1,
                          EVAL_COLUMNS_GPU1, args.force),
            ]
            results = [f.result() for f in futs]

    errors = [r for r in results if r.get("error")]
    if errors:
        print(json.dumps({"event": "eval_errors", "count": len(errors), "details": errors}), flush=True)

    # Parse payload
    payload_path = pathlib.Path(args.out_root) / "per_target" / f"{args.target}.json"
    scores = {}
    c7 = None
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        vals = [scores.get(c) for c in CHEAP_COLS]
        if all(v is not None for v in vals):
            c7 = float(mean(float(v) for v in vals))

    # Run pristine collation
    collation_result = None
    if not args.skip_collation and not args.cheap_only and not errors:
        collation_result = run_pristine_collation(
            args.run_dir, args.target, args.endpoint,
            args.out_root, args.collate_root, args.summary_dir,
        )

    # Compute deltas vs research 100M
    deltas = {}
    for c in list(reference_100M_REF.keys()):
        if c in ("Overall", "cheap7"):
            continue
        if scores.get(c) is not None:
            deltas[c] = round(float(scores[c]) - reference_100M_REF[c], 4)
    c7_delta = round(c7 - reference_100M_REF["cheap7"], 4) if c7 is not None else None

    # Extract Overall if available
    overall = None
    if collation_result and "overall_data" in collation_result:
        od = collation_result["overall_data"]
        overall = od.get("Overall") or od.get("scores", {}).get("Overall")

    # Summary
    summary = {
        "status": "SCALE1P75_100M_FULL_EVAL",
        "utc": now(),
        "run_dir": str(run_dir.relative_to(USER_ROOT)),
        "endpoint": args.endpoint,
        "scores": scores,
        "cheap7": c7,
        "overall": overall,
        "reference_100m_ref": reference_100M_REF,
        "deltas_vs_step35": deltas,
        "cheap7_delta": c7_delta,
        "overall_delta_vs_step35": round(overall - reference_100M_REF["Overall"], 4) if overall else None,
        "overall_margin_vs_41p8": round(overall - 41.8, 4) if overall else None,
        "aoa_ladder": ladder,
        "eval_results": results,
        "collation_result": collation_result,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_json = summary_dir / "scale1p75_100M_full_eval_summary.json"
    out_md = summary_dir / "scale1p75_100M_full_eval_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research Scale1.75 100M Full Nine-Column Evaluation",
        "",
    ]
    if overall is not None:
        lines.append(f"## Overall: **{overall:.4f}** (margin vs 41.8: **{overall - 41.8:+.4f}**)")
    else:
        lines.append("## Overall: pending collation")
    lines += [
        "",
        "## Scores vs research 100M",
        "",
        "| Column | Scale1.75 | research | Delta |",
        "|---|---:|---:|---:|",
    ]
    all_cols = CHEAP_COLS + ["SuperGLUE", "AoA"]
    for c in all_cols:
        u = scores.get(c)
        r = reference_100M_REF.get(c)
        d = deltas.get(c)
        if u is not None and r is not None and d is not None:
            lines.append(f"| {c} | {u:.3f} | {r:.3f} | {d:+.3f} |")
        else:
            lines.append(f"| {c} | {u} | {r} | {d} |")
    if c7 is not None:
        lines.append(f"| **cheap7** | **{c7:.4f}** | **{reference_100M_REF['cheap7']:.4f}** | **{c7_delta:+.4f}** |")
    if overall is not None:
        lines.append(f"| **Overall** | **{overall:.4f}** | **{reference_100M_REF['Overall']:.4f}** | **{overall - reference_100M_REF['Overall']:+.4f}** |")
    lines.append("")
    lines.append(f"Evidence: `{str(out_json.relative_to(USER_ROOT))}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
