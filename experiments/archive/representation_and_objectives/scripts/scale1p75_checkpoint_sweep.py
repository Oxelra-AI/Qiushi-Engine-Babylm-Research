#!/usr/bin/env python3
"""research: bounded cheap7 sweep around the scale1.75 80M peak.

This is not a new training route.  It establishes whether the existing
scale1.75 trajectory has an unevaluated neighboring checkpoint whose official
zero-shot/Reading surface is high enough that a complete score could exceed the
visible 41.80 Strict-Small frontier.  It evaluates only the non-SuperGLUE/AoA
columns needed for the cheap7 threshold, using the same official-compatible
evaluator with isolated writable caches and output roots.
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
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any


CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading",
]
DEFAULT_ENDPOINTS = ["chck_77M", "chck_78M", "chck_79M", "chck_81M", "chck_82M", "chck_83M"]
KNOWN_80M_SCORES = {
    "BLiMP": 68.11,
    "Supplement": 62.62,
    "EWoK": 49.24,
    "Entity": 28.20,
    "COMPS": 52.11,
    "SuperGLUE": 69.25971446477669,
    "GlobalPIQA": 38.105,
    "Reading": 8.30,
    "AoA": 0.0,
    "Overall": 41.77163494053074,
}
KNOWN_80M_SCORES["cheap7"] = float(mean(KNOWN_80M_SCORES[c] for c in CHEAP_COLUMNS))
ASSUMED_SUPERGLUE_80M = 69.25971446477669
ASSUMED_AOA = 0.0
FRONTIER = 41.80
CHEAP7_THRESHOLD_AT_80M_SUPERGLUE = (9.0 * FRONTIER - ASSUMED_SUPERGLUE_80M - ASSUMED_AOA) / 7.0


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
RUN_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
EVAL_SCRIPT = USER_ROOT / "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py"
DEFAULT_OUT_ROOT = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_m(endpoint: str) -> int:
    if not endpoint.startswith("chck_") or not endpoint.endswith("M"):
        raise ValueError(f"Unexpected endpoint format: {endpoint}")
    return int(endpoint[len("chck_"):-1])


def target_for(endpoint: str) -> str:
    return f"scale1p75_{endpoint}"


def per_target_path(out_root: Path, endpoint: str) -> Path:
    return out_root / "eval" / "per_target" / f"{target_for(endpoint)}.json"


def scores_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col)
        if not isinstance(rec, dict) or rec.get("returncode") != 0 or rec.get("score") is None:
            raise RuntimeError({"missing_or_failed_column": col, "target": payload.get("target"), "rec": rec})
        scores[col] = float(rec["score"])
    read_rec = tasks.get("Reading")
    if not isinstance(read_rec, dict) or read_rec.get("returncode") != 0:
        raise RuntimeError({"missing_or_failed_reading": payload.get("target"), "rec": read_rec})
    read_scores = read_rec.get("scores")
    if not isinstance(read_scores, dict) or read_scores.get("Reading") is None:
        raise RuntimeError({"missing_reading_score": payload.get("target"), "rec": read_rec})
    scores["Reading"] = float(read_scores["Reading"])
    scores["Reading_eye"] = float(read_scores.get("Reading_eye", math.nan))
    scores["Reading_self_paced"] = float(read_scores.get("Reading_self_paced", math.nan))
    scores["GlobalPIQA"] = (scores["GlobalPIQA_parallel"] + scores["GlobalPIQA_nonparallel"]) / 2.0
    scores["cheap7"] = float(mean(scores[c] for c in CHEAP_COLUMNS))
    scores["projected_overall_if_superglue_80M_aoa0"] = (7.0 * scores["cheap7"] + ASSUMED_SUPERGLUE_80M + ASSUMED_AOA) / 9.0
    scores["cheap7_margin_vs_threshold"] = scores["cheap7"] - CHEAP7_THRESHOLD_AT_80M_SUPERGLUE
    scores["projected_overall_margin_vs_41p8"] = scores["projected_overall_if_superglue_80M_aoa0"] - FRONTIER
    scores["superglue_required_for_41p8_with_aoa0"] = 9.0 * FRONTIER - 7.0 * scores["cheap7"]
    return scores


def completed_scores(out_root: Path, endpoint: str) -> dict[str, float] | None:
    p = per_target_path(out_root, endpoint)
    if not p.exists():
        return None
    try:
        return scores_from_payload(read_json(p))
    except Exception:
        return None


def run_endpoint(out_root: Path, endpoint: str, gpu: int, force: bool, timeout_sec: int) -> dict[str, Any]:
    ep_dir = out_root / "jobs" / endpoint
    ep_dir.mkdir(parents=True, exist_ok=True)
    log_stdout = ep_dir / "driver.stdout.log"
    log_stderr = ep_dir / "driver.stderr.log"
    existing = completed_scores(out_root, endpoint)
    if existing is not None and not force:
        return {
            "endpoint": endpoint,
            "target": target_for(endpoint),
            "gpu": gpu,
            "status": "skip_existing",
            "per_target_json": rel(per_target_path(out_root, endpoint)),
            "scores": existing,
        }

    model_path = RUN_DIR / "hf_model" / endpoint / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    cmd = [
        sys.executable,
        "-B",
        str(EVAL_SCRIPT),
        "--arm",
        "reinvest",
        "--run-dir",
        str(RUN_DIR),
        "--target",
        target_for(endpoint),
        "--endpoint",
        endpoint,
        "--out-root",
        str(out_root / "eval"),
        "--collate-root",
        str(out_root / "collate"),
        "--gpu",
        str(gpu),
        "--columns",
        *EVAL_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    # research's child commands set their own per-target writable cache.  These
    # outer variables protect imports or trust_remote_code operations before the
    # inner runner attaches its target cache.
    cache = ep_dir / "runtime_cache"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "TMPDIR": cache / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    t0 = time.time()
    print(json.dumps({"event": "endpoint_eval_start", "endpoint": endpoint, "gpu": gpu, "target": target_for(endpoint), "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=timeout_sec)
    elapsed = time.time() - t0
    log_stdout.write_text(proc.stdout, encoding="utf-8", errors="replace")
    log_stderr.write_text(proc.stderr, encoding="utf-8", errors="replace")
    rec: dict[str, Any] = {
        "endpoint": endpoint,
        "target": target_for(endpoint),
        "gpu": gpu,
        "cmd": [str(x) for x in cmd],
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_log": rel(log_stdout),
        "stderr_log": rel(log_stderr),
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-5000:],
        "per_target_json": rel(per_target_path(out_root, endpoint)),
    }
    if proc.returncode != 0:
        rec["status"] = "failed"
        write_json(ep_dir / "job_record.json", rec)
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    payload_path = per_target_path(out_root, endpoint)
    if not payload_path.exists():
        rec["status"] = "missing_payload"
        write_json(ep_dir / "job_record.json", rec)
        raise RuntimeError(rec)
    rec["scores"] = scores_from_payload(read_json(payload_path))
    rec["status"] = "done"
    write_json(ep_dir / "job_record.json", rec)
    print(json.dumps({"event": "endpoint_eval_done", "endpoint": endpoint, "gpu": gpu, "cheap7": rec["scores"]["cheap7"], "projected_overall": rec["scores"]["projected_overall_if_superglue_80M_aoa0"], "elapsed_sec": rec["elapsed_sec"]}), flush=True)
    return rec


def write_summary(out_root: Path, records: list[dict[str, Any]], include_known_80m: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        row = {
            "endpoint": rec["endpoint"],
            "status": rec.get("status"),
            "gpu": rec.get("gpu"),
            "elapsed_sec": rec.get("elapsed_sec"),
            "per_target_json": rec.get("per_target_json"),
        }
        row.update(rec["scores"])
        rows.append(row)
    if include_known_80m:
        known = {
            "endpoint": "chck_80M",
            "status": "known_step156_full_score_not_reevaluated_in_step165",
            "per_target_json": "experiments/archive/representation_and_objectives/data/scale1p75_score_collation/scale1p75_score_summary.json",
        }
        for k, v in KNOWN_80M_SCORES.items():
            known[k] = float(v)
        known["GlobalPIQA_parallel"] = 26.21
        known["GlobalPIQA_nonparallel"] = 50.00
        known["projected_overall_if_superglue_80M_aoa0"] = KNOWN_80M_SCORES["Overall"]
        known["cheap7_margin_vs_threshold"] = KNOWN_80M_SCORES["cheap7"] - CHEAP7_THRESHOLD_AT_80M_SUPERGLUE
        known["projected_overall_margin_vs_41p8"] = KNOWN_80M_SCORES["Overall"] - FRONTIER
        known["superglue_required_for_41p8_with_aoa0"] = 9.0 * FRONTIER - 7.0 * KNOWN_80M_SCORES["cheap7"]
        rows.append(known)
    rows.sort(key=lambda r: endpoint_m(r["endpoint"]))
    best = max(rows, key=lambda r: float(r["cheap7"])) if rows else None
    cheap_candidates = [r for r in rows if float(r["cheap7"]) >= CHEAP7_THRESHOLD_AT_80M_SUPERGLUE]
    projected_candidates = [r for r in rows if float(r["projected_overall_if_superglue_80M_aoa0"]) >= FRONTIER]
    summary = {
        "status": "SCALE1P75_CHECKPOINT_SWEEP_DONE",
        "created_utc": now(),
        "run_dir": rel(RUN_DIR),
        "evaluated_endpoint_window": [r["endpoint"] for r in rows],
        "newly_evaluated_records": records,
        "known_80M_reference": KNOWN_80M_SCORES,
        "thresholds": {
            "frontier_overall": FRONTIER,
            "assumed_superglue_for_screen": ASSUMED_SUPERGLUE_80M,
            "assumed_aoa_for_screen": ASSUMED_AOA,
            "cheap7_threshold_with_80M_superglue_aoa0": CHEAP7_THRESHOLD_AT_80M_SUPERGLUE,
            "interpretation": "A checkpoint only merits full SuperGLUE/AoA verification if cheap7 is high enough that plausible SuperGLUE can cross 41.80. Exposure selection alone is endpoint establishment, not a new learning principle.",
        },
        "rows": rows,
        "best_by_cheap7": best,
        "cheap7_threshold_candidates": cheap_candidates,
        "projected_80M_superglue_candidates": projected_candidates,
        "decision_signal": "launch_full_verification_for_candidate" if cheap_candidates else "no_neighbor_crosses_cheap7_threshold_stop_scanning",
    }
    write_json(out_root / "summary" / "scale1p75_checkpoint_sweep_summary.json", summary)

    md = [
        "# research Scale1.75 checkpoint sweep",
        "",
        f"Frontier Overall: {FRONTIER:.3f}",
        f"Cheap7 threshold assuming 80M SuperGLUE={ASSUMED_SUPERGLUE_80M:.6f} and AoA=0: **{CHEAP7_THRESHOLD_AT_80M_SUPERGLUE:.6f}**",
        "",
        "| endpoint | cheap7 | projected Overall at 80M SG | SG required for 41.80 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['endpoint']} | {float(r['cheap7']):.6f} | {float(r['projected_overall_if_superglue_80M_aoa0']):.6f} | "
            f"{float(r['superglue_required_for_41p8_with_aoa0']):.6f} | {float(r.get('BLiMP', math.nan)):.3f} | "
            f"{float(r.get('Supplement', math.nan)):.3f} | {float(r.get('EWoK', math.nan)):.3f} | {float(r.get('Entity', math.nan)):.3f} | "
            f"{float(r.get('COMPS', math.nan)):.3f} | {float(r.get('GlobalPIQA', math.nan)):.3f} | {float(r.get('Reading', math.nan)):.3f} | {r.get('status')} |"
        )
    md += [
        "",
        f"Best by cheap7: `{best['endpoint'] if best else 'none'}` at {float(best['cheap7']) if best else float('nan'):.6f}.",
        f"Decision signal: **{summary['decision_signal']}**.",
        "",
        f"JSON: `{rel(out_root / 'summary' / 'scale1p75_checkpoint_sweep_summary.json')}`",
    ]
    (out_root / "summary" / "scale1p75_checkpoint_sweep_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global RUN_DIR
    ap = argparse.ArgumentParser(description="Bounded cheap7 sweep for scale1.75 neighboring checkpoints")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout-sec-per-endpoint", type=int, default=14400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_root = args.out_root if args.out_root.is_absolute() else USER_ROOT / args.out_root
    RUN_DIR = args.run_dir if args.run_dir.is_absolute() else USER_ROOT / args.run_dir
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary").mkdir(parents=True, exist_ok=True)

    endpoints = list(dict.fromkeys(args.endpoints))
    if any(e == "chck_80M" for e in endpoints):
        raise ValueError("research sweep intentionally excludes chck_80M because it was already fully scored in research")
    for e in endpoints:
        if not (RUN_DIR / "hf_model" / e / "model.safetensors").exists():
            raise FileNotFoundError(RUN_DIR / "hf_model" / e / "model.safetensors")
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(EVAL_SCRIPT)
    plan = {
        "status": "SCALE1P75_CHECKPOINT_SWEEP_PLAN",
        "created_utc": now(),
        "out_root": rel(args.out_root),
        "run_dir": rel(RUN_DIR),
        "eval_script": rel(EVAL_SCRIPT),
        "endpoints": endpoints,
        "gpus": args.gpus,
        "workers": args.workers,
        "columns": EVAL_COLUMNS,
        "thresholds": {
            "frontier_overall": FRONTIER,
            "assumed_superglue_for_screen": ASSUMED_SUPERGLUE_80M,
            "cheap7_threshold_with_80M_superglue_aoa0": CHEAP7_THRESHOLD_AT_80M_SUPERGLUE,
        },
    }
    write_json(args.out_root / "summary" / "sweep_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.workers < 1:
        raise ValueError("workers must be >=1")
    if not args.gpus:
        raise ValueError("at least one gpu is required")
    max_workers = min(args.workers, len(args.gpus), len(endpoints))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, endpoint in enumerate(endpoints):
            gpu = args.gpus[i % len(args.gpus)]
            futs.append(ex.submit(run_endpoint, args.out_root, endpoint, gpu, args.force, args.timeout_sec_per_endpoint))
        for fut in as_completed(futs):
            try:
                records.append(fut.result())
            except Exception as exc:
                failures.append({"error": repr(exc), "utc": now()})
                print(json.dumps({"event": "endpoint_eval_failure", "error": repr(exc), "utc": now()}, ensure_ascii=False), flush=True)
    if failures:
        write_json(args.out_root / "summary" / "sweep_failures.json", {"failures": failures, "partial_records": records})
        raise RuntimeError({"failures": failures, "partial_records": [r.get("endpoint") for r in records]})
    summary = write_summary(args.out_root, records, include_known_80m=True)
    print(json.dumps({"status": summary["status"], "decision_signal": summary["decision_signal"], "best_endpoint": summary["best_by_cheap7"]["endpoint"], "best_cheap7": summary["best_by_cheap7"]["cheap7"], "summary_json": rel(args.out_root / "summary" / "scale1p75_checkpoint_sweep_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
