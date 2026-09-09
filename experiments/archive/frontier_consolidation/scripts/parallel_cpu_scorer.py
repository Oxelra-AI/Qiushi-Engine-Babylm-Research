#!/usr/bin/env python3
"""research: massively parallel CPU scoring for all trained arms.

The scoring bottleneck is binding: 6 empty stubs in 4170s.  With 208 cores and
448GB RAM we can run 16+ concurrent evaluation workers, each handling one
(arm, checkpoint, family) independently.

Safety: every child process runs with CUDA_VISIBLE_DEVICES=-1 and asserts
torch.cuda unavailable before any official module import.

Targets (in priority order):
  1. adultprose MAX register  (8 checkpoints × 6 families = 48 evals)
  2. quarter_1x subdose       (8 checkpoints × 6 families = 48 evals)
  3. half_1x subdose           (8 checkpoints × 6 families = 48 evals, skip existing)
  4. full_1x subdose           (available checkpoints × 6 families)

Output goes to existing readout-compatible directories.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY
STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
PRISTINE_FULL = STRICT / "evaluation_data/full_eval"
NLP_DATA_ROOT = ROOT / "experiments/archive/initial_model_studies/data/nltk_data"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
ZERO_SHOT_SPECS: dict[str, dict[str, Any]] = {
    "BLiMP":      {"task": "blimp",           "data_path": PRISTINE_FULL / "blimp_filtered",      "batch_size": 128},
    "Supplement":  {"task": "blimp",           "data_path": PRISTINE_FULL / "supplement_filtered",  "batch_size": 128},
    "EWoK":       {"task": "ewok",            "data_path": PRISTINE_FULL / "ewok_filtered",       "batch_size": 64},
    "Entity":     {"task": "entity_tracking",  "data_path": PRISTINE_FULL / "entity_tracking",     "batch_size": 128},
    "COMPS":      {"task": "comps",           "data_path": PRISTINE_FULL / "comps",               "batch_size": 128},
}
READING_DATA = PRISTINE_FULL / "reading/reading_data.csv"

STABLE_CKS = [f"chck_{i}M" for i in range(10, 81, 10)]  # 10M..80M

CPU_CHILD_CODE = r'''
import json, os, sys, time
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible != "-1":
    raise SystemExit(f"CPU safety failure: CUDA_VISIBLE_DEVICES={visible!r}")
import torch
rec = {
    "event": "cpu_child_assert",
    "cuda_visible_devices": visible,
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "torch_cuda_device_count": int(torch.cuda.device_count()),
}
print(json.dumps(rec), flush=True)
if torch.cuda.is_available() or torch.cuda.device_count() != 0:
    raise SystemExit("CPU safety failure: torch can see CUDA")
import runpy
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__")
'''


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def cpu_env(cache_root: pathlib.Path, worker_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = cache_root / f"hf_cache_{worker_id}"
    tmp = cache_root / f"tmp_{worker_id}"
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                 "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def parse_sentence_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5.0 <= val <= 105.0 else None
    return None


def parse_reading_scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"),
                       ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def latest_file(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime) if root.exists() else []
    return hits[-1] if hits else None


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def eval_one_task(
    model_path: pathlib.Path,
    target_label: str,
    family_col: str,
    out_root: pathlib.Path,
    worker_id: str,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Run one (model, family) evaluation in a CPU-safe subprocess."""
    task_out = out_root / "official_outputs" / target_label / family_col
    log_path = out_root / "logs" / target_label / f"zero_shot_{family_col}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rec: dict[str, Any] = {
        "column": family_col,
        "target": target_label,
        "worker_id": worker_id,
        "started_utc": now(),
    }

    env = cpu_env(out_root, worker_id)

    if family_col == "Reading":
        argv = [
            "--model_path_or_name", str(model_path.resolve()),
            "--backend", "mlm",
            "--data_path", str(READING_DATA.resolve()),
            "--revision_name", f"step290_{target_label}_Reading",
            "--output_dir", str(task_out.resolve()),
        ]
        module = "evaluation_pipeline.reading.run"
    else:
        spec = ZERO_SHOT_SPECS[family_col]
        argv = [
            "--model_path_or_name", str(model_path.resolve()),
            "--backend", "mlm",
            "--task", spec["task"],
            "--data_path", str(pathlib.Path(spec["data_path"]).resolve()),
            "--revision_name", f"step290_{target_label}_{family_col}",
            "--save_predictions",
            "--batch_size", str(spec["batch_size"]),
            "--non_causal_batch_size", "64",
            "--output_dir", str(task_out.resolve()),
        ]
        module = "evaluation_pipeline.sentence_zero_shot.run"

    cmd = [sys.executable, "-B", "-c", CPU_CHILD_CODE, module] + argv
    t0 = time.time()
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] worker={worker_id} {family_col}\n")
            proc = subprocess.run(
                cmd, cwd=str(STRICT.resolve()), env=env,
                stdout=f, stderr=subprocess.STDOUT, text=True,
                timeout=timeout
            )
        rec["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        rec["returncode"] = -999
        rec["error"] = f"timeout after {timeout}s"
    rec["elapsed_sec"] = round(time.time() - t0, 1)

    # Extract score
    if family_col == "Reading":
        rpt = latest_file(task_out, "report.txt")
        if rpt:
            scores = parse_reading_scores(rpt.read_text(encoding="utf-8", errors="replace"))
            rec["scores"] = scores
            rec["score"] = scores.get("Reading")
    else:
        rpt = latest_file(task_out, "best_temperature_report.txt")
        if rpt:
            rec["score"] = parse_sentence_score(rpt.read_text(encoding="utf-8", errors="replace"))

    # CPU assert check
    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    rec["cpu_assert_seen"] = ("cpu_child_assert" in log_text
                              and '"torch_cuda_available": false' in log_text)

    rec["finished_utc"] = now()
    rec["ok"] = (rec.get("returncode") == 0
                 and rec.get("cpu_assert_seen", False)
                 and finite(rec.get("score")))
    return rec


def build_job_list(arms: list[dict[str, Any]], out_root: pathlib.Path,
                   force: bool) -> list[dict[str, Any]]:
    """Build list of (arm_info, checkpoint, family) work items."""
    jobs = []
    for arm in arms:
        run_dir = pathlib.Path(arm["run_dir"])
        model_root = run_dir / "hf_model"
        label_prefix = arm["label_prefix"]

        for ck in STABLE_CKS:
            model_path = model_root / ck
            if not model_path.exists():
                continue
            target_label = f"{label_prefix}_{ck}"

            # Check existing scores
            existing_file = out_root / "per_target" / f"{target_label}.json"
            existing_cols: set[str] = set()
            if existing_file.exists() and not force:
                try:
                    payload = json.loads(existing_file.read_text())
                    for col in STABLE_COLUMNS:
                        task_rec = (payload.get("tasks") or {}).get(col, {})
                        if isinstance(task_rec, dict) and task_rec.get("returncode") == 0 and finite(task_rec.get("score") if col != "Reading" else (task_rec.get("scores", {}) or {}).get("Reading")):
                            existing_cols.add(col)
                except Exception:
                    pass

            for col in STABLE_COLUMNS:
                if col not in existing_cols:
                    jobs.append({
                        "arm": arm,
                        "checkpoint": ck,
                        "family": col,
                        "target_label": target_label,
                        "model_path": model_path,
                    })
    return jobs


def run_worker(job: dict[str, Any], out_root: pathlib.Path) -> dict[str, Any]:
    """Worker function for parallel execution."""
    target = job["target_label"]
    family = job["family"]
    worker_id = f"w_{target}_{family}"
    result = eval_one_task(
        model_path=job["model_path"],
        target_label=target,
        family_col=family,
        out_root=out_root,
        worker_id=worker_id,
    )
    return result


def merge_results(results: list[dict[str, Any]], out_root: pathlib.Path,
                  arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge individual task results into per-target payload files."""
    # Group by target_label
    by_target: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        tgt = r.get("target", "unknown")
        by_target.setdefault(tgt, []).append(r)

    scored_targets = 0
    total_ok = 0
    total_fail = 0

    for arm in arms:
        run_dir = pathlib.Path(arm["run_dir"])
        model_root = run_dir / "hf_model"
        label_prefix = arm["label_prefix"]

        for ck in STABLE_CKS:
            model_path = model_root / ck
            if not model_path.exists():
                continue
            target_label = f"{label_prefix}_{ck}"
            per_file = out_root / "per_target" / f"{target_label}.json"

            # Load or create payload
            if per_file.exists():
                payload = json.loads(per_file.read_text())
            else:
                payload = {
                    "target": target_label,
                    "description": arm.get("description", ""),
                    "family": arm.get("family_label", ""),
                    "run_dir": rel(run_dir),
                    "model_root": rel(model_root),
                    "model_path": rel(model_path),
                    "endpoint": ck,
                    "started_utc": now(),
                    "gpu": "cpu_safe_hidden",
                    "tasks": {},
                    "cpu_safety": "CUDA_VISIBLE_DEVICES=-1 asserted in child",
                }
                # Add run summary from metrics
                metrics_file = run_dir / "scientific_metrics.json"
                if metrics_file.exists():
                    try:
                        m = json.loads(metrics_file.read_text())
                        payload["run_summary"] = {
                            "variant": m.get("variant"),
                            "word_exposure": m.get("word_exposure"),
                            "actual_training_steps": m.get("actual_training_steps"),
                            "parameter_count": m.get("parameter_count"),
                            "vocab_size": m.get("vocab_size"),
                            "loss_last": m.get("loss_last"),
                            "seed": m.get("seed"),
                            "seq_length": m.get("seq_length"),
                            "n_layer": m.get("n_layer"),
                            "hidden_size": m.get("hidden_size"),
                            "n_head": m.get("n_head"),
                        }
                    except Exception:
                        pass

            # Merge new results
            for r in by_target.get(target_label, []):
                col = r.get("column")
                if col and r.get("ok"):
                    payload.setdefault("tasks", {})[col] = r
                    total_ok += 1
                elif col:
                    payload.setdefault("tasks", {})[col] = r
                    total_fail += 1

            # Compute aggregates
            tasks = payload.get("tasks", {})
            scores: dict[str, float | None] = {}
            for col in STABLE_COLUMNS:
                tr = tasks.get(col, {})
                if col == "Reading":
                    s = (tr.get("scores") or {}).get("Reading") if isinstance(tr, dict) else None
                else:
                    s = tr.get("score") if isinstance(tr, dict) else None
                scores[col] = float(s) if finite(s) else None

            vals6 = [scores[c] for c in STABLE_COLUMNS]
            vals5 = [scores[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]
            scores["cheap6_no_GlobalPIQA"] = statistics.mean([float(x) for x in vals6]) if all(finite(x) for x in vals6) else None
            scores["cheap5_no_GlobalPIQA_Reading"] = statistics.mean([float(x) for x in vals5]) if all(finite(x) for x in vals5) else None
            if finite(scores.get("EWoK")) and finite(scores.get("Entity")):
                scores["EWoK_plus_Entity_sum"] = float(scores["EWoK"]) + float(scores["Entity"])

            payload["stable_family_scores"] = scores
            payload["updated_utc"] = now()
            payload["no_globalpiqa_superglue_aoa_upload_or_leaderboard"] = True

            per_file.parent.mkdir(parents=True, exist_ok=True)
            per_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

            if all(finite(scores.get(c)) for c in STABLE_COLUMNS):
                scored_targets += 1

    return {
        "scored_targets": scored_targets,
        "total_ok": total_ok,
        "total_fail": total_fail,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--arms", nargs="+", default=["adultprose", "quarter", "half", "full"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Define all arms
    ARM_DEFS: dict[str, dict[str, Any]] = {
        "adultprose": {
            "run_dir": str(WS / "training/runs/regmax_adultprose_samefw_deberta100M_seed43022"),
            "label_prefix": "regmax_adultprose_seed43022",
            "description": "MAX register adultprose arm, scored CPU-safely",
            "family_label": "deberta_regmax_adultprose_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/register"),
        },
        "childspeech": {
            "run_dir": str(WS / "training/runs/regmax_childspeech_samefw_deberta100M_seed43022"),
            "label_prefix": "regmax_childspeech_seed43022",
            "description": "MAX register childspeech arm, scored CPU-safely",
            "family_label": "deberta_regmax_childspeech_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/register"),
        },
        "quarter": {
            "run_dir": str(WS / "training/runs/subdose_quarter_1x_deberta100M_seed43022"),
            "label_prefix": "subdose_quarter_1x_seed43022",
            "description": "research quarter_1x MAX-geometry view arm scored CPU-safely; rho≈0.0106.",
            "family_label": "deberta_subdose_quarter_1x_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/subdose"),
        },
        "half": {
            "run_dir": str(WS / "training/runs/subdose_half_1x_deberta100M_seed43022"),
            "label_prefix": "subdose_half_1x_seed43022",
            "description": "research half_1x MAX-geometry view arm scored CPU-safely; rho≈0.0212.",
            "family_label": "deberta_subdose_half_1x_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/subdose"),
        },
        "full": {
            "run_dir": str(WS / "training/runs/subdose_full_1x_deberta100M_seed43022"),
            "label_prefix": "subdose_full_1x_seed43022",
            "description": "research full_1x MAX-geometry view arm scored CPU-safely; rho≈0.0424.",
            "family_label": "deberta_subdose_full_1x_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/subdose"),
        },
        "incorpus": {
            "run_dir": str(WS / "training/runs/incorpus_adultprose_deberta100M_seed43022"),
            "label_prefix": "incorpus_adultprose_seed43022",
            "description": "In-corpus adult prose arm scored CPU-safely; rho≈0.0443.",
            "family_label": "deberta_incorpus_adultprose_seed43022_cpu_safe",
            "out_root": str(WS / "data/parallel_eval/incorpus"),
        },
    }

    # Group arms by output root to avoid cross-writing
    arms_by_out: dict[str, list[dict[str, Any]]] = {}
    for name in args.arms:
        if name not in ARM_DEFS:
            print(f"WARNING: unknown arm {name}, skipping")
            continue
        adef = ARM_DEFS[name]
        out_root = adef["out_root"]
        arms_by_out.setdefault(out_root, []).append(adef)

    grand_total = 0
    grand_ok = 0
    grand_fail = 0

    for out_root_str, arms in arms_by_out.items():
        out_root = pathlib.Path(out_root_str)
        out_root.mkdir(parents=True, exist_ok=True)

        jobs = build_job_list(arms, out_root, args.force)
        print(json.dumps({
            "event": "job_plan",
            "out_root": out_root_str,
            "arm_labels": [a["label_prefix"] for a in arms],
            "total_jobs": len(jobs),
            "jobs_by_family": {col: sum(1 for j in jobs if j["family"] == col) for col in STABLE_COLUMNS},
            "jobs_by_arm": {a["label_prefix"]: sum(1 for j in jobs if j["arm"] is a) for a in arms},
        }, indent=2), flush=True)

        if args.mode == "plan":
            grand_total += len(jobs)
            continue

        if not jobs:
            print(f"  No jobs needed for {out_root_str}")
            continue

        # Run in parallel
        results: list[dict[str, Any]] = []
        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_worker, job, out_root): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result(timeout=2400)
                    results.append(result)
                    status = "OK" if result.get("ok") else "FAIL"
                    score = result.get("score", "?")
                    elapsed = result.get("elapsed_sec", "?")
                    print(f"  [{status}] {result.get('target')} {result.get('column')} = {score} ({elapsed}s)",
                          flush=True)
                except Exception as exc:
                    results.append({
                        "target": job["target_label"],
                        "column": job["family"],
                        "ok": False,
                        "error": repr(exc),
                    })
                    print(f"  [ERROR] {job['target_label']} {job['family']}: {exc}", flush=True)

        merge_info = merge_results(results, out_root, arms)
        elapsed = round(time.time() - t0, 1)
        grand_total += len(jobs)
        grand_ok += merge_info["total_ok"]
        grand_fail += merge_info["total_fail"]

        print(json.dumps({
            "event": "batch_done",
            "out_root": out_root_str,
            "elapsed_sec": elapsed,
            **merge_info,
        }, indent=2), flush=True)

    summary = {
        "status": "PARALLEL_CPU_SCORER_COMPLETE" if args.mode == "run" else "PARALLEL_CPU_SCORER_PLAN",
        "mode": args.mode,
        "arms_requested": args.arms,
        "workers": args.workers,
        "grand_total_jobs": grand_total,
        "grand_ok": grand_ok,
        "grand_fail": grand_fail,
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
