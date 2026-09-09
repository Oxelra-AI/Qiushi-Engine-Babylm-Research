#!/usr/bin/env python3
"""research: full-score verification for scale1.75 chck_82M candidate.

The preceding bounded sweep found chck_82M cheap7=43.96, high enough to
justify complete-score verification.  This script evaluates the missing
SuperGLUE column by running the seven official-compatible finetune subtasks with
the same research one-task runner used for the verified 80M and 100M scale1.75
collations.  It then collates:
  * zero-shot/Reading from the research sweep payload for chck_82M;
  * SuperGLUE primary metrics from the newly run subtasks;
  * AoA=0 from the existing full-ladder AoA repair on the same
    hf_model root (AoA is a run-ladder measurement, independent of which saved
    endpoint is chosen for final predictions).

This is endpoint measurement only.  It does not train, alter the checkpoint, or
claim a new learning principle.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from statistics import mean
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402
from babylm_official_scoring import patch_payload_official_overall, normalize_aoa_record  # noqa: E402

RUN_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
ENDPOINT = "chck_82M"
TARGET = "scale1p75_chck_82M"
MODEL_PATH = RUN_DIR / "hf_model" / ENDPOINT
SWEEP_PAYLOAD = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/eval/per_target/scale1p75_chck_82M.json"
SWEEP_SUMMARY = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json"
AOA_REPAIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_aoa_repair/aoa_local_ckpts_minctx0.json"
SINGLE_TASK_RUNNER = USER_ROOT / "experiments/archive/representation_and_objectives/scripts/parallel_superglue_task.py"
DEFAULT_OUT_ROOT = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_verification"
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
ORDER = [s["task"] for s in base.SUPERGLUE_TASKS]
SPEC = {s["task"]: s for s in base.SUPERGLUE_TASKS}
OVERALL_FRONTIER = 41.80
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_file(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_results_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, sep, v = line.partition(":")
        if not sep:
            continue
        try:
            out[k.strip()] = float(v.strip()) * 100.0
        except Exception:
            continue
    return out


def result_root(out_root: Path, task: str) -> Path:
    return out_root / "superglue_results" / TARGET / task


def save_root(out_root: Path, task: str) -> Path:
    return out_root / "superglue_models" / TARGET / task


def task_complete(out_root: Path, task: str) -> bool:
    pred = latest_file(result_root(out_root, task), "predictions.json")
    res = latest_file(result_root(out_root, task), "results.txt")
    if pred is None or res is None:
        return False
    metrics = parse_results_txt(res)
    return PRIMARY[task] in metrics


def run_superglue_task(out_root: Path, task: str, gpu: int, timeout: int, force: bool) -> dict[str, Any]:
    rr = result_root(out_root, task)
    sr = save_root(out_root, task)
    log = out_root / "logs" / TARGET / f"superglue_{task}.log"
    cache = out_root / "runtime_cache" / task
    rr.mkdir(parents=True, exist_ok=True)
    sr.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if task_complete(out_root, task) and not force:
        rec = score_task(out_root, task)
        rec.update({"status": "skip_existing", "gpu": gpu})
        return rec
    cmd = [
        sys.executable,
        "-B",
        str(SINGLE_TASK_RUNNER),
        "--task",
        task,
        "--model-path",
        str(MODEL_PATH),
        "--results-dir",
        str(rr),
        "--save-dir",
        str(sr),
        "--log",
        str(log),
        "--cache-root",
        str(cache),
        "--gpu",
        str(gpu),
        "--timeout",
        str(timeout),
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(json.dumps({"event": "superglue_task_launch", "task": task, "gpu": gpu, "utc": now()}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=timeout + 120)
    elapsed = time.time() - t0
    driver_stdout = out_root / "driver_logs" / f"{task}.stdout.log"
    driver_stderr = out_root / "driver_logs" / f"{task}.stderr.log"
    driver_stdout.parent.mkdir(parents=True, exist_ok=True)
    driver_stdout.write_text(proc.stdout, encoding="utf-8", errors="replace")
    driver_stderr.write_text(proc.stderr, encoding="utf-8", errors="replace")
    rec: dict[str, Any] = {
        "task": task,
        "gpu": gpu,
        "cmd": [str(x) for x in cmd],
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "driver_stdout": rel(driver_stdout),
        "driver_stderr": rel(driver_stderr),
        "stdout_tail": proc.stdout[-2500:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        rec["status"] = "failed"
        write_json(out_root / "task_records" / f"{task}.json", rec)
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    score = score_task(out_root, task)
    score.update(rec)
    score["status"] = "done"
    write_json(out_root / "task_records" / f"{task}.json", score)
    print(json.dumps({"event": "superglue_task_verified", "task": task, "primary_metric": score["primary_metric"], "primary_metric_score": score["primary_metric_score"], "accuracy": score["accuracy"], "gpu": gpu, "elapsed_sec": round(elapsed, 1)}), flush=True)
    return score


def score_task(out_root: Path, task: str) -> dict[str, Any]:
    pred = latest_file(result_root(out_root, task), "predictions.json")
    res = latest_file(result_root(out_root, task), "results.txt")
    if pred is None or res is None:
        raise FileNotFoundError({"task": task, "pred": rel(result_root(out_root, task)), "res": rel(result_root(out_root, task))})
    acc = base.score_superglue_predictions(task, pred)
    metrics = parse_results_txt(res)
    primary = PRIMARY[task]
    if primary not in metrics:
        raise RuntimeError({"task": task, "missing_primary_metric": primary, "results_txt": rel(res), "metrics": metrics})
    spec = SPEC[task]
    return {
        "task": task,
        "num_labels": spec["num_labels"],
        "batch_size": spec["batch_size"],
        "epochs": spec["epochs"],
        "metric_for_valid": spec["metric_for_valid"],
        "results_dir": rel(result_root(out_root, task)),
        "save_dir": rel(save_root(out_root, task)),
        "predictions": rel(pred),
        "results_txt": rel(res),
        "num_examples": acc["num_examples"],
        "correct": acc["correct"],
        "accuracy": acc["accuracy"],
        "pred_counts": acc["pred_counts"],
        "metrics_percent": metrics,
        "primary_metric": primary,
        "primary_metric_score": metrics[primary],
        "returncode": 0,
    }


def run_all_superglue(out_root: Path, gpus: list[int], timeout: int, force: bool) -> list[dict[str, Any]]:
    pending = list(ORDER)
    available_gpus = list(gpus)
    active: dict[Any, tuple[str, int]] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        while pending or active:
            while pending and available_gpus and not failures:
                gpu = available_gpus.pop(0)
                task = pending.pop(0)
                fut = ex.submit(run_superglue_task, out_root, task, gpu, timeout, force)
                active[fut] = (task, gpu)
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for fut in done:
                task, gpu = active.pop(fut)
                available_gpus.append(gpu)
                try:
                    records.append(fut.result())
                except Exception as exc:
                    failures.append({"task": task, "gpu": gpu, "error": repr(exc), "utc": now()})
                    print(json.dumps({"event": "superglue_task_failure", "task": task, "gpu": gpu, "error": repr(exc)}), flush=True)
                    pending.clear()
        # If a failure happened while another GPU was still running, collect the
        # already-started jobs so their logs/records are not lost; no new jobs are launched.
        for fut, (task, gpu) in list(active.items()):
            try:
                records.append(fut.result())
            except Exception as exc:
                failures.append({"task": task, "gpu": gpu, "error": repr(exc), "utc": now()})
    if failures:
        write_json(out_root / "summary" / "superglue_failures.json", {"failures": failures, "records": records})
        raise RuntimeError({"superglue_failures": failures})
    by_task = {r["task"]: r for r in records}
    missing = [t for t in ORDER if t not in by_task]
    if missing:
        raise RuntimeError({"missing_superglue_records": missing, "records": [r["task"] for r in records]})
    return [by_task[t] for t in ORDER]


def build_superglue_record(task_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {r["task"]: r for r in task_records}
    ordered = [by_task[t] for t in ORDER]
    primary_vals = [float(r["primary_metric_score"]) for r in ordered]
    acc_vals = [float(r["accuracy"]) for r in ordered]
    return {
        "column": "SuperGLUE",
        "target": TARGET,
        "endpoint": ENDPOINT,
        "tasks": ordered,
        "started_utc": None,
        "finished_utc": now(),
        "superglue_mean_accuracy_only_legacy": sum(acc_vals) / len(acc_vals),
        "superglue_primary_metric_details": [
            {"task": r["task"], "metric": r["primary_metric"], "score": r["primary_metric_score"], "results_txt": r["results_txt"]}
            for r in ordered
        ],
        "superglue_mean": sum(primary_vals) / len(primary_vals),
        "superglue_coordinate": "current official primary metrics: f1 for MRPC/QQP, accuracy otherwise",
        "returncode": 0,
    }


def aoa_record() -> dict[str, Any]:
    d = read_json(AOA_REPAIR)
    raw = d.get("aoa")
    rec = {
        "column": "AoA",
        "status": "official_aoa_done" if d.get("status") == "AOA_LOCAL_CKPTS_MINCTX_DONE" else d.get("status"),
        "helper_status": d.get("status"),
        "aoa_helper_runner": "experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py via research writable-cache repair",
        "aoa_official": raw,
        "aoa_raw_correlation": raw,
        "aoa_leaderboard_score": 100.0 * float(raw) if raw is not None else None,
        "aoa_for_provisional_overall": 100.0 * float(raw) if raw is not None else None,
        "out_json": rel(AOA_REPAIR),
        "surprisal_path": d.get("surprisal_path"),
        "score_path": d.get("score_path"),
        "score_tokenizer_path": d.get("score_tokenizer_path"),
        "num_rows": d.get("num_rows"),
        "num_steps": d.get("num_steps"),
        "expected_steps": d.get("expected_steps"),
        "step_counts": d.get("step_counts"),
        "row_count_values": d.get("row_count_values"),
        "finite_surprisals": d.get("finite_surprisals"),
        "returncode": 0,
        "interpretation": "Reused exact research AoA repair from the same complete scale1.75 hf_model checkpoint ladder; endpoint selection changes final predictions but not the underlying saved AoA learning curve.",
    }
    return normalize_aoa_record(rec)


def collate(out_root: Path, task_records: list[dict[str, Any]]) -> dict[str, Any]:
    sweep = read_json(SWEEP_PAYLOAD)
    sweep_summary = read_json(SWEEP_SUMMARY)
    tasks = {k: v for k, v in sweep.get("tasks", {}).items() if k in {"BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"}}
    missing = [k for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"] if k not in tasks]
    if missing:
        raise RuntimeError({"missing_sweep_tasks": missing, "sweep_payload": rel(SWEEP_PAYLOAD)})
    tasks["SuperGLUE"] = build_superglue_record(task_records)
    tasks["AoA"] = aoa_record()
    metrics = read_json(RUN_DIR / "scientific_metrics.json")
    payload = {
        "target": TARGET,
        "endpoint": ENDPOINT,
        "created_utc": now(),
        "description": "scale1.75 adapter128 checkpoint at 82M exposure selected from a pre-existing 100M compliant ladder after bounded neighboring-checkpoint sweep; complete score verified with new SuperGLUE and existing full-ladder AoA.",
        "family": "scale1p75_residual_adapter_chck82M_endpoint_establishment",
        "run_dir": rel(RUN_DIR),
        "model_root": rel(RUN_DIR / "hf_model"),
        "model_path": rel(MODEL_PATH),
        "run_summary": {k: metrics.get(k) for k in ["variant", "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "seed", "extra_init_seed", "train_rng_seed", "mask_mode", "seq_length", "batch_size", "optimizer", "learning_rate", "n_layer", "hidden_size", "n_head", "intermediate_size"] if k in metrics},
        "tasks": tasks,
        "source_payloads": {
            "zero_shot_reading_sweep_payload": rel(SWEEP_PAYLOAD),
            "zero_shot_sweep_summary": rel(SWEEP_SUMMARY),
            "aoa_repair_same_model_root": rel(AOA_REPAIR),
            "single_superglue_task_runner": rel(SINGLE_TASK_RUNNER),
        },
        "threshold_context": {
            "frontier_overall": OVERALL_FRONTIER,
            "cheap7_from_sweep": sweep_summary.get("best_by_cheap7", {}).get("cheap7"),
            "superglue_required_for_41p8_from_sweep": sweep_summary.get("best_by_cheap7", {}).get("superglue_required_for_41p8_with_aoa0"),
            "screen_reason": "chck_82M was the only 77M-83M neighboring checkpoint with cheap7 >= 43.848612, so complete-score verification was launched immediately.",
        },
    }
    payload = patch_payload_official_overall(payload)
    scores = payload["official_overall"]["scores"]
    cheap7 = mean(float(scores[c]) for c in CHEAP_COLS)
    payload["cheap7"] = cheap7
    payload["overall_margin_vs_41p8"] = float(payload["official_overall"].get("Overall", float("nan"))) - OVERALL_FRONTIER
    payload["superglue_margin_vs_required"] = float(scores["SuperGLUE"]) - float(sweep_summary.get("best_by_cheap7", {}).get("superglue_required_for_41p8_with_aoa0"))
    payload["score_signal"] = "above_41p8" if payload["overall_margin_vs_41p8"] >= 0 else "below_41p8"
    payload["scientific_interpretation"] = "Endpoint-score candidate from an existing trajectory; even if above frontier, it establishes a checkpoint peak and must be kept separate from the broader generalizable learning-principle route."

    out_json = out_root / "summary" / "scale1p75_chck82_full_verification.json"
    write_json(out_json, payload)
    rows = [
        "# research scale1.75 chck_82M full verification",
        "",
        f"Overall: **{payload['official_overall'].get('Overall'):.6f}** (margin vs 41.80: **{payload['overall_margin_vs_41p8']:+.6f}**)",
        f"cheap7: **{cheap7:.6f}**; SuperGLUE: **{scores['SuperGLUE']:.6f}**; AoA: **{scores['AoA']}**",
        "",
        "| Column | Score |",
        "|---|---:|",
    ]
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]:
        rows.append(f"| {col} | {scores[col]:.6f} |")
    rows += [
        "",
        f"SuperGLUE required from sweep threshold: {sweep_summary.get('best_by_cheap7', {}).get('superglue_required_for_41p8_with_aoa0')}",
        f"SuperGLUE margin over required: {payload['superglue_margin_vs_required']:+.6f}",
        "",
        "This is a complete endpoint measurement of an already-trained checkpoint, not a mechanism claim.",
        f"JSON: `{rel(out_json)}`",
    ]
    out_md = out_root / "summary" / "scale1p75_chck82_full_verification.md"
    out_md.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CHCK82_FULL_VERIFICATION_DONE", "Overall": payload["official_overall"].get("Overall"), "margin_vs_41p8": payload["overall_margin_vs_41p8"], "SuperGLUE": scores["SuperGLUE"], "cheap7": cheap7, "summary_json": rel(out_json), "summary_md": rel(out_md)}, indent=2), flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Complete chck_82M SuperGLUE and score collation")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--timeout-sec-per-task", type=int, default=14400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="Do not run tasks; collate existing SuperGLUE outputs")
    args = ap.parse_args()
    args.out_root = args.out_root if args.out_root.is_absolute() else USER_ROOT / args.out_root
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary").mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    for p in [SWEEP_PAYLOAD, SWEEP_SUMMARY, AOA_REPAIR, SINGLE_TASK_RUNNER]:
        if not p.exists():
            raise FileNotFoundError(p)
    if not args.gpus:
        raise ValueError("at least one GPU id is required")
    plan = {
        "status": "CHCK82_FULL_VERIFICATION_PLAN",
        "created_utc": now(),
        "endpoint": ENDPOINT,
        "target": TARGET,
        "model_path": rel(MODEL_PATH),
        "out_root": rel(args.out_root),
        "gpus": args.gpus,
        "superglue_tasks": ORDER,
        "superglue_primary_metric": PRIMARY,
        "zero_reading_source": rel(SWEEP_PAYLOAD),
        "aoa_source": rel(AOA_REPAIR),
    }
    write_json(args.out_root / "summary" / "chck82_full_verification_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.merge_only:
        task_records = [score_task(args.out_root, t) for t in ORDER]
    else:
        task_records = run_all_superglue(args.out_root, args.gpus, args.timeout_sec_per_task, args.force)
    collate(args.out_root, task_records)


if __name__ == "__main__":
    main()
