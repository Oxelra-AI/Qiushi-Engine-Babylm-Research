#!/usr/bin/env python3
"""research: CPU-safe stable-family scoring for sub-dose checkpoints.

Uses the research CPU-safe official scorer, which hides CUDA before importing torch
or official BabyLM scoring modules in each child process.  This wrapper adds
live-checkpoint support through shadow run directories: if a training run has not
yet written final metrics, but a checkpoint directory has config/model/tokenizer
files, the scorer sees a local shadow `scientific_metrics.json` while the real
training directory is not modified.

Scientific role: score the already materialized half_1x and quarter_1x sub-dose
checkpoints while both H100s continue training, so the fixed-budget onset curve
can be read from stable families without blocking on GPU availability.

No training, GlobalPIQA, SuperGLUE, AoA, packaging, upload, or leaderboard code.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
RUNS = WS / "training/runs"
PATH = WS / "scripts/cpu_safe_scoring_worker.py"
OUT_ROOT = WS / "data/subdose_commonwindow_eval/eval"
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
ARMS: dict[str, dict[str, Any]] = {
    "quarter_1x": {
        "run_dir": RUNS / "subdose_quarter_1x_deberta100M_seed43022",
        "target_prefix": "subdose_quarter_1x_seed43022",
        "description": "research quarter_1x MAX-geometry view arm scored CPU-safely from materialized checkpoints; rho≈0.0106.",
        "family": "deberta_subdose_quarter_1x_seed43022_cpu_safe",
    },
    "half_1x": {
        "run_dir": RUNS / "subdose_half_1x_deberta100M_seed43022",
        "target_prefix": "subdose_half_1x_seed43022",
        "description": "research half_1x MAX-geometry view arm scored CPU-safely; rho≈0.0212.",
        "family": "deberta_subdose_half_1x_seed43022_cpu_safe",
    },
    "full_1x": {
        "run_dir": RUNS / "subdose_full_1x_deberta100M_seed43022",
        "target_prefix": "subdose_full_1x_seed43022",
        "description": "research full_1x MAX-geometry view arm scored CPU-safely from materialized checkpoints; rho≈0.0424.",
        "family": "deberta_subdose_full_1x_seed43022_cpu_safe",
    },
}
DEFAULT_CHECKPOINTS = [f"chck_{i}M" for i in range(10, 81, 10)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_step267():
    spec = importlib.util.spec_from_file_location("cpu_safe_scoring_worker", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def checkpoint_complete(model_path: pathlib.Path) -> bool:
    return (
        model_path.exists()
        and (model_path / "config.json").exists()
        and (model_path / "model.safetensors").exists()
        and (model_path / "tokenizer.json").exists()
    )


def run_ready(run_dir: pathlib.Path) -> dict[str, Any]:
    metrics = run_dir / "scientific_metrics.json"
    hf = run_dir / "hf_model"
    ck_names = sorted([p.name for p in hf.iterdir() if p.is_dir() and checkpoint_complete(p)]) if hf.exists() else []
    out: dict[str, Any] = {
        "run_dir": rel(run_dir),
        "metrics_exists": metrics.exists(),
        "checkpoint_names": ck_names,
        "checkpoint_count": len(ck_names),
        "stderr_size": (run_dir / "train_stderr.log").stat().st_size if (run_dir / "train_stderr.log").exists() else None,
    }
    if metrics.exists():
        try:
            m = read_json(metrics)
            out.update({
                "word_exposure": m.get("word_exposure"),
                "actual_training_steps": m.get("actual_training_steps"),
                "loss_last": m.get("loss_last"),
                "parameter_count": m.get("parameter_count"),
                "saved_checkpoint_count": len(m.get("saved_checkpoints", [])),
            })
        except Exception as exc:
            out["metrics_error"] = repr(exc)
    return out


def target_complete(research: Any, target: str, columns: list[str]) -> dict[str, Any]:
    path = OUT_ROOT / "per_target" / f"{target}.json"
    if not path.exists():
        return {"exists": False, "ready_cols": [], "complete": False}
    try:
        payload = read_json(path)
        ready = research.ready_cols(payload)
        return {"exists": True, "ready_cols": ready, "complete": all(c in ready for c in columns), "path": rel(path)}
    except Exception as exc:
        return {"exists": True, "ready_cols": [], "complete": False, "path": rel(path), "error": repr(exc)}


def make_shadow_run(arm: str, real_run: pathlib.Path) -> pathlib.Path:
    """Return a run dir with metrics for a live run; does not edit real_run."""
    metrics = real_run / "scientific_metrics.json"
    if metrics.exists():
        return real_run
    hf = real_run / "hf_model"
    ck_names = sorted([p.name for p in hf.iterdir() if p.is_dir() and checkpoint_complete(p)]) if hf.exists() else []
    shadow = OUT_ROOT.parent / "shadow_runs" / arm
    shadow.mkdir(parents=True, exist_ok=True)
    link = shadow / "hf_model"
    if not link.exists():
        try:
            link.symlink_to(hf, target_is_directory=True)
        except FileExistsError:
            pass
    stub = {
        "variant": "cpu_safe_live_checkpoint_shadow_stub",
        "parameter_count": 34467424,
        "vocab_size": 16384,
        "tokenizer_label": "compliant16k_reinvest10M",
        "seed": 43,
        "extra_init_seed": 43022,
        "train_rng_seed": 43023,
        "saved_checkpoints": [{"name": ck} for ck in ck_names],
        "source_run_dir": rel(real_run),
        "created_utc": now(),
        "note": "Shadow metadata for CPU-safe scoring of already materialized checkpoints; the real training run directory is not modified.",
    }
    write_json(shadow / "scientific_metrics.json", stub)
    return shadow


def build_jobs(research: Any, arms: list[str], checkpoints: list[str], columns: list[str], force: bool, max_targets: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}; choices={sorted(ARMS)}")
        cfg = ARMS[arm]
        real_run = pathlib.Path(cfg["run_dir"])
        status = run_ready(real_run)
        score_run = make_shadow_run(arm, real_run)
        for ck in checkpoints:
            target = f"{cfg['target_prefix']}_{ck}"
            mp = score_run / "hf_model" / ck
            ck_ok = checkpoint_complete(mp)
            completion = target_complete(research, target, columns)
            to_run = list(columns) if force else [c for c in columns if c not in completion.get("ready_cols", [])]
            row = {
                "arm": arm,
                "checkpoint": ck,
                "target": target,
                "real_run_dir": rel(real_run),
                "score_run_dir": rel(score_run),
                "model_path": rel(mp),
                "checkpoint_files_complete": ck_ok,
                "run_status": status,
                "existing_target": completion,
                "columns_requested": columns,
                "columns_to_run": to_run if ck_ok else [],
                "will_run": bool(ck_ok and to_run),
            }
            plan_rows.append(row)
            if row["will_run"]:
                jobs.append({
                    "arm": arm,
                    "checkpoint": ck,
                    "target": target,
                    "run_dir": score_run,
                    "endpoint": ck,
                    "columns": to_run,
                    "description": cfg["description"],
                    "family": cfg["family"],
                })
    if max_targets and max_targets > 0:
        allowed = {(j["arm"], j["checkpoint"]) for j in jobs[:max_targets]}
        jobs = [j for j in jobs if (j["arm"], j["checkpoint"]) in allowed]
        for r in plan_rows:
            if r["will_run"] and (r["arm"], r["checkpoint"]) not in allowed:
                r["will_run"] = False
                r["columns_to_run"] = []
                r["not_run_reason"] = "outside max_targets limit"
    return plan_rows, jobs


def worker(job: dict[str, Any], timeout_sec: int, force: bool) -> dict[str, Any]:
    # Keep official scoring children CPU-only and prevent BLAS oversubscription.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("QIUSHI_CPU_CHILD_THREADS", "2"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("QIUSHI_CPU_CHILD_THREADS", "2"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("QIUSHI_CPU_CHILD_THREADS", "2"))
    research = load_step267()
    started = now()
    try:
        res = research.run_official_target(
            job["target"],
            pathlib.Path(job["run_dir"]),
            job["endpoint"],
            OUT_ROOT,
            list(job["columns"]),
            force,
            job["description"],
            job["family"],
            timeout_sec,
            False,
        )
        return {"status": "target_done", "started_utc": started, "finished_utc": now(), "job": {k: rel(v) if isinstance(v, pathlib.Path) else v for k, v in job.items()}, "result": res}
    except Exception as exc:
        return {"status": "target_failed", "started_utc": started, "finished_utc": now(), "job": {k: rel(v) if isinstance(v, pathlib.Path) else v for k, v in job.items()}, "error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["half_1x", "quarter_1x"])
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--columns", nargs="*", default=STABLE_COLUMNS)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout-sec", type=int, default=7200)
    ap.add_argument("--child-threads", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-targets", type=int, default=0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["QIUSHI_CPU_CHILD_THREADS"] = str(args.child_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.child_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.child_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.child_threads)

    research = load_step267()
    plan_rows, jobs = build_jobs(research, args.arms, args.checkpoints, args.columns, args.force, args.max_targets)
    chunk_count = sum(len(r.get("columns_to_run", [])) for r in plan_rows if r.get("will_run"))
    plan = {
        "status": "CPU_SUBDOSE_COMMONWINDOW_PLAN" if args.plan_only else "CPU_SUBDOSE_COMMONWINDOW_START",
        "created_utc": now(),
        "out_root": rel(OUT_ROOT),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "columns": args.columns,
        "target_jobs_to_run": len(jobs),
        "official_family_chunks_to_run": chunk_count,
        "jobs": args.jobs,
        "child_threads": args.child_threads,
        "target_rows": plan_rows,
        "scientific_purpose": "Turn the completed half_1x and materialized quarter_1x checkpoints into stable-family evidence for the fixed-budget substitution onset curve while both H100s train.",
        "cpu_safety": "CUDA_VISIBLE_DEVICES=-1 is set in this wrapper and asserted again in each research scoring child before torch/evaluation imports.",
        "no_training_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(OUT_ROOT.parent / "cpu_subdose_commonwindow_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(worker, j, args.timeout_sec, args.force) for j in jobs]
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            if res.get("status") != "target_done":
                failures.append(res)
            print(json.dumps({k: (v if k != "result" else {"status": v.get("status"), "target": v.get("target"), "after_ready_cols": v.get("after_ready_cols"), "stable_family_scores": v.get("stable_family_scores"), "per_target": v.get("per_target")}) for k, v in res.items()}, indent=2, ensure_ascii=False), flush=True)

    final = {
        **{k: v for k, v in plan.items() if k != "target_rows"},
        "status": "CPU_SUBDOSE_COMMONWINDOW_DONE" if not failures else "CPU_SUBDOSE_COMMONWINDOW_FINISHED_WITH_FAILURES",
        "finished_utc": now(),
        "target_result_count": len(results),
        "failed_target_count": len(failures),
        "results": results,
        "failed_results": failures,
    }
    result_path = OUT_ROOT.parent / "cpu_subdose_commonwindow_result.json"
    write_json(result_path, final)
    print(json.dumps({"status": final["status"], "target_result_count": len(results), "failed_target_count": len(failures), "result_path": rel(result_path), "out_root": rel(OUT_ROOT)}, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
