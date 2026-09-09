#!/usr/bin/env python3
"""research: race-free full official-compatible evaluation for the scale1.75 100M endpoint.

The research draft evaluated cheap columns and SuperGLUE/AoA in two concurrent
processes that shared one per-target JSON and one output tree.  This script
removes that race: every independent column group writes to its own target and
root, then the predictions are staged into the exact path layout expected by the
research pristine official-coordinate collator.

Default resource plan after the endpoint appears:
  * run SuperGLUE on the first free GPU among [1, 0];
  * run zero-shot/Reading columns as split jobs on GPU0 with configurable slots;
  * run AoA on the first free GPU among [0, 1] after the split zero-shot jobs;
  * run pristine collation on CPU from staged artifacts.

The expensive work this orchestrates is the decisive endpoint measurement: the
80M scale1.75 trajectory already crossed the cheap7 level compatible with 41.8
if SuperGLUE/AoA stay flat, but the official nine-column endpoint is unknown.
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
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
EVAL_SCRIPT = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
COLLATE_SCRIPT = USER_ROOT / "experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py"

DEFAULT_RUN_DIR = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
DEFAULT_BASE_OUT = WORKSPACE / "data/scale1p75_100M_full_eval_hardened"
DEFAULT_TARGET = "scale1p75_100M_seed43022"
DEFAULT_ENDPOINT = "chck_100M"

AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SUPERGLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]

ZERO_REL_BY_COLUMN = {
    "BLiMP": "blimp/blimp_filtered/predictions.json",
    "Supplement": "blimp/supplement_filtered/predictions.json",
    "EWoK": "ewok/ewok_filtered/predictions.json",
    "Entity": "entity_tracking/entity_tracking/predictions.json",
    "COMPS": "comps/comps/predictions.json",
    "GlobalPIQA_parallel": "global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "reading/predictions.json",
}

reference_100M_REF: dict[str, float] = {
    "Overall": 41.257770896404615,
    "BLiMP": 65.8707181799453,
    "Supplement": 61.16566092036889,
    "EWoK": 50.39323748109589,
    "Entity": 27.400833994026197,
    "COMPS": 52.00834536316919,
    "SuperGLUE": 70.27986764740969,
    "GlobalPIQA": 36.0631067961165,
    "Reading": 8.13816768550987,
    "AoA": 0.0,
}
reference_100M_REF["cheap7"] = float(mean(reference_100M_REF[c] for c in CHEAP_COLS))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str], timeout: int, log_prefix: Path) -> dict[str, Any]:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout, encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr, encoding="utf-8")
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-5000:],
        "stdout_log": rel(log_prefix.with_suffix(".stdout.log")),
        "stderr_log": rel(log_prefix.with_suffix(".stderr.log")),
    }


def attach_writable_runtime_cache(env: dict[str, str], cache_root: Path) -> dict[str, str]:
    """Route HuggingFace dynamic modules/caches to a writable local tree.

    research showed direct trust_remote_code loads can fail if Transformers tries to
    create its dynamic-module cache under the runtime's read-only shared model
    cache.  The inherited official-eval runner already builds per-target writable
    caches for its child evaluation commands; this outer attachment protects the
    top-level subprocess boundary, pristine collation call, and any future
    merge/rerun use without changing model weights, data, scoring, or evaluation
    definitions.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "TMPDIR": cache_root / "tmp",
    }
    for key, path in mapping.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def gpu_free_mb(gpu: int) -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def wait_for_gpu(gpus: list[int], min_free_mb: int, timeout_sec: int, label: str) -> int:
    t0 = time.time()
    last: dict[int, int | None] = {}
    while True:
        for gpu in gpus:
            free = gpu_free_mb(gpu)
            last[gpu] = free
            if free is not None and free >= min_free_mb:
                print(json.dumps({"event": "gpu_selected", "label": label, "gpu": gpu, "free_mb": free, "utc": now()}), flush=True)
                return gpu
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"No GPU in {gpus} reached {min_free_mb} MiB free for {label}; last={last}")
        print(json.dumps({"event": "waiting_gpu_free", "label": label, "gpus": gpus, "last_free_mb": last, "utc": now()}), flush=True)
        time.sleep(30)


def verify_endpoint_ready(run_dir: Path, endpoint: str, timeout_sec: int, skip_wait: bool) -> dict[str, Any]:
    t0 = time.time()
    while True:
        metrics = run_dir / "scientific_metrics.json"
        endpoint_model = run_dir / "hf_model" / endpoint / "model.safetensors"
        if metrics.exists() and endpoint_model.exists():
            break
        if skip_wait:
            raise FileNotFoundError({"metrics": rel(metrics), "endpoint_model": rel(endpoint_model)})
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"Endpoint did not become ready within {timeout_sec}s: {run_dir}")
        print(json.dumps({"event": "waiting_endpoint", "metrics_exists": metrics.exists(), "endpoint_exists": endpoint_model.exists(), "utc": now()}), flush=True)
        time.sleep(30)

    data = read_json(run_dir / "scientific_metrics.json")
    saved = [x.get("name") for x in data.get("saved_checkpoints", []) if isinstance(x, dict)]
    present = []
    missing = []
    for step in AOA_STEPS:
        p = run_dir / "hf_model" / step / "model.safetensors"
        (present if p.exists() else missing).append(step)
    first_loss = data.get("loss_first")
    first_loss_ok = isinstance(first_loss, (int, float)) and abs(float(first_loss) - 9.837543487548828) < 1e-6
    out = {
        "run_dir": rel(run_dir),
        "endpoint": endpoint,
        "word_exposure": data.get("word_exposure"),
        "actual_training_steps": data.get("actual_training_steps"),
        "loss_first": first_loss,
        "loss_first_matches_step35": first_loss_ok,
        "loss_last": data.get("loss_last"),
        "saved_checkpoint_count": len(saved),
        "aoa_ladder_present": present,
        "aoa_ladder_missing": missing,
        "aoa_ladder_complete": len(missing) == 0,
        "parameter_count": data.get("parameter_count"),
        "tokenizer_label": data.get("tokenizer_label"),
        "example_jsonl": data.get("example_jsonl"),
    }
    if missing:
        raise RuntimeError({"incomplete_aoa_ladder": out})
    if not first_loss_ok:
        raise RuntimeError({"unexpected_first_loss": out})
    return out


def safe_col(column: str) -> str:
    return column.replace("GlobalPIQA_", "GP_").replace("/", "_")


def part_name(kind: str) -> str:
    return safe_col(kind)


def part_target(target: str, kind: str) -> str:
    return f"{target}__{part_name(kind)}"


def part_root(base_out: Path, kind: str) -> Path:
    return base_out / "parts" / part_name(kind)


def part_payload(base_out: Path, target: str, kind: str) -> Path:
    pt = part_target(target, kind)
    return part_root(base_out, kind) / "eval" / "per_target" / f"{pt}.json"


def part_complete(base_out: Path, target: str, kind: str) -> bool:
    p = part_payload(base_out, target, kind)
    if not p.exists():
        return False
    try:
        d = read_json(p)
        tasks = d.get("tasks", {})
        if kind in ZERO_COLUMNS:
            return isinstance(tasks.get(kind), dict) and tasks[kind].get("returncode") == 0
        if kind == "SuperGLUE":
            r = tasks.get("SuperGLUE")
            return isinstance(r, dict) and r.get("superglue_mean") is not None and len(r.get("tasks", [])) == len(SUPERGLUE_TASKS)
        if kind == "AoA":
            r = tasks.get("AoA")
            return isinstance(r, dict) and r.get("returncode") == 0 and r.get("aoa_leaderboard_score") is not None
    except Exception:
        return False
    return False


def eval_part(
    *,
    base_out: Path,
    run_dir: Path,
    target: str,
    endpoint: str,
    kind: str,
    columns: list[str],
    gpu_candidates: list[int],
    min_free_mb: int,
    gpu_wait_timeout_sec: int,
    force: bool,
    timeout_sec: int,
) -> dict[str, Any]:
    if not force and part_complete(base_out, target, kind):
        return {"kind": kind, "status": "skip_existing", "payload": rel(part_payload(base_out, target, kind))}

    gpu = wait_for_gpu(gpu_candidates, min_free_mb, gpu_wait_timeout_sec, kind)
    out = part_root(base_out, kind)
    eval_root = out / "eval"
    collate_root = out / "collate"
    log_prefix = out / "driver_logs" / f"{part_name(kind)}"
    pt = part_target(target, kind)
    cmd = [
        sys.executable,
        "-B",
        str(EVAL_SCRIPT),
        "--arm",
        "reinvest",
        "--run-dir",
        str(run_dir),
        "--target",
        pt,
        "--endpoint",
        endpoint,
        "--out-root",
        str(eval_root),
        "--collate-root",
        str(collate_root),
        "--gpu",
        str(gpu),
        "--columns",
        *columns,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    attach_writable_runtime_cache(env, out / "runtime_cache")
    print(json.dumps({"event": "eval_part_start", "kind": kind, "columns": columns, "target": pt, "gpu": gpu, "utc": now()}), flush=True)
    rec = run_cmd(cmd, USER_ROOT, env, timeout_sec, log_prefix)
    rec.update({"kind": kind, "columns": columns, "target": pt, "gpu": gpu, "payload": rel(part_payload(base_out, target, kind))})
    print(json.dumps({"event": "eval_part_done", "kind": kind, "returncode": rec["returncode"], "elapsed_sec": rec["elapsed_sec"], "payload": rec["payload"]}, ensure_ascii=False), flush=True)
    if rec["returncode"] != 0:
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    if not part_complete(base_out, target, kind):
        raise RuntimeError({"part_not_complete_after_eval": kind, "payload": rel(part_payload(base_out, target, kind))})
    return rec


def schedule_zero_parts(args: argparse.Namespace, base_out: Path, run_dir: Path) -> list[dict[str, Any]]:
    gpu_slots: list[list[int]] = []
    for gpu in args.zero_gpus:
        for _ in range(args.zero_slots_per_gpu):
            gpu_slots.append([gpu])
    if not gpu_slots:
        raise ValueError("zero_gpus/zero_slots_per_gpu produced no slots")
    jobs = [c for c in ZERO_COLUMNS if args.force or not part_complete(base_out, args.target, c)]
    rows: list[dict[str, Any]] = []
    print(json.dumps({"event": "zero_eval_plan", "jobs": jobs, "gpu_slots": gpu_slots, "utc": now()}), flush=True)
    if not jobs:
        return rows
    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as ex:
        futs = []
        for i, col in enumerate(jobs):
            futs.append(ex.submit(
                eval_part,
                base_out=base_out,
                run_dir=run_dir,
                target=args.target,
                endpoint=args.endpoint,
                kind=col,
                columns=[col],
                gpu_candidates=gpu_slots[i % len(gpu_slots)],
                min_free_mb=args.min_free_mb,
                gpu_wait_timeout_sec=args.gpu_wait_timeout_sec,
                force=args.force,
                timeout_sec=args.zero_timeout_sec,
            ))
        for fut in as_completed(futs):
            rows.append(fut.result())
    return rows


def as_path(pathlike: str | Path) -> Path:
    p = Path(pathlike)
    return p if p.is_absolute() else USER_ROOT / p


def link_or_copy(src: Path, dst: Path, copy: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        mode = "copy"
    else:
        os.symlink(src.resolve(), dst)
        mode = "symlink"
    return {"src": rel(src), "dst": rel(dst), "mode": mode, "size_bytes": src.stat().st_size}


def load_task(base_out: Path, target: str, kind: str, task_key: str) -> dict[str, Any]:
    p = part_payload(base_out, target, kind)
    d = read_json(p)
    task = d.get("tasks", {}).get(task_key)
    if not isinstance(task, dict):
        raise RuntimeError({"missing_task": task_key, "kind": kind, "payload": rel(p)})
    return task


def stage_predictions(base_out: Path, run_dir: Path, target: str, endpoint: str, copy: bool) -> dict[str, Any]:
    staged_root = base_out / "staged_full_eval"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged: list[dict[str, Any]] = []
    merged_tasks: dict[str, Any] = {}
    sources: dict[str, Any] = {}

    for col in ZERO_COLUMNS:
        task = load_task(base_out, target, col, col)
        src = as_path(task.get("predictions", ""))
        dst = staged_root / "official_outputs" / target / col / endpoint / f"full_{target}_{col}" / "zero_shot/mlm" / ZERO_REL_BY_COLUMN[col]
        staged.append(link_or_copy(src, dst, copy))
        merged_tasks[col] = task
        sources[col] = {"part_payload": rel(part_payload(base_out, target, col)), "staged_prediction": rel(dst)}

    sg_task = load_task(base_out, target, "SuperGLUE", "SuperGLUE")
    sg_sources: dict[str, Any] = {}
    for rec in sg_task.get("tasks", []):
        if not isinstance(rec, dict):
            continue
        task_name = rec.get("task")
        if task_name not in SUPERGLUE_TASKS:
            continue
        pred_src = as_path(rec.get("predictions", ""))
        res_src = pred_src.parent / "results.txt"
        pred_dst = staged_root / "superglue_results" / target / str(task_name) / endpoint / "main/finetune" / str(task_name) / "predictions.json"
        res_dst = pred_dst.parent / "results.txt"
        staged.append(link_or_copy(pred_src, pred_dst, copy))
        staged.append(link_or_copy(res_src, res_dst, copy))
        sg_sources[str(task_name)] = {"prediction": rel(pred_dst), "results": rel(res_dst)}
    if set(sg_sources) != set(SUPERGLUE_TASKS):
        raise RuntimeError({"superglue_tasks_missing_after_stage": sorted(set(SUPERGLUE_TASKS) - set(sg_sources))})
    merged_tasks["SuperGLUE"] = sg_task
    sources["SuperGLUE"] = {"part_payload": rel(part_payload(base_out, target, "SuperGLUE")), "tasks": sg_sources}

    aoa_task = load_task(base_out, target, "AoA", "AoA")
    aoa_part_target = part_target(target, "AoA")
    aoa_part_dir = part_root(base_out, "AoA") / "eval" / "aoa_outputs" / aoa_part_target / "hf_model_local_ckpts/main/zero_shot/mlm/AoA_word"
    surprisal_src = as_path(aoa_task.get("surprisal_path", aoa_part_dir / "surprisal.json"))
    score_path_record = aoa_task.get("score_path")
    score_src = as_path(score_path_record) if score_path_record else (aoa_part_dir / "aoa_score.json")
    if not score_src.exists():
        # Some helpers record only the summary JSON; the official layout should still exist.
        alt = part_root(base_out, "AoA") / "eval" / "aoa_outputs" / aoa_part_target / "hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json"
        score_src = alt
    aoa_dst_dir = staged_root / "aoa_outputs" / target / "hf_model_local_ckpts/main/zero_shot/mlm/AoA_word"
    staged.append(link_or_copy(surprisal_src, aoa_dst_dir / "surprisal.json", copy))
    staged.append(link_or_copy(score_src, aoa_dst_dir / "aoa_score.json", copy))
    merged_tasks["AoA"] = aoa_task
    sources["AoA"] = {"part_payload": rel(part_payload(base_out, target, "AoA")), "surprisal": rel(aoa_dst_dir / "surprisal.json"), "score": rel(aoa_dst_dir / "aoa_score.json")}

    merged_payload = {
        "target": target,
        "description": "scale1.75 adapter128 100M endpoint, official-compatible predictions staged from race-free split jobs",
        "family": "scale1p75_residual_adapter_100M_full_eval",
        "run_dir": rel(run_dir),
        "model_root": rel(run_dir / "hf_model"),
        "model_path": rel(run_dir / "hf_model" / endpoint),
        "endpoint": endpoint,
        "tasks": merged_tasks,
        "stage_sources": sources,
        "created_utc": now(),
    }
    write_json(staged_root / "per_target" / f"{target}.json", merged_payload)
    stage_summary = {"staged_root": rel(staged_root), "merged_payload": rel(staged_root / "per_target" / f"{target}.json"), "staged_files": staged, "sources": sources}
    write_json(base_out / "summary" / "stage_summary.json", stage_summary)
    return stage_summary


def run_pristine_collation(base_out: Path, run_dir: Path, target: str, endpoint: str, copy: bool, dry_run: bool) -> dict[str, Any]:
    staged_root = base_out / "staged_full_eval"
    collate_dir = base_out / "collate" / target
    ewok_pred = staged_root / "official_outputs" / target / "EWoK" / endpoint / f"full_{target}_EWoK" / "zero_shot/mlm/ewok/ewok_filtered/predictions.json"
    cmd = [
        sys.executable,
        "-B",
        str(COLLATE_SCRIPT),
        "--full-root",
        str(staged_root),
        "--target",
        target,
        "--model-root",
        str(run_dir / "hf_model"),
        "--out-dir",
        str(collate_dir),
        "--pristine-ewok-predictions",
        str(ewok_pred),
        "--aoa-dir",
        str(staged_root / "aoa_outputs" / target),
        "--endpoint",
        endpoint,
        "--tag",
        target,
    ]
    if copy:
        cmd.append("--copy")
    if dry_run:
        cmd.append("--dry-run")
    env = os.environ.copy()
    attach_writable_runtime_cache(env, base_out / "summary" / "collate_runtime_cache")
    rec = run_cmd(cmd, USER_ROOT, env, 7200, base_out / "summary/collate_driver")
    out_json = collate_dir / f"pristine_collate_{target}_summary.json"
    dry_json = collate_dir / f"pristine_collate_{target}_dryrun.json"
    if dry_run:
        rec["dryrun_json"] = rel(dry_json)
    else:
        rec["summary_json"] = rel(out_json)
        if out_json.exists():
            rec["summary"] = read_json(out_json)
    if rec["returncode"] != 0:
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    return rec


def extract_scores_from_collation(collation: dict[str, Any]) -> tuple[dict[str, float], float]:
    summary = collation.get("summary", {})
    score = summary.get("score_summary", {}) if isinstance(summary, dict) else {}
    scores = score.get("scores") if isinstance(score, dict) else None
    if not isinstance(scores, dict):
        raise RuntimeError({"missing_score_summary": collation.get("summary_json")})
    scores_f = {k: float(v) for k, v in scores.items()}
    overall = float(score.get("Overall"))
    return scores_f, overall


def summarize(base_out: Path, target: str, endpoint_ready: dict[str, Any], eval_rows: list[dict[str, Any]], stage_summary: dict[str, Any], collation: dict[str, Any]) -> dict[str, Any]:
    scores, overall = extract_scores_from_collation(collation)
    cheap7 = float(mean(scores[c] for c in CHEAP_COLS))
    deltas = {k: float(scores[k] - reference_100M_REF[k]) for k in scores if k in reference_100M_REF}
    overall_delta = overall - reference_100M_REF["Overall"]
    margin = overall - 41.8

    # Add the pristine-collated official score vector back to the staged per-target
    # payload.  The item-flip analyzer reconstructs discrete columns from
    # predictions but uses official_overall for cheap7/column deltas; without this
    # update the endpoint artifact would require manual patching before analysis.
    staged_payload_path = base_out / "staged_full_eval" / "per_target" / f"{target}.json"
    if staged_payload_path.exists():
        staged_payload = read_json(staged_payload_path)
        staged_payload["official_overall"] = {
            "scores": scores,
            "complete_for_provisional_overall": True,
            "Overall": overall,
            "NLP_average": float(mean(scores[k] for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"])),
            "Human_like_average": float(mean(scores[k] for k in ["Reading", "AoA"])),
            "official_like_arithmetic": "mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA_leaderboard_score)",
            "aoa_status": staged_payload.get("tasks", {}).get("AoA", {}).get("status"),
            "aoa_raw_correlation": staged_payload.get("tasks", {}).get("AoA", {}).get("aoa_raw_correlation"),
            "aoa_leaderboard_score": scores.get("AoA"),
        }
        staged_payload["finished_utc"] = now()
        write_json(staged_payload_path, staged_payload)

    summary = {
        "status": "SCALE1P75_100M_FULL_EVAL_HARDENED",
        "created_utc": now(),
        "target": target,
        "endpoint": endpoint_ready.get("endpoint"),
        "endpoint_ready": endpoint_ready,
        "scores": scores,
        "cheap7": cheap7,
        "Overall": overall,
        "reference_100M_ref": reference_100M_REF,
        "deltas_vs_step35": deltas,
        "cheap7_delta_vs_step35": cheap7 - reference_100M_REF["cheap7"],
        "overall_delta_vs_step35": overall_delta,
        "overall_margin_vs_41p8": margin,
        "score_signal": "above_41p8" if margin >= 0 else "below_41p8",
        "eval_rows": eval_rows,
        "stage_summary": stage_summary,
        "collation": {k: v for k, v in collation.items() if k != "summary"},
        "collation_summary_json": collation.get("summary_json"),
    }
    out_json = base_out / "summary" / "scale1p75_100M_full_eval_hardened_summary.json"
    out_md = base_out / "summary" / "scale1p75_100M_full_eval_hardened_summary.md"
    write_json(out_json, summary)
    lines = [
        "# research Scale1.75 100M Full Official-Compatible Evaluation",
        "",
        f"Overall: **{overall:.6f}** (margin vs 41.8: **{margin:+.6f}**)",
        f"cheap7: **{cheap7:.6f}** (delta vs research: **{cheap7 - reference_100M_REF['cheap7']:+.6f}**)",
        "",
        "| Column | scale1.75 100M | research 100M | Delta |",
        "|---|---:|---:|---:|",
    ]
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]:
        lines.append(f"| {col} | {scores[col]:.6f} | {reference_100M_REF[col]:.6f} | {deltas[col]:+.6f} |")
    lines.append(f"| **Overall** | **{overall:.6f}** | **{reference_100M_REF['Overall']:.6f}** | **{overall_delta:+.6f}** |")
    lines += [
        "",
        f"Evidence JSON: `{rel(out_json)}`",
        f"Pristine collation summary: `{collation.get('summary_json')}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "Overall": overall, "margin_vs_41p8": margin, "cheap7": cheap7, "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)
    return summary


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    base_out = args.base_out
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "summary").mkdir(parents=True, exist_ok=True)
    endpoint_ready = verify_endpoint_ready(args.run_dir, args.endpoint, args.endpoint_wait_timeout_sec, args.skip_wait)
    write_json(base_out / "summary" / "endpoint_ready.json", endpoint_ready)
    print(json.dumps({"event": "endpoint_ready", "word_exposure": endpoint_ready.get("word_exposure"), "steps": endpoint_ready.get("actual_training_steps"), "loss_last": endpoint_ready.get("loss_last")}), flush=True)

    eval_rows: list[dict[str, Any]] = []
    if not args.merge_only:
        with ThreadPoolExecutor(max_workers=2) as outer:
            sg_future = None
            if not args.no_superglue:
                sg_future = outer.submit(
                    eval_part,
                    base_out=base_out,
                    run_dir=args.run_dir,
                    target=args.target,
                    endpoint=args.endpoint,
                    kind="SuperGLUE",
                    columns=["SuperGLUE"],
                    gpu_candidates=args.superglue_gpus,
                    min_free_mb=args.min_free_mb,
                    gpu_wait_timeout_sec=args.gpu_wait_timeout_sec,
                    force=args.force,
                    timeout_sec=args.superglue_timeout_sec,
                )
            zero_future = outer.submit(schedule_zero_parts, args, base_out, args.run_dir)
            eval_rows.extend(zero_future.result())
            if not args.no_aoa:
                eval_rows.append(eval_part(
                    base_out=base_out,
                    run_dir=args.run_dir,
                    target=args.target,
                    endpoint=args.endpoint,
                    kind="AoA",
                    columns=["AoA"],
                    gpu_candidates=args.aoa_gpus,
                    min_free_mb=args.min_free_mb,
                    gpu_wait_timeout_sec=args.gpu_wait_timeout_sec,
                    force=args.force,
                    timeout_sec=args.aoa_timeout_sec,
                ))
            if sg_future is not None:
                eval_rows.append(sg_future.result())

    missing_parts = [c for c in ZERO_COLUMNS + ["SuperGLUE", "AoA"] if not part_complete(base_out, args.target, c)]
    if missing_parts:
        raise RuntimeError({"missing_completed_parts": missing_parts, "base_out": rel(base_out)})
    stage_summary = stage_predictions(base_out, args.run_dir, args.target, args.endpoint, args.copy_artifacts)
    collation = run_pristine_collation(base_out, args.run_dir, args.target, args.endpoint, args.copy_artifacts, args.collate_dry_run)
    if args.collate_dry_run:
        out = {"status": "SCALE1P75_100M_COLLATE_DRYRUN", "endpoint_ready": endpoint_ready, "eval_rows": eval_rows, "stage_summary": stage_summary, "collation": collation}
        write_json(base_out / "summary" / "collate_dryrun_summary.json", out)
        print(json.dumps({"status": out["status"], "out_json": rel(base_out / "summary" / "collate_dryrun_summary.json")}, indent=2), flush=True)
        return out
    return summarize(base_out, args.target, endpoint_ready, eval_rows, stage_summary, collation)


def main() -> None:
    p = argparse.ArgumentParser(description="Race-free full evaluation for scale1.75 100M endpoint")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--base-out", type=Path, default=DEFAULT_BASE_OUT)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--skip-wait", action="store_true")
    p.add_argument("--merge-only", action="store_true", help="Do not run eval jobs; stage/collate existing part outputs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--zero-gpus", nargs="*", type=int, default=[0])
    p.add_argument("--zero-slots-per-gpu", type=int, default=3)
    p.add_argument("--superglue-gpus", nargs="*", type=int, default=[1], help="GPU candidates for SuperGLUE; default waits for GPU1 to avoid racing the GPU0 zero-shot split jobs")
    p.add_argument("--aoa-gpus", nargs="*", type=int, default=[0, 1], help="GPU candidates for AoA after zero-shot jobs complete")
    p.add_argument("--min-free-mb", type=int, default=12000)
    p.add_argument("--gpu-wait-timeout-sec", type=int, default=21600)
    p.add_argument("--endpoint-wait-timeout-sec", type=int, default=21600)
    p.add_argument("--zero-timeout-sec", type=int, default=10800)
    p.add_argument("--superglue-timeout-sec", type=int, default=21600)
    p.add_argument("--aoa-timeout-sec", type=int, default=21600)
    p.add_argument("--no-superglue", action="store_true")
    p.add_argument("--no-aoa", action="store_true")
    p.add_argument("--copy-artifacts", action="store_true", help="Copy staged files instead of symlinking")
    p.add_argument("--collate-dry-run", action="store_true")
    args = p.parse_args()

    args.run_dir = args.run_dir if args.run_dir.is_absolute() else USER_ROOT / args.run_dir
    args.base_out = args.base_out if args.base_out.is_absolute() else USER_ROOT / args.base_out
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(EVAL_SCRIPT)
    if not COLLATE_SCRIPT.exists():
        raise FileNotFoundError(COLLATE_SCRIPT)
    run_full(args)


if __name__ == "__main__":
    main()
