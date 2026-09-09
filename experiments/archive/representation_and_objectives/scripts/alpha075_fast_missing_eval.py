#!/usr/bin/env python3
"""research: generate only the missing truthful alpha0.75 Strict-Small fast predictions.

The alpha0.75 practical endpoint has a truthful 19-revision model tree after the
82M-anchor replay. Pre-82M checkpoints are identical to the protected chck82 ladder,
so their 119 fast prediction files can be reused. The only missing fast files are
2 revisions (chck_90M, chck_100M) × 7 fast tasks.

This script is bounded evaluation work, not training. It prepares official fast data,
stages the alpha0.75 full-prediction source files for later unmodified collation,
links reusable pre-82 fast predictions, and evaluates only missing alpha-specific
fast tasks. It can resume from existing outputs and fails if an expected prediction
file is absent after a task.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
PRISTINE_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
PRISTINE_FAST_BASE = PRISTINE_STRICT / "evaluation_data/fast_eval"
FAST_GP = WORKSPACE / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/fast_eval"
EWOK_FAST_EXPANDED = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"
MODEL_TREE = WORKSPACE / "data/alpha075_fast_preflight_v2/hf_model_truthful_alpha075"
CHCK82_FAST = WORKSPACE / "data/chck82_fast_submission_materialization/collate_fast/results/hf_model"
CARRIER_MANIFEST = USER_ROOT / "experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json"
DEFAULT_BASE_OUT = WORKSPACE / "data/alpha075_fast_missing_eval"

FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
ALPHA_ONLY_REVISIONS = ["chck_90M", "chck_100M"]
FAST_TASKS: Dict[str, Dict[str, Any]] = {
    "BLiMP": {"runner": "sentence", "task": "blimp", "data_rel": "blimp_fast", "rel": "blimp/blimp_fast/predictions.json", "batch_size": 128},
    "Supplement": {"runner": "sentence", "task": "blimp", "data_rel": "supplement_fast", "rel": "blimp/supplement_fast/predictions.json", "batch_size": 128},
    "EWoK": {"runner": "sentence", "task": "ewok", "data_rel": "ewok_fast", "rel": "ewok/ewok_fast/predictions.json", "batch_size": 64},
    "Entity": {"runner": "sentence", "task": "entity_tracking", "data_rel": "entity_tracking_fast", "rel": "entity_tracking/entity_tracking_fast/predictions.json", "batch_size": 128},
    "GlobalPIQA_parallel": {"runner": "sentence", "task": "global_piqa_parallel", "data_rel": "global_piqa_parallel", "rel": "global_piqa_parallel/global_piqa_parallel/predictions.json", "batch_size": 128},
    "GlobalPIQA_nonparallel": {"runner": "sentence", "task": "global_piqa_nonparallel", "data_rel": "global_piqa_nonparallel", "rel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json", "batch_size": 128},
    "Reading": {"runner": "reading", "data_rel": "reading/reading_data.csv", "rel": "reading/predictions.json"},
}
FULL_ZERO_REL = {
    "blimp": "blimp/blimp_filtered/predictions.json",
    "blimp_supplement": "blimp/supplement_filtered/predictions.json",
    "ewok": "ewok/ewok_filtered/predictions.json",
    "entity_tracking_filtered": "entity_tracking/entity_tracking/predictions.json",
    "comps": "comps/comps/predictions.json",
    "global_piqa_parallel": "global_piqa_parallel/global_piqa_parallel/predictions.json",
    "global_piqa_nonparallel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "reading": "reading/predictions.json",
}
GLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        try:
            return str(p.relative_to(USER_ROOT))
        except Exception:
            return str(p)


def sha256_file(p: pathlib.Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_link(src: pathlib.Path, dst: pathlib.Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    rec = {"mode": "symlink", "src": rel(src), "dst": rel(dst), "is_dir": src.is_dir()}
    if src.is_file():
        rec.update({"size_bytes": src.stat().st_size, "sha256": sha256_file(src)})
    elif (src / "model.safetensors").exists():
        rec.update({"model_sha256": sha256_file(src / "model.safetensors"), "config_sha256": sha256_file(src / "config.json")})
    return rec


def count_jsonl(p: pathlib.Path) -> int:
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def prepare_fast_data(base_out: pathlib.Path) -> Dict[str, Any]:
    fast_root = base_out / "fast_eval_data"
    records: Dict[str, Any] = {}
    for name in ["blimp_fast", "supplement_fast", "entity_tracking_fast", "reading"]:
        records[name] = clean_link(PRISTINE_FAST_BASE / name, fast_root / name)
    for name in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        records[name] = clean_link(FAST_GP / name, fast_root / name)
    records["ewok_fast"] = clean_link(EWOK_FAST_EXPANDED, fast_root / "ewok_fast")
    counts: Dict[str, Any] = {}
    for name in ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast"]:
        counts[name] = {p.stem: count_jsonl(p) for p in sorted((fast_root / name).glob("*.jsonl"))}
    for name in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        counts[name] = {"eng_latn": count_jsonl(fast_root / name / "eng_latn.jsonl")}
    rfile = fast_root / "reading/reading_data.csv"
    counts["reading_rows_excluding_header"] = max(0, sum(1 for _ in rfile.open("r", encoding="utf-8", errors="replace")) - 1)
    return {"fast_root": rel(fast_root), "records": records, "counts": counts}


def expected_prediction_path(base_out: pathlib.Path, checkpoint: str, task_name: str) -> pathlib.Path:
    return base_out / "collate_fast" / "results" / "hf_model" / checkpoint / "zero_shot" / "mlm" / FAST_TASKS[task_name]["rel"]


def stage_reusable_fast(base_out: pathlib.Path) -> Dict[str, Any]:
    linked: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    existing: List[Dict[str, Any]] = []
    for ckpt in FAST_REVISIONS:
        for task in FAST_TASKS:
            dst = expected_prediction_path(base_out, ckpt, task)
            if ckpt in ALPHA_ONLY_REVISIONS:
                if dst.exists():
                    existing.append({"checkpoint": ckpt, "task": task, "path": rel(dst), "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})
                continue
            src = CHCK82_FAST / ckpt / "zero_shot" / "mlm" / FAST_TASKS[task]["rel"]
            if src.exists():
                linked.append(clean_link(src, dst))
            else:
                missing.append({"checkpoint": ckpt, "task": task, "source": rel(src), "target": rel(dst)})
    return {"linked_pre82": linked, "missing_pre82": missing, "existing_alpha_specific": existing}


def stage_full_main_from_manifest(base_out: pathlib.Path) -> Dict[str, Any]:
    manifest = read_json(CARRIER_MANIFEST)
    srcs = manifest.get("source_files", {}).get("source_prediction_files", {})
    results_root = base_out / "collate_fast" / "results" / "hf_model" / "main"
    records: Dict[str, Any] = {}
    for key, rel_dst in FULL_ZERO_REL.items():
        if key not in srcs:
            raise KeyError(f"missing source_prediction_files[{key!r}]")
        records[key] = clean_link(USER_ROOT / srcs[key], results_root / "zero_shot" / "mlm" / rel_dst)
    for task in GLUE_TASKS:
        skey = f"glue/{task}"
        if skey not in srcs:
            raise KeyError(f"missing source_prediction_files[{skey!r}]")
        records[skey] = clean_link(USER_ROOT / srcs[skey], results_root / "finetune" / task / "predictions.json")
    carrier = pathlib.Path(manifest["carrier_path"])
    if not carrier.is_absolute():
        carrier = USER_ROOT / carrier
    return {
        "records": records,
        "carrier_path": rel(carrier),
        "carrier_sha256": sha256_file(carrier),
        "manifest_status": manifest.get("status"),
        "overall_with_aoa0": manifest.get("score_arithmetic_candidate_native", {}).get("overall_with_aoa0"),
    }


def model_alias(base_out: pathlib.Path, checkpoint: str) -> pathlib.Path:
    src = MODEL_TREE / checkpoint
    if not (src / "model.safetensors").exists():
        raise FileNotFoundError(src / "model.safetensors")
    alias = base_out / "model_aliases" / checkpoint / "hf_model"
    clean_link(src, alias)
    return alias


def abs_no_symlink(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else USER_ROOT / path


def gpu_free_mb(gpu: int) -> int | None:
    try:
        proc = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def wait_for_gpu(gpu: int, min_free_mb: int, timeout_sec: int, label: str) -> None:
    t0 = time.time()
    last = None
    while True:
        last = gpu_free_mb(gpu)
        if last is not None and last >= min_free_mb:
            print(json.dumps({"event": "gpu_available", "gpu": gpu, "free_mb": last, "label": label, "utc": now()}), flush=True)
            return
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"GPU {gpu} did not reach {min_free_mb} MiB free for {label}; last_free={last}")
        print(json.dumps({"event": "waiting_gpu", "gpu": gpu, "free_mb": last, "need_mb": min_free_mb, "label": label, "utc": now()}), flush=True)
        time.sleep(30)


def make_env(base_out: pathlib.Path, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    cache = base_out / "runtime_cache" / f"gpu{gpu}"
    mapping = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "TMPDIR": cache / "tmp",
        "NLTK_DATA": USER_ROOT / "experiments/archive/initial_model_studies/data/nltk_data",
    }
    for key, path in mapping.items():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        env[key] = str(pathlib.Path(path).resolve())
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = str(PRISTINE_STRICT.resolve()) + os.pathsep + str((PRISTINE_STRICT / "evaluation_pipeline").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_cmd(cmd: List[str], cwd: pathlib.Path, env: Dict[str, str], timeout_sec: int, log_prefix: pathlib.Path) -> Dict[str, Any]:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout_sec)
    elapsed = time.time() - t0
    out_log = log_prefix.with_suffix(".stdout.log")
    err_log = log_prefix.with_suffix(".stderr.log")
    out_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    err_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {"cmd": cmd, "cwd": rel(cwd), "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-5000:], "stdout_log": rel(out_log), "stderr_log": rel(err_log)}


def inspect_prediction(p: pathlib.Path) -> Dict[str, Any]:
    rec = {"path": rel(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None, "sha256": sha256_file(p) if p.exists() else None}
    if p.exists():
        try:
            d = read_json(p)
            rec["top_keys"] = sorted(d.keys()) if isinstance(d, dict) else None
            rec["subtask_count"] = len(d) if isinstance(d, dict) else None
            rec["prediction_total"] = sum(len(v.get("predictions", [])) for v in d.values() if isinstance(v, dict)) if isinstance(d, dict) else None
        except Exception as exc:
            rec["read_error"] = repr(exc)
    return rec


def run_fast_task(base_out: pathlib.Path, checkpoint: str, task_name: str, gpu: int, force: bool, timeout_sec: int) -> Dict[str, Any]:
    spec = FAST_TASKS[task_name]
    out_pred = expected_prediction_path(base_out, checkpoint, task_name)
    if out_pred.exists() and not force:
        rec = {"checkpoint": checkpoint, "task_name": task_name, "gpu": gpu, "status": "skip_existing"}
        rec.update(inspect_prediction(out_pred))
        return rec
    alias = model_alias(base_out, checkpoint)
    env = make_env(base_out, gpu)
    if spec["runner"] == "sentence":
        cmd = [
            sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(abs_no_symlink(alias)),
            "--backend", "mlm",
            "--task", spec["task"],
            "--data_path", str((base_out / "fast_eval_data" / spec["data_rel"]).resolve()),
            "--revision_name", checkpoint,
            "--save_predictions",
            "--batch_size", str(spec["batch_size"]),
            "--non_causal_batch_size", "64",
            "--output_dir", str((base_out / "collate_fast" / "results").resolve()),
        ]
    elif spec["runner"] == "reading":
        cmd = [
            sys.executable, "-B", "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(abs_no_symlink(alias)),
            "--backend", "mlm",
            "--data_path", str((base_out / "fast_eval_data" / spec["data_rel"]).resolve()),
            "--revision_name", checkpoint,
            "--output_dir", str((base_out / "collate_fast" / "results").resolve()),
        ]
    else:
        raise ValueError(spec["runner"])
    log_prefix = base_out / "logs" / "fast_tasks" / f"{checkpoint}_{task_name}"
    print(json.dumps({"event": "alpha_fast_task_start", "checkpoint": checkpoint, "task": task_name, "gpu": gpu, "expected_prediction": rel(out_pred), "utc": now()}), flush=True)
    rec = run_cmd(cmd, PRISTINE_STRICT, env, timeout_sec, log_prefix)
    rec.update({"checkpoint": checkpoint, "task_name": task_name, "gpu": gpu, "prediction_path": rel(out_pred), "prediction_exists": out_pred.exists()})
    rec.update({f"prediction_{k}": v for k, v in inspect_prediction(out_pred).items()})
    print(json.dumps({"event": "alpha_fast_task_done", "checkpoint": checkpoint, "task": task_name, "returncode": rec["returncode"], "prediction_exists": out_pred.exists(), "elapsed_sec": rec["elapsed_sec"], "utc": now()}), flush=True)
    if rec["returncode"] != 0 or not out_pred.exists():
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    return rec


def pending_alpha_tasks(base_out: pathlib.Path, force: bool) -> List[Tuple[str, str]]:
    pending: List[Tuple[str, str]] = []
    for ckpt in ALPHA_ONLY_REVISIONS:
        for task in FAST_TASKS:
            p = expected_prediction_path(base_out, ckpt, task)
            if force or not p.exists():
                pending.append((ckpt, task))
    return pending


def run_full(args: argparse.Namespace) -> Dict[str, Any]:
    base_out = pathlib.Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prep = {
        "fast_data": prepare_fast_data(base_out),
        "full_main": stage_full_main_from_manifest(base_out),
        "reusable_fast": stage_reusable_fast(base_out),
    }
    errors: List[str] = []
    if prep["reusable_fast"]["missing_pre82"]:
        errors.append("missing_pre82_reusable_fast_predictions")
    for ckpt in FAST_REVISIONS:
        if not (MODEL_TREE / ckpt / "model.safetensors").exists():
            errors.append(f"missing_model_{ckpt}")
    if errors:
        out = {"status": "ALPHA075_FAST_PREFLIGHT_NEEDS_REPAIR", "created_utc": now(), "base_out": rel(base_out), "preparation": prep, "errors": errors}
        write_json(base_out / "fast_missing_eval_summary.json", out)
        return out

    pending = pending_alpha_tasks(base_out, args.force)
    gpus = [int(x) for x in args.gpus]
    if not gpus:
        raise ValueError("need at least one GPU")
    gpu_tasks: Dict[int, List[Tuple[str, str]]] = {g: [] for g in gpus}
    for i, item in enumerate(pending):
        gpu_tasks[gpus[i % len(gpus)]].append(item)

    def worker(gpu: int, items: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ckpt, task in items:
            wait_for_gpu(gpu, args.min_free_mb, args.gpu_wait_timeout_sec, f"alpha0.75 {ckpt}/{task}")
            rows.append(run_fast_task(base_out, ckpt, task, gpu, force=args.force, timeout_sec=args.task_timeout_sec))
        return rows

    print(json.dumps({"event": "alpha_fast_missing_plan", "pending_count": len(pending), "gpu_tasks": {str(k): len(v) for k, v in gpu_tasks.items()}, "base_out": rel(base_out), "utc": now()}), flush=True)
    task_records: List[Dict[str, Any]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futs = [ex.submit(worker, gpu, items) for gpu, items in gpu_tasks.items() if items]
            for fut in as_completed(futs):
                task_records.extend(fut.result())
    expected: List[Dict[str, Any]] = []
    missing_predictions: List[Dict[str, Any]] = []
    for ckpt in FAST_REVISIONS:
        for task in FAST_TASKS:
            rec = {"checkpoint": ckpt, "task": task}
            rec.update(inspect_prediction(expected_prediction_path(base_out, ckpt, task)))
            expected.append(rec)
            if not rec["exists"]:
                missing_predictions.append(rec)
    status = "ALPHA075_FAST_MISSING_EVAL_DONE" if not missing_predictions else "ALPHA075_FAST_MISSING_EVAL_NEEDS_REPAIR"
    out = {
        "status": status,
        "created_utc": now(),
        "purpose": {
            "question": "Can the strongest projected alpha0.75 endpoint be advanced toward a complete official-compatible Strict-Small artifact without model substitution?",
            "minimum_cost_action": "Reuse 119 pre-82M fast predictions from the identical protected chck82 ladder and generate only 14 alpha-specific chck_90M/chck_100M fast prediction files.",
            "decision_use": "If all 133 fast files exist and AoA later passes 19x8005 rows, final collation can use the existing alpha0.75 full carrier; otherwise the practical endpoint remains incomplete.",
        },
        "base_out": rel(base_out),
        "model_tree": rel(MODEL_TREE),
        "preparation": prep,
        "pending_at_start": len(pending),
        "gpu_plan": {str(k): [{"checkpoint": ck, "task": task} for ck, task in v] for k, v in gpu_tasks.items()},
        "task_records_count": len(task_records),
        "task_records": task_records,
        "expected_prediction_files": expected,
        "missing_prediction_files": missing_predictions,
        "elapsed_sec": round(time.time() - t0, 3),
        "next_after_fast": "Run official min_context=0 AoA on the same truthful model_tree; then link AoA into collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word and run unmodified collate_preds.py --fast.",
    }
    write_json(base_out / "fast_missing_eval_summary.json", out)
    md = [
        "# research alpha0.75 missing Fast evaluation",
        "",
        f"Status: `{status}`",
        f"Base out: `{rel(base_out)}`",
        f"Pending at start: {len(pending)}",
        f"Task records: {len(task_records)}",
        f"Missing prediction files after run: {len(missing_predictions)}",
        "",
        f"JSON: `{rel(base_out / 'fast_missing_eval_summary.json')}`",
    ]
    (base_out / "fast_missing_eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if missing_predictions:
        raise SystemExit(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-out", default=str(DEFAULT_BASE_OUT))
    ap.add_argument("--gpus", nargs="+", default=["1"], help="GPU ids; default uses one GPU so AoA can use the other")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--task-timeout-sec", type=int, default=1800)
    ap.add_argument("--min-free-mb", type=int, default=24000)
    ap.add_argument("--gpu-wait-timeout-sec", type=int, default=21600)
    args = ap.parse_args()
    out = run_full(args)
    print(json.dumps({"status": out["status"], "base_out": out["base_out"], "pending_at_start": out.get("pending_at_start"), "task_records_count": out.get("task_records_count"), "missing_prediction_files": len(out.get("missing_prediction_files", [])), "summary": rel(pathlib.Path(args.base_out) / "fast_missing_eval_summary.json")}, indent=2), flush=True)
    if out["status"].endswith("NEEDS_REPAIR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
