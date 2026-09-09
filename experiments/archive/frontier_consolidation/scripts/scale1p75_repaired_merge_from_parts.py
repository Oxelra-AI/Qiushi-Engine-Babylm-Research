#!/usr/bin/env python3
"""research: assemble a repaired scale1.75 100M full-eval merge from completed parts.

The original research scale1.75 full evaluator failed only because its AoA part hit a
read-only HF dynamic-module cache.  By research, zero-shot/Reading and SuperGLUE
parts in the original tree are complete, and research produced a valid isolated
AoA run with the official 19-checkpoint, 8005-row/checkpoint coordinate.

This script creates a fresh, isolated merge tree:
  * symlink completed non-AoA part directories from the research tree;
  * write a normalized AoA part payload pointing to the research repaired AoA
    surprisal/score files;
  * invoke the same hardened merge-only/pristine collation path.

It performs no model inference or finetuning and does not overwrite the original
failed research tree.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
SOURCE_BASE = WORKSPACE / "data/scale1p75_100M_full_eval_hardened"
OUT_BASE = WORKSPACE / "data/scale1p75_100M_repaired_merge"
REPAIR_AOA_JSON = WORKSPACE / "data/scale1p75_aoa_repair/aoa_outputs/scale1p75_100M_seed43022__AoA/aoa_local_ckpts_minctx0.json"
ORIG_AOA_PAYLOAD = SOURCE_BASE / "parts/AoA/eval/per_target/scale1p75_100M_seed43022__AoA.json"
MERGE_SCRIPT = WORKSPACE / "scripts/full_eval_scale1p75_100M_hardened.py"
RUN_DIR = WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
TARGET = "scale1p75_100M_seed43022"
ENDPOINT = "chck_100M"
PART_NAMES = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GP_parallel",
    "GP_nonparallel",
    "Reading",
    "SuperGLUE",
]
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


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


def ensure_symlink(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            raise RuntimeError(f"Refusing to replace existing non-symlink directory: {dst}")
    os.symlink(src.resolve(), dst)
    return {"src": rel(src), "dst": rel(dst), "mode": "symlink"}


def validate_superglue_payload() -> dict[str, Any]:
    sg_payload = SOURCE_BASE / "parts/SuperGLUE/eval/per_target/scale1p75_100M_seed43022__SuperGLUE.json"
    data = read_json(sg_payload)
    sg = data.get("tasks", {}).get("SuperGLUE")
    if not isinstance(sg, dict):
        raise RuntimeError({"missing_superglue_task": rel(sg_payload)})
    task_rows = [r for r in sg.get("tasks", []) if isinstance(r, dict)]
    task_names = sorted(r.get("task") for r in task_rows)
    expected = sorted(["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"])
    if task_names != expected:
        raise RuntimeError({"superglue_incomplete": task_names, "expected": expected, "payload": rel(sg_payload)})
    missing_preds = [r.get("task") for r in task_rows if not Path(str(r.get("predictions", ""))).is_absolute() and not (USER_ROOT / str(r.get("predictions", ""))).exists()]
    missing_preds += [r.get("task") for r in task_rows if Path(str(r.get("predictions", ""))).is_absolute() and not Path(str(r.get("predictions", ""))).exists()]
    if missing_preds:
        raise RuntimeError({"superglue_missing_predictions": missing_preds, "payload": rel(sg_payload)})
    return {"payload": rel(sg_payload), "task_names": task_names, "superglue_mean": sg.get("superglue_mean"), "primary_metric_details": sg.get("superglue_primary_metric_details")}


def build_aoa_payload() -> dict[str, Any]:
    repair = read_json(REPAIR_AOA_JSON)
    orig = read_json(ORIG_AOA_PAYLOAD)
    if repair.get("status") != "AOA_LOCAL_CKPTS_MINCTX_DONE":
        raise RuntimeError({"bad_repair_status": repair.get("status"), "path": rel(REPAIR_AOA_JSON)})
    if repair.get("num_steps") != len(AOA_STEPS):
        raise RuntimeError({"bad_num_steps": repair.get("num_steps")})
    if repair.get("expected_steps") != AOA_STEPS:
        raise RuntimeError({"expected_steps_mismatch": repair.get("expected_steps")})
    if repair.get("missing_steps") not in ([], None) or repair.get("unexpected_steps") not in ([], None):
        raise RuntimeError({"aoa_step_mismatch": {"missing": repair.get("missing_steps"), "unexpected": repair.get("unexpected_steps")}})
    if repair.get("row_count_values") != [8005] or repair.get("num_rows") != 152095:
        raise RuntimeError({"bad_aoa_rows": {"row_count_values": repair.get("row_count_values"), "num_rows": repair.get("num_rows")}})
    if repair.get("finite_surprisals") is not True:
        raise RuntimeError({"nonfinite_aoa_surprisals": repair.get("finite_surprisals")})
    surprisal = USER_ROOT / str(repair.get("surprisal_path"))
    score = USER_ROOT / str(repair.get("score_path"))
    if not surprisal.exists() or not score.exists():
        raise FileNotFoundError({"surprisal": str(surprisal), "score": str(score)})
    raw = float(repair.get("aoa", 0.0))
    lb = raw * 100.0
    curve = repair.get("curve_fitness_record") or {}
    task = {
        "column": "AoA",
        "status": "official_aoa_done",
        "helper_status": repair.get("status"),
        "aoa_helper_runner": rel(WORKSPACE / "scripts/aoa_local_ckpts_minctx.py"),
        "required_strict_small_steps": AOA_STEPS,
        "available_checkpoint_count": orig.get("tasks", {}).get("AoA", {}).get("available_checkpoint_count"),
        "missing_required_steps": [],
        "model_root": repair.get("model_root"),
        "min_context": repair.get("min_context"),
        "expected_rows_per_step": repair.get("expected_rows_per_step"),
        "returncode": 0,
        "elapsed_sec": repair.get("elapsed_sec"),
        "out_json": rel(REPAIR_AOA_JSON),
        "out_note": rel(REPAIR_AOA_JSON.with_suffix(".md")),
        "surprisal_path": repair.get("surprisal_path"),
        "score_path": repair.get("score_path"),
        "score_tokenizer_path": repair.get("score_tokenizer_path"),
        "aoa": raw,
        "aoa_official": raw,
        "aoa_raw_correlation": raw,
        "aoa_leaderboard_score": lb,
        "aoa_for_provisional_overall": lb,
        "aoa_unit_correction": "AoA raw correlation converted to leaderboard score by multiplying by 100 before Overall arithmetic.",
        "num_rows": repair.get("num_rows"),
        "num_steps": repair.get("num_steps"),
        "step_counts": repair.get("step_counts"),
        "expected_steps": repair.get("expected_steps"),
        "missing_steps": repair.get("missing_steps"),
        "unexpected_steps": repair.get("unexpected_steps"),
        "row_count_values": repair.get("row_count_values"),
        "finite_surprisals": repair.get("finite_surprisals"),
        "curve_fitness_record": curve,
        "aoa_p_value": curve.get("p_value") if isinstance(curve, dict) else None,
        "aoa_n_words": curve.get("n_words") if isinstance(curve, dict) else None,
        "aoa_validation": "passed full 19-step ladder, finite surprisal, 8005 rows/checkpoint, and finite raw-AoA checks in the isolated research repair",
    }
    payload = dict(orig)
    payload["target"] = f"{TARGET}__AoA"
    payload["started_utc"] = orig.get("started_utc") or now()
    payload["finished_utc"] = now()
    payload["tasks"] = {"AoA": task}
    payload["repair_source"] = {"original_failed_payload": rel(ORIG_AOA_PAYLOAD), "repaired_aoa_json": rel(REPAIR_AOA_JSON)}
    out_payload = OUT_BASE / "parts/AoA/eval/per_target" / f"{TARGET}__AoA.json"
    write_json(out_payload, payload)
    return {"out_payload": rel(out_payload), "aoa": raw, "aoa_leaderboard_score": lb, "surprisal_path": repair.get("surprisal_path"), "score_path": repair.get("score_path")}


def run_merge() -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        str(MERGE_SCRIPT),
        "--skip-wait",
        "--merge-only",
        "--base-out",
        str(OUT_BASE),
        "--run-dir",
        str(RUN_DIR),
        "--target",
        TARGET,
        "--endpoint",
        ENDPOINT,
    ]
    env = os.environ.copy()
    cache_root = OUT_BASE / "merge_runtime_cache"
    for key, path in {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home/hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home/hub",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "TMPDIR": cache_root / "tmp",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    log_dir = OUT_BASE / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    (log_dir / "merge.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "merge.stderr.log").write_text(proc.stderr, encoding="utf-8")
    rec = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "stdout_log": rel(log_dir / "merge.stdout.log"),
        "stderr_log": rel(log_dir / "merge.stderr.log"),
    }
    if proc.returncode != 0:
        raise RuntimeError(rec)
    return rec


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "summary").mkdir(parents=True, exist_ok=True)
    link_records = []
    for name in PART_NAMES:
        link_records.append(ensure_symlink(SOURCE_BASE / "parts" / name, OUT_BASE / "parts" / name))
    sg = validate_superglue_payload()
    aoa = build_aoa_payload()
    merge = run_merge()
    final_summary = OUT_BASE / "summary/scale1p75_100M_full_eval_hardened_summary.json"
    final = read_json(final_summary) if final_summary.exists() else None
    out = {
        "status": "SCALE1P75_REPAIRED_MERGE_DONE",
        "created_utc": now(),
        "source_base": rel(SOURCE_BASE),
        "out_base": rel(OUT_BASE),
        "target": TARGET,
        "endpoint": ENDPOINT,
        "links": link_records,
        "superglue_validation": sg,
        "aoa_repair": aoa,
        "merge_run": merge,
        "final_summary_json": rel(final_summary),
        "final_overall": final.get("overall") if isinstance(final, dict) else None,
        "final_margin_vs_41p8": final.get("overall_margin_vs_41p8") if isinstance(final, dict) else None,
        "final_scores": final.get("scores") if isinstance(final, dict) else None,
    }
    write_json(OUT_BASE / "summary/scale1p75_repaired_merge_from_parts.json", out)
    print(json.dumps({
        "status": out["status"],
        "Overall": out["final_overall"],
        "margin_vs_41p8": out["final_margin_vs_41p8"],
        "scores": out["final_scores"],
        "out_json": rel(OUT_BASE / "summary/scale1p75_repaired_merge_from_parts.json"),
        "summary_json": rel(final_summary),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
