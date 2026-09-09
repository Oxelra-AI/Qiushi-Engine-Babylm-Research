#!/usr/bin/env python3
"""research: CPU-parallel evaluation of ALL dose-family checkpoints.

Unblocks the dose-curve readout by running stable-family evaluations on CPU
in parallel, freeing GPUs for training.  Uses the same BabyLM evaluation
pipeline but forces CUDA_VISIBLE_DEVICES="" on every subprocess.

Evaluates: clean0 (10M-80M), dose1 view/repeat (10M-100M), dose1p82 view/repeat
(10M-100M), MAX view/repeat (10M-100M).  Reuses existing per-target JSONs from
prior evaluation steps where available.

Produces:
  - per_target/*.json           (one per arm-checkpoint, same schema as research)
  - dose_ladder_stable_rows.csv (unified CSV for integrator)
  - eval_manifest.json          (tracking)

Usage:
  python3 -B cpu_parallel_dose_eval.py [--workers N] [--plan-only]
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
BABYLM_STRICT = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
OUT_DIR = WS / "data/cpu_parallel_dose_eval"

# ─── Evaluation task definitions (same as research base runner) ───
ZERO_SHOT_FAMILIES = [
    {"column": "BLiMP", "task": "blimp", "data_path": "evaluation_data/full_eval/blimp_filtered", "batch_size": 64},
    {"column": "Supplement", "task": "blimp", "data_path": "evaluation_data/full_eval/supplement_filtered", "batch_size": 64},
    {"column": "EWoK", "task": "ewok", "data_path": "evaluation_data/full_eval/ewok_filtered", "batch_size": 32},
    {"column": "Entity", "task": "entity_tracking", "data_path": "evaluation_data/full_eval/entity_tracking", "batch_size": 64},
    {"column": "COMPS", "task": "comps", "data_path": "evaluation_data/full_eval/comps", "batch_size": 64},
]
# Reduced batch sizes for CPU (smaller than GPU defaults to control memory)

CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]

# ─── Dose arms ───
ARMS: Dict[str, Dict[str, Any]] = {
    "clean0": {
        "dose": 0.0, "rho": 0.0, "data_arm": "clean", "dose_name": "clean0",
        "run_dir": WS / "training/runs/complianttok_cleanqwen_seed43022_80M",
        "checkpoints": CKS_10_80, "max_words": 80_000_000,
    },
    "dose1_view": {
        "dose": 1.0, "rho": 0.042352, "data_arm": "view", "dose_name": "dose1",
        "run_dir": WS / "training/runs/complianttok_reinvest_seed43022_r2",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
    "dose1_repeat": {
        "dose": 1.0, "rho": 0.042352, "data_arm": "repeat", "dose_name": "dose1",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
    "dose1p82_view": {
        "dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "view", "dose_name": "dose1p82",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose1p82x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
    "dose1p82_repeat": {
        "dose": 1.8209293539856442, "rho": 0.07712, "data_arm": "repeat", "dose_name": "dose1p82",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose1p82x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
    "max_view": {
        "dose": 2.641480921798262, "rho": 0.111872, "data_arm": "view", "dose_name": "dose2p64",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
    "max_repeat": {
        "dose": 2.641480921798262, "rho": 0.111872, "data_arm": "repeat", "dose_name": "dose2p64",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100, "max_words": 100_000_000,
    },
}

# ─── Existing per-target results that can be reused ───
REUSE_SOURCES: Dict[str, Dict[str, pathlib.Path]] = {
    "clean0": {
        "chck_20M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json",
        "chck_70M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json",
        "chck_80M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json",
    },
}
# Also try to reuse from the research seed ladder and existing research eval
SEED_LADDER_DIR = WS / "data/full_deberta_seed_ladder_stable_eval/eval/per_target"
EVAL_DIR = WS / "data/dose_ladder_stable_eval/eval/per_target"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def checkpoint_words(ck: str) -> int:
    m = re.match(r"chck_(\d+)M", ck)
    return int(m.group(1)) * 1_000_000 if m else 0


def checkpoint_exists(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return d.is_dir() and (d / "config.json").is_file()


def parse_sentence_score(text: str) -> Optional[float]:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -5.0 <= val <= 105.0:
                return val
    return None


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    for p in sorted(task_out.rglob("best_temperature_report.txt"),
                    key=lambda p: (p.stat().st_mtime, str(p)), reverse=True):
        val = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if val is not None:
            return val
    return None


def parse_reading_scores(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"),
                       ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_reading_report(task_out: pathlib.Path) -> Dict[str, float]:
    for p in sorted(task_out.rglob("report.txt"),
                    key=lambda p: (p.stat().st_mtime, str(p)), reverse=True):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def load_existing_per_target(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load an existing per-target JSON and extract stable column scores."""
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        tasks = data.get("tasks", {})
        scores = {}
        for col in STABLE_COLUMNS:
            if col == "Reading":
                r = tasks.get("Reading", {})
                s = r.get("scores", {})
                if "Reading" in s:
                    scores["Reading"] = float(s["Reading"])
            else:
                r = tasks.get(col, {})
                if "score" in r and r["score"] is not None:
                    scores[col] = float(r["score"])
        if len(scores) == len(STABLE_COLUMNS):
            return scores
    except Exception:
        pass
    return None


def find_reusable(arm_name: str, ck: str) -> Optional[Tuple[pathlib.Path, Dict[str, Any]]]:
    """Try to find an existing per-target result for this arm/checkpoint."""
    # Check explicit reuse sources
    if arm_name in REUSE_SOURCES and ck in REUSE_SOURCES[arm_name]:
        p = REUSE_SOURCES[arm_name][ck]
        scores = load_existing_per_target(p)
        if scores:
            return p, scores

    # Check research eval output (might have some early results)
    p256 = EVAL_DIR / f"dose_{arm_name}_{ck}.json"
    scores = load_existing_per_target(p256)
    if scores:
        return p256, scores

    # Check research seed ladder for 1x arms
    if arm_name == "dose1_view":
        for pat in [f"ladder_seed43022_compact_{ck}.json",
                    f"complianttok_reinvest_seed43022_{ck.replace('chck_', '')}.json"]:
            p = SEED_LADDER_DIR / pat
            scores = load_existing_per_target(p)
            if scores:
                return p, scores
        # Also check research eval
        p49 = WS / f"data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_{ck.replace('chck_', '')}.json"
        scores = load_existing_per_target(p49)
        if scores:
            return p49, scores

    if arm_name == "dose1_repeat":
        p = SEED_LADDER_DIR / f"ladder_seed43022_repeat_{ck}.json"
        scores = load_existing_per_target(p)
        if scores:
            return p, scores

    # Check our own output
    p_own = OUT_DIR / "per_target" / f"step263_{arm_name}_{ck}.json"
    scores = load_existing_per_target(p_own)
    if scores:
        return p_own, scores

    return None


def make_cpu_env(target_name: str) -> Dict[str, str]:
    """Build subprocess environment forcing CPU-only evaluation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    hf = OUT_DIR / "hf_cache" / target_name
    tmp = OUT_DIR / "tmp" / target_name
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    nltk_data = ROOT / "experiments/archive/initial_model_studies/data/nltk_data"
    env["NLTK_DATA"] = str(nltk_data.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TMPDIR"] = str(tmp.resolve())
    env["OMP_NUM_THREADS"] = "4"  # Limit per-process threads
    env["MKL_NUM_THREADS"] = "4"
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
                "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def eval_one_family_cpu(model_path: pathlib.Path, family: Dict[str, Any],
                        output_dir: pathlib.Path, target_name: str,
                        env: Dict[str, str]) -> Optional[float]:
    """Evaluate one zero-shot family on CPU. Returns score or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", family["task"],
        "--data_path", family["data_path"],
        "--revision_name", target_name,
        "--save_predictions",
        "--batch_size", str(family["batch_size"]),
        "--non_causal_batch_size", "32",
        "--output_dir", str(output_dir.resolve()),
    ]
    log_path = OUT_DIR / "logs" / target_name / f"{family['column']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd, cwd=str(BABYLM_STRICT), env=env,
            capture_output=True, text=True, timeout=7200
        )
        log_path.write_text(
            f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-2000:] if proc.stdout else ''}\n"
            f"STDERR:\n{proc.stderr[-2000:] if proc.stderr else ''}\n",
            encoding="utf-8"
        )
        if proc.returncode != 0:
            return None
        return read_report_score(output_dir)
    except Exception as exc:
        log_path.write_text(f"EXCEPTION: {exc}\n", encoding="utf-8")
        return None


def eval_reading_cpu(model_path: pathlib.Path, output_dir: pathlib.Path,
                     target_name: str, env: Dict[str, str]) -> Optional[float]:
    """Evaluate Reading on CPU. Returns combined score or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/full_eval/reading/reading_data.csv",
        "--revision_name", target_name,
        "--output_dir", str(output_dir.resolve()),
    ]
    log_path = OUT_DIR / "logs" / target_name / "Reading.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd, cwd=str(BABYLM_STRICT), env=env,
            capture_output=True, text=True, timeout=7200
        )
        log_path.write_text(
            f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-2000:] if proc.stdout else ''}\n"
            f"STDERR:\n{proc.stderr[-2000:] if proc.stderr else ''}\n",
            encoding="utf-8"
        )
        if proc.returncode != 0:
            return None
        scores = read_reading_report(output_dir)
        return scores.get("Reading")
    except Exception as exc:
        log_path.write_text(f"EXCEPTION: {exc}\n", encoding="utf-8")
        return None


def eval_checkpoint_cpu(arm_name: str, ck: str, arm_info: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all 6 stable families for one arm/checkpoint on CPU."""
    target_name = f"step263_{arm_name}_{ck}"
    model_path = pathlib.Path(arm_info["run_dir"]) / "hf_model" / ck
    env = make_cpu_env(target_name)

    result = {
        "arm": arm_name, "checkpoint": ck,
        "words": checkpoint_words(ck),
        "model_path": str(model_path),
        "started_utc": now_utc(),
        "source_type": "cpu_eval",
    }

    scores: Dict[str, Optional[float]] = {}

    # Zero-shot families
    for fam in ZERO_SHOT_FAMILIES:
        col = fam["column"]
        fam_out = OUT_DIR / "eval" / "official_outputs" / target_name / col
        score = eval_one_family_cpu(model_path, fam, fam_out, target_name, env)
        scores[col] = score
        result[col] = score

    # Reading
    reading_out = OUT_DIR / "eval" / "official_outputs" / target_name / "Reading"
    reading_score = eval_reading_cpu(model_path, reading_out, target_name, env)
    scores["Reading"] = reading_score
    result["Reading"] = reading_score

    result["finished_utc"] = now_utc()
    result["complete"] = all(v is not None for v in scores.values())

    # Save per-target JSON (compatible with existing format)
    pt_path = OUT_DIR / "per_target" / f"{target_name}.json"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target_name,
        "model_path": str(model_path),
        "tasks": {},
    }
    for col, score in scores.items():
        if col == "Reading":
            payload["tasks"]["Reading"] = {"scores": {"Reading": score}, "column": "Reading"}
        else:
            payload["tasks"][col] = {"score": score, "column": col}
    with open(pt_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    result["per_target"] = str(pt_path)

    return result


def build_eval_plan() -> Tuple[List[Tuple[str, str, Dict]], List[Dict]]:
    """Build the evaluation plan: what to compute, what to reuse."""
    to_eval = []
    reused = []

    for arm_name, arm_info in ARMS.items():
        for ck in arm_info["checkpoints"]:
            # Check if already reusable
            r = find_reusable(arm_name, ck)
            if r is not None:
                path, scores = r
                reused.append({
                    "arm": arm_name, "checkpoint": ck,
                    "words": checkpoint_words(ck),
                    "source_type": "reuse",
                    "source_path": str(path),
                    **scores,
                })
                continue

            # Check if checkpoint exists
            if not checkpoint_exists(pathlib.Path(arm_info["run_dir"]), ck):
                # Checkpoint not yet available (maybe training)
                continue

            to_eval.append((arm_name, ck, arm_info))

    return to_eval, reused


def compute_derived(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add cheap6, cheap5, EWoK+Entity to a row."""
    try:
        vals = {c: float(row[c]) for c in STABLE_COLUMNS if row.get(c) is not None}
    except (ValueError, TypeError):
        return row

    if all(c in vals for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]):
        row["cheap6_no_GlobalPIQA"] = sum(vals[c] for c in
            ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]) / 6.0
    if all(c in vals for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]):
        row["cheap5_no_GlobalPIQA_Reading"] = sum(vals[c] for c in
            ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]) / 5.0
    if "EWoK" in vals and "Entity" in vals:
        row["EWoK_plus_Entity_sum"] = vals["EWoK"] + vals["Entity"]

    return row


def assemble_csv(all_rows: List[Dict[str, Any]]) -> pathlib.Path:
    """Write the unified ladder CSV."""
    csv_path = OUT_DIR / "dose_ladder_stable_rows.csv"
    fields = ["arm", "checkpoint", "words"] + STABLE_COLUMNS + [
        "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading",
        "EWoK_plus_Entity_sum", "source_type", "per_target"
    ]
    rows = sorted(all_rows, key=lambda r: (r.get("arm", ""), r.get("words", 0)))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12,
                    help="Max parallel CPU eval workers")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--wait-repeat", action="store_true",
                    help="Wait up to 30 min for midpoint repeat to finish training")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally wait for midpoint repeat to finish training
    if args.wait_repeat:
        repeat_dir = ARMS["dose1p82_repeat"]["run_dir"]
        for i in range(180):  # up to 30 min
            if checkpoint_exists(repeat_dir, "chck_100M"):
                print(json.dumps({"event": "repeat_ready", "utc": now_utc(),
                                  "waited_sec": i * 10}), flush=True)
                break
            time.sleep(10)

    to_eval, reused = build_eval_plan()

    plan = {
        "status": "CPU_PARALLEL_DOSE_EVAL_PLAN",
        "created_utc": now_utc(),
        "workers": args.workers,
        "to_eval": len(to_eval),
        "reused": len(reused),
        "eval_tasks": [(a, c) for a, c, _ in to_eval],
        "reused_tasks": [(r["arm"], r["checkpoint"]) for r in reused],
        "missing_checkpoints": [],
    }

    # Report missing checkpoints
    for arm_name, arm_info in ARMS.items():
        for ck in arm_info["checkpoints"]:
            if not checkpoint_exists(pathlib.Path(arm_info["run_dir"]), ck):
                exists_in_reuse = any(
                    r["arm"] == arm_name and r["checkpoint"] == ck for r in reused)
                if not exists_in_reuse:
                    plan["missing_checkpoints"].append((arm_name, ck))

    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    plan_path = OUT_DIR / "eval_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    if args.plan_only:
        return

    # Run evaluations in parallel on CPU
    all_rows = list(reused)  # Start with reused results

    completed = 0
    failed = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for arm_name, ck, arm_info in to_eval:
            fut = pool.submit(eval_checkpoint_cpu, arm_name, ck, arm_info)
            futures[fut] = (arm_name, ck)

        for fut in concurrent.futures.as_completed(futures):
            arm_name, ck = futures[fut]
            try:
                result = fut.result()
                if result.get("complete"):
                    completed += 1
                    all_rows.append(compute_derived(result))
                    print(json.dumps({
                        "event": "eval_done", "arm": arm_name, "checkpoint": ck,
                        "cheap6": result.get("cheap6_no_GlobalPIQA"),
                        "elapsed_sec": round(time.time() - t0, 1),
                        "completed": completed, "remaining": len(to_eval) - completed - failed,
                    }), flush=True)
                else:
                    failed += 1
                    print(json.dumps({
                        "event": "eval_incomplete", "arm": arm_name, "checkpoint": ck,
                        "result": {k: v for k, v in result.items() if k in STABLE_COLUMNS},
                    }), flush=True)
            except Exception as exc:
                failed += 1
                print(json.dumps({
                    "event": "eval_error", "arm": arm_name, "checkpoint": ck,
                    "error": repr(exc),
                }), flush=True)

    # Add derived columns to reused rows
    for r in all_rows:
        if "cheap6_no_GlobalPIQA" not in r:
            compute_derived(r)

    # Assemble CSV
    csv_path = assemble_csv(all_rows)

    manifest = {
        "status": "CPU_PARALLEL_DOSE_EVAL_DONE",
        "finished_utc": now_utc(),
        "total_elapsed_sec": round(time.time() - t0, 1),
        "completed_new": completed,
        "reused": len(reused),
        "failed": failed,
        "total_rows": len(all_rows),
        "csv_path": str(csv_path),
        "plan_path": str(plan_path),
    }
    manifest_path = OUT_DIR / "eval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
