#!/usr/bin/env python3
"""research: Evaluate D_V_43122 and D_C_43122 on BabyLM stable benchmarks.

This script runs BLiMP, Supplement, EWoK, Entity, COMPS, Reading evaluations
at 80M, 90M, 100M checkpoints for both VIEW and CLEAN arms at seed43122.
GPU 0 handles VIEW, GPU 1 handles CLEAN, running in parallel.

Scientific purpose: test whether V-C benchmark movement replicates across seeds.
Decision rule is pre-stated in notes/005_vc_seed_replication_prestate.md.

No training, upload, packaging, or leaderboard action.
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
import pathlib
import re
import statistics
import subprocess
import sys
import time
import concurrent.futures
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
STRICT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
PRISTINE_FULL = STRICT / "evaluation_data" / "full_eval"
NLP_DATA_ROOT = ROOT / "experiments/archive" / 'initial_model_studies' / "data" / "nltk_data"
OUT_BASE = WS / "data" / "vc_seed_replication"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
ZERO_SHOT_SPECS: dict[str, dict[str, Any]] = {
    "BLiMP": {"task": "blimp", "data_path": PRISTINE_FULL / "blimp_filtered", "batch_size": 128},
    "Supplement": {"task": "blimp", "data_path": PRISTINE_FULL / "supplement_filtered", "batch_size": 128},
    "EWoK": {"task": "ewok", "data_path": PRISTINE_FULL / "ewok_filtered", "batch_size": 64},
    "Entity": {"task": "entity_tracking", "data_path": PRISTINE_FULL / "entity_tracking", "batch_size": 128},
    "COMPS": {"task": "comps", "data_path": PRISTINE_FULL / "comps", "batch_size": 128},
}
READING_DATA = PRISTINE_FULL / "reading" / "reading_data.csv"

frontier_consolidation_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"

ARM_CONFIGS = {
    "D_V_43122": {
        "run_dir": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "description": "DeBERTa MAX VIEW seed43122",
        "data_arm": "view",
        "seed": 43122,
        "gpu": 0,
    },
    "D_C_43122": {
        "run_dir": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "description": "DeBERTa MAX CLEAN seed43122",
        "data_arm": "clean",
        "seed": 43122,
        "gpu": 1,
    },
    "D_R_43122": {
        "run_dir": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "description": "DeBERTa MAX REPEAT seed43122",
        "data_arm": "repeat",
        "seed": 43122,
        "gpu": 0,  # secondary, after VIEW finishes
    },
}

CHECKPOINTS = ["chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


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
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def gpu_env(arm: str, ck: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if NLP_DATA_ROOT.exists():
        env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = OUT_BASE / "hf_cache" / f"{arm}_{ck}"
    tmp = OUT_BASE / "tmp" / f"{arm}_{ck}"
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def target_name(arm: str, ck: str) -> str:
    return f"step005_{arm}_{ck}"


def per_target_path(target: str) -> pathlib.Path:
    return OUT_BASE / "per_target" / f"{target}.json"


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_or_create_payload(arm: str, ck: str) -> dict[str, Any]:
    target = target_name(arm, ck)
    p = per_target_path(target)
    if p.exists():
        payload = read_json(p)
        payload.setdefault("tasks", {})
        return payload
    cfg = ARM_CONFIGS[arm]
    model_path = cfg["run_dir"] / "hf_model" / ck
    payload = {
        "target": target,
        "arm": arm,
        "description": cfg["description"],
        "data_arm": cfg["data_arm"],
        "seed": cfg["seed"],
        "model_path": rel(model_path),
        "checkpoint": ck,
        "created_utc": now(),
        "tasks": {},
        "no_training_upload_leaderboard": True,
    }
    write_json(p, payload)
    return payload


def save_payload(arm: str, ck: str, payload: dict[str, Any]) -> None:
    target = target_name(arm, ck)
    payload["updated_utc"] = now()
    scores = {}
    for col in STABLE_COLUMNS:
        rec = (payload.get("tasks") or {}).get(col, {})
        if isinstance(rec, dict) and rec.get("returncode") == 0:
            val = rec.get("scores", {}).get("Reading") if col == "Reading" else rec.get("score")
            scores[col] = float(val) if val is not None and math.isfinite(float(val)) else None
        else:
            scores[col] = None
    payload["stable_scores"] = scores
    if all(scores.get(c) is not None for c in STABLE_COLUMNS):
        payload["cheap6"] = statistics.mean([scores[c] for c in STABLE_COLUMNS])
        payload["exEntity5"] = statistics.mean([scores[c] for c in ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]])
    write_json(per_target_path(target), payload)


def task_done(payload: dict[str, Any], col: str) -> bool:
    rec = (payload.get("tasks") or {}).get(col, {})
    if not isinstance(rec, dict) or rec.get("returncode") != 0:
        return False
    if col == "Reading":
        return isinstance(rec.get("scores"), dict) and "Reading" in rec["scores"]
    return rec.get("score") is not None and math.isfinite(float(rec["score"]))


def eval_zero_shot(arm: str, ck: str, col: str, gpu: int, timeout: int = 2400) -> dict[str, Any]:
    """Run a single zero-shot evaluation chunk."""
    cfg = ARM_CONFIGS[arm]
    model_path = cfg["run_dir"] / "hf_model" / ck
    target = target_name(arm, ck)
    
    payload = load_or_create_payload(arm, ck)
    if task_done(payload, col):
        score = payload["tasks"][col].get("score")
        print(f"  [SKIP] {arm} {ck} {col}: already scored = {score}", flush=True)
        return {"status": "skip", "arm": arm, "ck": ck, "col": col, "score": score}
    
    spec = ZERO_SHOT_SPECS[col]
    task_out = OUT_BASE / "outputs" / target / col
    log_path = OUT_BASE / "logs" / f"{target}_{col}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    revision = f"step005_{target}_{col}"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", str(spec["task"]),
        "--data_path", str(pathlib.Path(spec["data_path"]).resolve()),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(spec["batch_size"]),
        "--non_causal_batch_size", "64",
        "--output_dir", str(task_out.resolve()),
    ]
    
    env = gpu_env(arm, ck, gpu)
    print(f"  [RUN]  {arm} {ck} {col} on GPU {gpu}...", flush=True)
    t0 = time.time()
    rc = -999
    err = None
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "ck": ck, "col": col, "gpu": gpu}) + "\n")
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -124
        err = f"timeout after {timeout}s"
    elapsed = round(time.time() - t0, 1)
    
    # Parse score
    score = None
    report_files = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    if report_files:
        score = parse_sentence_score(report_files[-1].read_text(encoding="utf-8", errors="replace"))
    
    rec = {"column": col, "returncode": rc, "elapsed_sec": elapsed, "gpu": gpu, "score": score, "log": rel(log_path)}
    if err:
        rec["error"] = err
    
    payload.setdefault("tasks", {})[col] = rec
    save_payload(arm, ck, payload)
    
    status = "done" if score is not None else "failed"
    print(f"  [{status.upper():6s}] {arm} {ck} {col} = {score}  ({elapsed}s)", flush=True)
    return {"status": status, "arm": arm, "ck": ck, "col": col, "score": score, "elapsed": elapsed}


def eval_reading(arm: str, ck: str, gpu: int, timeout: int = 2400) -> dict[str, Any]:
    """Run Reading evaluation."""
    col = "Reading"
    cfg = ARM_CONFIGS[arm]
    model_path = cfg["run_dir"] / "hf_model" / ck
    target = target_name(arm, ck)
    
    payload = load_or_create_payload(arm, ck)
    if task_done(payload, col):
        score = payload["tasks"][col].get("scores", {}).get("Reading")
        print(f"  [SKIP] {arm} {ck} {col}: already scored = {score}", flush=True)
        return {"status": "skip", "arm": arm, "ck": ck, "col": col, "score": score}
    
    task_out = OUT_BASE / "outputs" / target / col
    log_path = OUT_BASE / "logs" / f"{target}_{col}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    revision = f"step005_{target}_Reading"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", str(READING_DATA.resolve()),
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    
    env = gpu_env(arm, ck, gpu)
    print(f"  [RUN]  {arm} {ck} {col} on GPU {gpu}...", flush=True)
    t0 = time.time()
    rc = -999
    err = None
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "ck": ck, "col": col, "gpu": gpu}) + "\n")
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -124
        err = f"timeout after {timeout}s"
    elapsed = round(time.time() - t0, 1)
    
    # Parse score
    scores = {}
    report_files = sorted(task_out.rglob("report.txt"), key=lambda p: p.stat().st_mtime)
    if report_files:
        scores = parse_reading_scores(report_files[-1].read_text(encoding="utf-8", errors="replace"))
    
    rec = {"column": col, "returncode": rc, "elapsed_sec": elapsed, "gpu": gpu, "scores": scores, "log": rel(log_path)}
    if err:
        rec["error"] = err
    
    payload.setdefault("tasks", {})[col] = rec
    save_payload(arm, ck, payload)
    
    reading_score = scores.get("Reading")
    status = "done" if reading_score is not None else "failed"
    print(f"  [{status.upper():6s}] {arm} {ck} {col} = {reading_score}  ({elapsed}s)", flush=True)
    return {"status": status, "arm": arm, "ck": ck, "col": col, "score": reading_score, "elapsed": elapsed}


def eval_one(arm: str, ck: str, col: str, gpu: int, timeout: int = 2400) -> dict[str, Any]:
    if col == "Reading":
        return eval_reading(arm, ck, gpu, timeout)
    return eval_zero_shot(arm, ck, col, gpu, timeout)


def run_arm_sequence(arm: str, gpu: int, checkpoints: list[str], columns: list[str], timeout: int) -> list[dict[str, Any]]:
    """Run all evaluations for one arm sequentially on its assigned GPU."""
    results = []
    for ck in checkpoints:
        for col in columns:
            try:
                r = eval_one(arm, ck, col, gpu, timeout)
                results.append(r)
            except Exception as exc:
                print(f"  [ERROR] {arm} {ck} {col}: {exc}", flush=True)
                results.append({"status": "error", "arm": arm, "ck": ck, "col": col, "error": str(exc)})
    return results


def compute_contrasts(checkpoints: list[str]) -> dict[str, Any]:
    """Compute V-C contrasts for seed43122 and compare with seed43022."""
    contrasts = {}
    for ck in checkpoints:
        v_target = target_name("D_V_43122", ck)
        c_target = target_name("D_C_43122", ck)
        v_path = per_target_path(v_target)
        c_path = per_target_path(c_target)
        
        if not v_path.exists() or not c_path.exists():
            continue
        
        v_data = read_json(v_path)
        c_data = read_json(c_path)
        v_scores = v_data.get("stable_scores", {})
        c_scores = c_data.get("stable_scores", {})
        
        ck_contrasts = {}
        for col in STABLE_COLUMNS:
            v = v_scores.get(col)
            c = c_scores.get(col)
            if v is not None and c is not None:
                ck_contrasts[col] = {"V": v, "C": c, "V_minus_C": round(v - c, 4)}
        
        contrasts[ck] = ck_contrasts
    return contrasts


# First-basin seed43022 V-C reference (from research contrast_rows.csv)
SEED43022_VC = {
    "chck_80M": {"BLiMP": -0.03, "Supplement": 2.30, "EWoK": 0.49, "Entity": 3.93, "COMPS": -0.49, "Reading": 0.52},
    "chck_90M": {"BLiMP": -0.09, "Supplement": 1.83, "EWoK": 0.03, "Entity": 3.24, "COMPS": -0.45, "Reading": 0.48},
    "chck_100M": {"BLiMP": -0.04, "Supplement": 1.67, "EWoK": -0.21, "Entity": 3.27, "COMPS": -0.70, "Reading": 0.47},
}


def print_comparison(contrasts: dict[str, Any]) -> str:
    """Print side-by-side V-C comparison across seeds."""
    lines = []
    lines.append("=" * 90)
    lines.append("V-C SEED REPLICATION: seed43022 (first basin) vs seed43122 (second basin)")
    lines.append("=" * 90)
    lines.append(f"{'Checkpoint':<12s} {'Column':<12s} {'s43022 V-C':>12s} {'s43122 V-C':>12s} {'Same sign?':>12s}")
    lines.append("-" * 60)
    
    sign_checks = {col: [] for col in STABLE_COLUMNS}
    
    for ck in CHECKPOINTS:
        ref = SEED43022_VC.get(ck, {})
        cur = contrasts.get(ck, {})
        for col in STABLE_COLUMNS:
            ref_val = ref.get(col)
            cur_rec = cur.get(col, {})
            cur_val = cur_rec.get("V_minus_C")
            
            ref_str = f"{ref_val:+.2f}" if ref_val is not None else "N/A"
            cur_str = f"{cur_val:+.2f}" if cur_val is not None else "N/A"
            
            same_sign = ""
            if ref_val is not None and cur_val is not None:
                if (ref_val > 0 and cur_val > 0) or (ref_val < 0 and cur_val < 0):
                    same_sign = "YES"
                elif abs(ref_val) < 0.15 or abs(cur_val) < 0.15:
                    same_sign = "~0"
                else:
                    same_sign = "**FLIP**"
                sign_checks[col].append(same_sign)
            
            lines.append(f"{ck:<12s} {col:<12s} {ref_str:>12s} {cur_str:>12s} {same_sign:>12s}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("SIGN STABILITY SUMMARY:")
    for col in STABLE_COLUMNS:
        checks = sign_checks[col]
        yes_count = sum(1 for c in checks if c == "YES")
        flip_count = sum(1 for c in checks if c == "**FLIP**")
        lines.append(f"  {col:<12s}: {yes_count}/3 same sign, {flip_count}/3 flips")
    
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["D_V_43122", "D_C_43122"])
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    ap.add_argument("--columns", nargs="+", default=STABLE_COLUMNS)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    
    # Verify model paths
    for arm in args.arms:
        cfg = ARM_CONFIGS[arm]
        for ck in args.checkpoints:
            model_path = cfg["run_dir"] / "hf_model" / ck
            assert model_path.exists(), f"Missing: {model_path}"
        print(f"[OK] {arm}: all checkpoints verified", flush=True)
    
    if args.plan_only:
        total = len(args.arms) * len(args.checkpoints) * len(args.columns)
        print(f"\nPlan: {total} evaluation chunks across {len(args.arms)} arms", flush=True)
        for arm in args.arms:
            cfg = ARM_CONFIGS[arm]
            print(f"  {arm} on GPU {cfg['gpu']}: {len(args.checkpoints)} × {len(args.columns)} = {len(args.checkpoints) * len(args.columns)} chunks", flush=True)
        print(json.dumps({"status": "PLAN_ONLY", "arms": args.arms, "checkpoints": args.checkpoints, "columns": args.columns, "total_chunks": total}, indent=2), flush=True)
        return
    
    # Run VIEW and CLEAN in parallel on different GPUs
    print(f"\n{'='*60}", flush=True)
    print(f"STARTING PARALLEL EVALUATION", flush=True)
    print(f"{'='*60}", flush=True)
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for arm in args.arms:
            cfg = ARM_CONFIGS[arm]
            gpu = cfg["gpu"]
            f = executor.submit(run_arm_sequence, arm, gpu, args.checkpoints, args.columns, args.timeout)
            futures[f] = arm
        
        for f in concurrent.futures.as_completed(futures):
            arm = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
                done = sum(1 for r in results if r.get("status") in ("done", "skip"))
                failed = sum(1 for r in results if r.get("status") in ("failed", "error"))
                print(f"\n[COMPLETE] {arm}: {done} done, {failed} failed", flush=True)
            except Exception as exc:
                print(f"\n[ERROR] {arm}: {exc}", flush=True)
    
    # Compute and print contrasts
    print(f"\n{'='*60}", flush=True)
    contrasts = compute_contrasts(args.checkpoints)
    comparison_text = print_comparison(contrasts)
    
    # Save summary
    summary = {
        "status": "VC_SEED_REPLICATION_DONE",
        "finished_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "columns": args.columns,
        "contrasts_seed43122": contrasts,
        "reference_seed43022": SEED43022_VC,
        "results": all_results,
        "scientific_purpose": "Test whether V-C benchmark movement replicates across DeBERTa seeds. Pre-stated decision rule in notes/005_vc_seed_replication_prestate.md.",
        "no_training_upload_leaderboard": True,
    }
    summary_path = OUT_BASE / "vc_seed_replication_summary.json"
    write_json(summary_path, summary)
    
    # Save comparison as markdown
    note_path = WS / "notes" / "vc_seed_replication_result.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(f"# research: V-C Seed Replication Result\n\n```\n{comparison_text}\n```\n\n## Raw contrasts\n\n```json\n{json.dumps(contrasts, indent=2)}\n```\n", encoding="utf-8")
    
    print(f"\nSaved: {rel(summary_path)}", flush=True)
    print(f"Saved: {rel(note_path)}", flush=True)
    print(json.dumps({"status": summary["status"], "summary": rel(summary_path), "note": rel(note_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
