#!/usr/bin/env python3
"""research: narrow GPU common-window scoring for DeBERTa MAX-geometry clean anchors.

Scientific purpose
------------------
The sub-dose V-C curve uses rho=0 clean as a load-bearing reference.  The old
monolithic clean scorer has been running for many steps and only targets one
basin.  This script uses the already validated official-compatible GPU chunking
style to score both trained clean basins on the stable BabyLM families across a
common trajectory window (default chck_10M..chck_80M).  It writes to an
independent research output tree to avoid racing older research/275 scorers.

It never trains and never runs GlobalPIQA, SuperGLUE, AoA, packaging, upload, or
leaderboard code.
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
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
PRISTINE_FULL = STRICT / "evaluation_data" / "full_eval"
NLP_DATA_ROOT = ROOT / "experiments/archive" / 'initial_model_studies' / "data" / "nltk_data"
OUT_ROOT = WS / "data" / "clean_anchor_commonwindow_eval" / "eval"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
EX_ENTITY_COLUMNS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
ZERO_SHOT_SPECS: dict[str, dict[str, Any]] = {
    "BLiMP": {"task": "blimp", "data_path": PRISTINE_FULL / "blimp_filtered", "batch_size": 128},
    "Supplement": {"task": "blimp", "data_path": PRISTINE_FULL / "supplement_filtered", "batch_size": 128},
    "EWoK": {"task": "ewok", "data_path": PRISTINE_FULL / "ewok_filtered", "batch_size": 64},
    "Entity": {"task": "entity_tracking", "data_path": PRISTINE_FULL / "entity_tracking", "batch_size": 128},
    "COMPS": {"task": "comps", "data_path": PRISTINE_FULL / "comps", "batch_size": 128},
}
READING_DATA = PRISTINE_FULL / "reading" / "reading_data.csv"

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "clean_seed43022": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "deberta_maxgeom_clean_seed43022",
        "family": "deberta_maxgeom_clean_seed43022",
        "seed": 43022,
    },
    "clean_seed43122": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "deberta_maxgeom_clean_seed43122",
        "family": "deberta_maxgeom_clean_seed43122",
        "seed": 43122,
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def ck_words(ck: str) -> int:
    m = re.match(r"chck_(\d+)M$", ck)
    if not m:
        raise ValueError(f"Bad checkpoint {ck!r}")
    return int(m.group(1)) * 1_000_000


def latest_file(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


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


def read_report_score(task_out: pathlib.Path) -> float | None:
    p = latest_file(task_out, "best_temperature_report.txt")
    if p is None:
        return None
    return parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))


def parse_reading_scores(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_reading_report(task_out: pathlib.Path) -> dict[str, float]:
    p = latest_file(task_out, "report.txt")
    if p is None:
        return {}
    return parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))


def score_value(rec: dict[str, Any], col: str) -> Any:
    if col == "Reading":
        scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        return scores.get("Reading") if scores else rec.get("score")
    return rec.get("score")


def task_done(payload: dict[str, Any], col: str) -> bool:
    rec = (payload.get("tasks") or {}).get(col)
    return isinstance(rec, dict) and rec.get("returncode") == 0 and finite(score_value(rec, col))


def ready_cols(payload: dict[str, Any]) -> list[str]:
    return [c for c in STABLE_COLUMNS if task_done(payload, c)]


def stable_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    out: dict[str, float | None] = {}
    for col in STABLE_COLUMNS:
        rec = tasks.get(col) or {}
        val = score_value(rec, col) if isinstance(rec, dict) else None
        out[col] = float(val) if finite(val) else None
    if all(finite(out.get(c)) for c in STABLE_COLUMNS):
        out["cheap6_no_GlobalPIQA"] = statistics.mean([float(out[c]) for c in STABLE_COLUMNS])
    else:
        out["cheap6_no_GlobalPIQA"] = None
    if all(finite(out.get(c)) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]):
        out["cheap5_no_GlobalPIQA_Reading"] = statistics.mean([float(out[c]) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]])
    else:
        out["cheap5_no_GlobalPIQA_Reading"] = None
    if all(finite(out.get(c)) for c in EX_ENTITY_COLUMNS):
        out["exEntity5_BLiMP_Supp_EWoK_COMPS_Reading"] = statistics.mean([float(out[c]) for c in EX_ENTITY_COLUMNS])
    else:
        out["exEntity5_BLiMP_Supp_EWoK_COMPS_Reading"] = None
    if finite(out.get("EWoK")) and finite(out.get("Entity")):
        out["EWoK_plus_Entity_sum"] = float(out["EWoK"]) + float(out["Entity"])
    else:
        out["EWoK_plus_Entity_sum"] = None
    return out


def target_name(cfg: dict[str, Any], ck: str) -> str:
    return f"{cfg['target_prefix']}_{ck}"


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


def load_or_new_payload(arm: str, checkpoint: str) -> dict[str, Any]:
    cfg = ARM_CONFIGS[arm]
    target = target_name(cfg, checkpoint)
    p = per_target_path(target)
    if p.exists():
        payload = read_json(p)
        payload.setdefault("tasks", {})
        return payload
    run_dir = pathlib.Path(cfg["run_dir"])
    model_path = run_dir / "hf_model" / checkpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload: dict[str, Any] = {
        "target": target,
        "description": "research independent common-window clean anchor for sub-dose V-C curve; official-compatible stable-family chunks only.",
        "family": cfg["family"],
        "architecture": "deberta",
        "data_arm": "clean",
        "seed": cfg["seed"],
        "run_dir": rel(run_dir),
        "model_path": rel(model_path),
        "endpoint": checkpoint,
        "started_utc": now(),
        "tasks": {},
        "gpu": "visible_gpu_clean_anchor_chunk",
        "hardware_reason": "Already-trained weights; narrow GPU scoring replaces a long monolithic dependency and gives a two-basin rho=0 anchor for the sub-dose admission curve.",
    }
    mf = run_dir / "scientific_metrics.json"
    if mf.exists():
        m = read_json(mf)
        payload["run_summary"] = {k: m.get(k) for k in ["word_exposure", "actual_training_steps", "parameter_count", "vocab_size", "loss_first", "loss_last", "seed", "extra_init_seed", "train_rng_seed"] if k in m}
        payload["run_summary"]["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", []) if isinstance(x, dict)]
    write_json(p, payload)
    return payload


def save_payload(target: str, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = now()
    payload["stable_family_scores"] = stable_scores(payload)
    payload["no_globalpiqa_superglue_aoa_upload_or_leaderboard"] = True
    write_json(per_target_path(target), payload)


def gpu_env(target: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = OUT_ROOT / "hf_cache_gpu_clean_anchor" / target
    tmp = OUT_ROOT / "tmp_gpu_clean_anchor" / target
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_zero_shot(arm: str, ck: str, col: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    cfg = ARM_CONFIGS[arm]
    target = target_name(cfg, ck)
    payload = load_or_new_payload(arm, ck)
    if task_done(payload, col) and not force:
        return {"status": "skip_existing", "arm": arm, "checkpoint": ck, "column": col, "target": target, "score": score_value(payload["tasks"][col], col), "per_target": rel(per_target_path(target))}
    spec = ZERO_SHOT_SPECS[col]
    run_dir = pathlib.Path(cfg["run_dir"])
    model_path = run_dir / "hf_model" / ck
    task_out = OUT_ROOT / "official_outputs" / target / col
    log = OUT_ROOT / "logs" / target / f"gpu_clean_anchor_{col}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    revision = f"clean_anchor_{target}_{col}"
    argv = [sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", str(model_path.resolve()), "--backend", "mlm", "--task", str(spec["task"]), "--data_path", str(pathlib.Path(spec["data_path"]).resolve()), "--revision_name", revision, "--save_predictions", "--batch_size", str(spec["batch_size"]), "--non_causal_batch_size", "64", "--output_dir", str(task_out.resolve())]
    header = {"event": "gpu_clean_anchor_zero_shot_start", "created_utc": now(), "arm": arm, "checkpoint": ck, "column": col, "gpu": gpu, "target": target, "cmd": argv, "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    t0 = time.time(); rc = -999; err = None
    try:
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=gpu_env(target, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        rc = -124; err = f"timeout after {timeout}s: {exc}"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "gpu_clean_anchor_zero_shot_timeout", "timeout_sec": timeout, "error": err}, ensure_ascii=False) + "\n")
    elapsed = round(time.time() - t0, 3)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_clean_anchor_zero_shot_finished", "returncode": rc, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
    score = read_report_score(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    rec: dict[str, Any] = {"column": col, "task": spec["task"], "data_path": rel(pathlib.Path(spec["data_path"])), "revision_name": revision, "output_dir": rel(task_out), "returncode": rc, "elapsed_sec": elapsed, "log": rel(log), "gpu": gpu, "score": score}
    if pred: rec["predictions"] = rel(pred)
    if report: rec["report"] = rel(report)
    if err: rec["error"] = err
    elif rc != 0: rec["error"] = f"zero-shot {col} failed rc={rc}"
    elif score is None: rec["error"] = f"zero-shot {col} produced no parseable score"
    payload.setdefault("tasks", {})[col] = rec
    save_payload(target, payload)
    result = {"status": "chunk_done" if not rec.get("error") else "chunk_failed", "arm": arm, "checkpoint": ck, "column": col, "target": target, "score": score, "returncode": rc, "elapsed_sec": elapsed, "per_target": rel(per_target_path(target)), "predictions": rec.get("predictions"), "report": rec.get("report"), "log": rel(log)}
    write_json(OUT_ROOT / "gpu_clean_anchor_chunk_results" / f"{target}_{col}.json", result)
    return result


def run_reading(arm: str, ck: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    col = "Reading"
    cfg = ARM_CONFIGS[arm]
    target = target_name(cfg, ck)
    payload = load_or_new_payload(arm, ck)
    if task_done(payload, col) and not force:
        return {"status": "skip_existing", "arm": arm, "checkpoint": ck, "column": col, "target": target, "score": score_value(payload["tasks"][col], col), "per_target": rel(per_target_path(target))}
    run_dir = pathlib.Path(cfg["run_dir"])
    model_path = run_dir / "hf_model" / ck
    task_out = OUT_ROOT / "official_outputs" / target / col
    log = OUT_ROOT / "logs" / target / "gpu_clean_anchor_Reading.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    revision = f"clean_anchor_{target}_Reading"
    argv = [sys.executable, "-B", "-m", "evaluation_pipeline.reading.run", "--model_path_or_name", str(model_path.resolve()), "--backend", "mlm", "--data_path", str(READING_DATA.resolve()), "--revision_name", revision, "--output_dir", str(task_out.resolve())]
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_clean_anchor_reading_start", "created_utc": now(), "arm": arm, "checkpoint": ck, "column": col, "gpu": gpu, "target": target, "cmd": argv, "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True}, ensure_ascii=False) + "\n")
    t0 = time.time(); rc = -999; err = None
    try:
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=gpu_env(target, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        rc = -124; err = f"timeout after {timeout}s: {exc}"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "gpu_clean_anchor_reading_timeout", "timeout_sec": timeout, "error": err}, ensure_ascii=False) + "\n")
    elapsed = round(time.time() - t0, 3)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_clean_anchor_reading_finished", "returncode": rc, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
    scores = read_reading_report(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "report.txt")
    rec: dict[str, Any] = {"column": col, "data_path": rel(READING_DATA), "revision_name": revision, "output_dir": rel(task_out), "returncode": rc, "elapsed_sec": elapsed, "log": rel(log), "gpu": gpu, "scores": scores, "score": scores.get("Reading") if scores else None}
    if pred: rec["predictions"] = rel(pred)
    if report: rec["report"] = rel(report)
    if err: rec["error"] = err
    elif rc != 0: rec["error"] = f"Reading failed rc={rc}"
    elif "Reading" not in scores: rec["error"] = "Reading produced no parseable combined score"
    payload.setdefault("tasks", {})[col] = rec
    save_payload(target, payload)
    result = {"status": "chunk_done" if not rec.get("error") else "chunk_failed", "arm": arm, "checkpoint": ck, "column": col, "target": target, "score": rec.get("score"), "returncode": rc, "elapsed_sec": elapsed, "per_target": rel(per_target_path(target)), "predictions": rec.get("predictions"), "report": rec.get("report"), "log": rel(log)}
    write_json(OUT_ROOT / "gpu_clean_anchor_chunk_results" / f"{target}_{col}.json", result)
    return result


def run_chunk(arm: str, ck: str, col: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm}; choices {sorted(ARM_CONFIGS)}")
    if col in ZERO_SHOT_SPECS:
        return run_zero_shot(arm, ck, col, gpu, force, timeout)
    if col == "Reading":
        return run_reading(arm, ck, gpu, force, timeout)
    raise ValueError(f"unknown or prohibited column {col}")


def parse_chunk_specs(vals: list[str] | None, arms: list[str], checkpoints: list[str], columns: list[str]) -> list[tuple[str, str, str]]:
    chunks: list[tuple[str, str, str]] = []
    for arm in arms:
        for ck in checkpoints:
            for col in columns:
                chunks.append((arm, ck, col))
    if vals:
        for v in vals:
            parts = v.split(":")
            if len(parts) != 3:
                raise ValueError(f"bad chunk spec {v!r}; expected arm:chck_80M:Entity")
            chunks.append((parts[0], parts[1], parts[2]))
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in chunks:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def plan_rows(chunks: list[tuple[str, str, str]], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        cfg = ARM_CONFIGS[arm]
        target = target_name(cfg, ck)
        p = per_target_path(target)
        existing = read_json(p) if p.exists() else {"tasks": {}}
        rows.append({
            "arm": arm,
            "checkpoint": ck,
            "words": ck_words(ck),
            "column": col,
            "target": target,
            "seed": cfg["seed"],
            "model_path": rel(pathlib.Path(cfg["run_dir"]) / "hf_model" / ck),
            "model_exists": (pathlib.Path(cfg["run_dir"]) / "hf_model" / ck).exists(),
            "metrics_exists": (pathlib.Path(cfg["run_dir"]) / "scientific_metrics.json").exists(),
            "per_target": rel(p),
            "existing_ready_cols": ready_cols(existing),
            "will_run": bool(force or not task_done(existing, col)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["clean_seed43022", "clean_seed43122"])
    ap.add_argument("--checkpoints", nargs="*", default=[f"chck_{i}M" for i in range(10, 81, 10)])
    ap.add_argument("--columns", nargs="*", default=STABLE_COLUMNS)
    ap.add_argument("--chunks", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    chunks = parse_chunk_specs(args.chunks, args.arms, args.checkpoints, args.columns)
    rows = plan_rows(chunks, args.force)
    will = [r for r in rows if r["will_run"]]
    if args.max_chunks and args.max_chunks > 0:
        allowed = {(r["arm"], r["checkpoint"], r["column"]) for r in will[:args.max_chunks]}
        chunks = [c for c in chunks if c in allowed or not next((r for r in rows if (r["arm"], r["checkpoint"], r["column"]) == c), {"will_run": False})["will_run"]]
        rows = plan_rows(chunks, args.force)
        will = [r for r in rows if r["will_run"]]
    plan = {
        "status": "CLEAN_ANCHOR_COMMONWINDOW_PLAN" if args.plan_only else "CLEAN_ANCHOR_COMMONWINDOW_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "requested_chunk_count": len(rows),
        "will_run_count": len(will),
        "chunks": rows,
        "scientific_purpose": "Score stable-family common-window clean anchors for both MAX-geometry clean basins so sub-dose V-C can be read against a measured rho=0 spread rather than one unresolved monolithic job.",
        "no_training_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        try:
            res = run_chunk(arm, ck, col, args.gpu, args.force, args.timeout_sec)
            results.append(res)
            if res.get("status") == "chunk_failed":
                failures.append(res)
                if not args.continue_on_error:
                    break
            print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
        except Exception as exc:
            err = {"status": "chunk_exception", "arm": arm, "checkpoint": ck, "column": col, "error": repr(exc)}
            results.append(err); failures.append(err)
            print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                break
    final = {**plan, "status": "CLEAN_ANCHOR_COMMONWINDOW_DONE" if not failures else "CLEAN_ANCHOR_COMMONWINDOW_FINISHED_WITH_FAILURES", "finished_utc": now(), "result_count": len(results), "failed_count": len(failures), "results": results}
    out = OUT_ROOT.parent / f"clean_anchor_commonwindow_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "failed_count": len(failures), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
