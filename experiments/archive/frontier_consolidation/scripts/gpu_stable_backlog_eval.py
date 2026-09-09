#!/usr/bin/env python3
"""research: GPU-score unread stable-family backlogs.

This script extends the validated research narrow official scorer from Entity/EWoK
chunks to the stable BabyLM families needed for the current fixed-budget
allocation study.  It never runs GlobalPIQA, SuperGLUE, AoA, upload, packaging,
or leaderboard code.

Immediate scientific purpose
----------------------------
* DeBERTa MAX breadth has ten trained checkpoints but only late Entity scores.
  Scoring BLiMP, Supplement, EWoK, COMPS, and Reading lets us read whether
  compact re-expression beats same-population sentence breadth outside Entity:
      V-R = (V-B) + (B-R)
  can then be tested on broad ex-Entity families rather than only on the
  operation-skewed Entity surface.
* RoBERTa MAX clean has the same 653,130-row geometry as RoBERTa MAX view.
  Scoring the missing stable families gives a cross-architecture, geometry-matched
  view-minus-clean total-effect leg.
* First-basin DeBERTa MAX repeat 100M lacks COMPS/Reading after earlier CPU
  timeouts.  Filling those two columns completes late broad comparisons involving
  repeat.

Use --plan-only first.  The default --preset backlog_decision is ordered so the
most interpretation-changing chunks are written early if the long task is later
interrupted.
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
    "first_basin_max_repeat": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "dose_max_repeat",
        "out_root": WS / "data" / "dose_ladder_stable_eval" / "eval",
        "description": "First-basin DeBERTa MAX repeat arm; fills late broad-family columns for V-B/B-R/V-R comparisons.",
        "family": "first_basin_deberta_max_repeat_seed43022",
        "architecture": "deberta",
        "data_arm": "repeat",
        "seed": 43022,
    },
    "max_breadth": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "max_breadth_seed43022",
        "out_root": WS / "data" / "breadth_entity_ewok_eval" / "eval",
        "description": "First-basin DeBERTa MAX same-population breadth arm; tests compact re-expression specificity beyond Entity.",
        "family": "deberta_max_breadth_seed43022",
        "architecture": "deberta",
        "data_arm": "breadth",
        "seed": 43022,
    },
    "roberta_max_clean": {
        "run_dir": WS / "training" / "runs" / "roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022",
        "target_prefix": "roberta_viewclean_total_clean",
        "out_root": WS / "data" / "roberta_viewclean_total_stable_eval" / "eval",
        "description": "RoBERTa MAX-geometry clean arm; cross-architecture geometry-matched total view-minus-clean reference.",
        "family": "roberta_max_clean_seed43022",
        "architecture": "roberta",
        "data_arm": "clean",
        "seed": 43022,
    },
    "deberta_maxgeom_clean_seed43022": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "deberta_maxgeom_clean_seed43022",
        "out_root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval",
        "description": "First-basin DeBERTa MAX-geometry clean arm; future V-C reference after training completes.",
        "family": "deberta_maxgeom_clean_seed43022",
        "architecture": "deberta",
        "data_arm": "clean",
        "seed": 43022,
    },
    "deberta_maxgeom_clean_seed43122": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "deberta_maxgeom_clean_seed43122",
        "out_root": WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval",
        "description": "Second-basin DeBERTa MAX-geometry clean arm; future V-C reference after training completes.",
        "family": "deberta_maxgeom_clean_seed43122",
        "architecture": "deberta",
        "data_arm": "clean",
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
        raise ValueError(f"Bad checkpoint name: {ck}")
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


def per_target_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / "per_target" / f"{target}.json"


def load_or_new_payload(target: str, cfg: dict[str, Any], checkpoint: str, force_new: bool = False) -> dict[str, Any]:
    out_root = pathlib.Path(cfg["out_root"])
    p = per_target_path(out_root, target)
    if p.exists() and not force_new:
        payload = read_json(p)
        payload.setdefault("target", target)
        payload.setdefault("tasks", {})
        return payload
    run_dir = pathlib.Path(cfg["run_dir"])
    model_root = run_dir / "hf_model"
    model_path = model_root / checkpoint
    payload: dict[str, Any] = {
        "target": target,
        "description": cfg["description"],
        "family": cfg["family"],
        "architecture": cfg["architecture"],
        "data_arm": cfg["data_arm"],
        "seed": cfg["seed"],
        "run_dir": rel(run_dir),
        "model_root": rel(model_root),
        "model_path": rel(model_path),
        "endpoint": checkpoint,
        "started_utc": now(),
        "gpu": "visible_gpu_stable_chunk",
        "tasks": {},
        "hardware_reason": "These are already-trained weights; GPU scoring is used only for stable BabyLM families after CPU official scoring timed out on related chunks.",
    }
    metrics = run_dir / "scientific_metrics.json"
    if metrics.exists():
        try:
            m = read_json(metrics)
            keys = ["variant", "word_exposure", "actual_training_steps", "parameter_count", "vocab_size", "tokenizer_label", "loss_first", "loss_last", "seed", "extra_init_seed", "train_rng_seed", "seq_length", "batch_size", "learning_rate", "n_layer", "hidden_size", "n_head", "intermediate_size"]
            payload["run_summary"] = {k: m.get(k) for k in keys if k in m}
            payload["run_summary"]["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", []) if isinstance(x, dict)]
        except Exception as exc:
            payload["run_summary_error"] = repr(exc)
    write_json(p, payload)
    return payload


def save_payload(out_root: pathlib.Path, target: str, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = now()
    payload["stable_family_scores"] = stable_scores(payload)
    payload["no_globalpiqa_superglue_aoa_upload_or_leaderboard"] = True
    write_json(per_target_path(out_root, target), payload)


def gpu_env(out_root: pathlib.Path, target: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NLTK_DATA"] = str(NLP_DATA_ROOT.resolve())
    cache = out_root / "hf_cache_gpu_stable" / target
    tmp = out_root / "tmp_gpu_stable" / target
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_zero_shot(arm: str, checkpoint: str, column: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    cfg = ARM_CONFIGS[arm]
    run_dir = pathlib.Path(cfg["run_dir"])
    model_path = run_dir / "hf_model" / checkpoint
    metrics = run_dir / "scientific_metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metrics.exists():
        raise FileNotFoundError(metrics)
    out_root = pathlib.Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    target = target_name(cfg, checkpoint)
    payload = load_or_new_payload(target, cfg, checkpoint)
    if task_done(payload, column) and not force:
        return {"status": "skip_existing", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": score_value(payload["tasks"][column], column), "per_target": rel(per_target_path(out_root, target))}

    spec = ZERO_SHOT_SPECS[column]
    task_out = out_root / "official_outputs" / target / column
    log = out_root / "logs" / target / f"gpu_stable_{column}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    revision = f"full_{target}_{column}"
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
    env = gpu_env(out_root, target, gpu)
    header = {"event": "gpu_stable_zero_shot_start", "created_utc": now(), "arm": arm, "checkpoint": checkpoint, "column": column, "gpu": gpu, "target": target, "cmd": argv, "scientific_role": cfg["description"], "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    t0 = time.time()
    rc = -999
    err: str | None = None
    try:
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        rc = -124
        err = f"timeout after {timeout}s: {exc}"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "gpu_stable_zero_shot_timeout", "timeout_sec": timeout, "error": err}, ensure_ascii=False) + "\n")
    elapsed = round(time.time() - t0, 3)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_stable_zero_shot_finished", "returncode": rc, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
    score = read_report_score(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    rec: dict[str, Any] = {
        "column": column,
        "task": spec["task"],
        "data_path": rel(pathlib.Path(spec["data_path"])),
        "revision_name": revision,
        "output_dir": rel(task_out),
        "returncode": rc,
        "elapsed_sec": elapsed,
        "log": rel(log),
        "gpu": gpu,
        "score": score,
    }
    if pred:
        rec["predictions"] = rel(pred)
    if report:
        rec["report"] = rel(report)
    if err:
        rec["error"] = err
    elif rc != 0:
        rec["error"] = f"zero-shot {column} failed rc={rc}"
    elif score is None:
        rec["error"] = f"zero-shot {column} produced no parseable score"
    payload.setdefault("tasks", {})[column] = rec
    save_payload(out_root, target, payload)
    result = {"status": "chunk_done" if not rec.get("error") else "chunk_failed", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": score, "returncode": rc, "elapsed_sec": elapsed, "per_target": rel(per_target_path(out_root, target)), "predictions": rec.get("predictions"), "report": rec.get("report"), "log": rel(log)}
    write_json(out_root / "gpu_stable_chunk_results" / f"{target}_{column}.json", result)
    return result


def run_reading(arm: str, checkpoint: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    column = "Reading"
    cfg = ARM_CONFIGS[arm]
    run_dir = pathlib.Path(cfg["run_dir"])
    model_path = run_dir / "hf_model" / checkpoint
    metrics = run_dir / "scientific_metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not metrics.exists():
        raise FileNotFoundError(metrics)
    out_root = pathlib.Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    target = target_name(cfg, checkpoint)
    payload = load_or_new_payload(target, cfg, checkpoint)
    if task_done(payload, column) and not force:
        return {"status": "skip_existing", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": score_value(payload["tasks"][column], column), "per_target": rel(per_target_path(out_root, target))}

    task_out = out_root / "official_outputs" / target / column
    log = out_root / "logs" / target / "gpu_stable_Reading.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    revision = f"full_{target}_Reading"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", str(READING_DATA.resolve()),
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    env = gpu_env(out_root, target, gpu)
    header = {"event": "gpu_stable_reading_start", "created_utc": now(), "arm": arm, "checkpoint": checkpoint, "column": column, "gpu": gpu, "target": target, "cmd": argv, "scientific_role": cfg["description"], "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    t0 = time.time()
    rc = -999
    err: str | None = None
    try:
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        rc = -124
        err = f"timeout after {timeout}s: {exc}"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "gpu_stable_reading_timeout", "timeout_sec": timeout, "error": err}, ensure_ascii=False) + "\n")
    elapsed = round(time.time() - t0, 3)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_stable_reading_finished", "returncode": rc, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
    scores = read_reading_report(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "report.txt")
    rec: dict[str, Any] = {"column": column, "data_path": rel(READING_DATA), "revision_name": revision, "output_dir": rel(task_out), "returncode": rc, "elapsed_sec": elapsed, "log": rel(log), "gpu": gpu, "scores": scores, "score": scores.get("Reading") if scores else None}
    if pred:
        rec["predictions"] = rel(pred)
    if report:
        rec["report"] = rel(report)
    if err:
        rec["error"] = err
    elif rc != 0:
        rec["error"] = f"Reading failed rc={rc}"
    elif "Reading" not in scores:
        rec["error"] = "Reading produced no parseable combined score"
    payload.setdefault("tasks", {})[column] = rec
    save_payload(out_root, target, payload)
    result = {"status": "chunk_done" if not rec.get("error") else "chunk_failed", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": rec.get("score"), "returncode": rc, "elapsed_sec": elapsed, "per_target": rel(per_target_path(out_root, target)), "predictions": rec.get("predictions"), "report": rec.get("report"), "log": rel(log)}
    write_json(out_root / "gpu_stable_chunk_results" / f"{target}_{column}.json", result)
    return result


def run_chunk(arm: str, checkpoint: str, column: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm}; choices {sorted(ARM_CONFIGS)}")
    if column in ZERO_SHOT_SPECS:
        return run_zero_shot(arm, checkpoint, column, gpu, force, timeout)
    if column == "Reading":
        return run_reading(arm, checkpoint, gpu, force, timeout)
    raise ValueError(f"unknown or prohibited column {column}")


def all_cks() -> list[str]:
    return [f"chck_{i}M" for i in range(10, 101, 10)]


def stable_chunks_for(arm: str, checkpoints: list[str], columns: list[str]) -> list[tuple[str, str, str]]:
    return [(arm, ck, col) for ck in checkpoints for col in columns]


def preset_chunks(name: str) -> list[tuple[str, str, str]]:
    if name == "backlog_decision":
        chunks: list[tuple[str, str, str]] = []
        # Complete the 100M repeat anchor first.
        chunks += stable_chunks_for("first_basin_max_repeat", ["chck_100M"], ["COMPS", "Reading"])
        # Breadth late first, then the earlier common-window breadth rows.  Entity
        # was already scored at 80/90/100 and is deliberately not requested here.
        chunks += stable_chunks_for("max_breadth", ["chck_80M", "chck_90M", "chck_100M"], EX_ENTITY_COLUMNS)
        chunks += stable_chunks_for("max_breadth", [f"chck_{i}M" for i in range(10, 80, 10)], EX_ENTITY_COLUMNS)
        # RoBERTa clean late first, then 50M-70M, then already-scored early rows
        # are included only so skip_existing records keep the plan self-contained.
        chunks += stable_chunks_for("roberta_max_clean", ["chck_80M", "chck_90M", "chck_100M"], STABLE_COLUMNS)
        chunks += stable_chunks_for("roberta_max_clean", ["chck_50M", "chck_60M", "chck_70M"], STABLE_COLUMNS)
        chunks += stable_chunks_for("roberta_max_clean", ["chck_10M", "chck_20M", "chck_30M", "chck_40M"], STABLE_COLUMNS)
        return chunks
    if name == "breadth_exentity_all":
        return stable_chunks_for("max_breadth", all_cks(), EX_ENTITY_COLUMNS)
    if name == "roberta_clean_all":
        return stable_chunks_for("roberta_max_clean", all_cks(), STABLE_COLUMNS)
    if name == "deberta_clean_seed43022_all":
        return stable_chunks_for("deberta_maxgeom_clean_seed43022", all_cks(), STABLE_COLUMNS)
    if name == "deberta_clean_seed43122_all":
        return stable_chunks_for("deberta_maxgeom_clean_seed43122", all_cks(), STABLE_COLUMNS)
    if name == "deberta_clean_seed43022_decision":
        return stable_chunks_for("deberta_maxgeom_clean_seed43022", ["chck_80M", "chck_90M", "chck_100M"], STABLE_COLUMNS) + stable_chunks_for("deberta_maxgeom_clean_seed43022", [f"chck_{i}M" for i in range(10, 80, 10)], STABLE_COLUMNS)
    if name == "deberta_clean_seed43122_decision":
        return stable_chunks_for("deberta_maxgeom_clean_seed43122", ["chck_80M", "chck_90M", "chck_100M"], STABLE_COLUMNS) + stable_chunks_for("deberta_maxgeom_clean_seed43122", [f"chck_{i}M" for i in range(10, 80, 10)], STABLE_COLUMNS)
    raise ValueError(f"unknown preset {name}; choices backlog_decision, breadth_exentity_all, roberta_clean_all, deberta_clean_seed43022_all, deberta_clean_seed43122_all, deberta_clean_seed43022_decision, deberta_clean_seed43122_decision")


def parse_chunk_specs(vals: list[str] | None, preset: str | None) -> list[tuple[str, str, str]]:
    chunks: list[tuple[str, str, str]] = []
    if preset:
        chunks.extend(preset_chunks(preset))
    if vals:
        for v in vals:
            parts = v.split(":")
            if len(parts) != 3:
                raise ValueError(f"bad chunk spec {v!r}; expected arm:chck_80M:Entity")
            chunks.append((parts[0], parts[1], parts[2]))
    # Stable de-duplication preserving order.
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in chunks:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def plan_rows(chunks: list[tuple[str, str, str]], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        cfg = ARM_CONFIGS[arm]
        target = target_name(cfg, ck)
        out_root = pathlib.Path(cfg["out_root"])
        p = per_target_path(out_root, target)
        existing = read_json(p) if p.exists() else {"tasks": {}}
        rows.append({
            "arm": arm,
            "checkpoint": ck,
            "words": ck_words(ck),
            "column": col,
            "target": target,
            "architecture": cfg["architecture"],
            "data_arm": cfg["data_arm"],
            "run_dir": rel(pathlib.Path(cfg["run_dir"])),
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
    ap.add_argument("--preset", default=None, help="Preset chunk set. Main choice: backlog_decision.")
    ap.add_argument("--chunks", nargs="*", default=None, help="Additional chunk specs as arm:chck_80M:Entity")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0, help="Optional cap for a pilot run after skip filtering; 0 means no cap.")
    ap.add_argument("--continue-on-error", action="store_true", help="Record failed chunks and continue to later chunks.")
    ap.add_argument("--summary-dir", default=str(WS / "data" / "gpu_stable_backlog_eval"))
    args = ap.parse_args()

    chunks = parse_chunk_specs(args.chunks, args.preset)
    rows = plan_rows(chunks, args.force)
    will = [r for r in rows if r["will_run"]]
    if args.max_chunks and args.max_chunks > 0:
        allowed = {(r["arm"], r["checkpoint"], r["column"]) for r in will[: args.max_chunks]}
        chunks = [c for c in chunks if c in allowed or not next((r for r in rows if (r["arm"], r["checkpoint"], r["column"]) == c), {"will_run": False})["will_run"]]
        rows = plan_rows(chunks, args.force)
        will = [r for r in rows if r["will_run"]]
    plan = {
        "status": "GPU_STABLE_BACKLOG_PLAN" if args.plan_only else "GPU_STABLE_BACKLOG_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "preset": args.preset,
        "requested_chunk_count": len(rows),
        "will_run_count": len(will),
        "chunks": rows,
        "scientific_purpose": "Score already-trained stable-family checkpoints needed to read RoBERTa geometry-matched V-C and DeBERTa breadth V-B/B-R outside Entity while one H100 trains the first MAX-geometry clean DeBERTa control.",
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
            results.append(err)
            failures.append(err)
            print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                break
    final = {**plan, "status": "GPU_STABLE_BACKLOG_DONE" if not failures else "GPU_STABLE_BACKLOG_FINISHED_WITH_FAILURES", "finished_utc": now(), "result_count": len(results), "failed_count": len(failures), "results": results}
    summary_dir = pathlib.Path(args.summary_dir)
    if not summary_dir.is_absolute():
        summary_dir = ROOT / summary_dir
    out = summary_dir / f"stable_backlog_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "failed_count": len(failures), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
