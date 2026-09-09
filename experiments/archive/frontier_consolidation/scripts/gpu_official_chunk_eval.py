#!/usr/bin/env python3
"""research: narrow GPU official-compatible scoring chunks for stalled Entity/EWoK readouts.

CPU-safe full-arm scoring timed out on Entity and even EWoK for the second-basin
full ladder. This script deliberately uses an H100 for a small number of
predeclared official-compatible zero-shot chunks, never training and never
running GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard actions.

Scientific purpose:
  * first_basin_max_repeat_100M Entity repairs the missing late MAX repeat score
    needed for full-exposure dose decomposition and operation splits.
  * second_basin_{view,repeat}_{80,90,100M} Entity tests the frozen prediction
    that the first-basin +Entity carrier is a basin-specific propensity shift and
    should not reproduce as a large positive V-R in basin 2.
  * optional EWoK chunks read whether world-knowledge-like near-chance movement
    accompanies or opposes Entity in the same late window.

The worker wraps the official BabyLM sentence_zero_shot.run entry point with the
same fixed model paths and output JSON format used by research. It only changes
hardware from CPU to a chosen visible GPU because CPU throughput was empirically
insufficient for the decision window.
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

ZERO_SHOT_SPECS: dict[str, dict[str, Any]] = {
    "EWoK": {"task": "ewok", "data_path": PRISTINE_FULL / "ewok_filtered", "batch_size": 64},
    "Entity": {"task": "entity_tracking", "data_path": PRISTINE_FULL / "entity_tracking", "batch_size": 128},
}
STABLE_COLUMNS = ["EWoK", "Entity"]

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "first_basin_max_repeat": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "dose_max_repeat",
        "out_root": WS / "data" / "dose_ladder_stable_eval" / "eval",
        "description": "First-basin MAX 2.64x repeat arm; repairs missing 100M Entity for late fixed-budget decomposition.",
        "family": "first_basin_max_repeat_seed43022",
        "data_arm": "repeat",
        "seed": 43022,
    },
    "second_basin_repeat": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "second_basin_max_repeat_seed43122",
        "out_root": WS / "data" / "second_basin_entity_ewok_eval" / "repeat_eval",
        "description": "Second-basin MAX 2.64x repeat arm; late Entity/EWoK counterfactual for basin reproducibility.",
        "family": "second_basin_max_repeat_seed43122",
        "data_arm": "repeat",
        "seed": 43122,
    },
    "second_basin_view": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
        "target_prefix": "second_basin_max_view_seed43122",
        "out_root": WS / "data" / "second_basin_entity_ewok_eval" / "view_eval",
        "description": "Second-basin MAX 2.64x compact-view arm; late Entity/EWoK treatment half for basin reproducibility.",
        "family": "second_basin_max_view_seed43122",
        "data_arm": "view",
        "seed": 43122,
    },
    "max_breadth": {
        "run_dir": WS / "training" / "runs" / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "target_prefix": "max_breadth_seed43022",
        "out_root": WS / "data" / "breadth_entity_ewok_eval" / "eval",
        "description": "First-basin MAX same-population breadth arm; late Entity/EWoK specificity against compact re-expression.",
        "family": "max_breadth_seed43022",
        "data_arm": "breadth",
        "seed": 43022,
    },
    "roberta_max_repeat": {
        "run_dir": WS / "training" / "runs" / "roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022",
        "target_prefix": "roberta_max_repeat_seed43022",
        "out_root": WS / "data" / "roberta_maxdose_repeat_eval" / "eval",
        "description": "RoBERTa MAX repeat arm; EWoK/Entity cross-architecture semantic leg.",
        "family": "roberta_max_repeat_seed43022",
        "data_arm": "repeat",
        "seed": 43022,
    },
}

DEFAULT_CHUNKS = [
    ("first_basin_max_repeat", "chck_100M", "Entity"),
    ("second_basin_repeat", "chck_80M", "Entity"),
    ("second_basin_view", "chck_80M", "Entity"),
    ("second_basin_repeat", "chck_90M", "Entity"),
    ("second_basin_view", "chck_90M", "Entity"),
    ("second_basin_repeat", "chck_100M", "Entity"),
    ("second_basin_view", "chck_100M", "Entity"),
]


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


def latest_file(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def score_value(rec: dict[str, Any], col: str) -> Any:
    return rec.get("score")


def stable_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    out: dict[str, float | None] = {"EWoK": None, "Entity": None, "EWoK_plus_Entity_sum": None}
    for col in ["EWoK", "Entity"]:
        rec = tasks.get(col) or {}
        val = score_value(rec, col) if isinstance(rec, dict) else None
        out[col] = float(val) if finite(val) else None
    if finite(out.get("EWoK")) and finite(out.get("Entity")):
        out["EWoK_plus_Entity_sum"] = float(out["EWoK"]) + float(out["Entity"])
    return out


def task_done(payload: dict[str, Any], col: str) -> bool:
    rec = (payload.get("tasks") or {}).get(col)
    return isinstance(rec, dict) and rec.get("returncode") == 0 and finite(score_value(rec, col))


def ready_cols(payload: dict[str, Any]) -> list[str]:
    return [c for c in STABLE_COLUMNS if task_done(payload, c)]


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


def target_name(cfg: dict[str, Any], ck: str) -> str:
    return f"{cfg['target_prefix']}_{ck}"


def per_target_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / "per_target" / f"{target}.json"


def load_or_new_payload(target: str, cfg: dict[str, Any], endpoint: str, force: bool) -> dict[str, Any]:
    out_root = pathlib.Path(cfg["out_root"])
    p = per_target_path(out_root, target)
    if p.exists() and not force:
        return read_json(p)
    run_dir = pathlib.Path(cfg["run_dir"])
    model_root = run_dir / "hf_model"
    model_path = model_root / endpoint
    payload: dict[str, Any] = {
        "target": target,
        "description": cfg["description"],
        "family": cfg["family"],
        "run_dir": rel(run_dir),
        "model_root": rel(model_root),
        "model_path": rel(model_path),
        "endpoint": endpoint,
        "started_utc": now(),
        "gpu": "visible_gpu_chunk",
        "tasks": {},
        "hardware_reason": "CPU official scoring timed out for the required Entity/EWoK chunks; this run uses only a narrow predeclared GPU scoring chunk, not training.",
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
    cache = out_root / "hf_cache_gpu_chunk" / target
    tmp = out_root / "tmp_gpu_chunk" / target
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_chunk(arm: str, checkpoint: str, column: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm}; choices {sorted(ARM_CONFIGS)}")
    if column not in ZERO_SHOT_SPECS:
        raise ValueError(f"column {column} not allowed; choices {sorted(ZERO_SHOT_SPECS)}")
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
    payload = load_or_new_payload(target, cfg, checkpoint, force=False)
    if task_done(payload, column) and not force:
        return {"status": "skip_existing", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": score_value(payload["tasks"][column], column), "per_target": rel(per_target_path(out_root, target))}

    spec = ZERO_SHOT_SPECS[column]
    task_out = out_root / "official_outputs" / target / column
    log = out_root / "logs" / target / f"gpu_chunk_{column}.log"
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
    header = {"event": "gpu_chunk_start", "created_utc": now(), "arm": arm, "checkpoint": checkpoint, "column": column, "gpu": gpu, "target": target, "cmd": argv, "scientific_role": cfg["description"], "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    t0 = time.time()
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = round(time.time() - t0, 3)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "gpu_chunk_finished", "returncode": proc.returncode, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")
    score = read_report_score(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    rec: dict[str, Any] = {
        "column": column,
        "task": spec["task"],
        "data_path": rel(pathlib.Path(spec["data_path"])),
        "revision_name": revision,
        "output_dir": rel(task_out),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log": rel(log),
        "gpu": gpu,
        "score": score,
    }
    if pred:
        rec["predictions"] = rel(pred)
    if report:
        rec["report"] = rel(report)
    if proc.returncode != 0:
        rec["error"] = f"zero-shot {column} GPU chunk failed rc={proc.returncode}"
    elif score is None:
        rec["error"] = f"zero-shot {column} produced no parseable score"
    payload.setdefault("tasks", {})[column] = rec
    save_payload(out_root, target, payload)
    result = {"status": "chunk_done" if not rec.get("error") else "chunk_failed", "arm": arm, "checkpoint": checkpoint, "column": column, "target": target, "score": score, "returncode": proc.returncode, "elapsed_sec": elapsed, "per_target": rel(per_target_path(out_root, target)), "predictions": rec.get("predictions"), "report": rec.get("report"), "log": rel(log)}
    write_json(out_root / "gpu_chunk_results" / f"{target}_{column}.json", result)
    if rec.get("error"):
        raise RuntimeError(rec["error"])
    return result


def parse_chunk_specs(vals: list[str] | None) -> list[tuple[str, str, str]]:
    if not vals:
        return DEFAULT_CHUNKS
    out = []
    for v in vals:
        parts = v.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad chunk spec {v!r}; expected arm:checkpoint:column")
        out.append((parts[0], parts[1], parts[2]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", nargs="*", default=None, help="Chunk specs as arm:chck_80M:Entity. Default is decision-critical Entity set.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--summary-dir", default=str(WS / "data" / "gpu_official_chunk_eval"), help="Directory for this wrapper's aggregate result JSON; use distinct subdirectories for parallel runs.")
    args = ap.parse_args()
    chunks = parse_chunk_specs(args.chunks)
    plan_rows: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        cfg = ARM_CONFIGS[arm]
        target = target_name(cfg, ck)
        out_root = pathlib.Path(cfg["out_root"])
        p = per_target_path(out_root, target)
        existing = read_json(p) if p.exists() else {"tasks": {}}
        plan_rows.append({
            "arm": arm,
            "checkpoint": ck,
            "column": col,
            "target": target,
            "run_dir": rel(pathlib.Path(cfg["run_dir"])),
            "model_path": rel(pathlib.Path(cfg["run_dir"]) / "hf_model" / ck),
            "model_exists": (pathlib.Path(cfg["run_dir"]) / "hf_model" / ck).exists(),
            "metrics_exists": (pathlib.Path(cfg["run_dir"]) / "scientific_metrics.json").exists(),
            "per_target": rel(p),
            "existing_ready_cols": ready_cols(existing),
            "will_run": bool(args.force or not task_done(existing, col)),
        })
    plan = {
        "status": "GPU_OFFICIAL_CHUNK_EVAL_PLAN" if args.plan_only else "GPU_OFFICIAL_CHUNK_EVAL_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "chunks": plan_rows,
        "scientific_purpose": "Use a narrow official-compatible GPU scoring path only after CPU official scoring timed out, to decide second-basin Entity reproducibility and the first-basin 100M late boundary before any new H100 training.",
        "no_training_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    results: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        res = run_chunk(arm, ck, col, args.gpu, args.force, args.timeout_sec)
        results.append(res)
        print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    final = {**plan, "status": "GPU_OFFICIAL_CHUNK_EVAL_DONE", "finished_utc": now(), "results": results}
    summary_dir = pathlib.Path(args.summary_dir)
    if not summary_dir.is_absolute():
        summary_dir = ROOT / summary_dir
    out = summary_dir / f"chunk_eval_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
