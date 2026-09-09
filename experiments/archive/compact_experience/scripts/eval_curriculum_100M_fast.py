#!/usr/bin/env python3
"""research: official-compatible fast evaluation for 100M masking-curriculum checkpoints.

This is a checkpoint-general version of the repaired research evaluator.  It uses
exactly the known-good BabyLM strict evaluation invocation style:
- cwd is experiments/archive/initial_model_studies/repos/babylm-eval/strict;
- data paths are the fast_eval/full_eval paths that parsed correctly in Steps 8/10;
- Supplement is BLiMP task over supplement_fast;
- each work tag has isolated HF caches and output directories;
- per-target JSONs are resumable.

Targets are named ARM@CHECKPOINT, e.g. wwm_fixed@chck_100M or
amlm_hard_switch@chck_70M.  The default target set is the four final chck_100M
models from the regime-correct research screen.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/curriculum_100M_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/masking_curriculum_100M_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "curriculum_100M_eval_summary.json"
TRAJ_JSON_PATH = OUT_ROOT / "curriculum_trajectory_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "wwm_fixed": RUN_BASE / "wwm_fixed_100M_seed43",
    "wwm_to_token": RUN_BASE / "wwm_to_token_100M_seed43",
    "amlm_hard": RUN_BASE / "amlm_hard_100M_seed43",
    "amlm_hard_switch": RUN_BASE / "amlm_hard_switch_100M_seed43",
}
DEFAULT_FINAL_TARGETS = [f"{arm}@chck_100M" for arm in RUNS]
DEFAULT_TRAJ_TARGETS = [
    f"{arm}@{ck}"
    for arm in ["wwm_fixed", "wwm_to_token", "amlm_hard_switch"]
    for ck in ["chck_60M", "chck_70M", "chck_80M", "chck_100M"]
]

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COL = {c: (t, d, b) for c, t, d, b in TASKS}
TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced",
]
DELTA_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def parse_target(spec: str) -> Tuple[str, str, str]:
    if "@" not in spec:
        # Back-compatible shorthand: arm means arm@chck_100M
        arm, ck = spec, "chck_100M"
    else:
        arm, ck = spec.split("@", 1)
    if arm not in RUNS:
        raise KeyError(f"unknown arm {arm}; choices={sorted(RUNS)}")
    label = f"{arm}__{ck}"
    return arm, ck, label


def model_path_for(spec: str) -> pathlib.Path:
    arm, ck, _ = parse_target(spec)
    return RUNS[arm] / "hf_model" / ck


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def parse_sentence_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def parse_reading_scores(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [
        ("EYE TRACKING SCORE", "Reading_eye"),
        ("SELF-PACED READING SCORE", "Reading_self_paced"),
    ]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not candidates:
        candidates = sorted(task_out.rglob("*.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        score = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score
    return None


def read_reading_report(task_out: pathlib.Path) -> Dict[str, float]:
    candidates = sorted(task_out.rglob("report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    with log_path.open("a", encoding="utf-8") as f:
        f.write(p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr)
        f.write(f"\n[returncode={p.returncode} elapsed_sec={elapsed:.1f}]\n")
    return p


def eval_sentence_column(target: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    arm, ck, label = parse_target(target)
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / label / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{label}_{column}.log"
    revision = f"step011_{label}_{column}"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data_path,
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=1800)
    stdout_score = parse_sentence_score(p.stdout)
    report_score = read_report_score(task_out)
    score = stdout_score if stdout_score is not None else report_score
    return {
        "column": column,
        "task": task,
        "data_path": data_path,
        "score": score,
        "stdout_score": stdout_score,
        "report_score": report_score,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 3),
        "log": str(log),
        "output_dir": str(task_out),
    }


def eval_reading(target: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    arm, ck, label = parse_target(target)
    task_out = OUT_ROOT / "eval_outputs" / work_tag / label / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{label}_Reading.log"
    revision = f"step011_{label}_Reading"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=1800)
    scores = parse_reading_scores(p.stdout)
    if not scores:
        scores = read_reading_report(task_out)
    return {
        "column": "Reading",
        "scores": scores,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 3),
        "log": str(log),
        "output_dir": str(task_out),
    }


def per_target_path(target: str) -> pathlib.Path:
    _, _, label = parse_target(target)
    return OUT_ROOT / "per_target" / f"{label}.json"


def load_existing_target(target: str) -> Optional[Dict[str, Any]]:
    p = per_target_path(target)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    arm, ck, label = parse_target(target)
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    existing = load_existing_target(target) if not force else None
    payload: Dict[str, Any] = existing or {
        "target": target,
        "arm": arm,
        "checkpoint": ck,
        "label": label,
        "model_path": str(model_path),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
        "work_tag": work_tag,
        "tasks": {},
        "reading": None,
    }
    env = setup_env(work_tag, gpu)
    for column in list(columns):
        if column == "Reading":
            if payload.get("reading") and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_reading(target, model_path, work_tag, env)
            payload["reading"] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if payload.get("tasks", {}).get(column) and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            if column not in TASK_BY_COL:
                raise KeyError(f"unknown column {column}")
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_sentence_column(target, model_path, column, work_tag, env)
            payload.setdefault("tasks", {})[column] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(target, payload)
    return payload


def target_to_table(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    tasks = rec.get("tasks", {})
    for column, _, _, _ in TASKS:
        out[column] = tasks.get(column, {}).get("score")
    gp_p = out.get("GlobalPIQA_parallel")
    gp_n = out.get("GlobalPIQA_nonparallel")
    out["GlobalPIQA_mean"] = (gp_p + gp_n) / 2.0 if gp_p is not None and gp_n is not None else None
    rs = (rec.get("reading") or {}).get("scores") or {}
    out["Reading"] = rs.get("Reading")
    out["Reading_eye"] = rs.get("Reading_eye")
    out["Reading_self_paced"] = rs.get("Reading_self_paced")
    return out


def delta_rows(table: Dict[str, Dict[str, Optional[float]]], a: str, b: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in TABLE_KEYS:
        av = table.get(a, {}).get(k)
        bv = table.get(b, {}).get(k)
        out[k] = av - bv if av is not None and bv is not None else None
    return out


def weighted_proxy(row_or_delta: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [row_or_delta.get(k) for k in NLP_PROXY_KEYS]
    if any(v is None for v in vals):
        return None
    reading = row_or_delta.get("Reading") or 0.0
    return (3.0 / 28.0) * sum(v for v in vals if v is not None) + (1.0 / 8.0) * reading


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def load_jsonl_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dyn_scalar(d: Dict[str, Any], top_key: str, band: Optional[str] = None, subkey: str = "last") -> Optional[float]:
    if band is None:
        val = d.get(top_key)
        return float(val) if isinstance(val, (int, float)) else None
    block = (d.get(top_key) or {}).get(band) or {}
    val = block.get(subkey)
    return float(val) if isinstance(val, (int, float)) else None


def load_run_summaries() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm, run_dir in RUNS.items():
        rec: Dict[str, Any] = {"run_dir": str(run_dir)}
        metrics_p = run_dir / "scientific_metrics.json"
        dyn_p = run_dir / "dynamics_traces.jsonl"
        rec["scientific_metrics_path"] = str(metrics_p)
        rec["dynamics_path"] = str(dyn_p)
        if metrics_p.exists():
            try:
                m = json.loads(metrics_p.read_text(encoding="utf-8"))
                for k in ["word_exposure", "loss_first", "loss_last", "masking_curriculum", "n_amlm_updates", "checkpoints", "param_count"]:
                    rec[k] = m.get(k)
            except Exception as e:
                rec["metrics_error"] = str(e)
        dyn = load_jsonl_records(dyn_p)
        rec["n_dynamics_records"] = len(dyn)
        rec["dynamics_by_checkpoint"] = []
        for d in dyn:
            rec["dynamics_by_checkpoint"].append({
                "checkpoint": d.get("checkpoint"),
                "step": d.get("step"),
                "mask_mode_at_checkpoint": d.get("mask_mode_at_checkpoint"),
                "mask_prob_at_checkpoint": d.get("mask_prob_at_checkpoint"),
                "effective_mask_rate_mean": d.get("effective_mask_rate_mean"),
                "prediction_entropy_mean": d.get("prediction_entropy_mean"),
                "n_amlm_updates_total": d.get("n_amlm_updates_total"),
                "low_loss_last": dyn_scalar(d, "loss_by_freq_band", "low"),
                "mid_loss_last": dyn_scalar(d, "loss_by_freq_band", "mid"),
                "high_loss_last": dyn_scalar(d, "loss_by_freq_band", "high"),
                "low_acc_last": dyn_scalar(d, "accuracy_by_freq_band", "low"),
                "mid_acc_last": dyn_scalar(d, "accuracy_by_freq_band", "mid"),
                "high_acc_last": dyn_scalar(d, "accuracy_by_freq_band", "high"),
                "amlm_weight_mean": (d.get("amlm_weight_stats") or {}).get("mean"),
                "amlm_weight_max": (d.get("amlm_weight_stats") or {}).get("max"),
            })
        out[arm] = rec
    return out


def load_targets(targets: Iterable[str]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Optional[float]]]]:
    raw: Dict[str, Any] = {}
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for t in targets:
        _, _, label = parse_target(t)
        p = per_target_path(t)
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            raw[label] = rec
            table[label] = target_to_table(rec)
    return raw, table


def aggregate_final() -> Dict[str, Any]:
    raw, table = load_targets(DEFAULT_FINAL_TARGETS)
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    pairs = [
        ("wwm_to_token_minus_wwm_fixed", "wwm_to_token__chck_100M", "wwm_fixed__chck_100M"),
        ("amlm_hard_minus_wwm_fixed", "amlm_hard__chck_100M", "wwm_fixed__chck_100M"),
        ("amlm_hard_switch_minus_wwm_fixed", "amlm_hard_switch__chck_100M", "wwm_fixed__chck_100M"),
        ("amlm_hard_switch_minus_amlm_hard", "amlm_hard_switch__chck_100M", "amlm_hard__chck_100M"),
        ("amlm_hard_minus_wwm_to_token", "amlm_hard__chck_100M", "wwm_to_token__chck_100M"),
        ("amlm_hard_switch_minus_wwm_to_token", "amlm_hard_switch__chck_100M", "wwm_to_token__chck_100M"),
    ]
    for name, a, b in pairs:
        if a in table and b in table:
            deltas[name] = delta_rows(table, a, b)
    proxies = {name: weighted_proxy(d) for name, d in deltas.items()}
    absolute_proxy = {label: weighted_proxy(row) for label, row in table.items()}
    payload = {
        "status": "CURRICULUM_100M_FAST_EVAL_DONE" if len(raw) == len(DEFAULT_FINAL_TARGETS) else "CURRICULUM_100M_FAST_EVAL_PARTIAL",
        "note": "Fast mechanism screen only: SuperGLUE and AoA absent; most zero-shot columns use fast_eval, COMPS full_eval/comps. Same evaluator style as research/research.",
        "targets_completed": sorted(raw),
        "table": table,
        "absolute_weighted_fast_proxy": absolute_proxy,
        "deltas": deltas,
        "weighted_fast_proxy_delta": proxies,
        "run_summaries": load_run_summaries(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    return payload


def aggregate_trajectory() -> Dict[str, Any]:
    raw, table = load_targets(DEFAULT_TRAJ_TARGETS)
    abs_proxy = {label: weighted_proxy(row) for label, row in table.items()}
    checkpoints = ["chck_60M", "chck_70M", "chck_80M", "chck_100M"]
    arms = ["wwm_fixed", "wwm_to_token", "amlm_hard_switch"]
    contrasts: Dict[str, Dict[str, Any]] = {}
    for ck in checkpoints:
        wf = f"wwm_fixed__{ck}"
        wt = f"wwm_to_token__{ck}"
        ahs = f"amlm_hard_switch__{ck}"
        if wt in table and wf in table:
            d = delta_rows(table, wt, wf)
            contrasts[f"wwm_to_token_minus_wwm_fixed__{ck}"] = {"delta": d, "weighted_fast_proxy_delta": weighted_proxy(d)}
        if ahs in table and wf in table:
            d = delta_rows(table, ahs, wf)
            contrasts[f"amlm_hard_switch_minus_wwm_fixed__{ck}"] = {"delta": d, "weighted_fast_proxy_delta": weighted_proxy(d)}
        if ahs in table and wt in table:
            d = delta_rows(table, ahs, wt)
            contrasts[f"amlm_hard_switch_minus_wwm_to_token__{ck}"] = {"delta": d, "weighted_fast_proxy_delta": weighted_proxy(d)}
    payload = {
        "status": "CURRICULUM_TRAJECTORY_FAST_EVAL_DONE" if len(raw) == len(DEFAULT_TRAJ_TARGETS) else "CURRICULUM_TRAJECTORY_FAST_EVAL_PARTIAL",
        "targets_completed": sorted(raw),
        "table": table,
        "absolute_weighted_fast_proxy": abs_proxy,
        "trajectory_contrasts": contrasts,
        "run_summaries": load_run_summaries(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRAJ_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_note(payload: Dict[str, Any]) -> None:
    order = ["wwm_fixed__chck_100M", "wwm_to_token__chck_100M", "amlm_hard__chck_100M", "amlm_hard_switch__chck_100M"]
    table = payload.get("table", {})
    abs_proxy = payload.get("absolute_weighted_fast_proxy", {})
    runs = payload.get("run_summaries", {})
    lines = [
        "# research — Regime-correct masking-curriculum 100M fast evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This is a mechanism-screen evaluation of the four 100M-exposure official-corpus runs. It is not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses `full_eval/comps`.",
        "",
        "## chck_100M scores",
        "",
        "| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy | loss_last | late mode/eff mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in order:
        if label not in table:
            continue
        arm = label.split("__", 1)[0]
        s = table[label]
        run = runs.get(arm, {})
        dyn = run.get("dynamics_by_checkpoint") or []
        late = dyn[-1] if dyn else {}
        late_desc = f"{late.get('mask_mode_at_checkpoint','?')}/{fmt(late.get('effective_mask_rate_mean'))}"
        lines.append(
            f"| {label} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA_mean'))} | {fmt(s.get('Reading'))} | {fmt(abs_proxy.get(label))} | {fmt(run.get('loss_last'))} | {late_desc} |"
        )
    lines += ["", "## chck_100M contrasts", "", "| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in payload.get("deltas", {}).items():
        lines.append(f"| {name} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA_mean'))} | {fmt(d.get('Reading'))} | {fmt(payload.get('weighted_fast_proxy_delta', {}).get(name))} |")
    lines += [
        "",
        "Weighted fast proxy = (3/28) * (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA_mean) + (1/8) * Reading. It is a screening statistic only.",
        "",
        "## Dynamics paths",
        "",
    ]
    for arm, run in runs.items():
        lines.append(f"- `{arm}`: metrics `{run.get('scientific_metrics_path')}`, dynamics `{run.get('dynamics_path')}`")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_columns(spec: str) -> List[str]:
    if spec == "all":
        return [c for c, _, _, _ in TASKS] + ["Reading"]
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_targets(spec: str, mode_default: str = "final") -> List[str]:
    if spec == "final":
        return DEFAULT_FINAL_TARGETS
    if spec == "trajectory":
        return DEFAULT_TRAJ_TARGETS
    return [x.strip() for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "aggregate_final", "aggregate_trajectory"], default="eval")
    ap.add_argument("--targets", default="final", help="final, trajectory, or comma-separated ARM@CHECKPOINT targets")
    ap.add_argument("--columns", default="all", help="comma-separated columns or all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--work_tag", default="gpu0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate_final":
        payload = aggregate_final()
        print(json.dumps({
            "status": payload["status"],
            "summary": str(FINAL_JSON_PATH),
            "note": str(NOTE_PATH),
            "weighted_fast_proxy_delta": payload.get("weighted_fast_proxy_delta"),
            "absolute_weighted_fast_proxy": payload.get("absolute_weighted_fast_proxy"),
        }, indent=2))
        return
    if args.mode == "aggregate_trajectory":
        payload = aggregate_trajectory()
        print(json.dumps({
            "status": payload["status"],
            "summary": str(TRAJ_JSON_PATH),
            "targets_completed": payload.get("targets_completed"),
        }, indent=2))
        return
    columns = parse_columns(args.columns)
    targets = parse_targets(args.targets)
    for target in targets:
        t0 = time.time()
        rec = eval_target(target, args.gpu, args.work_tag, columns, force=args.force)
        print(json.dumps({"event": "target_done", "target": target, "elapsed_sec": round(time.time()-t0, 1), "per_target": str(per_target_path(target))}), flush=True)


if __name__ == "__main__":
    main()
