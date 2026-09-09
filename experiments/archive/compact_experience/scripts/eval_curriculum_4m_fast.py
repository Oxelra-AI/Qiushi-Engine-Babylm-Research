#!/usr/bin/env python3
"""research: repaired official-compatible fast evaluation for research masking-curriculum 4M arms.

This script deliberately copies the known-good research invocation style:
- cwd is the `strict/` package inside the INITIAL_MODEL_STUDIES BabyLM eval repo;
- sentence tasks receive the exact fast_eval/full_eval data directories, not the
  top-level evaluation_data directory;
- Supplement is evaluated as the BLiMP task over supplement_fast;
- score parsing reads official stdout/report files;
- each work_tag has isolated HF caches and output directories.

It evaluates chck_4M checkpoints for four official-corpus 4M masking-curriculum
arms: fixed WWM, WWM→token switch, AMLM-hard, and AMLM-hard+switch.
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
from typing import Any, Dict, Iterable, List, Optional

ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/curriculum_4m_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/masking_curriculum_4m_eval.md')
JSON_PATH = OUT_ROOT / "curriculum_4m_eval_summary.json"

ARMS: Dict[str, pathlib.Path] = {
    "wwm_fixed": RUN_BASE / "wwm_fixed_4m_seed43/hf_model/chck_4M",
    "wwm_to_token": RUN_BASE / "wwm_to_token_4m_seed43/hf_model/chck_4M",
    "amlm_hard": RUN_BASE / "amlm_hard_4m_seed43/hf_model/chck_4M",
    "amlm_hard_switch": RUN_BASE / "amlm_hard_switch_4m_seed43/hf_model/chck_4M",
}

# column, task, data_path relative to STRICT, batch size. These are the
# Evaluator paths with previously verified parsing.
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
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "GlobalPIQA_mean",
    "Reading",
    "Reading_eye",
    "Reading_self_paced",
]
DELTA_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]

DYNAMICS_FILES: Dict[str, pathlib.Path] = {
    "wwm_fixed": RUN_BASE / "wwm_fixed_4m_seed43/dynamics_traces.jsonl",
    "wwm_to_token": RUN_BASE / "wwm_to_token_4m_seed43/dynamics_traces.jsonl",
    "amlm_hard": RUN_BASE / "amlm_hard_4m_seed43/dynamics_traces.jsonl",
    "amlm_hard_switch": RUN_BASE / "amlm_hard_switch_4m_seed43/dynamics_traces.jsonl",
}
METRICS_FILES: Dict[str, pathlib.Path] = {
    "wwm_fixed": RUN_BASE / "wwm_fixed_4m_seed43/scientific_metrics.json",
    "wwm_to_token": RUN_BASE / "wwm_to_token_4m_seed43/scientific_metrics.json",
    "amlm_hard": RUN_BASE / "amlm_hard_4m_seed43/scientific_metrics.json",
    "amlm_hard_switch": RUN_BASE / "amlm_hard_switch_4m_seed43/scientific_metrics.json",
}


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
        txt = p.read_text(encoding="utf-8", errors="replace")
        score = parse_sentence_score(txt)
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


def eval_sentence_column(arm: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / arm / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{arm}_{column}.log"
    revision = f"step010_{arm}_{column}"
    cmd = [
        sys.executable,
        "-m",
        "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name",
        str(model_path.resolve()),
        "--backend",
        "mlm",
        "--task",
        task,
        "--data_path",
        data_path,
        "--revision_name",
        revision,
        "--save_predictions",
        "--batch_size",
        str(batch),
        "--output_dir",
        str(task_out.resolve()),
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


def eval_reading(arm: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task_out = OUT_ROOT / "eval_outputs" / work_tag / arm / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{arm}_Reading.log"
    revision = f"step010_{arm}_Reading"
    cmd = [
        sys.executable,
        "-m",
        "evaluation_pipeline.reading.run",
        "--model_path_or_name",
        str(model_path.resolve()),
        "--backend",
        "mlm",
        "--data_path",
        "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name",
        revision,
        "--output_dir",
        str(task_out.resolve()),
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


def load_existing_arm(arm: str) -> Optional[Dict[str, Any]]:
    p = OUT_ROOT / "per_arm" / f"{arm}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_arm(arm: str, payload: Dict[str, Any]) -> None:
    p = OUT_ROOT / "per_arm" / f"{arm}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def eval_arm(arm: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm}; choices={sorted(ARMS)}")
    model_path = ARMS[arm]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    existing = load_existing_arm(arm) if not force else None
    payload: Dict[str, Any] = existing or {
        "arm": arm,
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
                print(json.dumps({"event": "resume", "arm": arm, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "arm": arm, "column": column, "gpu": gpu}), flush=True)
            rec = eval_reading(arm, model_path, work_tag, env)
            payload["reading"] = rec
            save_arm(arm, payload)
            print(json.dumps({"event": "eval_done", "arm": arm, "column": column, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if payload.get("tasks", {}).get(column) and not force:
                print(json.dumps({"event": "resume", "arm": arm, "column": column}), flush=True)
                continue
            if column not in TASK_BY_COL:
                raise KeyError(f"unknown column {column}")
            print(json.dumps({"event": "eval_start", "arm": arm, "column": column, "gpu": gpu}), flush=True)
            rec = eval_sentence_column(arm, model_path, column, work_tag, env)
            payload.setdefault("tasks", {})[column] = rec
            save_arm(arm, payload)
            print(json.dumps({"event": "eval_done", "arm": arm, "column": column, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_arm(arm, payload)
    return payload


def arm_to_table(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
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


def weighted_proxy(d: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [d.get(k) for k in NLP_PROXY_KEYS]
    if any(v is None for v in vals):
        return None
    reading = d.get("Reading") or 0.0
    return (3.0 / 28.0) * sum(v for v in vals if v is not None) + (1.0 / 8.0) * reading


def fast_proxy_absolute(row: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [row.get(k) for k in NLP_PROXY_KEYS]
    if any(v is None for v in vals):
        return None
    reading = row.get("Reading") or 0.0
    return (3.0 / 28.0) * sum(v for v in vals if v is not None) + (1.0 / 8.0) * reading


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def load_jsonl_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_training_summaries() -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for arm, p in METRICS_FILES.items():
        rec: Dict[str, Any] = {"scientific_metrics_path": str(p)}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for key in ["loss_first", "loss_last", "word_exposure", "masking_curriculum", "n_amlm_updates", "checkpoints", "param_count"]:
                    if key in data:
                        rec[key] = data[key]
            except Exception as e:
                rec["metrics_error"] = str(e)
        dyn = load_jsonl_records(DYNAMICS_FILES[arm])
        rec["dynamics_path"] = str(DYNAMICS_FILES[arm])
        rec["n_dynamics_records"] = len(dyn)
        if dyn:
            rec["dynamics_by_checkpoint"] = [
                {
                    "checkpoint": d.get("checkpoint"),
                    "word_exposure": d.get("word_exposure"),
                    "mask_mode": d.get("mask_mode"),
                    "mask_prob": d.get("mask_prob"),
                    "effective_mask_rate": d.get("effective_mask_rate"),
                    "prediction_entropy": d.get("prediction_entropy"),
                    "freq_low_loss": (d.get("freq_band_metrics") or {}).get("low", {}).get("loss"),
                    "freq_low_acc": (d.get("freq_band_metrics") or {}).get("low", {}).get("acc"),
                    "freq_high_loss": (d.get("freq_band_metrics") or {}).get("high", {}).get("loss"),
                    "freq_high_acc": (d.get("freq_band_metrics") or {}).get("high", {}).get("acc"),
                    "amlm_weight_mean": (d.get("amlm_weight_stats") or {}).get("mean"),
                    "amlm_weight_max": (d.get("amlm_weight_stats") or {}).get("max"),
                }
                for d in dyn
            ]
        summaries[arm] = rec
    return summaries


def aggregate() -> Dict[str, Any]:
    table: Dict[str, Dict[str, Optional[float]]] = {}
    raw: Dict[str, Any] = {}
    for arm in ARMS:
        p = OUT_ROOT / "per_arm" / f"{arm}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            raw[arm] = rec
            table[arm] = arm_to_table(rec)
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for name, a, b in [
        ("wwm_to_token_minus_wwm_fixed", "wwm_to_token", "wwm_fixed"),
        ("amlm_hard_minus_wwm_fixed", "amlm_hard", "wwm_fixed"),
        ("amlm_hard_switch_minus_wwm_fixed", "amlm_hard_switch", "wwm_fixed"),
        ("amlm_hard_switch_minus_amlm_hard", "amlm_hard_switch", "amlm_hard"),
        ("amlm_hard_minus_wwm_to_token", "amlm_hard", "wwm_to_token"),
    ]:
        if a in table and b in table:
            deltas[name] = delta_rows(table, a, b)
    proxies = {name: weighted_proxy(d) for name, d in deltas.items()}
    absolute_proxy = {arm: fast_proxy_absolute(row) for arm, row in table.items()}
    training = load_training_summaries()
    payload = {
        "status": "CURRICULUM_4M_FAST_EVAL" if len(raw) < len(ARMS) else "CURRICULUM_4M_FAST_EVAL_DONE",
        "note": "Repaired research evaluation: exact research strict-package cwd, fast_eval paths, report parsers. Fast mechanism screen only; SuperGLUE and AoA are absent, most zero-shot columns use fast_eval, COMPS uses full_eval/comps.",
        "arms_completed": sorted(raw),
        "model_paths": {k: str(v) for k, v in ARMS.items()},
        "raw_per_arm_files": {k: str(OUT_ROOT / "per_arm" / f"{k}.json") for k in raw},
        "table": table,
        "absolute_weighted_fast_proxy": absolute_proxy,
        "deltas": deltas,
        "weighted_fast_proxy_delta": proxies,
        "training_dynamics": training,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    order = ["wwm_fixed", "wwm_to_token", "amlm_hard", "amlm_hard_switch"]
    lines = [
        "# research — Masking-curriculum 4M fast evaluation",
        "",
        f"Summary JSON: `{JSON_PATH}`",
        "",
        "This repairs the failed research evaluator by reusing the known-good research strict-package invocation and fast-data paths. It is a mechanism-screen evaluation, not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses full_eval/comps.",
        "",
        "## Scores",
        "",
        "| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy | loss_last | late mask mode/rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in order:
        if arm not in table:
            continue
        s = table[arm]
        tr = training.get(arm, {})
        dyn = tr.get("dynamics_by_checkpoint") or []
        late = dyn[-1] if dyn else {}
        lines.append(
            f"| {arm} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA_mean'))} | {fmt(s.get('Reading'))} | {fmt(absolute_proxy.get(arm))} | {fmt(tr.get('loss_last'))} | {late.get('mask_mode','?')}/{fmt(late.get('effective_mask_rate'))} |"
        )
    lines += [
        "",
        "## Contrasts",
        "",
        "| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [
        "wwm_to_token_minus_wwm_fixed",
        "amlm_hard_minus_wwm_fixed",
        "amlm_hard_switch_minus_wwm_fixed",
        "amlm_hard_switch_minus_amlm_hard",
        "amlm_hard_minus_wwm_to_token",
    ]:
        if name not in deltas:
            continue
        d = deltas[name]
        lines.append(
            f"| {name} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA_mean'))} | {fmt(d.get('Reading'))} | {fmt(proxies.get(name))} |"
        )
    lines += [
        "",
        "Weighted fast proxy = (3/28) * delta(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA_mean) + (1/8) * delta(Reading). SuperGLUE and AoA are not included.",
        "",
        "## Dynamics paths",
        "",
    ]
    for arm in order:
        tr = training.get(arm, {})
        lines.append(f"- `{arm}`: metrics `{tr.get('scientific_metrics_path')}`, dynamics `{tr.get('dynamics_path')}`")
    lines.append("")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_columns(spec: str) -> List[str]:
    if spec == "all":
        return [c for c, _, _, _ in TASKS] + ["Reading"]
    return [x.strip() for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "aggregate"], default="eval")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--columns", default="all", help="comma-separated columns or all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--work_tag", default="gpu0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        payload = aggregate()
        print(json.dumps({"status": payload["status"], "summary": str(JSON_PATH), "note": str(NOTE_PATH), "arms_completed": payload["arms_completed"], "weighted_fast_proxy_delta": payload["weighted_fast_proxy_delta"], "absolute_weighted_fast_proxy": payload["absolute_weighted_fast_proxy"]}, indent=2), flush=True)
        return
    cols = parse_columns(args.columns)
    for arm in args.arms:
        t0 = time.time()
        rec = eval_arm(arm, args.gpu, args.work_tag, cols, force=args.force)
        print(json.dumps({"event": "arm_done", "arm": arm, "elapsed_sec": round(time.time() - t0, 1), "per_arm": str(OUT_ROOT / "per_arm" / f"{arm}.json")}), flush=True)


if __name__ == "__main__":
    main()
