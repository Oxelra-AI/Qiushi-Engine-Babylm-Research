#!/usr/bin/env python3
"""research: official-compatible fast evaluation for COMPACT_EXPERIENCE C/S residualized 4M screen.

Evaluates DeBERTa-v2 MLM checkpoints trained in research on the BabyLM strict
pipeline tasks used for mechanism triage: BLiMP fast, Supplement fast, EWoK fast,
Entity fast, COMPS full, GlobalPIQA parallel/nonparallel fast, and Reading fast.

The script is intentionally resumable and safe for two concurrent processes: each
process writes only per-arm JSON files and per-work-tag logs/output directories.
Run `--mode aggregate` after the per-arm jobs finish to compute tables and deltas.
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/cs_4m_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/cs_4m_residualized_eval.md')
JSON_PATH = OUT_ROOT / "cs_4m_residualized_eval_summary.json"

ARMS: Dict[str, pathlib.Path] = {
    "r_strat_a": RUN_BASE / "r_strat_a_4m_seed43/hf_model/chck_4M",
    "r_strat_b": RUN_BASE / "r_strat_b_4m_seed43/hf_model/chck_4M",
    "c_unique_high": RUN_BASE / "c_unique_high_4m_seed43/hf_model/chck_4M",
    "c_unique_control": RUN_BASE / "c_unique_control_4m_seed43/hf_model/chck_4M",
    "s_unique_high": RUN_BASE / "s_unique_high_4m_seed43/hf_model/chck_4M",
    "s_unique_control": RUN_BASE / "s_unique_control_4m_seed43/hf_model/chck_4M",
}

# column, task, data_path relative to STRICT, batch size
TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    # COMPS has no small fast_eval directory in this repo snapshot; use the official full_eval COMPS column as in INITIAL_MODEL_STUDIES screens.
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


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    # Use COMPACT_EXPERIENCE-local caches to avoid read-only shared cache and to avoid parallel cache collisions.
    hf = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    # Reuse the downloaded punkt/material from INITIAL_MODEL_STUDIES if available.
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def parse_sentence_score(text: str) -> Optional[float]:
    # Official scripts often print an AVERAGE ACCURACY block.
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    # Some fast scripts print a final two-column line "temperature score".
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
                return val
    # Last-resort named score extraction.
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


def read_report_score(task_out: pathlib.Path, column: str) -> Optional[float]:
    # Find a score report only inside this task-specific output directory.
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


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    header = f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(cmd) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header)
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
    revision = f"step008_{arm}_{column}"
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
    report_score = read_report_score(task_out, column)
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
    revision = f"step008_{arm}_Reading"
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
    cols = list(columns)
    for column in cols:
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
    reading = rec.get("reading") or {}
    rs = reading.get("scores") or {}
    out["Reading"] = rs.get("Reading")
    out["Reading_eye"] = rs.get("Reading_eye")
    out["Reading_self_paced"] = rs.get("Reading_self_paced")
    return out


def mean_rows(table: Dict[str, Dict[str, Optional[float]]], arms: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in TABLE_KEYS:
        vals = [table[a].get(k) for a in arms if a in table and table[a].get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
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


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def aggregate() -> Dict[str, Any]:
    table: Dict[str, Dict[str, Optional[float]]] = {}
    raw: Dict[str, Any] = {}
    for arm in ARMS:
        p = OUT_ROOT / "per_arm" / f"{arm}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            raw[arm] = rec
            table[arm] = arm_to_table(rec)
    if "r_strat_a" in table and "r_strat_b" in table:
        table["random_mean"] = mean_rows(table, ["r_strat_a", "r_strat_b"])
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for name, a, b in [
        ("c_high_minus_c_control", "c_unique_high", "c_unique_control"),
        ("s_high_minus_s_control", "s_unique_high", "s_unique_control"),
        ("c_high_minus_random_mean", "c_unique_high", "random_mean"),
        ("s_high_minus_random_mean", "s_unique_high", "random_mean"),
        ("c_control_minus_random_mean", "c_unique_control", "random_mean"),
        ("s_control_minus_random_mean", "s_unique_control", "random_mean"),
        ("r_b_minus_r_a", "r_strat_b", "r_strat_a"),
    ]:
        if a in table and b in table:
            deltas[name] = delta_rows(table, a, b)
    proxies = {name: weighted_proxy(d) for name, d in deltas.items()}
    payload = {
        "status": "CS_4M_RESIDUALIZED_FAST_EVAL" if len(raw) < len(ARMS) else "CS_4M_RESIDUALIZED_FAST_EVAL_DONE",
        "note": "Fast mechanism screen: BLiMP/Supplement/EWoK/Entity/GPIQA/Reading use fast_eval; COMPS uses full_eval/comps; SuperGLUE and AoA are not run and this is not official Overall.",
        "arms_completed": sorted(raw),
        "model_paths": {k: str(v) for k, v in ARMS.items()},
        "raw_per_arm_files": {k: str(OUT_ROOT / "per_arm" / f"{k}.json") for k in raw},
        "table": table,
        "deltas": deltas,
        "weighted_fast_proxy_delta": proxies,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    order = ["r_strat_a", "r_strat_b", "random_mean", "c_unique_control", "c_unique_high", "s_unique_control", "s_unique_high"]
    lines = [
        "# research — Residualized C/S 4M fast evaluation",
        "",
        f"Summary JSON: `{JSON_PATH}`",
        "",
        "This is a mechanism-screen evaluation, not official Overall: SuperGLUE and AoA are absent, fast subsets are used for most zero-shot columns, and COMPS is evaluated from the full COMPS set because no fast COMPS directory exists in this repo snapshot.",
        "",
        "## Scores",
        "",
        "| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in order:
        if arm not in table:
            continue
        s = table[arm]
        lines.append(
            f"| {arm} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA_mean'))} | {fmt(s.get('Reading'))} |"
        )
    lines += [
        "",
        "## Deltas",
        "",
        "| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | weighted fast proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [
        "c_high_minus_c_control",
        "s_high_minus_s_control",
        "c_high_minus_random_mean",
        "s_high_minus_random_mean",
        "c_control_minus_random_mean",
        "s_control_minus_random_mean",
        "r_b_minus_r_a",
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
    ]
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
        print(json.dumps({"status": payload["status"], "summary": str(JSON_PATH), "note": str(NOTE_PATH), "arms_completed": payload["arms_completed"], "weighted_fast_proxy_delta": payload["weighted_fast_proxy_delta"]}, indent=2), flush=True)
        return
    cols = parse_columns(args.columns)
    for arm in args.arms:
        t0 = time.time()
        rec = eval_arm(arm, args.gpu, args.work_tag, cols, force=args.force)
        print(json.dumps({"event": "arm_done", "arm": arm, "elapsed_sec": round(time.time() - t0, 1), "per_arm": str(OUT_ROOT / "per_arm" / f"{arm}.json")}), flush=True)


if __name__ == "__main__":
    main()
