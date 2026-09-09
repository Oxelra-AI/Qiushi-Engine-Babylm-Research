#!/usr/bin/env python3
"""research: fast BabyLM evaluation for paired-alignment Wave 1.

This script uses the known-good strict-package invocation pattern from Steps 8/10/11/15:
`evaluation_pipeline.sentence_zero_shot.run` for BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA
and `evaluation_pipeline.reading.run` for Reading.  It evaluates the critical Wave-1
contrast: aligned vs mismatched within-sequence paired text, plus an inherited INITIAL_MODEL_STUDIES
reference checkpoint.

The output is a mechanism-screen record, not official Overall: SuperGLUE and AoA are
not included here; several columns use fast subsets as in earlier COMPACT_EXPERIENCE screens.
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/paired_alignment_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/paired_alignment_wave1_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "paired_alignment_wave1_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "aligned": RUN_BASE / "aligned_100M_seed43",
    "mismatched": RUN_BASE / "mismatched_100M_seed43",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
}
DEFAULT_TARGETS = ["aligned@chck_100M", "mismatched@chck_100M", "initial_model_baseline@chck_100M"]

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
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean",
]
DELTA_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean"]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def parse_target(spec: str) -> Tuple[str, str, str]:
    if "@" not in spec:
        arm, ck = spec, "chck_100M"
    else:
        arm, ck = spec.split("@", 1)
    if arm not in RUNS:
        raise KeyError(f"unknown arm {arm}; choices={sorted(RUNS)}")
    return arm, ck, f"{arm}__{ck}"


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
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
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
    _, _, label = parse_target(target)
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / label / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{label}_{column}.log"
    revision = f"step018_{label}_{column}"
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
    _, _, label = parse_target(target)
    task_out = OUT_ROOT / "eval_outputs" / work_tag / label / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{label}_Reading.log"
    revision = f"step018_{label}_Reading"
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


def read_run_summary(arm: str) -> Dict[str, Any]:
    run_dir = RUNS[arm]
    out: Dict[str, Any] = {"run_dir": str(run_dir)}
    p = run_dir / "scientific_metrics.json"
    out["scientific_metrics_path"] = str(p)
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "word_exposure", "loss_first", "loss_last", "actual_training_steps", "parameter_count",
            "example_jsonl_total_words", "example_jsonl_total_rows", "selected_for_training_words",
            "mask_mode", "mask_prob", "seq_length", "batch_size", "seed", "tokenizer_label",
            "source_words_consumed", "saved_checkpoints",
        ]:
            if k in m:
                out[k] = m[k]
    return out


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
        "run_summary": read_run_summary(arm),
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
    payload["run_summary"] = read_run_summary(arm)
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
    vals = [out.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]]
    out["equal7_mean"] = sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
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


def build_summary(targets: List[str]) -> Dict[str, Any]:
    raw, table = load_targets(targets)
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    pair_defs = [
        ("aligned_minus_mismatched", "aligned__chck_100M", "mismatched__chck_100M"),
        ("aligned_minus_initial_model_studies_baseline", "aligned__chck_100M", "initial_model_baseline__chck_100M"),
        ("mismatched_minus_initial_model_studies_baseline", "mismatched__chck_100M", "initial_model_baseline__chck_100M"),
    ]
    for name, a, b in pair_defs:
        if a in table and b in table:
            deltas[name] = delta_rows(table, a, b)
    payload = {
        "status": "PAIRED_ALIGNMENT_WAVE1_FAST_EVAL_DONE" if len(raw) == len(targets) else "PAIRED_ALIGNMENT_WAVE1_FAST_EVAL_PARTIAL",
        "note": "Fast mechanism screen: SuperGLUE and AoA absent; most zero-shot columns use fast subsets; COMPS uses full_eval/comps. Same evaluator family as Steps 8/10/11/15.",
        "targets_requested": targets,
        "targets_completed": sorted(raw),
        "table": table,
        "absolute_weighted_fast_proxy": {label: weighted_proxy(row) for label, row in table.items()},
        "deltas": deltas,
        "weighted_fast_proxy_delta": {name: weighted_proxy(d) for name, d in deltas.items()},
        "run_summaries": {arm: read_run_summary(arm) for arm in RUNS},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    return payload


def write_note(payload: Dict[str, Any]) -> None:
    table = payload.get("table", {})
    abs_proxy = payload.get("absolute_weighted_fast_proxy", {})
    deltas = payload.get("deltas", {})
    proxy_delta = payload.get("weighted_fast_proxy_delta", {})
    order = ["aligned__chck_100M", "mismatched__chck_100M", "initial_model_baseline__chck_100M"]
    lines = [
        "# research — Paired-alignment Wave 1 fast evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This evaluates the critical Wave-1 contrast: ALIGNED and MISMATCHED have the same sentence inventory, same alternating packed structure, same word exposure, and differ in whether adjacent orig/simp text is meaning-matched. This is a fast mechanism screen, not official Overall: SuperGLUE and AoA are absent, most zero-shot columns use fast subsets, and COMPS uses `full_eval/comps`.",
        "",
        "## chck_100M scores",
        "",
        "| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | equal-7 mean | weighted fast proxy | loss_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runs = payload.get("run_summaries", {})
    for label in order:
        if label not in table:
            continue
        arm = label.split("__", 1)[0]
        s = table[label]
        run = runs.get(arm, {})
        lines.append(
            f"| {label} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA_mean'))} | {fmt(s.get('Reading'))} | {fmt(s.get('equal7_mean'))} | {fmt(abs_proxy.get(label))} | {fmt(run.get('loss_last'))} |"
        )
    lines += [
        "",
        "## Contrasts",
        "",
        "| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal-7 | weighted fast proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in deltas.items():
        lines.append(
            f"| {name} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA_mean'))} | {fmt(d.get('Reading'))} | {fmt(d.get('equal7_mean'))} | {fmt(proxy_delta.get(name))} |"
        )
    lines += [
        "",
        "Weighted fast proxy = (3/28) * (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA_mean) + (1/8) * Reading. It is a screening statistic only.",
        "",
        "## Training exposure notes",
        "",
    ]
    for arm in ["aligned", "mismatched"]:
        run = runs.get(arm, {})
        source_words = run.get("source_words_consumed") or {}
        lines.append(f"- `{arm}`: word_exposure={run.get('word_exposure')}, loss_first={fmt(run.get('loss_first'))}, loss_last={fmt(run.get('loss_last'))}, source_words_consumed={source_words}")
    lines.append("- The ALIGNED and MISMATCHED base pools each contain 9,999,840 words; exact 100,000,000 exposure equals 10 full passes plus 1,600 words. This is acceptable for the matched mechanism contrast but should be rerun at exactly 99,998,400 exposure before treating it as a strict official candidate.")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--columns", nargs="*", default=[c for c, _, _, _ in TASKS] + ["Reading"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work_tag", default="wave1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = args.targets or DEFAULT_TARGETS
    if args.summary_only:
        payload = build_summary(targets)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for target in targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
        print(json.dumps({"event": "target_done", "target": target}), flush=True)
    payload = build_summary(targets)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
