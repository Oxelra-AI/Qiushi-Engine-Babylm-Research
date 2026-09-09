#!/usr/bin/env python3
"""research: direct fast evaluation of the research paired continuation.

Targets:
- wwm_cont: INITIAL_MODEL_STUDIES research chck_70M continued for final 30M words with WWM.
- token_cont: same chck_70M / same data / same fresh optimizer schedule, token masking only.
- initial_model_chck100: original INITIAL_MODEL_STUDIES research WWM chck_100M reference.

The evaluator follows the known-good BabyLM strict invocation used in Steps 8/10/11
and keeps exact checkpoint directories as --model_path_or_name.  It avoids
parent+revision model loading and writes isolated output/log/cache directories.
"""
from __future__ import annotations

import argparse
import hashlib
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/paired_continuation_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/paired_continuation_eval.md')
SUMMARY_JSON = OUT_ROOT / "paired_continuation_eval_summary.json"

TARGETS: Dict[str, pathlib.Path] = {
    "wwm_cont": ROOT_COMPACT_EXPERIENCE / "training/runs/continuation_wwm_seed43/hf_model/chck_100M",
    "token_cont": ROOT_COMPACT_EXPERIENCE / "training/runs/continuation_token_seed43/hf_model/chck_100M",
    "initial_model_chck100": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M",
}

# Data/task paths intentionally mirror the validated research/010/011 strict calls.
TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COLUMN = {c: (task, data, batch) for c, task, data, batch in TASKS}
SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
TABLE_COLUMNS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced",
]


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    cache = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def sha16(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_sentence_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
                return val
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
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(map(str, cmd)) + "\n")
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    with log_path.open("a", encoding="utf-8") as f:
        f.write(p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr)
        f.write(f"\n[returncode={p.returncode} elapsed_sec={elapsed:.1f}]\n")
    if p.returncode != 0:
        raise RuntimeError("command failed rc=%s\n%s\n%s" % (p.returncode, " ".join(map(str, cmd)), (p.stdout + p.stderr)[-6000:]))
    return p


def eval_sentence_column(target: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str], force: bool = False) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COLUMN[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_{column}.log"
    report_score = None if force else read_report_score(task_out)
    if report_score is None:
        cmd = [
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path.resolve()),
            "--backend", "mlm",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--batch_size", str(batch),
            "--output_dir", str(task_out.resolve()),
        ]
        t0 = time.time()
        p = run_cmd(cmd, env, log, timeout=1800)
        stdout_score = parse_sentence_score(p.stdout)
        report_score = read_report_score(task_out)
        score = stdout_score if stdout_score is not None else report_score
        elapsed = round(time.time() - t0, 3)
        rc = p.returncode
    else:
        score = report_score
        stdout_score = None
        elapsed = 0.0
        rc = 0
    if score is None:
        raise RuntimeError(f"score parse failed for {target} {column}; output={task_out}")
    return {
        "column": column,
        "task": task,
        "data_path": data_path,
        "score": float(score),
        "stdout_score": stdout_score,
        "report_score": report_score,
        "returncode": rc,
        "elapsed_sec": elapsed,
        "log": str(log),
        "output_dir": str(task_out),
    }


def eval_reading(target: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str], force: bool = False) -> Dict[str, Any]:
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_Reading.log"
    scores = {} if force else read_reading_report(task_out)
    if not scores:
        cmd = [
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(model_path.resolve()),
            "--backend", "mlm",
            "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
            "--output_dir", str(task_out.resolve()),
        ]
        t0 = time.time()
        p = run_cmd(cmd, env, log, timeout=1800)
        scores = parse_reading_scores(p.stdout) or read_reading_report(task_out)
        elapsed = round(time.time() - t0, 3)
        rc = p.returncode
    else:
        elapsed = 0.0
        rc = 0
    if not scores or "Reading" not in scores:
        raise RuntimeError(f"reading score parse failed for {target}; output={task_out}")
    return {"column": "Reading", "scores": scores, "returncode": rc, "elapsed_sec": elapsed, "log": str(log), "output_dir": str(task_out)}


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


def load_existing(target: str) -> Optional[Dict[str, Any]]:
    p = per_target_path(target)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def eval_target(target: str, gpu: int, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    if target not in TARGETS:
        raise KeyError(f"unknown target {target}; choices={sorted(TARGETS)}")
    model_path = TARGETS[target]
    if not model_path.exists() or not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(model_path)
    payload = {} if force else (load_existing(target) or {})
    if not payload:
        payload = {
            "target": target,
            "model_path": str(model_path),
            "sha16": sha16(model_path / "model.safetensors"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": gpu,
            "tasks": {},
            "reading": None,
        }
    env = setup_env(target, gpu)
    for column in columns:
        if column == "Reading":
            if payload.get("reading") and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_reading(target, model_path, target, env, force=force)
            payload["reading"] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "scores": rec.get("scores"), "rc": rec.get("returncode")}), flush=True)
        else:
            if column not in TASK_BY_COLUMN:
                raise KeyError(f"unknown column {column}")
            if payload.get("tasks", {}).get(column) and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_sentence_column(target, model_path, column, target, env, force=force)
            payload.setdefault("tasks", {})[column] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "score": rec.get("score"), "rc": rec.get("returncode")}), flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(target, payload)
    return payload


def target_to_row(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    row: Dict[str, Optional[float]] = {}
    tasks = rec.get("tasks", {}) or {}
    for column, _, _, _ in TASKS:
        val = (tasks.get(column) or {}).get("score")
        row[column] = float(val) if val is not None else None
    gp_p = row.get("GlobalPIQA_parallel")
    gp_n = row.get("GlobalPIQA_nonparallel")
    row["GlobalPIQA_mean"] = (gp_p + gp_n) / 2.0 if gp_p is not None and gp_n is not None else None
    rs = (rec.get("reading") or {}).get("scores") or {}
    for key in ["Reading", "Reading_eye", "Reading_self_paced"]:
        row[key] = float(rs[key]) if key in rs else None
    return row


def diff_row(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: (a.get(k) - b.get(k) if a.get(k) is not None and b.get(k) is not None else None) for k in TABLE_COLUMNS}


def legacy_weighted_screen(row: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [row.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]]
    if any(v is None for v in vals):
        return None
    reading = row.get("Reading") or 0.0
    return (3.0 / 28.0) * sum(float(v) for v in vals if v is not None) + (1.0 / 8.0) * float(reading)


def equal7_mean(row: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [row.get(k) for k in SCORE_COLUMNS]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals if v is not None) / len(vals)


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def summarize() -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for target in TARGETS:
        p = per_target_path(target)
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        profiles[target] = rec
        row = target_to_row(rec)
        row["legacy_weighted_screen"] = legacy_weighted_screen(row)
        row["equal7_mean"] = equal7_mean(row)
        table[target] = row
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for a, b, name in [
        ("token_cont", "wwm_cont", "token_minus_wwm_cont"),
        ("wwm_cont", "initial_model_chck100", "wwm_cont_minus_initial_model_studies_ref"),
        ("token_cont", "initial_model_chck100", "token_cont_minus_initial_model_studies_ref"),
    ]:
        if a in table and b in table:
            d = diff_row(table[a], table[b])
            d["legacy_weighted_screen"] = (table[a].get("legacy_weighted_screen") - table[b].get("legacy_weighted_screen")
                                             if table[a].get("legacy_weighted_screen") is not None and table[b].get("legacy_weighted_screen") is not None else None)
            d["equal7_mean"] = (table[a].get("equal7_mean") - table[b].get("equal7_mean")
                                if table[a].get("equal7_mean") is not None and table[b].get("equal7_mean") is not None else None)
            deltas[name] = d
    payload = {
        "status": "PAIRED_CONTINUATION_FAST_EVAL_DONE" if len(table) == len(TARGETS) else "PAIRED_CONTINUATION_FAST_EVAL_PARTIAL",
        "method": "exact checkpoint directories; BabyLM strict fast/full paths matching Steps 8/10/11; no parent+revision model loading",
        "targets": {k: str(v) for k, v in TARGETS.items()},
        "table": table,
        "deltas": deltas,
        "profiles": profiles,
        "summary_written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    return payload


def write_note(payload: Dict[str, Any]) -> None:
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = payload.get("table", {})
    lines: List[str] = [
        "# research — research paired continuation evaluation",
        "",
        f"JSON: `{SUMMARY_JSON}`",
        "",
        "## Design carried into evaluation",
        "",
        "Both research arms load the same INITIAL_MODEL_STUDIES research `chck_70M`, use the same post-70M example order and the same fresh optimizer schedule, and differ only in late masking granularity. The WWM continuation is the restart control; token continuation is the treatment. The INITIAL_MODEL_STUDIES original `chck_100M` is the old full-run reference.",
        "",
        "## Scores",
        "",
        "| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA | Reading | legacy weighted screen | equal-7 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ["initial_model_chck100", "wwm_cont", "token_cont"]:
        if target not in table:
            continue
        r = table[target]
        lines.append(
            f"| {target} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('Entity'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('GlobalPIQA_mean'))} | {fmt(r.get('Reading'))} | {fmt(r.get('legacy_weighted_screen'))} | {fmt(r.get('equal7_mean'))} |"
        )
    lines += [
        "",
        "Legacy weighted screen = (3/28)·(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA) + (1/8)·Reading, preserved for continuity with INITIAL_MODEL_STUDIES/COMPACT_EXPERIENCE fast screens. Equal-7 mean is the arithmetic mean of the seven shown columns.",
        "",
        "## Deltas",
        "",
        "| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA | Reading | legacy weighted screen | equal-7 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["token_minus_wwm_cont", "wwm_cont_minus_initial_model_studies_ref", "token_cont_minus_initial_model_studies_ref"]:
        d = (payload.get("deltas") or {}).get(name)
        if not d:
            continue
        lines.append(
            f"| {name} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA_mean'))} | {fmt(d.get('Reading'))} | {fmt(d.get('legacy_weighted_screen'))} | {fmt(d.get('equal7_mean'))} |"
        )
    if "token_minus_wwm_cont" in (payload.get("deltas") or {}):
        d = payload["deltas"]["token_minus_wwm_cont"]
        lines += [
            "",
            "## Interpretation for the granularity mechanism",
            "",
            f"The isolated late-token effect is token minus WWM: BLiMP {fmt(d.get('BLiMP'))}, Supplement {fmt(d.get('Supplement'))}, EWoK {fmt(d.get('EWoK'))}, Entity {fmt(d.get('Entity'))}, COMPS {fmt(d.get('COMPS'))}, GlobalPIQA {fmt(d.get('GlobalPIQA_mean'))}, Reading {fmt(d.get('Reading'))}.",
            "This row is the causal comparison for masking granularity under the restart-controlled research design; the comparison to INITIAL_MODEL_STUDIES original separates restart drift from the token-vs-WWM effect.",
        ]
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list_arg(text: str, default: Iterable[str]) -> List[str]:
    if text.strip().lower() in {"all", "default"}:
        return list(default)
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "summary", "all"], default="all")
    ap.add_argument("--targets", default="all", help="comma-separated target names or all")
    ap.add_argument("--columns", default="all", help="comma-separated columns or all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_columns = [c for c, _, _, _ in TASKS] + ["Reading"]
    targets = parse_list_arg(args.targets, TARGETS.keys())
    columns = parse_list_arg(args.columns, all_columns)

    if args.mode in {"eval", "all"}:
        for target in targets:
            t0 = time.time()
            eval_target(target, args.gpu, columns, force=args.force)
            print(json.dumps({"event": "target_done", "target": target, "elapsed_sec": round(time.time() - t0, 1), "per_target": str(per_target_path(target))}), flush=True)
    if args.mode in {"summary", "all"}:
        payload = summarize()
        print(json.dumps({
            "status": payload["status"],
            "summary": str(SUMMARY_JSON),
            "note": str(NOTE_PATH),
            "deltas": payload.get("deltas", {}),
        }, indent=2))


if __name__ == "__main__":
    main()
