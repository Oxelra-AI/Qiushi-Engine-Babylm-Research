#!/usr/bin/env python3
"""research: repaired mixture dose-response evaluator.

The research evaluator used non-existent module names (evaluation_pipeline.blimp.run,
...); this script uses the known-good strict-package invocation pattern from
Steps 18/19:
  - evaluation_pipeline.sentence_zero_shot.run for BLiMP, Supplement, EWoK,
    Entity, COMPS, and GlobalPIQA fast/full-screen columns
  - evaluation_pipeline.reading.run for Reading fast

It evaluates the 0/25/50/75/100% official+aligned dose-response while preserving
both fast-Entity equal-7 (comparable with prior COMPACT_EXPERIENCE screens) and full-Entity
equal-7 (more reliable for the Entity signal). It also records the initialization
provenance, because research showed that missing --extra_init_seed makes seed43 not
a fixed initialization control for the old fullcycle trainer.
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/mixture_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/mixture_eval_repaired.md')
FINAL_JSON_PATH = OUT_ROOT / "mixture_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
    "mix_25pct": RUN_BASE / "mix_25pct_100M_seed43",
    "mix_50pct": RUN_BASE / "mix_50pct_100M_seed43",
    "mix_75pct": RUN_BASE / "mix_75pct_100M_seed43",
    "aligned_100pct": RUN_BASE / "aligned_100M_seed43",
}
DEFAULT_TARGETS = ["initial_model_baseline", "mix_25pct", "mix_50pct", "mix_75pct", "aligned_100pct"]
ALIGNED_FRACTIONS = {
    "initial_model_baseline": 0.0,
    "mix_25pct": 0.25,
    "mix_50pct": 0.50,
    "mix_75pct": 0.75,
    "aligned_100pct": 1.0,
}

# Known-good task/data invocations from research/research fast screens.
TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("Entity_full", "entity_tracking", "evaluation_data/full_eval/entity_tracking", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COL = {c: (t, d, b) for c, t, d, b in TASKS}
DEFAULT_COLUMNS = [c for c, _, _, _ in TASKS] + ["Reading"]
TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean",
    "equal7_full_entity", "weighted_fast_proxy",
]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def model_path_for(target: str) -> pathlib.Path:
    return RUNS[target] / "hf_model" / "chck_100M"


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT_ROOT / "hf_cache_repaired" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmp = OUT_ROOT / "tmp_repaired" / work_tag
    env["TMPDIR"] = str(tmp.resolve())
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
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
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs_repaired" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs_repaired" / work_tag / f"{target}_{column}.log"
    revision = f"mix_{target}_{column}"
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
    task_out = OUT_ROOT / "eval_outputs_repaired" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs_repaired" / work_tag / f"{target}_Reading.log"
    revision = f"mix_{target}_Reading"
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
    return {"column": "Reading", "scores": scores, "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target_repaired" / f"{target}.json"


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_run_summary(target: str) -> Dict[str, Any]:
    run_dir = RUNS[target]
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "word_exposure", "loss_first", "loss_last", "actual_training_steps", "parameter_count",
            "example_pool_words_actual", "example_jsonl_total_words", "selected_for_training_words",
            "mask_mode", "mask_prob", "seq_length", "batch_size", "seed", "extra_init_seed",
            "train_rng_seed", "tokenizer_label", "tokenizer_path", "source_words_consumed",
        ]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
    return out


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = per_target_path(target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {
            "target": target,
            "aligned_fraction": ALIGNED_FRACTIONS.get(target),
            "model_path": str(model_path),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": gpu,
            "work_tag": work_tag,
            "tasks": {},
            "reading": None,
            "run_summary": read_run_summary(target),
        }
    env = setup_env(f"{work_tag}_{target}", gpu)
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
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_sentence_column(target, model_path, column, work_tag, env)
            payload.setdefault("tasks", {})[column] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["run_summary"] = read_run_summary(target)
    save_target(target, payload)
    return payload


def target_to_table(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    tasks = rec.get("tasks", {})
    for col, _, _, _ in TASKS:
        out[col] = tasks.get(col, {}).get("score")
    gp = out.get("GlobalPIQA_parallel")
    gn = out.get("GlobalPIQA_nonparallel")
    out["GlobalPIQA_mean"] = (gp + gn) / 2.0 if gp is not None and gn is not None else None
    rs = (rec.get("reading") or {}).get("scores") or {}
    out["Reading"] = rs.get("Reading")
    out["Reading_eye"] = rs.get("Reading_eye")
    out["Reading_self_paced"] = rs.get("Reading_self_paced")
    vals_fast = [out.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]]
    vals_full = [out.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]]
    valid_fast = [v for v in vals_fast if v is not None]
    valid_full = [v for v in vals_full if v is not None]
    out["equal7_mean"] = sum(valid_fast) / len(valid_fast) if valid_fast else None
    out["equal7_full_entity"] = sum(valid_full) / len(valid_full) if valid_full else None
    vals_proxy = [out.get(k) for k in NLP_PROXY_KEYS]
    if all(v is not None for v in vals_proxy) and out.get("Reading") is not None:
        out["weighted_fast_proxy"] = (3.0 / 28.0) * sum(v for v in vals_proxy if v is not None) + (1.0 / 8.0) * out["Reading"]
    else:
        out["weighted_fast_proxy"] = None
    return out


def delta_row(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: (round(a.get(k) - b.get(k), 4) if a.get(k) is not None and b.get(k) is not None else None) for k in TABLE_KEYS}


def build_summary(targets: List[str]) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for t in targets:
        p = per_target_path(t)
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            raw[t] = rec
            table[t] = target_to_table(rec)

    dose_response = []
    for t in sorted(targets, key=lambda x: ALIGNED_FRACTIONS.get(x, 999)):
        if t in table:
            row = {"target": t, "aligned_fraction": ALIGNED_FRACTIONS.get(t)}
            row.update(table[t])
            dose_response.append(row)

    contrasts: Dict[str, Dict[str, Optional[float]]] = {}
    baseline = table.get("initial_model_baseline", {})
    for t in ["mix_25pct", "mix_50pct", "mix_75pct", "aligned_100pct"]:
        if t in table and baseline:
            contrasts[f"{t}_minus_initial_model_studies_baseline"] = delta_row(table[t], baseline)

    best_fast = max((r for r in dose_response if r.get("equal7_mean") is not None), key=lambda r: r["equal7_mean"], default=None)
    best_full = max((r for r in dose_response if r.get("equal7_full_entity") is not None), key=lambda r: r["equal7_full_entity"], default=None)

    payload = {
        "status": "REPAIRED_MIXTURE_EVAL_DONE" if len(raw) == len(targets) else "REPAIRED_MIXTURE_EVAL_PARTIAL",
        "note": "Repaired evaluator after the original research script used nonexistent module entrypoints. equal7_mean uses fast Entity for comparability with prior COMPACT_EXPERIENCE fast screens; equal7_full_entity substitutes full Entity Tracking. This is not official nine-column Overall.",
        "initialization_caveat": "The mixture arms were trained with extra_init_seed=-1, so seed43 does not guarantee shared model initialization under the old fullcycle trainer. Do not select Phase 2 solely from this single-init fast curve; replicate key endpoints with fixed extra_init_seed and keep an official-data Phase 2 control under the same 40k tokenizer/architecture/LAMB.",
        "targets_requested": targets,
        "targets_completed": list(raw.keys()),
        "table": table,
        "dose_response": dose_response,
        "contrasts_vs_initial_model_studies_baseline": contrasts,
        "best_by_equal7_fast_entity": best_fast,
        "best_by_equal7_full_entity": best_full,
        "run_summaries": {t: raw[t].get("run_summary", {}) for t in raw},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Optional[float], n: int = 2) -> str:
        return "NA" if x is None else f"{x:.{n}f}"

    lines = [
        "# research — Repaired mixture dose-response evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "The original research evaluator was cancelled after it used non-existent task modules. This repaired run uses the research/research `sentence_zero_shot` and `reading` invocations.",
        "",
        "Important interpretation constraint: the three new mixture arms record `extra_init_seed=-1`; research showed this means seed43 does not fix model initialization in the inherited fullcycle trainer. This curve is exploratory evidence for a dose-response, not a basis for locking Phase 2 without shared-initialization replication and an official-data Phase 2 control.",
        "",
        "## Dose-response table",
        "",
        "| target | frac | BLiMP | Supp | EWoK | Entity_fast | Entity_full | COMPS | GPIQA_mean | Reading | equal7_fast | equal7_fullEnt | wproxy | extra_init_seed | loss_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dose_response:
        t = row["target"]
        rs = payload["run_summaries"].get(t, {})
        lines.append(
            f"| {t} | {row.get('aligned_fraction'):.0%} | {fmt(row.get('BLiMP'))} | {fmt(row.get('Supplement'))} | {fmt(row.get('EWoK'))} | "
            f"{fmt(row.get('Entity'))} | {fmt(row.get('Entity_full'))} | {fmt(row.get('COMPS'))} | {fmt(row.get('GlobalPIQA_mean'))} | {fmt(row.get('Reading'),3)} | "
            f"{fmt(row.get('equal7_mean'),3)} | {fmt(row.get('equal7_full_entity'),3)} | {fmt(row.get('weighted_fast_proxy'),3)} | "
            f"{rs.get('extra_init_seed','NA')} | {fmt(rs.get('loss_last'))} |"
        )
    lines += ["", "## Contrasts vs INITIAL_MODEL_STUDIES baseline", ""]
    for name, d in contrasts.items():
        lines.append(
            f"- **{name}**: equal7_fast {d.get('equal7_mean'):+.3f}, equal7_fullEnt {d.get('equal7_full_entity'):+.3f}, "
            f"Entity_fast {d.get('Entity'):+.2f}, Entity_full {d.get('Entity_full'):+.2f}, BLiMP {d.get('BLiMP'):+.2f}, Supplement {d.get('Supplement'):+.2f}, wproxy {d.get('weighted_fast_proxy'):+.3f}"
        )
    if best_fast:
        lines.append("")
        lines.append(f"Best by fast-Entity equal7: `{best_fast['target']}` = {best_fast['equal7_mean']:.3f}.")
    if best_full:
        lines.append(f"Best by full-Entity equal7: `{best_full['target']}` = {best_full['equal7_full_entity']:.3f}.")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--work_tag", default="mix_repaired")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        if target not in RUNS:
            raise KeyError(f"unknown target {target}; choices={sorted(RUNS)}")
        model_path = model_path_for(target)
        if not model_path.exists():
            print(json.dumps({"event": "skip", "target": target, "reason": "missing_model", "path": str(model_path)}), flush=True)
            continue
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
        print(json.dumps({"event": "target_done", "target": target}), flush=True)
    summary = build_summary(args.targets)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
