#!/usr/bin/env python3
"""research: evaluate the fixed-initialization official-vs-mix25 replication.

This is a focused copy of the repaired research BabyLM fast-screen evaluator.  It
keeps only the two research fixed-init 16k runs and writes a separate summary so
that the causal data-complementarity check is not mixed with the exploratory
single-init dose-response artifacts.
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/fixedinit_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/fixedinit_replication_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "fixedinit_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "official_fixedinit": RUN_BASE / "official_fixedinit16k_seed43022",
    "mix25_fixedinit": RUN_BASE / "mix25_fixedinit16k_seed43022",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
}
DEFAULT_TARGETS = ["official_fixedinit", "mix25_fixedinit"]

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
    hf = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmp = OUT_ROOT / "tmp" / work_tag
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
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"fixedinit_{target}_{column}"
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
    return {"column": column, "task": task, "data_path": data_path, "score": score,
            "stdout_score": stdout_score, "report_score": report_score,
            "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3),
            "log": str(log), "output_dir": str(task_out)}


def eval_reading(target: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_Reading.log"
    revision = f"fixedinit_{target}_Reading"
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
    return {"column": "Reading", "scores": scores, "returncode": p.returncode,
            "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


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
            "tokenization_coupling_summary",
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
        payload = {"target": target, "model_path": str(model_path),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpu": gpu, "work_tag": work_tag, "tasks": {}, "run_summary": read_run_summary(target)}
    env = setup_env(f"{work_tag}_{target}", gpu)
    for col in columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"]:
                rec = eval_reading(target, model_path, work_tag, env)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"]:
                rec = eval_sentence_column(target, model_path, col, work_tag, env)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        save_target(target, payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(target, payload)
    return payload


def target_scores(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    tasks = raw.get("tasks", {})
    for col in [c for c, _, _, _ in TASKS]:
        if col in tasks:
            out[col] = tasks[col].get("score")
    if "Reading" in tasks:
        for k, v in tasks["Reading"].get("scores", {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_keys):
        out["equal7_mean"] = sum(out[k] for k in eq_keys) / len(eq_keys)  # type: ignore[arg-type]
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(out[k] for k in eq_full_keys) / len(eq_full_keys)  # type: ignore[arg-type]
    if all(out.get(k) is not None for k in NLP_PROXY_KEYS):
        out["weighted_fast_proxy"] = sum(out[k] for k in NLP_PROXY_KEYS) / len(NLP_PROXY_KEYS)  # type: ignore[arg-type]
    return out


def build_summary(targets: List[str]) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(t).read_text(encoding="utf-8")) for t in targets if per_target_path(t).exists()}
    table = {t: target_scores(raw[t]) for t in raw}
    contrasts: Dict[str, Dict[str, float]] = {}
    if "official_fixedinit" in table and "mix25_fixedinit" in table:
        delta: Dict[str, float] = {}
        for k in TABLE_KEYS:
            a = table["mix25_fixedinit"].get(k)
            b = table["official_fixedinit"].get(k)
            if a is not None and b is not None:
                delta[k] = round(a - b, 4)
        contrasts["mix25_fixedinit_minus_official_fixedinit"] = delta
    if "initial_model_baseline" in table:
        for t in ["official_fixedinit", "mix25_fixedinit"]:
            if t in table:
                delta = {}
                for k in TABLE_KEYS:
                    a = table[t].get(k)
                    b = table["initial_model_baseline"].get(k)
                    if a is not None and b is not None:
                        delta[k] = round(a - b, 4)
                contrasts[f"{t}_minus_initial_model_studies_baseline"] = delta
    payload = {
        "status": "FIXEDINIT_REPLICATION_EVAL_DONE",
        "note": "Causal official-only vs 25%-aligned replication under shared initialization on the 8x480/baseline16k WWM backbone. This is a fast-screen plus full Entity, not complete official nine-column Overall.",
        "targets_evaluated": list(raw.keys()),
        "table": table,
        "contrasts": contrasts,
        "raw_paths": {t: str(per_target_path(t)) for t in raw},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    return payload


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(summary: Dict[str, Any]) -> None:
    lines = [
        "# research — Fixed-initialization mix25 replication evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This evaluates whether the 25% aligned-data gain from research survives a shared model initialization and shared training RNG on the inherited 8×480/baseline16k WWM recipe.",
        "",
        "| target | BLiMP | Supp | EWoK | Entity_fast | Entity_full | COMPS | GPIQA | Reading | equal7_fast | equal7_fullEnt | wproxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, scores in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity", "weighted_fast_proxy",
        ]) + " |")
    lines += ["", "## Contrasts", ""]
    for name, delta in summary.get("contrasts", {}).items():
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v:+.3f}" for k, v in delta.items() if k in ["equal7_mean", "equal7_full_entity", "weighted_fast_proxy", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]))
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, choices=list(RUNS.keys()))
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--work_tag", default="fixedinit")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
    summary = build_summary(args.targets)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
