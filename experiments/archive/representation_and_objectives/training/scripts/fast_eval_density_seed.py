#!/usr/bin/env python3
"""Fast no-AoA BabyLM task-family screen for a density seed run.

Used for independent-seed checks of the compact-view density endpoint after the
seed43022 full evaluation shows that the endpoint is worth confirming. It runs
BLiMP, Supplement, EWoK, Entity fast/full, COMPS, GlobalPIQA, and Reading through
the same local strict evaluation code used by the reference fast screens. SuperGLUE and
AoA remain separate full-evaluation work.
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

USER_ROOT = pathlib.Path(".").resolve()
ROOT_INITIAL_MODEL_STUDIES = USER_ROOT / "experiments/archive" / 'initial_model_studies'
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos" / "babylm-eval" / "strict"
PYTHON_EXE = sys.executable

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
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]

A02_REINVEST_FAST = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_reinvest" / "density_noaoa_eval_summary.json"
REFERENCE = {
    "visible_leader_card": {
        "source": "Visible BabyLM 2026 Strict-Small leaderboard/model card for wwm_curriculum_simplification_40k",
        "BLiMP": 67.20, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
        "COMPS": 53.57, "GlobalPIQA_mean": 39.67, "SuperGLUE": 69.79,
        "Reading": 5.42, "AoA": 0.0, "Overall": 41.80,
    },
    "compact_experience_clean_qwen_full": {
        "source": "COMPACT_EXPERIENCE research clean-Qwen full official-compatible result at 100M exposure",
        "Overall": 41.3443, "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19,
        "Entity": 25.76, "COMPS": 51.78, "GlobalPIQA_mean": 36.62,
        "SuperGLUE": 70.3086, "Reading": 7.76, "AoA": 0.0,
    },
}


def parse_sentence_score(text: str) -> Optional[float]:
    for pat in [
        r"### AVERAGE [A-Z_ '\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if -5.0 <= val <= 105.0:
                return val
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


def latest_file(root: pathlib.Path, pattern: str) -> Optional[pathlib.Path]:
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    for pat in ["best_temperature_report.txt", "*.txt"]:
        candidates = sorted(task_out.rglob(pat), key=lambda p: (p.stat().st_mtime, str(p)))
        for p in reversed(candidates):
            val = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
            if val is not None:
                return val
    return None


def read_reading_report(task_out: pathlib.Path) -> Dict[str, float]:
    candidates = sorted(task_out.rglob("report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def setup_env(out_root: pathlib.Path, work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / work_tag
    tmp = out_root / "tmp" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data" / "nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TMPDIR"] = str(tmp.resolve())
    for key in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 2400) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
        f.write(f"\n[returncode={proc.returncode}]\n")
    return proc


def eval_sentence(model_path: pathlib.Path, target_label: str, column: str, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = out_root / "eval_outputs" / work_tag / target_label / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target_label}_{column}.log"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data_path,
        "--revision_name", f"step025_{target_label}_{column}",
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    proc = run_cmd(cmd, env, log, timeout=2400)
    score = read_report_score(task_out)
    rec: Dict[str, Any] = {
        "column": column, "task": task, "data_path": data_path, "score": score,
        "returncode": proc.returncode, "elapsed_sec": round(time.time() - t0, 3),
        "log": str(log), "output_dir": str(task_out),
    }
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    if pred:
        rec["predictions"] = str(pred)
    if report:
        rec["report"] = str(report)
    if proc.returncode != 0:
        rec["error"] = f"returncode {proc.returncode}"
    if score is None:
        rec["score_parse_error"] = True
    return rec


def eval_reading(model_path: pathlib.Path, target_label: str, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task_out = out_root / "eval_outputs" / work_tag / target_label / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target_label}_Reading.log"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", f"step025_{target_label}_Reading",
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    proc = run_cmd(cmd, env, log, timeout=2400)
    scores = read_reading_report(task_out)
    rec: Dict[str, Any] = {"column": "Reading", "scores": scores, "returncode": proc.returncode,
                           "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "report.txt")
    if pred:
        rec["predictions"] = str(pred)
    if report:
        rec["report"] = str(report)
    if proc.returncode != 0:
        rec["error"] = f"returncode {proc.returncode}"
    if not scores or "Reading" not in scores:
        rec["score_parse_error"] = True
    return rec


def read_run_summary(run_dir: pathlib.Path, endpoint: str) -> Dict[str, Any]:
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in ["variant", "word_exposure", "loss_first", "loss_last", "actual_training_steps", "parameter_count", "vocab_size", "tokenizer_label", "masking_curriculum", "learning_rate", "warmup_fraction", "weight_decay", "n_layer", "hidden_size", "n_head", "seed", "extra_init_seed", "train_rng_seed", "example_jsonl", "example_jsonl_label"]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
    model_path = run_dir / "hf_model" / endpoint
    out["endpoint_path"] = str(model_path)
    out["endpoint_exists"] = (model_path / "model.safetensors").exists()
    return out


def scores_from_payload(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    tasks = payload.get("tasks", {})
    for col, _, _, _ in TASKS:
        rec = tasks.get(col)
        if isinstance(rec, dict):
            out[col] = rec.get("score")
    r = tasks.get("Reading")
    if isinstance(r, dict):
        for k, v in (r.get("scores") or {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    eq = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq):
        out["equal7_mean"] = sum(float(out[k]) for k in eq) / len(eq)
    eqf = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eqf):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eqf) / len(eqf)
    return out


def diff_scores(a: Dict[str, Optional[float]], b: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in TABLE_KEYS:
        av, bv = a.get(k), b.get(k)
        if av is not None and bv is not None:
            out[k] = round(float(av) - float(bv), 4)
    return out


def build_summary(target_label: str, payload: Dict[str, Any], out_root: pathlib.Path) -> Dict[str, Any]:
    scores = scores_from_payload(payload)
    table: Dict[str, Any] = {target_label: scores}
    contrasts: Dict[str, Any] = {}
    a02_fast: Dict[str, Any] = {}
    if A02_REINVEST_FAST.exists():
        a02_fast = json.loads(A02_REINVEST_FAST.read_text(encoding="utf-8"))
        for name in ["compact_view_reinvest", "compact_view_core"]:
            if name in a02_fast.get("table", {}):
                table[f"a02_seed43022_{name}"] = a02_fast["table"][name]
                contrasts[f"{target_label}_minus_a02_seed43022_{name}"] = diff_scores(scores, a02_fast["table"][name])
    for ref, ref_scores in REFERENCE.items():
        contrasts[f"{target_label}_minus_{ref}"] = diff_scores(scores, ref_scores)
    summary = {
        "status": "DENSITY_SEED_FAST_SUMMARY",
        "note": "Fast official-compatible no-AoA screen; SuperGLUE and AoA not included.",
        "target_label": target_label,
        "run_dir": payload.get("run_dir"),
        "model_path": payload.get("model_path"),
        "table": table,
        "contrasts": contrasts,
        "reference": REFERENCE,
        "a02_fast_summary": str(A02_REINVEST_FAST),
        "a02_fast": a02_fast,
        "raw_paths": {target_label: str(out_root / "per_target" / f"{target_label}.json")},
        "out_root": str(out_root),
    }
    (out_root / f"{target_label}_fast_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(summary: Dict[str, Any], target_label: str, out_root: pathlib.Path) -> None:
    lines = [
        f"# research {target_label} fast no-AoA screen",
        "",
        f"Summary JSON: `{out_root / f'{target_label}_fast_summary.json'}`",
        "",
        "This is an independent-seed task-family screen. It does not include SuperGLUE or AoA.",
        "",
        "| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scores in summary.get("table", {}).items():
        lines.append("| " + name + " | " + " | ".join(fmt(scores.get(k)) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity"]) + " |")
    lines += ["", "## Contrasts", ""]
    keep = ["equal7_mean", "equal7_full_entity", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    for name, rec in summary.get("contrasts", {}).items():
        vals = ", ".join(f"{k} {rec[k]:+.3f}" for k in keep if k in rec)
        lines.append(f"- **{name}**: {vals}")
    (out_root / f"{target_label}_fast_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-label", required=True)
    ap.add_argument("--endpoint", default="chck_100M")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    model_path = run_dir / "hf_model" / args.endpoint
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"endpoint missing: {model_path}")
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    per_target = out_root / "per_target" / f"{args.target_label}.json"
    per_target.parent.mkdir(parents=True, exist_ok=True)
    if per_target.exists() and not args.force:
        payload = json.loads(per_target.read_text(encoding="utf-8"))
    else:
        payload = {
            "target": args.target_label,
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "endpoint": args.endpoint,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": args.gpu,
            "tasks": {},
            "run_summary": read_run_summary(run_dir, args.endpoint),
        }
    work_tag = f"step025_{args.target_label}"
    env = setup_env(out_root, work_tag, args.gpu)
    for col in args.columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or args.force:
                rec = eval_reading(model_path, args.target_label, work_tag, env, out_root)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": args.target_label, "column": col, "scores": rec.get("scores"), "rc": rec.get("returncode")}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or args.force:
                rec = eval_sentence(model_path, args.target_label, col, work_tag, env, out_root)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": args.target_label, "column": col, "score": rec.get("score"), "rc": rec.get("returncode")}), flush=True)
        per_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    per_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = build_summary(args.target_label, payload, out_root)
    write_note(summary, args.target_label, out_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
