#!/usr/bin/env python3
"""Fast official-compatible no-AoA screen for independent-seed replication of
compact_view_core at seed 43122.

This script evaluates a specified run directory rather than the
fixed reference TARGETS table, while reusing the same local BabyLM strict evaluation
calls and score parsing.  It compares the replication to:
  - seed43022 compact_view_core fast result;
  - seed43022 compact_repeat_core fast result;
  - COMPACT_EXPERIENCE clean-Qwen full endpoint columns;
  - the visible 41.8 leader card columns.

It does not run SuperGLUE or AoA; those remain full official-compatible evidence.
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
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
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

A02_FAST_SUMMARY = pathlib.Path("experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json")

REFERENCE = {
    "visible_leader_card": {
        "source": "Visible BabyLM 2026 Strict-Small leaderboard/model card for wwm_curriculum_simplification_40k",
        "BLiMP": 67.20,
        "Supplement": 56.01,
        "EWoK": 56.07,
        "Entity": 28.45,
        "COMPS": 53.57,
        "GlobalPIQA_mean": 39.67,
        "SuperGLUE": 69.79,
        "Reading": 5.42,
        "AoA": 0.0,
        "Overall": 41.80,
    },
    "compact_experience_clean_qwen_full": {
        "source": "COMPACT_EXPERIENCE research clean-Qwen full official-compatible result at 100M exposure",
        "Overall": 41.3443,
        "BLiMP": 66.84,
        "Supplement": 62.84,
        "EWoK": 50.19,
        "Entity": 25.76,
        "COMPS": 51.78,
        "GlobalPIQA_mean": 36.62,
        "SuperGLUE": 70.3086,
        "Reading": 7.76,
        "AoA": 0.0,
    },
}


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


def setup_env(out_root: pathlib.Path, work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmp = out_root / "tmp" / work_tag
    env["TMPDIR"] = str(tmp.resolve())
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 2400) -> subprocess.CompletedProcess[str]:
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


def eval_sentence_column(model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    target = "compact_view_core_seed43122"
    task_out = out_root / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"repl_{column}"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.sentence_zero_shot.run",
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
    p = run_cmd(cmd, env, log, timeout=2400)
    stdout_score = parse_sentence_score(p.stdout)
    report_score = read_report_score(task_out)
    score = stdout_score if stdout_score is not None else report_score
    return {"column": column, "task": task, "data_path": data_path, "score": score,
            "stdout_score": stdout_score, "report_score": report_score,
            "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3),
            "log": str(log), "output_dir": str(task_out)}


def eval_reading(model_path: pathlib.Path, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    target = "compact_view_core_seed43122"
    task_out = out_root / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_Reading.log"
    revision = "repl_Reading"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=2400)
    scores = parse_reading_scores(p.stdout)
    if not scores:
        scores = read_reading_report(task_out)
    return {"column": "Reading", "scores": scores, "returncode": p.returncode,
            "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def target_scores_from_payload(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    tasks = raw.get("tasks", {})
    for col in [c for c, _, _, _ in TASKS]:
        if col in tasks:
            out[col] = tasks[col].get("score")
    if "Reading" in tasks:
        for k, v in tasks["Reading"].get("scores", {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_keys):
        out["equal7_mean"] = sum(float(out[k]) for k in eq_keys) / len(eq_keys)
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eq_full_keys) / len(eq_full_keys)
    return out


def diff_scores(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in TABLE_KEYS:
        av = a.get(k); bv = b.get(k)
        if av is not None and bv is not None:
            out[k] = round(float(av) - float(bv), 4)
    return out


def read_run_summary(run_dir: pathlib.Path, endpoint: str) -> Dict[str, Any]:
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "variant", "word_exposure", "loss_first", "loss_last", "actual_training_steps",
            "parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path",
            "masking_curriculum", "mask_prob_start", "mask_prob_end", "learning_rate",
            "warmup_fraction", "weight_decay", "n_layer", "hidden_size", "n_head",
            "max_position_embeddings", "max_relative_positions", "seed", "extra_init_seed", "train_rng_seed",
            "data_source_type", "example_jsonl", "example_jsonl_label",
        ]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
    model_path = run_dir / "hf_model" / endpoint
    out["endpoint_path"] = str(model_path)
    out["endpoint_exists"] = (model_path / "model.safetensors").exists()
    return out


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(summary: Dict[str, Any], out_root: pathlib.Path) -> None:
    table = summary.get("table", {})
    lines = [
        "# research compact_view_core seed43122 replication fast screen",
        "",
        f"Summary JSON: `{out_root / 'compact_view_core_seed43122_fast_summary.json'}`",
        "",
        "This is A01's independent-seed fast no-AoA screen of A02's strongest compact-view density candidate.",
        "It does not include SuperGLUE or AoA; A02's running full official-compatible evaluation remains the full-score evidence.",
        "",
        "| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, scores in table.items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity",
        ]) + " |")
    lines += ["", "## Contrasts", ""]
    keep = ["equal7_mean", "equal7_full_entity", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    for name, delta in summary.get("contrasts", {}).items():
        vals = ", ".join(f"{k} {delta[k]:+.3f}" for k in keep if k in delta)
        lines.append(f"- **{name}**: {vals}")
    lines += ["", "## Interpretation rule", "",
              "If seed43122 remains close to A02 seed43022 compact_view_core (equal7_full_entity 43.928) and stays well above the A02 compact_repeat_core (41.766), the compact-view density mechanism is reproducible enough to prioritize full evaluation and submission engineering. If it collapses toward the repeat baseline, the seed43022 fast-screen surface is fragile and the route must be reinterpreted before more GPU work."]
    (out_root / "compact_view_core_seed43122_fast_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--endpoint", default="chck_100M")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default="experiments/archive/representation_and_objectives/data/compact_view_core_replication")
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    model_path = run_dir / "hf_model" / args.endpoint
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"replication endpoint missing: {model_path}")
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    per_target = out_root / "per_target" / "compact_view_core_seed43122.json"
    per_target.parent.mkdir(parents=True, exist_ok=True)
    payload = None if args.force or not per_target.exists() else json.loads(per_target.read_text(encoding="utf-8"))
    if payload is None:
        payload = {
            "target": "compact_view_core_seed43122",
            "model_path": str(model_path),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": args.gpu,
            "work_tag": "compact_view_core_repl_seed43122",
            "tasks": {},
            "run_summary": read_run_summary(run_dir, args.endpoint),
        }
    env = setup_env(out_root, "compact_view_core_repl_seed43122", args.gpu)
    for col in args.columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or args.force:
                rec = eval_reading(model_path, "compact_view_core_repl_seed43122", env, out_root)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or args.force:
                rec = eval_sentence_column(model_path, col, "compact_view_core_repl_seed43122", env, out_root)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        per_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    per_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    repl_scores = target_scores_from_payload(payload)
    table = {"compact_view_core_seed43122": repl_scores}
    contrasts: Dict[str, Dict[str, float]] = {}
    a02_summary: Dict[str, Any] = {}
    if A02_FAST_SUMMARY.exists():
        a02_summary = json.loads(A02_FAST_SUMMARY.read_text(encoding="utf-8"))
        for name in ["compact_view_core", "compact_repeat_core"]:
            if name in a02_summary.get("table", {}):
                table[f"a02_seed43022_{name}"] = a02_summary["table"][name]
        if "compact_view_core" in a02_summary.get("table", {}):
            contrasts["seed43122_minus_a02_seed43022_compact_view_core"] = diff_scores(repl_scores, a02_summary["table"]["compact_view_core"])
        if "compact_repeat_core" in a02_summary.get("table", {}):
            contrasts["seed43122_minus_a02_seed43022_compact_repeat_core"] = diff_scores(repl_scores, a02_summary["table"]["compact_repeat_core"])
    for ref_name, ref_scores in REFERENCE.items():
        contrasts[f"seed43122_minus_{ref_name}_columns"] = diff_scores(repl_scores, ref_scores)  # type: ignore[arg-type]

    summary = {
        "status": "COMPACT_VIEW_CORE_SEED43122_FAST_REPLICATION_SUMMARY",
        "note": "Fast official-compatible no-AoA replication screen; SuperGLUE and AoA not included.",
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "reference": REFERENCE,
        "a02_fast_summary": str(A02_FAST_SUMMARY),
        "table": table,
        "contrasts": contrasts,
        "raw_paths": {"compact_view_core_seed43122": str(per_target)},
        "out_root": str(out_root),
    }
    out_json = out_root / "compact_view_core_seed43122_fast_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(summary, out_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
