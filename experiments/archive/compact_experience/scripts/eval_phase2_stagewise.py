#!/usr/bin/env python3
"""research: evaluate corrected Phase-2 stagewise official vs mix25 runs.

This script uses the known-good BabyLM strict fast-screen invocation pattern from
Steps 18/21/23:
  - evaluation_pipeline.sentence_zero_shot.run for BLiMP, Supplement, EWoK,
    Entity, COMPS, GlobalPIQA, and optional full Entity;
  - evaluation_pipeline.reading.run for Reading.

It evaluates the two corrected research stagewise Phase-2 checkpoints and compares
against the previously evaluated INITIAL_MODEL_STUDIES inherited baseline using the same pattern.
The output is a fast-screen plus full-Entity comparison, not a complete official
nine-column Overall because SuperGLUE and AoA are not run here.
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
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/phase2_stagewise_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/phase2_stagewise_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "phase2_stagewise_eval_summary.json"
CACHED_INITIAL_MODEL_STUDIES_SUMMARY = ROOT_COMPACT_EXPERIENCE / "data/mixture_eval/mixture_eval_summary.json"

RUN_DIRS: Dict[str, pathlib.Path] = {
    "phase2s_official": RUN_BASE / "phase2s_official_40k_12x384_seed43200",
    "phase2s_mix25": RUN_BASE / "phase2s_mix25_40k_12x384_seed43200",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
}
DEFAULT_TARGETS = ["phase2s_official", "phase2s_mix25"]
ALL_TARGETS = DEFAULT_TARGETS + ["initial_model_baseline"]

LEADER_FAST_REFERENCE = {
    "source": "BabyLM 2026 leaderboard/model card for wwm_curriculum_simplification_40k; columns are official/card values, not local fast-screen reruns.",
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
}

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
    "Reading", "Reading_eye", "Reading_self_paced",
    "equal7_mean", "equal7_full_entity", "weighted_fast_proxy",
]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def model_path_for(target: str) -> pathlib.Path:
    return RUN_DIRS[target] / "hf_model" / "chck_100M"


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
    revision = f"phase2s_{target}_{column}"
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
    p = run_cmd(cmd, env, log, timeout=2400)
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
    revision = f"phase2s_{target}_Reading"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
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


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_run_summary(target: str) -> Dict[str, Any]:
    run_dir = RUN_DIRS[target]
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "variant", "word_exposure", "loss_first", "loss_last", "actual_training_steps",
            "total_scheduled_steps", "parameter_count", "embedding_parameter_count",
            "non_embedding_parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path",
            "mask_mode", "mask_switch_mode", "mask_switch_words", "mask_prob", "optimizer",
            "learning_rate", "warmup_fraction", "weight_decay", "n_layer", "hidden_size", "n_head",
            "intermediate_size", "share_att_key", "position_biased_input", "seed", "extra_init_seed",
            "train_rng_seed", "stage_config", "stage_metrics",
        ]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
        out["chck_100M_path"] = str(run_dir / "hf_model" / "chck_100M")
        out["chck_100M_exists"] = (run_dir / "hf_model" / "chck_100M" / "model.safetensors").exists()
    return out


def inspect_mask_switch(target: str) -> Dict[str, Any]:
    p = RUN_DIRS[target] / "training_log.jsonl"
    out: Dict[str, Any] = {"training_log": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    counts: Dict[str, int] = {}
    first_token: Optional[Dict[str, Any]] = None
    last_wwm: Optional[Dict[str, Any]] = None
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        mode = rec.get("mask_mode")
        counts[mode] = counts.get(mode, 0) + 1
        slim = {"step": rec.get("step"), "stage": rec.get("stage"), "cumulative_word_exposure": rec.get("cumulative_word_exposure"), "mask_mode": mode, "loss": rec.get("loss")}
        if mode == "token" and first_token is None:
            first_token = slim
        if mode == "wwm":
            last_wwm = slim
    out.update({"records": n, "mode_counts": counts, "first_token_record": first_token, "last_wwm_record": last_wwm})
    return out


def check_phase2_contract(target: str) -> Dict[str, Any]:
    m = read_run_summary(target)
    expected_tokenizer = "experiments/archive/compact_experience/data/shared_tokenizer/hf_tokenizer_40k_shared"
    checks = {
        "word_exposure_100M": m.get("word_exposure") == 100_000_000,
        "steps_5984": m.get("actual_training_steps") == 5984,
        "param_count_34677184": m.get("parameter_count") == 34_677_184,
        "vocab_40000": m.get("vocab_size") == 40_000,
        "extra_init_seed_43200": m.get("extra_init_seed") == 43200,
        "train_rng_seed_43201": m.get("train_rng_seed") == 43201,
        "shared_tokenizer_path": m.get("tokenizer_path") == expected_tokenizer,
        "mask_switch_words_70M": m.get("mask_switch_words") == 70_000_000,
        "mask_switch_mode_token": m.get("mask_switch_mode") == "token",
        "chck_100M_exists": bool(m.get("chck_100M_exists")),
    }
    sm = m.get("stage_metrics") or []
    if len(sm) == 3:
        checks["stage_words_30_30_40M"] = [s.get("words") for s in sm] == [30_000_000, 30_000_000, 40_000_000]
        checks["stage_visible_groups_near_one"] = all((s.get("visible_word_groups_per_word") or 0) >= 0.97 for s in sm)
    else:
        checks["stage_words_30_30_40M"] = False
        checks["stage_visible_groups_near_one"] = False
    switch = inspect_mask_switch(target)
    checks["training_log_records_5984"] = switch.get("records") == 5984
    first_token = switch.get("first_token_record") or {}
    checks["observed_token_after_70M"] = first_token.get("cumulative_word_exposure", 0) >= 70_000_000
    return {"target": target, "checks": checks, "all_passed": all(checks.values()), "mask_switch_inspection": switch, "run_summary": m}


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = per_target_path(target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"target": target, "model_path": str(model_path),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpu": gpu, "work_tag": work_tag, "tasks": {},
                   "run_summary": read_run_summary(target),
                   "phase2_contract": check_phase2_contract(target)}
    env = setup_env(f"{work_tag}_{target}", gpu)
    for col in columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or force:
                rec = eval_reading(target, model_path, work_tag, env)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or force:
                rec = eval_sentence_column(target, model_path, col, work_tag, env)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        save_target(target, payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(target, payload)
    return payload


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


def cached_initial_model_studies_scores() -> Dict[str, Any]:
    summary = json.loads(CACHED_INITIAL_MODEL_STUDIES_SUMMARY.read_text(encoding="utf-8"))
    return summary["table"]["initial_model_baseline"]


def build_summary(targets: List[str], include_cached_initial_model_studies: bool = True) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(t).read_text(encoding="utf-8")) for t in targets if per_target_path(t).exists()}
    table = {t: target_scores_from_payload(raw[t]) for t in raw}
    if include_cached_initial_model_studies:
        table["initial_model_baseline"] = cached_initial_model_studies_scores()
    elif "initial_model_baseline" in raw:
        table["initial_model_baseline"] = target_scores_from_payload(raw["initial_model_baseline"])

    contrasts: Dict[str, Dict[str, float]] = {}
    if "phase2s_official" in table and "phase2s_mix25" in table:
        delta: Dict[str, float] = {}
        for k in TABLE_KEYS:
            a = table["phase2s_mix25"].get(k)
            b = table["phase2s_official"].get(k)
            if a is not None and b is not None:
                delta[k] = round(a - b, 4)
        contrasts["phase2s_mix25_minus_phase2s_official"] = delta
    if "initial_model_baseline" in table:
        for t in ["phase2s_official", "phase2s_mix25"]:
            if t in table:
                delta = {}
                for k in TABLE_KEYS:
                    a = table[t].get(k)
                    b = table["initial_model_baseline"].get(k)
                    if a is not None and b is not None:
                        delta[k] = round(a - b, 4)
                contrasts[f"{t}_minus_initial_model_studies_baseline"] = delta
    for t in ["phase2s_official", "phase2s_mix25"]:
        if t in table:
            delta = {}
            for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]:
                a = table[t].get(k)
                b = LEADER_FAST_REFERENCE.get(k)
                if a is not None and b is not None:
                    delta[k] = round(a - b, 4)  # type: ignore[operator]
            contrasts[f"{t}_minus_leader_local_columns"] = delta

    verification = {t: check_phase2_contract(t) for t in ["phase2s_official", "phase2s_mix25"] if t in RUN_DIRS}
    best_by_eq7 = None
    if table:
        candidates = [(t, s.get("equal7_mean")) for t, s in table.items() if s.get("equal7_mean") is not None]
        if candidates:
            best_by_eq7 = max(candidates, key=lambda x: x[1])
    payload = {
        "status": "PHASE2_STAGEWISE_EVAL_DONE",
        "note": "Corrected stagewise Phase-2 fast-screen evaluation. This is not complete official nine-column Overall because SuperGLUE and AoA are not evaluated here; Entity_full is included as a reliability check.",
        "targets_evaluated": list(raw.keys()),
        "initial_model_studies_baseline_source": str(CACHED_INITIAL_MODEL_STUDIES_SUMMARY) if include_cached_initial_model_studies else str(per_target_path("initial_model_baseline")),
        "leader_reference": LEADER_FAST_REFERENCE,
        "phase2_training_verification": verification,
        "table": table,
        "contrasts": contrasts,
        "best_by_equal7_fast": best_by_eq7,
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
        "# research — Corrected Phase 2 stagewise fast evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This evaluates the corrected 12×384/LAMB/40k stagewise Phase-2 runs from research. It uses the same sentence-zero-shot and Reading invocation pattern as the repaired research/research screens. It is not a complete official nine-column Overall because SuperGLUE and AoA are not yet evaluated.",
        "",
        "## Training verification",
        "",
    ]
    for t, ver in summary.get("phase2_training_verification", {}).items():
        checks = ver.get("checks", {})
        failed = [k for k, v in checks.items() if not v]
        run = ver.get("run_summary", {})
        modes = ver.get("mask_switch_inspection", {}).get("mode_counts", {})
        first_token = ver.get("mask_switch_inspection", {}).get("first_token_record")
        lines.append(f"- `{t}`: all_passed={ver.get('all_passed')}, words={run.get('word_exposure')}, steps={run.get('actual_training_steps')}, params={run.get('parameter_count')}, vocab={run.get('vocab_size')}, loss {fmt(run.get('loss_first'))}→{fmt(run.get('loss_last'))}, modes={modes}, first_token={first_token}, failed={failed}")
    lines += [
        "",
        "## Fast-screen table",
        "",
        "| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt | wproxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, scores in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity", "weighted_fast_proxy",
        ]) + " |")
    lines += ["", "## Contrasts", ""]
    for name, delta in summary.get("contrasts", {}).items():
        keep = ["equal7_mean", "equal7_full_entity", "weighted_fast_proxy", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v:+.3f}" for k, v in delta.items() if k in keep))
    lines += [
        "",
        "## Scientific reading",
        "",
        "The summary JSON should be used for the authoritative numbers. The comparison tests whether the low-dose aligned-data gain survives when tokenizer, architecture, optimizer, sequence curriculum, initialization and training RNG are held fixed in the stronger 12×384/LAMB/40k recipe.",
    ]
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, choices=ALL_TARGETS)
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--work_tag", default="phase2_stagewise")
    p.add_argument("--force", action="store_true")
    p.add_argument("--eval_initial_model_studies", action="store_true", help="rerun the INITIAL_MODEL_STUDIES baseline instead of using the cached research same-pattern scores")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = list(args.targets)
    if args.eval_initial_model_studies and "initial_model_baseline" not in targets:
        targets.append("initial_model_baseline")
    for target in targets:
        if target == "initial_model_baseline" and not args.eval_initial_model_studies:
            continue
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
    summary = build_summary(targets, include_cached_initial_model_studies=not args.eval_initial_model_studies)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
