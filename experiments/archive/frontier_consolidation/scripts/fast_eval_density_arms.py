#!/usr/bin/env python3
"""Fast official-compatible no-AoA evaluation for research density-overlay arms.

This is a research screen, not a final BabyLM submission package. It reuses the
same local BabyLM strict evaluation calls as earlier REPRESENTATION_FRONTIER_STUDIES/COMPACT_EXPERIENCE screens for:
BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS, GlobalPIQA, and Reading.
SuperGLUE and AoA remain separate full-evaluation work for promising endpoints.
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
ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT / "training/runs"
OUT_ROOT_DEFAULT = ROOT / "data/density_noaoa_eval"
PYTHON_EXE = sys.executable

TARGETS: Dict[str, Dict[str, Any]] = {
    "near_repeat": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_repeat_near_core_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_repeat_near_core",
        "description": "Clean-Qwen row-holdout base with medium risk-hard FineWeb near-core source repeated locally; source/control for near-view value.",
    },
    "near_view": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_near_view_core_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_near_view_core",
        "description": "Same near-core FineWeb sources paired with near-length Qwen3.5 views; isolates generated near-view structure against near_repeat.",
    },
    "compact_repeat_core": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_repeat_compact_core_neutral",
        "description": "Compact-core source repetition plus neutral clean-Qwen top-up; source/control for compact core view value.",
    },
    "compact_view_core": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_compact_view_core_neutral",
        "description": "Same compact core sources with compact Qwen3.5 views plus neutral clean-Qwen top-up; isolates compact view structure against compact_repeat_core.",
    },
    "compact_view_reinvest": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_compact_view_reinvest",
        "description": "Compact views with saved word budget reinvested into additional compact source-view packets; tests source diversity gain over compact_view_core.",
    },
    "compact_repeat_reinvest": {
        "run_dir": RUN_BASE / "cleanqwen_fineweb_repeat_compact_reinvest_16k_seed43022",
        "endpoint": "chck_100M",
        "arm": "cleanqwen_fineweb_repeat_compact_reinvest",
        "description": "Optional repeat counterpart for compact reinvest; isolates added-source breadth without generated view transform.",
    },
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
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]
REFERENCE = {
    "leader_card": {
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


def model_path_for(target: str) -> pathlib.Path:
    t = TARGETS[target]
    return pathlib.Path(t["run_dir"]) / "hf_model" / str(t["endpoint"])


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


def eval_sentence_column(target: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = out_root / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"step016_{target}_{column}"
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


def eval_reading(target: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str], out_root: pathlib.Path) -> Dict[str, Any]:
    task_out = out_root / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_Reading.log"
    revision = f"step016_{target}_Reading"
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


def per_target_path(out_root: pathlib.Path, target: str) -> pathlib.Path:
    return out_root / "per_target" / f"{target}.json"


def save_target(out_root: pathlib.Path, target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(out_root, target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_run_summary(target: str) -> Dict[str, Any]:
    cfg = TARGETS[target]
    run_dir = pathlib.Path(cfg["run_dir"])
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"target": target, "arm": cfg.get("arm"), "run_dir": str(run_dir), "description": cfg.get("description", ""), "scientific_metrics_path": str(p)}
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
    model_path = model_path_for(target)
    out["endpoint_path"] = str(model_path)
    out["endpoint_exists"] = (model_path / "model.safetensors").exists()
    return out


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], out_root: pathlib.Path, force: bool = False) -> Dict[str, Any]:
    model_path = model_path_for(target)
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"endpoint missing for {target}: {model_path}")
    p = per_target_path(out_root, target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"target": target, "model_path": str(model_path),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpu": gpu, "work_tag": work_tag, "tasks": {},
                   "run_summary": read_run_summary(target)}
    env = setup_env(out_root, f"{work_tag}_{target}", gpu)
    for col in columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or force:
                rec = eval_reading(target, model_path, work_tag, env, out_root)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or force:
                rec = eval_sentence_column(target, model_path, col, work_tag, env, out_root)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        save_target(out_root, target, payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(out_root, target, payload)
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
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0  # type: ignore[operator]
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_keys):
        out["equal7_mean"] = sum(out[k] for k in eq_keys) / len(eq_keys)  # type: ignore[arg-type]
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(out[k] for k in eq_full_keys) / len(eq_full_keys)  # type: ignore[arg-type]
    return out


def diff_scores(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in TABLE_KEYS:
        av = a.get(k); bv = b.get(k)
        if av is not None and bv is not None:
            out[k] = round(float(av) - float(bv), 4)
    return out


def build_summary(targets: List[str], out_root: pathlib.Path) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(out_root, t).read_text(encoding="utf-8")) for t in targets if per_target_path(out_root, t).exists()}
    table = {t: target_scores_from_payload(raw[t]) for t in raw}
    contrasts: Dict[str, Dict[str, float]] = {}
    if "near_view" in table and "near_repeat" in table:
        contrasts["near_view_minus_near_repeat"] = diff_scores(table["near_view"], table["near_repeat"])
    if "compact_view_core" in table and "compact_repeat_core" in table:
        contrasts["compact_view_core_minus_compact_repeat_core"] = diff_scores(table["compact_view_core"], table["compact_repeat_core"])
    if "compact_view_reinvest" in table and "compact_view_core" in table:
        contrasts["compact_view_reinvest_minus_compact_view_core"] = diff_scores(table["compact_view_reinvest"], table["compact_view_core"])
    if "compact_view_reinvest" in table and "compact_repeat_reinvest" in table:
        contrasts["compact_view_reinvest_minus_compact_repeat_reinvest"] = diff_scores(table["compact_view_reinvest"], table["compact_repeat_reinvest"])
    for t, scores in table.items():
        leader_delta = {}
        for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]:
            if scores.get(k) is not None and REFERENCE["leader_card"].get(k) is not None:
                leader_delta[k] = round(float(scores[k]) - float(REFERENCE["leader_card"][k]), 4)  # type: ignore[index]
        if leader_delta:
            contrasts[f"{t}_minus_visible_leader_columns"] = leader_delta
    best = None
    cands = [(t, s.get("equal7_mean")) for t, s in table.items() if s.get("equal7_mean") is not None]
    if cands:
        best = max(cands, key=lambda x: x[1])
    payload = {
        "status": "DENSITY_NOAOA_EVAL_SUMMARY",
        "note": "Fast official-compatible research screen; SuperGLUE and AoA are not included.",
        "targets_evaluated": list(raw.keys()),
        "reference": REFERENCE,
        "table": table,
        "contrasts": contrasts,
        "best_by_equal7_fast": best,
        "raw_paths": {t: str(per_target_path(out_root, t)) for t in raw},
        "out_root": str(out_root),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "density_noaoa_eval_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload, out_root)
    return payload


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(summary: Dict[str, Any], out_root: pathlib.Path) -> None:
    lines = [
        "# research density-overlay no-AoA task-family screen",
        "",
        f"Summary JSON: `{out_root / 'density_noaoa_eval_summary.json'}`",
        "",
        "This screen asks whether FineWeb second views on a clean-Qwen row-holdout base improve BabyLM task families before any full official evaluation.",
        "",
        "| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, scores in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity",
        ]) + " |")
    lines += ["", "## Mechanism contrasts", ""]
    for name, delta in summary.get("contrasts", {}).items():
        keep = ["equal7_mean", "equal7_full_entity", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
        vals = ", ".join(f"{k} {delta[k]:+.3f}" for k in keep if k in delta)
        lines.append(f"- **{name}**: {vals}")
    (out_root / "density_noaoa_eval_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=["near_repeat", "near_view"], choices=sorted(TARGETS))
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--work_tag", default="density_noaoa_eval")
    p.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--list-targets", action="store_true")
    args = p.parse_args()
    if args.list_targets:
        print(json.dumps({t: {"run_dir": str(v["run_dir"]), "endpoint": v["endpoint"], "description": v["description"]} for t, v in TARGETS.items()}, indent=2, ensure_ascii=False))
        return
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu, "model_path": str(model_path_for(target))}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, out_root, force=args.force)
    summary = build_summary(list(args.targets), out_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
