#!/usr/bin/env python3
"""research fast official-compatible evaluation for repaired pilot runs.

This script evaluates local repaired training runs with the same BabyLM sentence
zero-shot and Reading calls used in prior COMPACT_EXPERIENCE screens.  It writes per-run JSON,
logs, predictions, and a compact summary under data.  The result is a
fast task screen for research decisions, not the final complete BabyLM submission
package.
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
ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT / "training/runs"
OUT_ROOT = ROOT / "data/repaired_pilot_eval"
NOTE_PATH = (ROOT.parents[2] / 'research/notes/frontier_consolidation/repaired_pilot_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "repaired_pilot_eval_summary.json"

TARGETS: Dict[str, Dict[str, Any]] = {
    "stagewise_qwen10_12x384_40k_seed44011": {
        "run_dir": RUN_BASE / "qwen10_stagewise_rowchunk_40k_12x384_lamb_bf16_seed44011",
        "endpoint": "chck_10M",
        "description": "Repaired low-truncation clean-Qwen 10M row-chunk stagewise run; DeBERTa-v2 12x384, 40k tokenizer, LAMB lr0.007, WWM->token at 7M, bf16 forward.",
    },
    "pairfit_qwen10_12x384_40k_seed44011": {
        "run_dir": RUN_BASE / "qwen10_stagewise_pairfit_40k_12x384_lamb_bf16_seed44011",
        "endpoint": "chck_10M",
        "description": "Pair-boundary-aware clean-Qwen 10M stagewise run; same DeBERTa-v2 12x384, 40k tokenizer, LAMB lr0.007, WWM->token at 7M, bf16 forward as row-chunked arm, but every Qwen original+rewrite pair remains atomic in one attention window.",
    },
    "baseline_qwen10_fixed256_8x480_16k_seed44021": {
        "run_dir": RUN_BASE / "qwen10_fixed256_8x480_16k_adamw_bf16_seed44021",
        "endpoint": "chck_10M",
        "description": "Matched clean-Qwen 10M fixed-seq256 baseline to be trained; DeBERTa-v2 8x480, 16k tokenizer, AdamW, WWM fixed, bf16 forward.",
    },
    "stagewise_qwen10_8x480_16k_seed44031": {
        "run_dir": RUN_BASE / "qwen10_stagewise_rowchunk_16k_8x480_lamb_bf16_seed44031",
        "endpoint": "chck_10M",
        "description": "Repaired low-truncation clean-Qwen 10M row-chunk stagewise run planned for architecture/tokenizer isolation; DeBERTa-v2 8x480, 16k tokenizer, LAMB lr0.007.",
    },
}

REFERENCE = {
    "leader_card": {
        "source": "BabyLM 2026 leaderboard/model card values for wwm_curriculum_simplification_40k",
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
        "source": "COMPACT_EXPERIENCE research full official-compatible result at 100M exposure",
        "summary": "experiments/archive/compact_experience/data/full_eval/full_eval_summary.json",
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
    "compact_experience_stagewise_phase2_mix25_fast": {
        "source": "COMPACT_EXPERIENCE research corrected stagewise 12x384/LAMB/40k mix25 fast screen",
        "summary": "experiments/archive/compact_experience/data/phase2_stagewise_eval/phase2_stagewise_eval_summary.json",
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


def model_path_for(target: str) -> pathlib.Path:
    t = TARGETS[target]
    return pathlib.Path(t["run_dir"]) / "hf_model" / str(t["endpoint"])


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


def eval_sentence_column(target: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"step004_{target}_{column}"
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
    revision = f"step004_{target}_Reading"
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
    cfg = TARGETS[target]
    run_dir = pathlib.Path(cfg["run_dir"])
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "description": cfg.get("description", ""), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "variant", "word_exposure", "loss_first", "loss_last", "actual_training_steps",
            "total_scheduled_steps", "parameter_count", "embedding_parameter_count",
            "non_embedding_parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path",
            "mask_mode", "mask_switch_mode", "mask_switch_words", "mask_prob", "optimizer",
            "learning_rate", "warmup_fraction", "weight_decay", "amp_dtype", "n_layer", "hidden_size", "n_head",
            "intermediate_size", "share_att_key", "position_biased_input", "max_position_embeddings",
            "max_relative_positions", "seed", "extra_init_seed", "train_rng_seed", "stage_config", "stage_metrics",
        ]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
    model_path = model_path_for(target)
    out["endpoint_path"] = str(model_path)
    out["endpoint_exists"] = (model_path / "model.safetensors").exists()
    return out


def inspect_mask_switch(target: str) -> Dict[str, Any]:
    p = pathlib.Path(TARGETS[target]["run_dir"]) / "training_log.jsonl"
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
        slim = {"step": rec.get("step"), "stage": rec.get("stage"),
                "cumulative_word_exposure": rec.get("cumulative_word_exposure"),
                "mask_mode": mode, "loss": rec.get("loss")}
        if mode == "token" and first_token is None:
            first_token = slim
        if mode == "wwm":
            last_wwm = slim
    out.update({"records": n, "mode_counts": counts, "first_token_record": first_token, "last_wwm_record": last_wwm})
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
                   "gpu": gpu, "work_tag": work_tag, "tasks": {},
                   "run_summary": read_run_summary(target),
                   "mask_switch_inspection": inspect_mask_switch(target)}
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
    return out


def build_summary(targets: List[str]) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(t).read_text(encoding="utf-8")) for t in targets if per_target_path(t).exists()}
    table = {t: target_scores_from_payload(raw[t]) for t in raw}
    contrasts: Dict[str, Dict[str, float]] = {}
    leader = REFERENCE["leader_card"]
    for t, scores in table.items():
        delta: Dict[str, float] = {}
        for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]:
            a = scores.get(k)
            b = leader.get(k)
            if a is not None and b is not None:
                delta[k] = round(a - float(b), 4)
        if delta:
            contrasts[f"{t}_minus_leader_card_columns"] = delta
    if "stagewise_qwen10_12x384_40k_seed44011" in table and "baseline_qwen10_fixed256_8x480_16k_seed44021" in table:
        delta = {}
        a = table["stagewise_qwen10_12x384_40k_seed44011"]
        b = table["baseline_qwen10_fixed256_8x480_16k_seed44021"]
        for k in TABLE_KEYS:
            av = a.get(k); bv = b.get(k)
            if av is not None and bv is not None:
                delta[k] = round(av - bv, 4)
        contrasts["stagewise12_40k_minus_fixed8_16k"] = delta
    best = None
    cands = [(t, s.get("equal7_mean")) for t, s in table.items() if s.get("equal7_mean") is not None]
    if cands:
        best = max(cands, key=lambda x: x[1])
    payload = {
        "status": "REPAIRED_PILOT_EVAL_DONE",
        "note": "Fast official-compatible research screen; SuperGLUE and AoA are not included here.",
        "targets_evaluated": list(raw.keys()),
        "reference": REFERENCE,
        "table": table,
        "contrasts": contrasts,
        "best_by_equal7_fast": best,
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
        "# research — Repaired pilot fast evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This uses the official-compatible fast sentence-zero-shot and Reading calls to decide whether the repaired leader-style training dynamics deserve larger H100 runs. It does not include SuperGLUE or AoA.",
        "",
        "## Fast-screen table",
        "",
        "| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, scores in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity",
        ]) + " |")
    lines += ["", "## Contrasts", ""]
    for name, delta in summary.get("contrasts", {}).items():
        keep = ["equal7_mean", "equal7_full_entity", "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v:+.3f}" for k, v in delta.items() if k in keep))
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=["stagewise_qwen10_12x384_40k_seed44011"], choices=sorted(TARGETS))
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--work_tag", default="repaired_pilot_eval")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
    summary = build_summary(list(args.targets))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
