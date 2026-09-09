#!/usr/bin/env python3
"""research: Evaluate mixture arms on BabyLM fast eval + full Entity.

Evaluates chck_100M for each mixture arm (25%, 50%, 75% aligned) plus
the existing endpoints (0% = INITIAL_MODEL_STUDIES baseline, 100% = research ALIGNED).

Output: dose-response curve showing how Entity, BLiMP, Supplement, etc.
change as a function of paired-aligned data fraction.

Usage:
  python eval_mixture_fast.py --gpu 0
  python eval_mixture_fast.py --gpu 0 --targets mix_25pct mix_50pct
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

# Paths
ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
EVAL_CWD = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/mixture_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/20_mixture_eval.md')
FINAL_JSON = OUT_ROOT / "mixture_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
    "aligned_100pct": ROOT_COMPACT_EXPERIENCE / "training/runs/aligned_100M_seed43",
    "mix_25pct": ROOT_COMPACT_EXPERIENCE / "training/runs/mix_25pct_100M_seed43",
    "mix_50pct": ROOT_COMPACT_EXPERIENCE / "training/runs/mix_50pct_100M_seed43",
    "mix_75pct": ROOT_COMPACT_EXPERIENCE / "training/runs/mix_75pct_100M_seed43",
}

ALIGNED_FRACTIONS = {
    "initial_model_baseline": 0.0,
    "mix_25pct": 0.25,
    "mix_50pct": 0.50,
    "mix_75pct": 0.75,
    "aligned_100pct": 1.0,
}

# Evaluation columns
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
           "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

TABLE_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
              "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
              "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean"]

# Evaluation task configs
EVAL_TASKS = {
    "BLiMP": {
        "task": "blimp", "data_path": "evaluation_data/fast_eval/blimp_fast",
        "parse": "accuracy"},
    "Supplement": {
        "task": "blimp", "data_path": "evaluation_data/fast_eval/supplement_fast",
        "parse": "accuracy"},
    "EWoK": {
        "task": "ewok", "data_path": "evaluation_data/fast_eval/ewok_fast",
        "parse": "accuracy"},
    "Entity": {
        "task": "entity_tracking", "data_path": "evaluation_data/fast_eval/entity_tracking_fast",
        "parse": "score"},
    "Entity_full": {
        "task": "entity_tracking", "data_path": "evaluation_data/full_eval/entity_tracking",
        "parse": "score"},
    "COMPS": {
        "task": "comps", "data_path": "evaluation_data/full_eval/comps",
        "parse": "accuracy"},
    "GlobalPIQA_parallel": {
        "task": "glue_filtered", "data_path": "evaluation_data/full_eval/piqa/parallel",
        "parse": "accuracy"},
    "GlobalPIQA_nonparallel": {
        "task": "glue_filtered", "data_path": "evaluation_data/full_eval/piqa/non_parallel",
        "parse": "accuracy"},
    "Reading": {
        "task": "reading", "data_path": "evaluation_data/fast_eval/reading_fast",
        "parse": "reading"},
}


def model_path_for(target: str) -> pathlib.Path:
    return RUNS[target] / "hf_model" / "chck_100M"


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmpdir = OUT_ROOT / "tmp" / work_tag
    tmpdir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmpdir.resolve())
    for d in [hf, hf / "hub", hf / "datasets", hf / "transformers"]:
        d.mkdir(parents=True, exist_ok=True)
    return env


def parse_score(text: str, mode: str) -> Optional[float]:
    if mode == "reading":
        return parse_reading_scores(text)
    vals = re.findall(r"(?:Overall accuracy|accuracy|score)[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        if val <= 1.0:
            val *= 100
        if val < 1000:
            return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def parse_reading_scores(text: str) -> Optional[Dict[str, float]]:
    eye = re.search(r"eye[_ ]tracking[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    sp = re.search(r"self[_ ]paced[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if eye and sp:
        e = float(eye.group(1))
        s = float(sp.group(1))
        if e <= 1: e *= 100
        if s <= 1: s *= 100
        return {"eye": e, "self_paced": s, "mean": (e + s) / 2}
    return None


def run_eval_task(model_path: pathlib.Path, task_cfg: dict, env: dict) -> dict:
    task = task_cfg["task"]
    data_path = task_cfg["data_path"]
    
    cmd = [sys.executable, "-m", f"evaluation_pipeline.{task}.run",
           "--model_path", str(model_path.resolve()),
           "--data_path", data_path]
    
    out_dir = OUT_ROOT / "eval_outputs" / model_path.parent.parent.name / task / data_path.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
        cwd=str(EVAL_CWD.resolve()), timeout=600)
    
    combined = result.stdout + "\n" + result.stderr
    log_path = out_dir / "output.log"
    log_path.write_text(combined, encoding="utf-8")
    
    return {"rc": result.returncode, "output": combined, "log": str(log_path)}


def eval_target(target: str, gpu: int, columns: List[str], force: bool = False) -> Dict[str, Any]:
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    per_target = OUT_ROOT / "per_target" / f"{target}.json"
    per_target.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(per_target.read_text()) if (per_target.exists() and not force) else None
    
    payload: Dict[str, Any] = existing or {"target": target, "model_path": str(model_path), "scores": {}}
    env = setup_env(target, gpu)
    
    for col in columns:
        if col in payload.get("scores", {}) and not force:
            continue
        if col not in EVAL_TASKS:
            continue
        
        cfg = EVAL_TASKS[col]
        print(json.dumps({"event": "eval_start", "target": target, "column": col, "gpu": gpu}), flush=True)
        
        result = run_eval_task(model_path, cfg, env)
        
        if cfg["parse"] == "reading":
            scores = parse_reading_scores(result["output"])
            if scores:
                payload["scores"]["Reading"] = scores["mean"]
                payload["scores"]["Reading_eye"] = scores["eye"]
                payload["scores"]["Reading_self_paced"] = scores["self_paced"]
                print(json.dumps({"event": "eval_done", "target": target, "column": "Reading",
                                  "score": scores["mean"], "rc": result["rc"]}), flush=True)
            else:
                payload["scores"]["Reading"] = None
                print(json.dumps({"event": "eval_fail", "target": target, "column": "Reading",
                                  "rc": result["rc"]}), flush=True)
        else:
            score = parse_score(result["output"], cfg["parse"])
            payload["scores"][col] = score
            print(json.dumps({"event": "eval_done", "target": target, "column": col,
                              "score": score, "rc": result["rc"]}), flush=True)
    
    # Compute derived scores
    scores = payload["scores"]
    gp = scores.get("GlobalPIQA_parallel")
    gn = scores.get("GlobalPIQA_nonparallel")
    if gp is not None and gn is not None:
        scores["GlobalPIQA_mean"] = (gp + gn) / 2
    
    # equal-7 mean (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_mean, Reading)
    # Use Entity_full if available, else Entity (fast)
    entity = scores.get("Entity_full") or scores.get("Entity")
    cols7 = [scores.get("BLiMP"), scores.get("Supplement"), scores.get("EWoK"),
             entity, scores.get("COMPS"), scores.get("GlobalPIQA_mean"), scores.get("Reading")]
    valid7 = [c for c in cols7 if c is not None]
    scores["equal7_mean"] = sum(valid7) / len(valid7) if valid7 else None
    
    # Save per-target
    per_target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "target_done", "target": target}), flush=True)
    return payload


def build_summary(targets: List[str]) -> dict:
    """Build the final summary with dose-response table and contrasts."""
    raw = {}
    for t in targets:
        p = OUT_ROOT / "per_target" / f"{t}.json"
        if p.exists():
            raw[t] = json.loads(p.read_text())
    
    # Build table
    table = {}
    for t, data in raw.items():
        table[t] = data.get("scores", {})
    
    # Dose-response: aligned fraction → scores
    dose_response = []
    for t in sorted(ALIGNED_FRACTIONS.keys(), key=lambda x: ALIGNED_FRACTIONS[x]):
        if t in table:
            entry = {"target": t, "aligned_fraction": ALIGNED_FRACTIONS[t]}
            entry.update(table[t])
            dose_response.append(entry)
    
    # Contrasts vs baseline
    contrasts = {}
    baseline = table.get("initial_model_baseline", {})
    for t in ["mix_25pct", "mix_50pct", "mix_75pct", "aligned_100pct"]:
        if t in table and baseline:
            delta = {}
            for k in TABLE_KEYS:
                bv = baseline.get(k)
                tv = table[t].get(k)
                if bv is not None and tv is not None:
                    delta[k] = round(tv - bv, 4)
            contrasts[f"{t}_minus_baseline"] = delta
    
    # Find optimal mixture (highest equal7_mean)
    best_target = None
    best_eq7 = -999
    for t, scores in table.items():
        eq7 = scores.get("equal7_mean")
        if eq7 is not None and eq7 > best_eq7:
            best_eq7 = eq7
            best_target = t
    
    payload = {
        "status": "MIXTURE_EVAL_DONE",
        "note": "Dose-response curve: 0% (official), 25%, 50%, 75%, 100% (aligned) paired data.",
        "targets_evaluated": list(raw.keys()),
        "table": table,
        "dose_response": dose_response,
        "contrasts_vs_baseline": contrasts,
        "best_target": best_target,
        "best_equal7_mean": best_eq7,
        "aligned_fractions": ALIGNED_FRACTIONS,
    }
    
    # Save
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Write note
    lines = [
        "# research — Mixture dose-response evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON}`",
        "",
        "## Dose-response table",
        "",
        "| target | frac | BLiMP | Supp | EWoK | Entity_full | COMPS | GPIQA_mean | Reading | equal-7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in dose_response:
        t = entry["target"]
        f = entry["aligned_fraction"]
        lines.append(
            f"| {t} | {f:.0%} | "
            f"{entry.get('BLiMP', '—'):.2f} | "
            f"{entry.get('Supplement', '—'):.2f} | "
            f"{entry.get('EWoK', '—'):.2f} | "
            f"{entry.get('Entity_full', entry.get('Entity', '—')):.2f} | "
            f"{entry.get('COMPS', '—'):.2f} | "
            f"{entry.get('GlobalPIQA_mean', '—'):.2f} | "
            f"{entry.get('Reading', '—'):.3f} | "
            f"{entry.get('equal7_mean', '—'):.3f} |"
        )
    
    lines += ["", "## Contrasts vs INITIAL_MODEL_STUDIES baseline", ""]
    for k, v in contrasts.items():
        eq7d = v.get("equal7_mean", "?")
        entd = v.get("Entity_full", v.get("Entity", "?"))
        lines.append(f"- **{k}**: equal-7 {eq7d:+.3f}, Entity {entd:+.2f}")
    
    lines += ["", f"## Best arm: **{best_target}** (equal-7 = {best_eq7:.3f})", ""]
    
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--targets", nargs="*",
                        default=["initial_model_baseline", "mix_25pct", "mix_50pct", "mix_75pct", "aligned_100pct"])
    parser.add_argument("--columns", nargs="*", default=COLUMNS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--work_tag", default="mix_eval")
    args = parser.parse_args()
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    for target in args.targets:
        if target not in RUNS:
            print(f"WARNING: Unknown target {target}, skipping")
            continue
        model_path = model_path_for(target)
        if not model_path.exists():
            print(json.dumps({"event": "skip", "target": target, "reason": "no checkpoint"}), flush=True)
            continue
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.columns, force=args.force)
    
    summary = build_summary(args.targets)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
