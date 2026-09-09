#!/usr/bin/env python3
"""research: trajectory probe for corrected Phase-2 stagewise checkpoints.

Evaluates selected intermediate checkpoints with the same fast-screen columns as
`eval_phase2_stagewise.py`, excluding full Entity by default to keep the
causal trajectory probe lightweight. The main scientific purpose is to test
whether the 70M WWM-to-token switch and the final 30M stage improve or damage the
Phase-2 endpoints.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Dict, Iterable, Optional

import eval_phase2_stagewise as ev  # reuse known-good invocation/parsers

ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/phase2_trajectory_eval"
FINAL_JSON_PATH = OUT_ROOT / "phase2_trajectory_eval_summary.json"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/24_phase2_trajectory_eval.md')
PHASE2_100M_SUMMARY = ROOT_COMPACT_EXPERIENCE / "data/phase2_stagewise_eval/phase2_stagewise_eval_summary.json"

MODEL_PATHS = {
    "official_70M": RUN_BASE / "phase2s_official_40k_12x384_seed43200/hf_model/chck_70M",
    "official_80M": RUN_BASE / "phase2s_official_40k_12x384_seed43200/hf_model/chck_80M",
    "mix25_70M": RUN_BASE / "phase2s_mix25_40k_12x384_seed43200/hf_model/chck_70M",
    "mix25_80M": RUN_BASE / "phase2s_mix25_40k_12x384_seed43200/hf_model/chck_80M",
    "mix25_90M": RUN_BASE / "phase2s_mix25_40k_12x384_seed43200/hf_model/chck_90M",
}
DEFAULT_TARGETS = ["official_70M", "mix25_70M", "mix25_80M", "mix25_90M"]
DEFAULT_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
TABLE_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean", "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "weighted_fast_proxy"]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]

# Redirect reused evaluator output paths to the trajectory directory.
ev.OUT_ROOT = OUT_ROOT


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parent_run_summary(target: str) -> Dict[str, Any]:
    if target.startswith("official"):
        run = RUN_BASE / "phase2s_official_40k_12x384_seed43200"
    else:
        run = RUN_BASE / "phase2s_mix25_40k_12x384_seed43200"
    ckpt = target.split("_")[-1]
    m = json.loads((run / "scientific_metrics.json").read_text(encoding="utf-8"))
    return {
        "parent_run": str(run),
        "checkpoint": ckpt,
        "checkpoint_path": str(MODEL_PATHS[target]),
        "checkpoint_exists": (MODEL_PATHS[target] / "model.safetensors").exists(),
        "word_exposure_final": m.get("word_exposure"),
        "actual_training_steps_final": m.get("actual_training_steps"),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_path": m.get("tokenizer_path"),
        "mask_switch_words": m.get("mask_switch_words"),
        "stage_metrics": m.get("stage_metrics"),
    }


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    model_path = MODEL_PATHS[target]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = per_target_path(target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"target": target, "model_path": str(model_path), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "work_tag": work_tag, "tasks": {}, "run_summary": parent_run_summary(target)}
    env = ev.setup_env(f"{work_tag}_{target}", gpu)
    for col in columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or force:
                rec = ev.eval_reading(target, model_path, work_tag, env)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in ev.TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or force:
                rec = ev.eval_sentence_column(target, model_path, col, work_tag, env)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        save_target(target, payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(target, payload)
    return payload


def scores(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        if col in raw.get("tasks", {}):
            out[col] = raw["tasks"][col].get("score")
    if "Reading" in raw.get("tasks", {}):
        for k, v in raw["tasks"]["Reading"].get("scores", {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0
    eq = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq):
        out["equal7_mean"] = sum(out[k] for k in eq) / len(eq)  # type: ignore[arg-type]
    if all(out.get(k) is not None for k in NLP_PROXY_KEYS):
        out["weighted_fast_proxy"] = sum(out[k] for k in NLP_PROXY_KEYS) / len(NLP_PROXY_KEYS)  # type: ignore[arg-type]
    return out


def add_cached_100m(table: Dict[str, Dict[str, Any]]) -> None:
    if not PHASE2_100M_SUMMARY.exists():
        return
    s = json.loads(PHASE2_100M_SUMMARY.read_text(encoding="utf-8"))
    table["official_100M"] = {k: v for k, v in s["table"].get("phase2s_official", {}).items() if k in TABLE_KEYS}
    table["mix25_100M"] = {k: v for k, v in s["table"].get("phase2s_mix25", {}).items() if k in TABLE_KEYS}
    table["initial_model_baseline"] = {k: v for k, v in s["table"].get("initial_model_baseline", {}).items() if k in TABLE_KEYS}


def contrast(table: Dict[str, Dict[str, Any]], a: str, b: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if a not in table or b not in table:
        return d
    for k in TABLE_KEYS:
        av = table[a].get(k); bv = table[b].get(k)
        if av is not None and bv is not None:
            d[k] = round(av - bv, 4)
    return d


def build_summary(targets: list[str]) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(t).read_text(encoding="utf-8")) for t in targets if per_target_path(t).exists()}
    table = {t: scores(raw[t]) for t in raw}
    add_cached_100m(table)
    contrasts = {
        "mix25_70M_minus_official_70M": contrast(table, "mix25_70M", "official_70M"),
        "mix25_100M_minus_official_100M": contrast(table, "mix25_100M", "official_100M"),
        "official_100M_minus_official_70M": contrast(table, "official_100M", "official_70M"),
        "mix25_80M_minus_mix25_70M": contrast(table, "mix25_80M", "mix25_70M"),
        "mix25_90M_minus_mix25_70M": contrast(table, "mix25_90M", "mix25_70M"),
        "mix25_100M_minus_mix25_70M": contrast(table, "mix25_100M", "mix25_70M"),
        "mix25_70M_minus_initial_model_studies_baseline": contrast(table, "mix25_70M", "initial_model_baseline"),
        "mix25_100M_minus_initial_model_studies_baseline": contrast(table, "mix25_100M", "initial_model_baseline"),
    }
    best = None
    cands = [(t, s.get("equal7_mean")) for t, s in table.items() if s.get("equal7_mean") is not None]
    if cands:
        best = max(cands, key=lambda x: x[1])
    payload = {
        "status": "PHASE2_TRAJECTORY_EVAL_DONE",
        "note": "Fast trajectory probe for corrected Phase-2 stagewise checkpoints. Full Entity is omitted here unless explicitly requested; 100M values are imported from phase2_stagewise_eval.",
        "targets_evaluated": list(raw.keys()),
        "cached_100M_source": str(PHASE2_100M_SUMMARY),
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
        "# research — Phase 2 checkpoint trajectory probe",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
        "This lightweight trajectory probe asks whether the corrected 12×384/LAMB/40k stagewise run was better before the WWM→token switch or before the final 30M words. It uses the same fast-screen columns as research, omitting full Entity by default.",
        "",
        "| target | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal7 | wproxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, s in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(s.get(k)) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "weighted_fast_proxy"]) + " |")
    lines += ["", "## Contrasts", ""]
    for name, delta in summary.get("contrasts", {}).items():
        keep = ["equal7_mean", "weighted_fast_proxy", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
        lines.append(f"- **{name}**: " + ", ".join(f"{k} {v:+.3f}" for k, v in delta.items() if k in keep))
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--work_tag", default="phase2_trajectory")
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
