#!/usr/bin/env python3
"""research generic cheap-surface evaluator for dual-view 20M runs.

The script does no training.  It evaluates an already-finished run directory with the
current official-compatible research wrapper on the seven fast columns used for route
selection: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading.  It
also records charged exposure from the run's scientific_metrics.json so aligned,
shuffled, and mlm_only arms can be compared under the corrected charged-budget
semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from statistics import mean
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL_SCRIPT = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
DEFAULT_OUT_ROOT = WORKSPACE / "data/dualview_20m_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/dualview_20m_collate"
SUMMARY_ROOT = WORKSPACE / "data/dualview_20m_summary"

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

reference_20M_REF = {
    "label": "research legal 20M, original 100M LR horizon",
    "scores": {
        "BLiMP": 59.69,
        "Supplement": 55.45,
        "EWoK": 50.73,
        "Entity": 18.65,
        "COMPS": 50.26,
        "GlobalPIQA": 34.195,
        "Reading": 8.67,
    },
}
reference_20M_REF["cheap7"] = float(mean(reference_20M_REF["scores"][c] for c in CHEAP_COLUMNS))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def choose_endpoint(run_dir: pathlib.Path, requested: str) -> str:
    if requested != "auto":
        return requested
    model_root = run_dir / "hf_model"
    for name in ["chck_20M", "final"]:
        if (model_root / name / "model.safetensors").exists() or (model_root / name / "pytorch_model.bin").exists():
            return name
    raise FileNotFoundError(f"No evaluable checkpoint found under {model_root}; looked for chck_20M and final")


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {c: None for c in CHEAP_COLUMNS}
    official = payload.get("official_overall", {}).get("scores", {})
    for c in CHEAP_COLUMNS:
        v = official.get(c)
        if v is not None:
            scores[c] = float(v)
    tasks = payload.get("tasks", {})
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        if scores.get(c) is None:
            r = tasks.get(c, {})
            if isinstance(r, dict) and r.get("score") is not None:
                scores[c] = float(r["score"])
    if scores.get("GlobalPIQA") is None:
        vals = []
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            r = tasks.get(c, {})
            if isinstance(r, dict) and r.get("score") is not None:
                vals.append(float(r["score"]))
        if len(vals) == 2:
            scores["GlobalPIQA"] = float(mean(vals))
    if scores.get("Reading") is None:
        r = tasks.get("Reading", {})
        if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
            scores["Reading"] = float(r["scores"]["Reading"])
        elif isinstance(r, dict) and r.get("score") is not None:
            scores["Reading"] = float(r["score"])
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def read_training_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
    path = run_dir / "scientific_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Training metrics missing: {path}")
    metrics = read_json(path)
    return {
        "path": rel(path),
        "status": metrics.get("status"),
        "mode": metrics.get("mode"),
        "updates": metrics.get("updates"),
        "total_main_word_exposure": metrics.get("total_main_word_exposure"),
        "total_aux_word_exposure": metrics.get("total_aux_word_exposure"),
        "total_charged_words": metrics.get("total_charged_words", metrics.get("total_words")),
        "stopped_before_cap": metrics.get("stopped_before_cap"),
        "schedule_total": metrics.get("schedule_total"),
        "first_loss": metrics.get("first_loss"),
        "final_loss": metrics.get("final_loss"),
        "mean_aux_loss": metrics.get("mean_aux_loss"),
        "aux_loss_batches": metrics.get("aux_loss_batches"),
        "gradient_checkpointing": metrics.get("gradient_checkpointing"),
        "aux_micro_batch_size": metrics.get("aux_micro_batch_size"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--endpoint", default="auto", help="Use chck_20M when present, otherwise final")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--summary-root", default=str(SUMMARY_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    endpoint = choose_endpoint(run_dir, args.endpoint)
    model_path = run_dir / "hf_model" / endpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    train_metrics = read_training_metrics(run_dir)

    out_root = pathlib.Path(args.out_root)
    collate_root = pathlib.Path(args.collate_root)
    summary_root = pathlib.Path(args.summary_root)
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", args.target,
        "--endpoint", endpoint,
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(args.gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    if args.force:
        cmd.append("--force")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache = summary_root / "runtime_cache" / args.target
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())

    log_dir = summary_root / "logs" / args.target
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"event": "dualview_eval_start", "target": args.target, "endpoint": endpoint,
                      "run_dir": rel(run_dir), "gpu": args.gpu, "columns": EVAL_COLUMNS,
                      "charged_words": train_metrics.get("total_charged_words"), "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    (log_dir / "eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "eval_stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({"event": "dualview_eval_finished", "target": args.target, "returncode": proc.returncode,
                      "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {args.target} rc={proc.returncode}\n{proc.stderr[-4000:]}")

    payload_path = out_root / "per_target" / f"{args.target}.json"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    scores = extract_scores(payload)
    c7 = cheap7(scores)
    deltas = {c: (None if scores.get(c) is None else float(scores[c] - reference_20M_REF["scores"][c])) for c in CHEAP_COLUMNS}
    c7_delta = None if c7 is None else float(c7 - reference_20M_REF["cheap7"])

    summary = {
        "status": "DUALVIEW_CHEAP_EVAL",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(run_dir),
        "endpoint": endpoint,
        "model_path": rel(model_path),
        "training_metrics": train_metrics,
        "scores": scores,
        "cheap7": c7,
        "reference_20M_reference": reference_20M_REF,
        "deltas_vs_step35_20M": deltas,
        "cheap7_delta_vs_step35_20M": c7_delta,
        "payload_path": rel(payload_path),
        "eval_stdout": rel(log_dir / "eval_stdout.log"),
        "eval_stderr": rel(log_dir / "eval_stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = summary_root / f"{args.target}_summary.json"
    out_md = summary_root / f"{args.target}_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research dual-view cheap evaluation — {args.target}",
        "",
        f"Run: `{rel(run_dir)}`; endpoint: `{endpoint}`; charged words: `{train_metrics.get('total_charged_words')}`.",
        "",
        "| Column | score | research 20M | delta |",
        "|---|---:|---:|---:|",
    ]
    for c in CHEAP_COLUMNS:
        v = scores.get(c); r = reference_20M_REF["scores"][c]; d = deltas[c]
        lines.append(f"| {c} | {v:.4f} | {r:.4f} | {d:+.4f} |" if v is not None and d is not None else f"| {c} | {v} | {r:.4f} | {d} |")
    if c7 is not None and c7_delta is not None:
        lines.append(f"| **cheap7** | **{c7:.6f}** | **{reference_20M_REF['cheap7']:.6f}** | **{c7_delta:+.6f}** |")
    lines += ["", f"Payload: `{rel(payload_path)}`", f"Summary JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "target": args.target, "endpoint": endpoint,
                      "cheap7": c7, "cheap7_delta_vs_step35_20M": c7_delta,
                      "summary_json": rel(out_json), "payload_path": rel(payload_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
