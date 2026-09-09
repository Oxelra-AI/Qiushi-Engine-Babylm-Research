#!/usr/bin/env python3
"""Cheap7 evaluator for frozen-82M private-tail branches.

This is the research analogue of dualview_eval_one.py, but the reference
is the verified scale1.75 chck_82M function rather than the 20M research baseline.
It evaluates an already-finished frozen-tail run through the current
research official-compatible wrapper on the seven route-selection columns:
BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading.

No training is performed here.  The purpose is to test whether a reversible
private-tail branch preserves the mature above-leader broad competence surface.
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
DEFAULT_OUT_ROOT = WORKSPACE / "data/frozen82_tail_cheap_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/frozen82_tail_collate"
SUMMARY_ROOT = WORKSPACE / "data/frozen82_tail_summary"
CHCK82_VERIFY = WORKSPACE / "data/chck82_independent_verification/chck82_independent_verification.json"

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chck82_reference() -> dict[str, Any]:
    data = read_json(CHCK82_VERIFY)
    scores = {c: float(data["score_arithmetic"]["scores"][c]) for c in CHEAP_COLUMNS}
    return {
        "label": "verified scale1.75 chck_82M protected reference",
        "source": rel(CHCK82_VERIFY),
        "scores": scores,
        "cheap7": float(mean(scores[c] for c in CHEAP_COLUMNS)),
        "overall": float(data["score_arithmetic"]["overall_reported"]),
        "actual_word_exposure": int(data["legal_exposure_and_provenance"]["selected_actual_words"]),
        "model_sha256": data["artifact_identity"]["expected_model_sha256"],
    }


def choose_endpoint(run_dir: pathlib.Path, requested: str) -> str:
    if requested != "auto":
        return requested
    model_root = run_dir / "hf_model"
    for name in ["final", "chck_total_86012495w", "chck_86M", "chck_20M"]:
        if (model_root / name / "model.safetensors").exists() or (model_root / name / "pytorch_model.bin").exists():
            return name
    raise FileNotFoundError(f"No evaluable checkpoint found under {model_root}")


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
    m = read_json(path)
    return {
        "path": rel(path),
        "status": m.get("status"),
        "mode": m.get("mode"),
        "endpoint": m.get("endpoint"),
        "updates": m.get("updates"),
        "schedule_total": m.get("schedule_total"),
        "initial_consumed_words": m.get("initial_consumed_words"),
        "skip_rows": m.get("skip_rows"),
        "tail_main_word_exposure": m.get("tail_main_word_exposure"),
        "tail_aux_word_exposure": m.get("tail_aux_word_exposure"),
        "tail_charged_words": m.get("tail_charged_words"),
        "total_consumed_words": m.get("total_consumed_words"),
        "max_tail_charged_words": m.get("max_tail_charged_words"),
        "stopped_before_cap": m.get("stopped_before_cap"),
        "trainable": m.get("trainable"),
        "first_aux_loss": m.get("first_aux_loss"),
        "final_aux_loss": m.get("final_aux_loss"),
        "mean_aux_loss": m.get("mean_aux_loss"),
        "aux_loss_batches": m.get("aux_loss_batches"),
        "first_neutral_loss": m.get("first_neutral_loss"),
        "final_neutral_loss": m.get("final_neutral_loss"),
        "mean_neutral_loss": m.get("mean_neutral_loss"),
        "deterministic_neutrality_eval_mode": m.get("deterministic_neutrality_eval_mode"),
        "total_params": m.get("total_params"),
        "private_params": m.get("private_params"),
        "frozen_slow_params": m.get("frozen_slow_params"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--endpoint", default="auto")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--summary-root", default=str(SUMMARY_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ref = load_chck82_reference()
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
    print(json.dumps({"event": "frozen82_tail_eval_start", "target": args.target, "endpoint": endpoint,
                      "run_dir": rel(run_dir), "gpu": args.gpu, "columns": EVAL_COLUMNS,
                      "tail_charged_words": train_metrics.get("tail_charged_words"),
                      "total_consumed_words": train_metrics.get("total_consumed_words"),
                      "chck82_ref_cheap7": ref["cheap7"], "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    (log_dir / "eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "eval_stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({"event": "frozen82_tail_eval_finished", "target": args.target, "returncode": proc.returncode,
                      "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {args.target} rc={proc.returncode}\n{proc.stderr[-4000:]}")

    payload_path = out_root / "per_target" / f"{args.target}.json"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    scores = extract_scores(payload)
    c7 = cheap7(scores)
    deltas = {c: (None if scores.get(c) is None else float(scores[c] - ref["scores"][c])) for c in CHEAP_COLUMNS}
    c7_delta = None if c7 is None else float(c7 - ref["cheap7"])

    summary = {
        "status": "FROZEN82_TAIL_CHEAP_EVAL",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(run_dir),
        "endpoint": endpoint,
        "model_path": rel(model_path),
        "training_metrics": train_metrics,
        "scores": scores,
        "cheap7": c7,
        "chck82_reference": ref,
        "deltas_vs_chck82": deltas,
        "cheap7_delta_vs_chck82": c7_delta,
        "payload_path": rel(payload_path),
        "eval_stdout": rel(log_dir / "eval_stdout.log"),
        "eval_stderr": rel(log_dir / "eval_stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = summary_root / f"{args.target}_summary.json"
    out_md = summary_root / f"{args.target}_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research frozen-82M private-tail cheap evaluation — {args.target}",
        "",
        f"Run: `{rel(run_dir)}`; endpoint: `{endpoint}`; total consumed words: `{train_metrics.get('total_consumed_words')}`.",
        f"Protected reference: chck_82M cheap7 `{ref['cheap7']}`; Overall `{ref['overall']}`.",
        "",
        "| Column | score | chck_82M | delta |",
        "|---|---:|---:|---:|",
    ]
    for c in CHEAP_COLUMNS:
        v = scores.get(c); r = ref["scores"][c]; d = deltas[c]
        lines.append(f"| {c} | {v:.4f} | {r:.4f} | {d:+.4f} |" if v is not None and d is not None else f"| {c} | {v} | {r:.4f} | {d} |")
    if c7 is not None and c7_delta is not None:
        lines.append(f"| **cheap7** | **{c7:.6f}** | **{ref['cheap7']:.6f}** | **{c7_delta:+.6f}** |")
    lines += ["", f"Payload: `{rel(payload_path)}`", f"Summary JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "target": args.target, "endpoint": endpoint,
                      "cheap7": c7, "cheap7_delta_vs_chck82": c7_delta,
                      "summary_json": rel(out_json), "payload_path": rel(payload_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
