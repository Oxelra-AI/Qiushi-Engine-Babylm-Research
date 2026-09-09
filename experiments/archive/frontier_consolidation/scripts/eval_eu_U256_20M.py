#!/usr/bin/env python3
"""research: Wait for U256 20M training, then evaluate cheap columns on GPU1.

Waits for the U256 experience-utilization 20M screen to produce a checkpoint,
evaluates all cheap columns using evaluate_compliant_endpoint.py,
and compares with the canonical research legal 20M reference.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from statistics import mean

USER_ROOT = pathlib.Path(".").resolve()
U256_RUN = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/eu_U256_legal16k_seed43022_20M"
OUT_ROOT = USER_ROOT / "experiments/archive/frontier_consolidation/data/eu_U256_20M_eval"
COLLATE_ROOT = USER_ROOT / "experiments/archive/frontier_consolidation/data/eu_U256_20M_collate"
EVAL_SCRIPT = str(USER_ROOT / "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py")
SUMMARY_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/eu_U256_20M_summary"
TARGET = "eu_U256_20M_seed43022"
ENDPOINT = "chck_20M"
GPU = 1

# research legal 20M reference (from research disabled-adapter exact reproduction)
reference_20M_REF = {
    "BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73,
    "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67,
    "cheap7": 39.66357142857142,
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_scores(payload: dict) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        r = tasks.get(c, {})
        scores[c] = float(r["score"]) if isinstance(r, dict) and r.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = tasks.get(c, {})
        if isinstance(r, dict) and r.get("score") is not None:
            gp.append(float(r["score"]))
    scores["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    r = tasks.get("Reading", {})
    if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
        scores["Reading"] = float(r["scores"]["Reading"])
    elif isinstance(r, dict) and r.get("score") is not None:
        scores["Reading"] = float(r["score"])
    else:
        scores["Reading"] = None
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def main() -> None:
    t0 = time.time()
    # Wait for training completion (scientific_metrics.json + checkpoint)
    print(json.dumps({"event": "waiting_for_training", "run": str(U256_RUN.relative_to(USER_ROOT)), "utc": now()}), flush=True)
    while True:
        sm_ok = (U256_RUN / "scientific_metrics.json").exists()
        cp_ok = (U256_RUN / "hf_model" / ENDPOINT / "model.safetensors").exists()
        if sm_ok and cp_ok:
            break
        time.sleep(15)
        if time.time() - t0 > 3600:
            raise TimeoutError("U256 training did not complete within 1 hour")
    wait_sec = time.time() - t0
    print(json.dumps({"event": "training_complete", "wait_sec": round(wait_sec, 1), "utc": now()}), flush=True)

    # Read training summary
    with open(U256_RUN / "scientific_metrics.json") as f:
        metrics = json.load(f)
    total_words = metrics.get("verification", {}).get("total_charged_words", metrics.get("total_words"))
    print(json.dumps({"event": "training_metrics", "total_words": total_words, "arm": metrics.get("arm")}), flush=True)

    # Evaluate cheap columns on GPU1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
               "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
    cmd = [
        sys.executable, "-B", EVAL_SCRIPT,
        "--arm", "reinvest",
        "--run-dir", str(U256_RUN),
        "--target", TARGET,
        "--endpoint", ENDPOINT,
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(GPU),
        "--columns", *columns,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(json.dumps({"event": "eval_start", "target": TARGET, "endpoint": ENDPOINT, "gpu": GPU, "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)

    # Save logs
    log_dir = SUMMARY_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "eval_stderr.log").write_text(proc.stderr, encoding="utf-8")

    print(json.dumps({"event": "eval_done", "returncode": proc.returncode,
                       "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed rc={proc.returncode}\nstderr: {proc.stderr[-5000:]}")

    # Parse results
    payload_path = OUT_ROOT / "per_target" / f"{TARGET}.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload not found: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    scores = extract_scores(payload)
    c7 = cheap7(scores)

    # Compare with research 20M reference
    deltas = {}
    for c in CHEAP_COLS:
        if scores.get(c) is not None:
            deltas[c] = round(float(scores[c]) - reference_20M_REF[c], 4)
        else:
            deltas[c] = None
    c7_delta = round(c7 - reference_20M_REF["cheap7"], 4) if c7 is not None else None

    # Route signal
    if c7_delta is not None and c7_delta > 0.3:
        signal = "positive_continue_to_mature"
    elif c7_delta is not None and c7_delta > -0.3:
        signal = "marginal_inspect_columns"
    else:
        signal = "negative_close_U256"

    summary = {
        "status": "EU_U256_20M_EVAL",
        "utc": now(),
        "endpoint": ENDPOINT,
        "model_path": str((U256_RUN / "hf_model" / ENDPOINT).relative_to(USER_ROOT)),
        "total_words": total_words,
        "u256_scores": scores,
        "u256_cheap7": c7,
        "reference_20m_ref": reference_20M_REF,
        "deltas": deltas,
        "cheap7_delta": c7_delta,
        "route_signal": signal,
        "payload_path": str(payload_path.relative_to(USER_ROOT)),
        "eval_elapsed_sec": round(time.time() - t0 - wait_sec, 1),
        "total_elapsed_sec": round(time.time() - t0, 1),
    }

    out_json = SUMMARY_DIR / "eu_U256_20M_eval_summary.json"
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/eu_U256_20M_summary/eu_U256_20M_eval_summary.md')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research U256 Experience-Utilization 20M Screen",
        "",
        f"Route signal: **{signal}**",
        f"cheap7 delta: **{c7_delta:+.4f}**" if c7_delta is not None else "cheap7 delta: N/A",
        "",
        "## Scores",
        "",
        "| Column | U256 20M | research 20M | Delta |",
        "|---|---:|---:|---:|",
    ]
    for c in CHEAP_COLS:
        u = scores.get(c)
        r = reference_20M_REF.get(c)
        d = deltas.get(c)
        if u is not None and r is not None and d is not None:
            lines.append(f"| {c} | {u:.3f} | {r:.3f} | {d:+.3f} |")
        else:
            lines.append(f"| {c} | {u} | {r} | {d} |")
    if c7 is not None and c7_delta is not None:
        lines.append(f"| **cheap7** | **{c7:.4f}** | **{reference_20M_REF['cheap7']:.4f}** | **{c7_delta:+.4f}** |")
    lines += [
        "",
        "## Context",
        "",
        "U256 faithful experience-utilization: same legal corpus, tokenizer, architecture,",
        "recipe, and seeds as research; changed only the training example object so all charged",
        "word tokens are visible (14.66M active tokens vs research's ~14.30M, +2.59%).",
        "",
        f"Evidence: `{str(out_json.relative_to(USER_ROOT))}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
