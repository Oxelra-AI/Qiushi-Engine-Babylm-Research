#!/usr/bin/env python3
"""research: evaluate rephase restart checkpoints and continuous reference on cheap7.

Evaluates:
  1. Continuous chck_90M (free reference, from existing research run)
  2. Matched-LR restart: chck_85M, chck_90M, chck_95M, chck_100M
  3. Base-LR restart: chck_85M, chck_90M, chck_95M, chck_100M

Then writes a comparison summary against the existing research 100M legal endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL_SCRIPT = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
OUT_ROOT = WORKSPACE / "data/rephase_eval"
COLLATE_ROOT = WORKSPACE / "data/rephase_collate"

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                 "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

# Models to evaluate
EVAL_TARGETS = [
    # Continuous 90M reference
    {"label": "continuous_90M",
     "target": "continuous_chck_90M",
     "run_dir": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2",
     "endpoint": "chck_90M"},
    # Matched-LR restart checkpoints
    {"label": "matched_lr_85M",
     "target": "matched_lr_chck_85M",
     "run_dir": WORKSPACE / "training/runs/rephase_matched_lr_seed43044",
     "endpoint": "chck_85M"},
    {"label": "matched_lr_90M",
     "target": "matched_lr_chck_90M",
     "run_dir": WORKSPACE / "training/runs/rephase_matched_lr_seed43044",
     "endpoint": "chck_90M"},
    {"label": "matched_lr_95M",
     "target": "matched_lr_chck_95M",
     "run_dir": WORKSPACE / "training/runs/rephase_matched_lr_seed43044",
     "endpoint": "chck_95M"},
    {"label": "matched_lr_100M",
     "target": "matched_lr_chck_100M",
     "run_dir": WORKSPACE / "training/runs/rephase_matched_lr_seed43044",
     "endpoint": "chck_100M"},
    # Base-LR restart checkpoints
    {"label": "base_lr_85M",
     "target": "base_lr_chck_85M",
     "run_dir": WORKSPACE / "training/runs/rephase_base_lr_seed43044",
     "endpoint": "chck_85M"},
    {"label": "base_lr_90M",
     "target": "base_lr_chck_90M",
     "run_dir": WORKSPACE / "training/runs/rephase_base_lr_seed43044",
     "endpoint": "chck_90M"},
    {"label": "base_lr_95M",
     "target": "base_lr_chck_95M",
     "run_dir": WORKSPACE / "training/runs/rephase_base_lr_seed43044",
     "endpoint": "chck_95M"},
    {"label": "base_lr_100M",
     "target": "base_lr_chck_100M",
     "run_dir": WORKSPACE / "training/runs/rephase_base_lr_seed43044",
     "endpoint": "chck_100M"},
]

# Legal research 100M reference scores (from research)
LEGAL_REF = {
    "BLiMP": 65.8707, "Supplement": 61.1657, "EWoK": 50.3932,
    "Entity": 27.4008, "COMPS": 52.0083,
    "GlobalPIQA_parallel": 26.13, "GlobalPIQA_nonparallel": 46.0,
    "Reading_eye": 10.47, "Reading_self_paced": 5.81,
    "GlobalPIQA": 36.065, "Reading": 8.14,
    "cheap7_mean": 43.006,
    "Overall_9col": 41.258,
}

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_eval(target_spec: dict, gpu: int, force: bool = False) -> dict[str, Any]:
    """Run cheap evaluation for one target and return parsed scores."""
    target = target_spec["target"]
    run_dir = pathlib.Path(target_spec["run_dir"])
    endpoint = target_spec["endpoint"]
    label = target_spec["label"]

    model_path = run_dir / "hf_model" / endpoint
    if not model_path.exists():
        return {"label": label, "status": "MISSING_CHECKPOINT", "path": str(model_path)}

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--target", target,
        "--run-dir", str(run_dir),
        "--endpoint", endpoint,
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *CHEAP_COLUMNS,
    ]
    if force:
        cmd.append("--force")

    print(f"\n{'='*60}\nEvaluating: {label} ({endpoint} from {run_dir.name})\n{'='*60}", flush=True)
    p = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=3600)
    if p.returncode != 0:
        print(f"STDERR: {p.stderr[-2000:]}", flush=True)
        return {"label": label, "status": "EVAL_FAILED", "returncode": p.returncode,
                "stderr_tail": p.stderr[-2000:]}

    # Parse scores from per-target JSON. Official wrapper records zero-shot columns
    # as `score` in this coordinate, not always `accuracy`; Reading stores a nested
    # `scores` dict. Do not treat missing parser fields as real zeroes.
    per_target = OUT_ROOT / "per_target" / f"{target}.json"
    if not per_target.exists():
        return {"label": label, "status": "NO_OUTPUT", "stdout_tail": p.stdout[-2000:]}

    payload = json.loads(per_target.read_text())
    tasks = payload.get("tasks", {})
    scores: dict[str, float] = {}

    def scalar_score(rec: dict, *, nested_key: str | None = None) -> float | None:
        if rec.get("accuracy") is not None:
            return float(rec["accuracy"])
        if rec.get("score") is not None:
            return float(rec["score"])
        nested = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        if nested_key and nested.get(nested_key) is not None:
            return float(nested[nested_key])
        return None

    missing_scores = []
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        task_rec = tasks.get(col, {})
        val = scalar_score(task_rec)
        if val is None:
            missing_scores.append(col)
            val = 0.0
        scores[col] = val

    gpiqa_par = scalar_score(tasks.get("GlobalPIQA_parallel", {}))
    gpiqa_nonpar = scalar_score(tasks.get("GlobalPIQA_nonparallel", {}))
    if gpiqa_par is None:
        missing_scores.append("GlobalPIQA_parallel")
        gpiqa_par = 0.0
    if gpiqa_nonpar is None:
        missing_scores.append("GlobalPIQA_nonparallel")
        gpiqa_nonpar = 0.0
    scores["GlobalPIQA_parallel"] = gpiqa_par
    scores["GlobalPIQA_nonparallel"] = gpiqa_nonpar
    scores["GlobalPIQA"] = (gpiqa_par + gpiqa_nonpar) / 2.0

    reading_rec = tasks.get("Reading", {})
    nested_reading = reading_rec.get("scores") if isinstance(reading_rec.get("scores"), dict) else {}
    scores["Reading_eye"] = float(reading_rec.get("reading_eye", nested_reading.get("Reading_eye", 0.0)))
    scores["Reading_self_paced"] = float(reading_rec.get("reading_self_paced", nested_reading.get("Reading_self_paced", 0.0)))
    r_val = reading_rec.get("reading", nested_reading.get("Reading"))
    if r_val is None:
        missing_scores.append("Reading")
        r_val = 0.0
    scores["Reading"] = float(r_val)

    cheap7_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    scores["cheap7_mean"] = sum(scores.get(k, 0) for k in cheap7_keys) / 7.0
    if missing_scores:
        return {"label": label, "status": "INCOMPLETE_SCORES", "missing_scores": missing_scores,
                "scores": scores, "per_target_json": str(per_target)}

    return {"label": label, "status": "OK", "scores": scores, "per_target_json": str(per_target)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--only", nargs="*", help="Evaluate only these labels")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)

    targets = EVAL_TARGETS
    if args.only:
        targets = [t for t in targets if t["label"] in args.only]

    results: list[dict] = []
    for spec in targets:
        res = run_eval(spec, args.gpu, args.force)
        results.append(res)
        if res["status"] == "OK":
            s = res["scores"]
            delta = s["cheap7_mean"] - LEGAL_REF["cheap7_mean"]
            print(f"  {res['label']}: cheap7={s['cheap7_mean']:.4f} (Δ={delta:+.4f} vs legal100M)", flush=True)
        else:
            print(f"  {res['label']}: {res['status']}", flush=True)

    # Write comparison summary
    summary = {
        "status": "REPHASE_EVAL_SUMMARY",
        "created_utc": now_utc(),
        "legal_ref": LEGAL_REF,
        "results": results,
    }

    # Build comparison table
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        table_lines = ["# research rephase evaluation comparison", "",
                       "| Model | BLiMP | Suppl | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 | Δcheap7 |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        ref_line = f"| legal100M_ref | {LEGAL_REF['BLiMP']:.2f} | {LEGAL_REF['Supplement']:.2f} | {LEGAL_REF['EWoK']:.2f} | {LEGAL_REF['Entity']:.2f} | {LEGAL_REF['COMPS']:.2f} | {LEGAL_REF['GlobalPIQA']:.2f} | {LEGAL_REF['Reading']:.2f} | {LEGAL_REF['cheap7_mean']:.4f} | — |"
        table_lines.append(ref_line)
        for r in ok_results:
            s = r["scores"]
            d = s["cheap7_mean"] - LEGAL_REF["cheap7_mean"]
            line = f"| {r['label']} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Reading']:.2f} | {s['cheap7_mean']:.4f} | {d:+.4f} |"
            table_lines.append(line)
        summary["comparison_table"] = "\n".join(table_lines)

        # Identify best
        best = max(ok_results, key=lambda r: r["scores"]["cheap7_mean"])
        summary["best_model"] = best["label"]
        summary["best_cheap7"] = best["scores"]["cheap7_mean"]
        summary["best_delta_vs_legal100M"] = best["scores"]["cheap7_mean"] - LEGAL_REF["cheap7_mean"]

    out_json = OUT_ROOT / "rephase_eval_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    out_md = OUT_ROOT / "rephase_eval_summary.md"
    if "comparison_table" in summary:
        out_md.write_text(summary["comparison_table"] + "\n")

    print(f"\nSummary: {out_json}", flush=True)
    if "comparison_table" in summary:
        print(summary["comparison_table"], flush=True)


if __name__ == "__main__":
    main()
