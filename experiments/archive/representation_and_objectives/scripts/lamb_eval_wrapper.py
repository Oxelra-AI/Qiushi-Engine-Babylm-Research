#!/usr/bin/env python3
"""research: Evaluation wrapper for LAMB+curriculum training endpoints.

Calls the validated evaluate_compliant_endpoint.py evaluator
for cheap7 columns (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading).

Usage:
  python3 lamb_eval_wrapper.py --gpu 0 [--arm 12x384] [--arm 8x480] [--force]
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, subprocess, sys, time
from statistics import mean

def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_root()
A01 = ROOT / "experiments/archive/representation_and_objectives"
EVALUATOR = ROOT / "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py"
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

ARMS = {
    "12x384": {
        "run_dir": A01 / "training/runs/lamb_curriculum_12x384_legal40k_seed43022",
        "endpoint": "chck_100M",
        "target": "lamb_curriculum_12x384",
        "label": "LAMB+curriculum 12x384/12h/FFN1280 (leader match)",
    },
    "8x480": {
        "run_dir": A01 / "training/runs/lamb_curriculum_8x480_legal40k_seed43022",
        "endpoint": "chck_100M",
        "target": "lamb_curriculum_8x480",
        "label": "LAMB+curriculum 8x480/8h/FFN1920 (our architecture)",
    },
}

OUT_ROOT = A01 / "data/lamb_eval"
NOTE_PATH = A01 / "notes/131_lamb_eval_results.md"

def run_eval(arm_key: str, gpu: int, force: bool = False, timeout: int = 3600):
    spec = ARMS[arm_key]
    chck = spec["run_dir"] / "hf_model" / spec["endpoint"]
    if not chck.exists():
        # Try chck_final instead
        chck_final = spec["run_dir"] / "hf_model" / "chck_final"
        if chck_final.exists():
            spec = {**spec, "endpoint": "chck_final"}
        else:
            return {"arm": arm_key, "status": "no_checkpoint",
                    "expected": str(chck), "alt": str(chck_final)}

    out = OUT_ROOT / arm_key
    collate = OUT_ROOT / f"{arm_key}_collate"
    out.mkdir(parents=True, exist_ok=True)
    collate.mkdir(parents=True, exist_ok=True)
    per_target = out / "per_target" / f"{spec['target']}.json"

    if per_target.exists() and not force:
        return {"arm": arm_key, "status": "exists", "per_target": str(per_target)}

    cmd = [
        sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
        "--run-dir", str(spec["run_dir"]),
        "--target", spec["target"],
        "--endpoint", spec["endpoint"],
        "--out-root", str(out),
        "--collate-root", str(collate),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    if force:
        cmd.append("--force")

    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm_key}.log"

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(json.dumps({"event": "eval_start", "arm": arm_key, "cmd": " ".join(cmd[:8])}), flush=True)
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(proc.stdout[-5000:] if len(proc.stdout) > 5000 else proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr[-3000:] if len(proc.stderr) > 3000 else proc.stderr)
        f.write(f"\n[rc={proc.returncode} elapsed={time.time()-t0:.1f}s]\n")

    if proc.returncode != 0:
        return {"arm": arm_key, "status": "failed", "rc": proc.returncode,
                "log": str(log_path), "stderr_tail": proc.stderr[-1000:]}
    if not per_target.exists():
        return {"arm": arm_key, "status": "missing_output", "log": str(log_path)}
    return {"arm": arm_key, "status": "done", "per_target": str(per_target),
            "elapsed_sec": round(time.time()-t0, 1)}


def extract_cheap7(per_target_path: pathlib.Path):
    data = json.loads(per_target_path.read_text())
    tasks = data.get("tasks", {})
    scores = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        scores[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp_vals = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    scores["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        scores["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        scores["Reading"] = float(rd["score"])
    else:
        scores["Reading"] = None
    vals = [scores.get(c) for c in SCORE_COLUMNS]
    c7 = mean(v for v in vals if v is not None) if all(v is not None for v in vals) else None
    return scores, c7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arm", action="append", default=[],
                        help="Arm keys to evaluate (default: all ready)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    arms_to_run = args.arm if args.arm else list(ARMS.keys())
    results = {}

    for arm_key in arms_to_run:
        if arm_key not in ARMS:
            print(f"Unknown arm: {arm_key}", file=sys.stderr)
            continue
        result = run_eval(arm_key, args.gpu, args.force)
        results[arm_key] = result
        print(json.dumps({"event": "eval_result", "arm": arm_key, **result}), flush=True)

    # Summarize
    summary = {"status": "LAMB_EVAL", "arms": {}}
    for arm_key, result in results.items():
        entry = {"label": ARMS[arm_key]["label"], "eval_status": result["status"]}
        if result["status"] in ("done", "exists"):
            pt = pathlib.Path(result["per_target"])
            if pt.exists():
                scores, c7 = extract_cheap7(pt)
                entry["scores"] = scores
                entry["cheap7"] = c7
        summary["arms"][arm_key] = entry

    summary_path = OUT_ROOT / "lamb_eval_summary.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "summary", "path": str(summary_path),
                      **{k: v.get("cheap7") for k, v in summary["arms"].items()
                         if isinstance(v.get("cheap7"), (int, float))}},
                     default=str), flush=True)

if __name__ == "__main__":
    main()
