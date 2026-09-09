#!/usr/bin/env python3
"""research: evaluate LAMB 20M screens vs research legal 20M baseline.

Uses the research evaluator with --arm reinvest (for correct eval infrastructure)
but overrides --run-dir and --target to point to LAMB checkpoints.

Usage:
  python eval_lamb_20m.py --gpu 0   # eval both arms serially on gpu 0
  python eval_lamb_20m.py --gpu 0 --arm lr005  # just one arm
  python eval_lamb_20m.py --summarize  # just print existing results
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, subprocess, sys
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
BASELINE_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json')

ARMS = {
    "lr005": {
        "label": "LAMB lr=0.005",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M')),
        "target": "lamb_lr005_seed43022_20M",
    },
    "lr007": {
        "label": "LAMB lr=0.007",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr007_seed43022_20M')),
        "target": "lamb_lr007_seed43022_20M",
    },
}

# Reference scores for comparison table
reference_20M_SCORES = {
    "BLiMP": 63.53, "Supplement": 56.17, "EWoK": 49.85,
    "Entity": 24.11, "COMPS": 50.24, "GlobalPIQA": 33.47, "Reading": 0.18,
}
MUON_20M_SCORES = {
    "BLiMP": 64.99, "Supplement": 58.82, "EWoK": 49.82,
    "Entity": 26.36, "COMPS": 50.43, "GlobalPIQA": 37.88, "Reading": -1.23,
}

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_collate')


def extract_scores(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    out = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if "score" in rec and rec["score"] is not None:
            gp.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores):
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def run_eval(arm_key, arm, gpu):
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model" / "chck_20M"
    if not model_path.exists():
        model_path = run_dir / "hf_model"
    if not (model_path / "model.safetensors").exists():
        print(f"SKIP {arm_key}: no checkpoint at {model_path}", flush=True)
        return None
    target = arm["target"]
    per_target = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval/per_target') / f"{target}.json"
    if per_target.exists():
        print(f"EXISTS {arm_key}: {per_target}", flush=True)
        return per_target
    cmd = [sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
           "--run-dir", str(run_dir), "--target", target, "--endpoint", "chck_20M",
           "--out-root", str(OUT_ROOT), "--collate-root", str(COLLATE_ROOT),
           "--gpu", str(gpu), "--columns", *EVAL_COLUMNS]
    print(f"RUN {arm_key}: {target} gpu={gpu}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"FAIL {arm_key}:\nSTDOUT {proc.stdout[-2000:]}\nSTDERR {proc.stderr[-3000:]}", flush=True)
        return None
    return per_target if per_target.exists() else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arm", type=str, default=None, choices=list(ARMS.keys()),
                    help="Evaluate only this arm; default evaluates both")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval/per_target')).mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)

    # research baseline
    baseline = json.loads(BASELINE_JSON.read_text()) if BASELINE_JSON.exists() else None
    b_scores = extract_scores(baseline) if baseline else reference_20M_SCORES
    b_c7 = cheap7(b_scores)

    rows = {}
    rows["reference_20M"] = {"label": "research legal 20M", "scores": b_scores, "cheap7": b_c7, "deltas": None}
    rows["muon_20M"] = {"label": "Muon wd-matched 20M", "scores": MUON_20M_SCORES,
                        "cheap7": cheap7(MUON_20M_SCORES),
                        "deltas": {c: MUON_20M_SCORES[c] - b_scores[c] for c in CHEAP_COLUMNS if b_scores.get(c) is not None}}

    # Evaluate or load LAMB arms
    arms_to_eval = {args.arm: ARMS[args.arm]} if args.arm else ARMS
    for k, arm in arms_to_eval.items():
        if not args.summarize:
            rp = run_eval(k, arm, args.gpu)
        else:
            rp = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval/per_target') / f"{arm['target']}.json"
            if not rp.exists():
                rp = None
        if rp and rp.exists():
            sc = extract_scores(json.loads(rp.read_text()))
            c7 = cheap7(sc)
            d = {c: (sc[c] - b_scores[c]) if sc.get(c) is not None and b_scores.get(c) is not None else None
                 for c in CHEAP_COLUMNS}
            d["cheap7"] = (c7 - b_c7) if c7 and b_c7 else None
            rows[k] = {"label": arm["label"], "scores": sc, "cheap7": c7, "deltas": d}
        else:
            rows[k] = {"label": arm["label"], "scores": None, "cheap7": None, "deltas": None}

    out = {"status": "LAMB_20M_EVAL", "baseline_cheap7": b_c7, "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval/lamb_20M_comparison.json')).write_text(json.dumps(out, indent=2) + "\n")

    # Build markdown table
    lines = ["# research LAMB 20M comparison", "",
             f"Baseline (research legal 20M) cheap7: **{b_c7:.4f}**" if b_c7 else "Baseline: N/A", "",
             "| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rows.items():
        s = r.get("scores") or {}
        c7v = r.get("cheap7")
        dc = (r.get("deltas") or {}).get("cheap7") if r.get("deltas") else None
        def f(c): return f"{s[c]:.2f}" if s.get(c) is not None else "—"
        if c7v is not None:
            dc_str = f"{dc:+.4f}" if dc is not None else ""
            lines.append(f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7v:.4f} | {dc_str} |")
        else:
            lines.append(f"| {r['label']} | — | — | — | — | — | — | — | — | (pending) |")
    lines.append("")
    (_public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval/lamb_20M_comparison.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
