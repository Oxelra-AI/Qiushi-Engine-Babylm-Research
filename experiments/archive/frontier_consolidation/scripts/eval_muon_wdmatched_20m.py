#!/usr/bin/env python3
"""research: evaluate the wd-matched Muon 20M arm against research legal 20M and the research arms.

Reuses the research evaluator and the same baseline/target extraction as research.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')

EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
BASELINE_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json')

ARMS = {
    "muon_lr008_wd00125": {
        "label": "Muon LR=0.008 wd=0.00125",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_20M')),
        "target": "muon_lr008_wd00125_20M",
    },
}

# Prior arms for the consolidated table (already evaluated in research).
PRIOR = {
    "muon_lr008_wd01": _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr008_20M.json'),
    "muon_lr012_wd01": _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr012_20M.json'),
}

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_collate')


def extract_scores(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    out = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp = []
    for c in GP_COLS:
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
    if not (model_path / "model.safetensors").exists():
        print(f"SKIP {arm_key}: no checkpoint at {model_path}", flush=True)
        return None
    target = arm["target"]
    per_target = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval/per_target') / f"{target}.json"
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
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(BASELINE_JSON.read_text()) if BASELINE_JSON.exists() else None
    b_scores = extract_scores(baseline) if baseline else None
    b_c7 = cheap7(b_scores) if b_scores else None

    rows = {}
    # baseline row
    rows["baseline"] = {"label": "research legal 20M", "scores": b_scores, "cheap7": b_c7, "deltas": None}
    # prior arms
    for k, p in PRIOR.items():
        if Path(p).exists():
            sc = extract_scores(json.loads(Path(p).read_text()))
            c7 = cheap7(sc)
            d = {c: (sc[c] - b_scores[c]) if sc.get(c) is not None and b_scores.get(c) is not None else None for c in CHEAP_COLUMNS}
            d["cheap7"] = (c7 - b_c7) if c7 and b_c7 else None
            rows[k] = {"label": k, "scores": sc, "cheap7": c7, "deltas": d}
    # new arm
    for k, arm in ARMS.items():
        rp = run_eval(k, arm, args.gpu)
        if rp and rp.exists():
            sc = extract_scores(json.loads(rp.read_text()))
            c7 = cheap7(sc)
            d = {c: (sc[c] - b_scores[c]) if sc.get(c) is not None and b_scores.get(c) is not None else None for c in CHEAP_COLUMNS}
            d["cheap7"] = (c7 - b_c7) if c7 and b_c7 else None
            rows[k] = {"label": arm["label"], "scores": sc, "cheap7": c7, "deltas": d}
        else:
            rows[k] = {"label": arm["label"], "scores": None, "cheap7": None, "deltas": None}

    out = {"status": "MUON_WDMATCHED_20M_EVAL", "baseline_cheap7": b_c7, "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.json')).write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# research Muon wd-matched 20M screen", "",
             f"Baseline (research legal 20M) cheap7: **{b_c7:.4f}**" if b_c7 else "Baseline: N/A", "",
             "| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rows.items():
        s = r.get("scores") or {}
        c7 = r.get("cheap7")
        dc = (r.get("deltas") or {}).get("cheap7") if r.get("deltas") else None
        def f(c): return f"{s[c]:.2f}" if s.get(c) is not None else ""
        lines.append(f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7:.4f} | {dc:+.4f} |" if c7 is not None and dc is not None else f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7 if c7 else ''} | |")
    (_public_path('research/documents/frontier_consolidation/data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
