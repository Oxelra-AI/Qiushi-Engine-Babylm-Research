#!/usr/bin/env python3
"""research: evaluate the matched-decay Muon 70M/80M mature readout.

This script runs official-compatible cheap-column evaluation for the bounded
80M run at chck_70M and chck_80M, then compares with the research legal compact
reinvest trajectory at the same exposures. It does not run SuperGLUE or AoA and
it does not make a submission endpoint.
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
RUN_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_collate')

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

BASELINES = {
    "70M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
    "80M": _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
}
TARGETS = {
    "70M": {"endpoint": "chck_70M", "target": "muon_lr008_wd00125_70M"},
    "80M": {"endpoint": "chck_80M", "target": "muon_lr008_wd00125_80M"},
}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def extract_scores(payload: dict | None):
    if payload is None: return None
    tasks = payload.get("tasks", {})
    out = {}
    for c in ZERO:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp = []
    for c in GP:
        rec = tasks.get(c, {})
        if rec.get("score") is not None: gp.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(sc):
    if sc is None: return None
    vals = [sc.get(c) for c in CHEAP]
    if any(v is None for v in vals): return None
    return mean(float(v) for v in vals)


def run_one(label: str, gpu: int):
    spec = TARGETS[label]
    ck = _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model') / spec["endpoint"]
    if not (ck / "model.safetensors").exists():
        print(f"SKIP {label}: missing {ck}", flush=True)
        return None
    pt = _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/per_target') / f"{spec['target']}.json"
    if pt.exists():
        print(f"EXISTS {label}: {pt}", flush=True)
        return pt
    cmd = [sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
           "--run-dir", str(RUN_DIR), "--target", spec["target"], "--endpoint", spec["endpoint"],
           "--out-root", str(OUT_ROOT), "--collate-root", str(COLLATE_ROOT), "--gpu", str(gpu),
           "--columns", *EVAL_COLUMNS]
    print(f"RUN {label}: {spec['target']} {spec['endpoint']} gpu={gpu}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"FAIL {label}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-4000:]}", flush=True)
        return None
    print(proc.stdout[-3000:], flush=True)
    return pt if pt.exists() else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", choices=["70M", "80M", ""], default="")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    labels = [args.only] if args.only else ["70M", "80M"]

    rows = {}
    for label in labels:
        pt = run_one(label, args.gpu)
        mu = extract_scores(read_json(pt)) if pt else None
        bs = extract_scores(read_json(BASELINES[label]))
        mu_c7, bs_c7 = cheap7(mu), cheap7(bs)
        deltas = None
        if mu and bs:
            deltas = {c: mu[c] - bs[c] for c in CHEAP}
            deltas["cheap7"] = mu_c7 - bs_c7
        rows[label] = {
            "baseline_path": str(BASELINES[label]),
            "muon_path": str(pt) if pt else None,
            "baseline_scores": bs,
            "baseline_cheap7": bs_c7,
            "muon_scores": mu,
            "muon_cheap7": mu_c7,
            "deltas": deltas,
        }

    out = {"status": "MUON_WDMATCHED_MATURE_EVAL", "run_dir": str(RUN_DIR), "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.json')).write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# research Muon wd-matched mature cheap-column readout", "", "| ckpt | arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in rows.items():
        for arm, sc, c7 in [("research", r["baseline_scores"], r["baseline_cheap7"]), ("Muon wd-match", r["muon_scores"], r["muon_cheap7"])]:
            if sc:
                lines.append(f"| {label} | {arm} | {sc['BLiMP']:.2f} | {sc['Supplement']:.2f} | {sc['EWoK']:.2f} | {sc['Entity']:.2f} | {sc['COMPS']:.2f} | {sc['GlobalPIQA']:.2f} | {sc['Reading']:.3f} | {c7:.4f} |")
        d = r.get("deltas")
        if d:
            lines.append(f"| {label} | Δ | {d['BLiMP']:+.2f} | {d['Supplement']:+.2f} | {d['EWoK']:+.2f} | {d['Entity']:+.2f} | {d['COMPS']:+.2f} | {d['GlobalPIQA']:+.2f} | {d['Reading']:+.3f} | {d['cheap7']:+.4f} |")
    (_public_path('research/documents/frontier_consolidation/data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
