#!/usr/bin/env python3
"""Merge isolated research train-time scaled-adapter 20M evaluations."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged')
SOURCES = [_public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p75'), _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s2p00')]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
research = {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67}
LIVE = {"BLiMP": 60.38, "Supplement": 56.23, "EWoK": 49.49, "Entity": 18.65, "COMPS": 50.44, "GlobalPIQA": 32.225, "Reading": 8.31}
INFER = {
    "scale1p75": {"BLiMP": 60.56, "Supplement": 57.12, "EWoK": 50.74, "Entity": 18.47, "COMPS": 50.46, "GlobalPIQA": 32.71, "Reading": 8.225},
    "scale2p00": {"BLiMP": 60.61, "Supplement": 57.35, "EWoK": 51.08, "Entity": 18.46, "COMPS": 50.33, "GlobalPIQA": 32.71, "Reading": 8.195},
}


def c7(scores):
    return float(mean(scores[c] for c in CHEAP))


def main():
    records = []
    for src in SOURCES:
        p = src / "scaled_train_20M_summary.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data.get("records", []):
            r = dict(r)
            r["source_summary"] = str(p)
            records.append(r)
    rows = {
        "reference_20M": {"label": "research legal 20M", "scores": research, "cheap7": c7(research)},
        "train_scale1_eval_scale1": {"label": "research trained scale1 evaluated scale1", "scores": LIVE, "cheap7": c7(LIVE)},
        "infer_scale1p75": {"label": "research scale1 checkpoint evaluated at scale1.75", "scores": INFER["scale1p75"], "cheap7": c7(INFER["scale1p75"])},
        "infer_scale2p00": {"label": "research scale1 checkpoint evaluated at scale2.00", "scores": INFER["scale2p00"], "cheap7": c7(INFER["scale2p00"])},
    }
    successes = []
    failures = []
    for r in records:
        if r.get("returncode") == 0 and r.get("scores"):
            successes.append(r)
            rows[f"train_{r['arm']}"] = {"label": r["label"], "scores": r["scores"], "cheap7": r["cheap7"], "delta_vs_step35": r.get("delta_vs_step35"), "delta_vs_inference_scale_map": r.get("delta_vs_inference_scale_map")}
        else:
            failures.append(r)
    best = max(successes, key=lambda r: r["cheap7"]) if successes else None
    interp = []
    if best:
        if best["cheap7"] > c7(research):
            interp.append("A train-time scaled adapter 20M arm exceeds research 20M cheap7, so residual amplitude is a trajectory-level signal rather than only post-hoc endpoint sensitivity.")
        else:
            interp.append("No train-time scaled adapter arm exceeds research 20M; post-hoc endpoint sensitivity does not transfer to the learning trajectory at 20M.")
        for r in successes:
            if r.get("delta_vs_inference_scale_map", {}).get("cheap7") is not None:
                if r["delta_vs_inference_scale_map"]["cheap7"] < -0.2:
                    interp.append(f"{r['arm']} is materially worse than evaluating the research scale1 checkpoint at the same amplitude; fixed high branch amplitude during learning may perturb the trajectory rather than just reveal missing output gain.")
    out = {"status": "SCALED_TRAIN_20M_MERGED", "rows": rows, "records": records, "successes": successes, "failures": failures, "best_success": best, "interpretation": interp}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research train-time scaled adapter 20M merged", "", "| row | cheap7 | Δ vs research | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows.values():
        sc = row["scores"]
        cheap = row["cheap7"]
        lines.append(f"| {row['label']} | {cheap:.4f} | {cheap - c7(research):+.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    lines += ["", "## Interpretation"] + [f"- {x}" for x in interp]
    if failures:
        lines += ["", "## Failed records"] + [f"- {r.get('arm')}: rc={r.get('returncode')} stderr_tail={r.get('stderr_tail','')[-500:]}" for r in failures]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "n_success": len(successes), "best": None if not best else {"arm": best["arm"], "cheap7": best["cheap7"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
