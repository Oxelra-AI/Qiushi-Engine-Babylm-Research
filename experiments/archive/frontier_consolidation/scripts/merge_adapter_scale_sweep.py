#!/usr/bin/env python3
"""Merge and interpret research adapter scale-sweep batches."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep/adapter_scale_sweep_summary.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/adapter_scale_sweep/adapter_scale_sweep_summary.md')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
research = {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67}
C7 = mean(research[c] for c in CHEAP)
LIVE103 = {"BLiMP": 60.38, "Supplement": 56.23, "EWoK": 49.49, "Entity": 18.65, "COMPS": 50.44, "GlobalPIQA": 32.225, "Reading": 8.31}
LIVE103_C7 = mean(LIVE103[c] for c in CHEAP)


def load_records():
    records = []
    search_roots = [
        ROOT,
        _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep_high'),
        _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep_mid'),
        _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep_extended'),
    ]
    for sr in search_roots:
        for p in sorted(sr.glob("scale_batch_*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in data.get("records", []):
                r = dict(r)
                r["batch_summary"] = str(p)
                records.append(r)
    # Deduplicate by scale, preferring successful records and then later filename order.
    by_scale = {}
    for r in records:
        s = float(r.get("scale"))
        if s not in by_scale or (by_scale[s].get("returncode") != 0 and r.get("returncode") == 0):
            by_scale[s] = r
    return [by_scale[s] for s in sorted(by_scale)]


def main():
    records = load_records()
    if not records:
        raise SystemExit(f"No batch records found under {ROOT}")
    for r in records:
        scores = r.get("scores") or {}
        c7 = r.get("cheap7")
        if c7 is None and scores:
            vals = [scores.get(c) for c in CHEAP]
            if all(v is not None for v in vals):
                c7 = mean(vals)
                r["cheap7"] = c7
        if c7 is not None:
            r["delta_vs_step35_cheap7"] = float(c7 - C7)
            r["delta_vs_live103_cheap7"] = float(c7 - LIVE103_C7)
            r["scores_delta_vs_step35"] = {c: (float(scores[c] - research[c]) if scores.get(c) is not None else None) for c in CHEAP}
    success = [r for r in records if r.get("returncode") == 0 and r.get("cheap7") is not None]
    best = max(success, key=lambda r: r["cheap7"]) if success else None
    interior = [r for r in success if 0.0 < float(r["scale"]) < 1.5]
    best_interior = max(interior, key=lambda r: r["cheap7"]) if interior else None
    # A compact route judgment from this no-training readout only.
    interpretation = []
    if best:
        if best["cheap7"] > C7:
            interpretation.append("At least one inference scale exceeds research 20M cheap7; adapter amplitude control deserves construction/training follow-up.")
        elif best["cheap7"] > LIVE103_C7:
            interpretation.append("A scale improves over the live scale=1 endpoint but still does not exceed research; this supports amplitude sensitivity but not a promoted ordinary-adapter route by itself.")
        else:
            interpretation.append("No tested scale improves over the live endpoint; the scale map gives no evidence that simple post-hoc amplitude rescales rescue ordinary adapters.")
        if best_interior and best_interior is best and best["cheap7"] > LIVE103_C7:
            interpretation.append("The best point is interior, so any follow-up should control branch amplitude during training rather than only train longer at fixed scale=1.")
    failures = [r for r in records if r.get("returncode") != 0]
    out = {
        "status": "ADAPTER_SCALE_SWEEP_SUMMARY",
        "reference_20M": {"scores": research, "cheap7": C7},
        "live128_scale1_reference": {"scores": LIVE103, "cheap7": LIVE103_C7},
        "records": records,
        "best_success": best,
        "best_interior": best_interior,
        "failures": failures,
        "interpretation": interpretation,
        "note": "Inference-scale shadows are co-adapted endpoint interventions; they are not a clean decomposition of direct adapter output from backbone drift.",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research adapter scale-sweep summary", "", "Inference-time adapter scaling is treated as an endpoint sensitivity map, not as a clean branch-vs-backbone causal separation.", "", f"research/disabled 20M cheap7: {C7:.4f}; research live scale=1 cheap7: {LIVE103_C7:.4f}.", "", "| scale | cheap7 | Δ vs research | Δ vs live1 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in success:
        sc = r.get("scores") or {}
        def f(c):
            v = sc.get(c)
            return "" if v is None else f"{v:.3f}"
        lines.append(f"| {float(r['scale']):.2f} | {r['cheap7']:.4f} | {r['delta_vs_step35_cheap7']:+.4f} | {r['delta_vs_live103_cheap7']:+.4f} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} |")
    if failures:
        lines += ["", "## Failed records"]
        for r in failures:
            lines.append(f"- scale {r.get('scale')}: returncode {r.get('returncode')}, stderr_tail={r.get('stderr_tail','')[-500:]}")
    lines += ["", "## Interpretation"]
    if interpretation:
        lines += [f"- {x}" for x in interpretation]
    else:
        lines.append("- Insufficient successful records for route interpretation.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD), "n_success": len(success), "best_scale": None if not best else best.get("scale"), "best_cheap7": None if not best else best.get("cheap7")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
