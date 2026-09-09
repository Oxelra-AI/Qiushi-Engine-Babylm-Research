#!/usr/bin/env python3
"""Merge research scaled-adapter interpolation and 50M maturation evidence when present."""
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/scaled_adapter_route_synthesis')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
reference_20 = {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67}


def c7(scores):
    return float(mean(scores[c] for c in CHEAP))


def read_record(p: Path):
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if "record" in data:
        return data["record"]
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    p = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json')
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, row in d.get("rows", {}).items():
            records[k] = row
    p1625 = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p625/adapter128_scale1p625_h100M20M_seed43022_summary.json')
    r1625 = read_record(p1625)
    if r1625 and r1625.get("scores"):
        records["train_scale1p625_20M"] = {"label": "train-time adapter scale 1.625 20M", "scores": r1625["scores"], "cheap7": r1625["cheap7"]}
    p35_50 = _public_path('experiments/archive/frontier_consolidation/data/50M_eval/legal_chck50M_summary.json')
    r35 = read_record(p35_50)
    if r35 and r35.get("scores"):
        records["reference_50M"] = {"label": "research legal 50M", "scores": r35["scores"], "cheap7": r35["cheap7"]}
    p175_50 = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/adapter128_scale1p75_h100M50M_seed43022_summary.json')
    r175 = read_record(p175_50)
    if r175 and r175.get("scores"):
        records["train_scale1p75_50M"] = {"label": "train-time adapter scale1.75 50M", "scores": r175["scores"], "cheap7": r175["cheap7"]}
    out = {"status": "SCALED_ADAPTER_ROUTE_SYNTHESIS", "records": records}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scaled_adapter_route_synthesis/scaled_adapter_route_synthesis.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scaled_adapter_route_synthesis/scaled_adapter_route_synthesis.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scaled adapter route synthesis", "", "| row | cheap7 | Δ vs research 20M | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ref = c7(reference_20)
    for k, row in records.items():
        sc = row["scores"]; cheap = row["cheap7"]
        lines.append(f"| {row.get('label', k)} | {cheap:.4f} | {cheap-ref:+.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    if "reference_50M" in records and "train_scale1p75_50M" in records:
        delta = records["train_scale1p75_50M"]["cheap7"] - records["reference_50M"]["cheap7"]
        lines += ["", f"50M scale1.75 minus research cheap7: {delta:+.4f}."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "n": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
