#!/usr/bin/env python3
"""research: merge parallel-safe LAMB 20M evaluations and compare with baselines."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from statistics import mean

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
ROOT_OUT = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval_parts')
OUT = _public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval_merged')
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

# Exact research and Muon matched-decay 20M scores already validated in previous steps.
BASELINES = {
    "reference_20M": {
        "label": "research legal AdamW 20M",
        "scores": {"BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73, "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67},
    },
    "muon_wdmatched_20M": {
        "label": "Muon wd-matched 20M",
        "scores": {"BLiMP": 61.15, "Supplement": 58.10, "EWoK": 50.70, "Entity": 20.90, "COMPS": 50.45, "GlobalPIQA": 38.605, "Reading": 7.26},
    },
}
ARMS = ["lr005", "lr007"]


def cheap7(scores: dict) -> float:
    return float(mean(float(scores[c]) for c in CHEAP_COLUMNS))


def load_arm(arm: str) -> dict | None:
    path = ROOT_OUT / arm / "lamb_arm_20M_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    scores = BASELINES["reference_20M"]["scores"]
    c7 = cheap7(scores)
    for key, rec in BASELINES.items():
        sc = rec["scores"]
        c7 = cheap7(sc)
        rows[key] = {
            "label": rec["label"],
            "scores": sc,
            "cheap7": c7,
            "deltas_vs_step35": {c: sc[c] - scores[c] for c in CHEAP_COLUMNS} | {"cheap7": c7 - c7},
            "complete": True,
        }
    for arm in ARMS:
        rec = load_arm(arm)
        if rec is None:
            rows[arm] = {"label": f"LAMB {arm}", "scores": None, "cheap7": None, "deltas_vs_step35": None, "complete": False}
            continue
        sc = rec["scores"]
        missing = [c for c in CHEAP_COLUMNS if sc.get(c) is None]
        if missing:
            rows[arm] = {"label": rec.get("label", arm), "scores": sc, "cheap7": None, "deltas_vs_step35": None, "complete": False, "missing": missing}
            continue
        c7 = cheap7(sc)
        endpoint = rec.get("endpoint", "")
        label = rec.get("label", arm)
        if endpoint and endpoint != "chck_20M":
            label = f"{label} ({endpoint} partial-shadow)"
        rows[arm] = {
            "label": label,
            "scores": sc,
            "cheap7": c7,
            "deltas_vs_step35": {c: sc[c] - scores[c] for c in CHEAP_COLUMNS} | {"cheap7": c7 - c7},
            "complete": True,
            "endpoint": endpoint,
            "run_dir": rec.get("run_dir"),
            "per_target_json": rec.get("per_target_json"),
        }
    out = {"status": "LAMB_20M_EVAL_MERGED", "baseline_cheap7": c7, "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/lamb_20M_eval_merged/lamb_20M_eval_merged.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = ["# research LAMB 20M merged cheap-column comparison", "",
             f"research legal AdamW 20M cheap7: **{c7:.4f}**", "",
             "| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs research | complete |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in rows.items():
        sc = r.get("scores") or {}
        def f(c): return f"{sc[c]:.2f}" if sc.get(c) is not None else "—"
        c7 = r.get("cheap7")
        d = (r.get("deltas_vs_step35") or {}).get("cheap7")
        lines.append(f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7:.4f} | {d:+.4f} | {r.get('complete')} |" if c7 is not None else f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | — | — | {r.get('complete')} |")
    lines += ["", "## Column deltas vs research", "", "| arm | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGP | ΔRead | Δcheap7 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in rows.items():
        d = r.get("deltas_vs_step35") or {}
        if not d:
            continue
        def fd(c): return f"{d[c]:+.2f}" if d.get(c) is not None else "—"
        lines.append(f"| {r['label']} | {fd('BLiMP')} | {fd('Supplement')} | {fd('EWoK')} | {fd('Entity')} | {fd('COMPS')} | {fd('GlobalPIQA')} | {fd('Reading')} | {d['cheap7']:+.4f} |")
    lines += ["", "Interpretation should combine these scores with `data/lamb_update_allocation_probe/lamb_update_allocation_probe.md`. A 20M cheap7 gain alone is not enough after Muon's mature reversal; only one LAMB arm should advance if broad columns and measured update allocation both match the intended Adam-direction-preserving mechanism."]
    (_public_path('research/documents/frontier_consolidation/data/lamb_20M_eval_merged/lamb_20M_eval_merged.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "rows": {k: {"cheap7": v.get("cheap7"), "complete": v.get("complete")} for k, v in rows.items()}, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
