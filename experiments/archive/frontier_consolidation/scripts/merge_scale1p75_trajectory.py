#!/usr/bin/env python3
"""Merge scale1.75 vs research cheap-column trajectory at 20/30/40/50M.

This reads already-computed evaluation summaries and produces a compact time-course for
the residual-amplitude route. It is intentionally tolerant of missing 30M/40M files so
it can be run before or after background evals finish.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

USER_ROOT = _public_path('.')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_trajectory')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

P20_MERGED = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json')
P50_STEP35 = _public_path('experiments/archive/frontier_consolidation/data/50M_eval/legal_chck50M_summary.json')
P50_S175 = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/adapter128_scale1p75_h100M50M_seed43022_summary.json')


def load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap(path: Path) -> Optional[Dict[str, Any]]:
    p = load(path)
    if not p:
        return None
    return p.get("record", p)


def cheap7(scores: Dict[str, float]) -> float:
    return float(mean(float(scores[c]) for c in CHEAP))


def add(rows: Dict[str, Dict[str, Any]], arm: str, exposure: str, label: str, rec: Optional[Dict[str, Any]]):
    if rec and rec.get("scores"):
        rows.setdefault(arm, {})[exposure] = {"label": label, "scores": rec["scores"], "cheap7": float(rec.get("cheap7", cheap7(rec["scores"])))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, Any]] = {"research": {}, "scale1p75": {}}

    p20 = load(P20_MERGED)
    if p20:
        r35 = p20.get("rows", {}).get("reference_20m") or p20.get("rows", {}).get("research legal 20M")
        # Actual keys in research merged are preserved in JSON; fall back by label search.
        if not r35:
            for k, v in p20.get("rows", {}).items():
                if "research" in v.get("label", k) and "20" in v.get("label", k):
                    r35 = v
        r175 = None
        for k, v in p20.get("rows", {}).items():
            lab = v.get("label", k)
            if "scale 1.75" in lab or "scale1.75" in lab:
                if "train-time" in lab:
                    r175 = v
        if r35:
            rows["research"]["20M"] = {"label": r35.get("label", "research legal 20M"), "scores": r35["scores"], "cheap7": float(r35["cheap7"])}
        if r175:
            rows["scale1p75"]["20M"] = {"label": r175.get("label", "adapter scale1.75 20M"), "scores": r175["scores"], "cheap7": float(r175["cheap7"])}

    for exposure in ["30M", "40M"]:
        ep = f"chck_{exposure}"
        add(rows, "research", exposure, f"research legal {exposure}", unwrap(DATA / f"trajectory_eval_30_40/{ep}/legal_{ep}_summary.json"))
        add(rows, "scale1p75", exposure, f"adapter128 scale1.75 {exposure}", unwrap(DATA / f"trajectory_eval_scale1p75_30_40/{ep}/scale1p75_{ep}_summary.json"))

    add(rows, "research", "50M", "research legal 50M", unwrap(P50_STEP35))
    add(rows, "scale1p75", "50M", "adapter128 scale1.75 50M", unwrap(P50_S175))

    deltas: Dict[str, Dict[str, Any]] = {}
    for exposure in ["20M", "30M", "40M", "50M"]:
        a = rows["research"].get(exposure)
        b = rows["scale1p75"].get(exposure)
        if a and b:
            deltas[exposure] = {
                "cheap7_delta": b["cheap7"] - a["cheap7"],
                "column_deltas": {c: float(b["scores"][c]) - float(a["scores"][c]) for c in CHEAP},
            }

    interpretation = []
    if all(e in deltas for e in ["20M", "30M", "40M", "50M"]):
        seq = [deltas[e]["cheap7_delta"] for e in ["20M", "30M", "40M", "50M"]]
        interpretation.append(f"cheap7 delta trajectory 20/30/40/50M = {[round(x, 4) for x in seq]}")
        if seq[-1] > 0 and seq[-1] >= min(seq[:-1]):
            interpretation.append("The 50M advantage is not an isolated one-checkpoint artifact; it remains positive after the exact reproduced 20M prefix.")
        if deltas["50M"]["column_deltas"].get("Entity", 0) > 0 and deltas["50M"]["column_deltas"].get("BLiMP", 0) > 0:
            interpretation.append("Entity and BLiMP are positive at 50M, unlike the scale1.75 20M Entity weakness.")
        neg50 = {c: v for c, v in deltas["50M"]["column_deltas"].items() if v < -0.5}
        if neg50:
            interpretation.append(f"Remaining 50M negative columns below -0.5 are {neg50}; mature continuation should test whether these recover or deepen.")
    else:
        missing = [e for e in ["20M", "30M", "40M", "50M"] if e not in deltas]
        interpretation.append(f"Missing matched exposures: {missing}")

    out = {"status": "SCALE1P75_TRAJECTORY", "rows": rows, "deltas": deltas, "interpretation": interpretation}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_trajectory/scale1p75_trajectory.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_trajectory/scale1p75_trajectory.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research scale1.75 trajectory", "", "## Scores", "", "| exposure | arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exposure in ["20M", "30M", "40M", "50M"]:
        for arm in ["research", "scale1p75"]:
            r = rows[arm].get(exposure)
            if r:
                sc = r["scores"]
                lines.append(f"| {exposure} | {arm} | {r['cheap7']:.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    lines += ["", "## Deltas scale1.75 minus research", "", "| exposure | cheap7 Δ | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exposure in ["20M", "30M", "40M", "50M"]:
        d = deltas.get(exposure)
        if d:
            cd = d["column_deltas"]
            lines.append(f"| {exposure} | {d['cheap7_delta']:+.4f} | {cd['BLiMP']:+.3f} | {cd['Supplement']:+.3f} | {cd['EWoK']:+.3f} | {cd['Entity']:+.3f} | {cd['COMPS']:+.3f} | {cd['GlobalPIQA']:+.3f} | {cd['Reading']:+.3f} |")
    lines += ["", "## Interpretation", ""]
    for s in interpretation:
        lines.append(f"- {s}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "available_deltas": sorted(deltas), "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
