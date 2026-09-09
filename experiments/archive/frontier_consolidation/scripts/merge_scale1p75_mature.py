#!/usr/bin/env python3
"""Merge scale1.75 mature 70M/80M evals against research research references."""
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_decision')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap(path: Path) -> Optional[Dict[str, Any]]:
    p = load(path)
    if not p:
        return None
    return p.get("record", p)


def extract_step50_ref(exposure: str) -> Optional[Dict[str, Any]]:
    p = _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target') / f"complianttok_reinvest_seed43022_{exposure}.json"
    d = load(p)
    if not d:
        return None
    scores = d.get("official_overall", {}).get("scores", {})
    scores = {c: scores[c] for c in CHEAP if scores.get(c) is not None}
    if len(scores) != len(CHEAP):
        return None
    return {"label": f"research legal {exposure}", "scores": scores, "cheap7": float(mean(scores[c] for c in CHEAP)), "path": str(p)}


def extract_adapter(exposure: str) -> Optional[Dict[str, Any]]:
    ep = f"chck_{exposure}"
    p = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_70_80_eval') / ep / f"adapter128_scale1p75_{ep}_summary.json"
    r = unwrap(p)
    if not r or not r.get("scores"):
        return None
    return {"label": f"adapter128 scale1.75 {exposure}", "scores": r["scores"], "cheap7": float(r["cheap7"]), "path": str(p)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Dict[str, Any]] = {}
    deltas: Dict[str, Dict[str, Any]] = {}
    for exposure in ["70M", "80M"]:
        ref = extract_step50_ref(exposure)
        ad = extract_adapter(exposure)
        if ref:
            rows[f"Step35_{exposure}"] = ref
        if ad:
            rows[f"scale1p75_{exposure}"] = ad
        if ref and ad:
            cd = {c: float(ad["scores"][c]) - float(ref["scores"][c]) for c in CHEAP}
            deltas[exposure] = {"cheap7_delta": float(ad["cheap7"] - ref["cheap7"]), "column_deltas": cd}
    route = "pending"
    interpretation = []
    if "80M" in deltas:
        d80 = deltas["80M"]
        neg80 = {c: v for c, v in d80["column_deltas"].items() if v < -0.5}
        if d80["cheap7_delta"] >= 0.35 and len(neg80) <= 2:
            route = "continue_to_100M_and_full_eval_candidate"
            interpretation.append(f"80M retains a strong cheap7 advantage {d80['cheap7_delta']:+.4f}; negative columns below -0.5: {neg80}.")
        elif d80["cheap7_delta"] > 0.0:
            route = "positive_but_analyze_before_100M"
            interpretation.append(f"80M is positive but weaker/mixed ({d80['cheap7_delta']:+.4f}); inspect column anatomy before endpoint work.")
        else:
            route = "stop_fixed_scale_adapter"
            interpretation.append(f"80M is not positive ({d80['cheap7_delta']:+.4f}); fixed scale1.75 behaves like another mature redistribution route.")
    else:
        interpretation.append("70M/80M adapter scores are not both present yet.")
    out = {"status": "SCALE1P75_MATURE_DECISION", "rows": rows, "deltas": deltas, "route_signal": route, "interpretation": interpretation}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_decision/scale1p75_mature_decision.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_mature_decision/scale1p75_mature_decision.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scale1.75 mature 70/80M decision", "", "## Scores", "", "| exposure | arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exposure in ["70M", "80M"]:
        for arm in ["research", "scale1p75"]:
            r = rows.get(f"{arm}_{exposure}")
            if r:
                sc = r["scores"]
                lines.append(f"| {exposure} | {arm} | {r['cheap7']:.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    lines += ["", "## Deltas scale1.75 minus research", "", "| exposure | cheap7 Δ | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exposure, d in deltas.items():
        cd = d["column_deltas"]
        lines.append(f"| {exposure} | {d['cheap7_delta']:+.4f} | {cd['BLiMP']:+.3f} | {cd['Supplement']:+.3f} | {cd['EWoK']:+.3f} | {cd['Entity']:+.3f} | {cd['COMPS']:+.3f} | {cd['GlobalPIQA']:+.3f} | {cd['Reading']:+.3f} |")
    lines += ["", "## Interpretation", ""]
    for s in interpretation:
        lines.append(f"- {s}")
    lines += ["", f"Route signal: `{route}`", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "route_signal": route, "available_deltas": sorted(deltas)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
