#!/usr/bin/env python3
"""research late readout for adapter128 scale1.75.

Merges the existing 20/30/40/50M trajectory with the pending 70/80M cheap-column
evaluations once they appear.  The interpretation criterion is: do not decide from the 80M sign alone; read 70M+80M together with domain
movements and the short remaining distance to the official 100M endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

USER_ROOT = _public_path('.')
DATA = _public_path('experiments/archive/frontier_consolidation/data')
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_late_readout')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
PRESSURE = ["EWoK", "GlobalPIQA", "Reading"]
PROTECTED_POSITIVE = ["BLiMP", "Entity", "COMPS"]


def load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cheap7(scores: Dict[str, float]) -> float:
    return float(mean(float(scores[c]) for c in CHEAP))


def load_step48() -> Dict[str, Any]:
    p = _public_path('experiments/archive/frontier_consolidation/data/compliant_endpoint_results/compliant_endpoint_results_summary.json')
    d = load(p)
    if not d:
        raise FileNotFoundError(p)
    rec = next(r for r in d["records"] if r["name"] == "same_pool_tokenizer")
    return {
        "live_leader_overall": float(d["live_leader_overall"]),
        "overall": float(rec["overall"]),
        "scores": rec["scores"],
        "path": str(p),
    }


def load_trajectory_20_50() -> Dict[str, Any]:
    p = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_trajectory/scale1p75_trajectory.json')
    d = load(p)
    if not d:
        raise FileNotFoundError(p)
    return d


def ref(exposure: str) -> Optional[Dict[str, Any]]:
    p = _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target') / f"complianttok_reinvest_seed43022_{exposure}.json"
    d = load(p)
    if not d:
        return None
    scores0 = d.get("official_overall", {}).get("scores", {})
    scores = {c: float(scores0[c]) for c in CHEAP if scores0.get(c) is not None}
    if len(scores) != len(CHEAP):
        return None
    return {"label": f"research legal {exposure}", "scores": scores, "cheap7": cheap7(scores), "path": str(p)}


def adapter_eval(exposure: str) -> Optional[Dict[str, Any]]:
    ep = f"chck_{exposure}"
    p = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_70_80_eval') / ep / f"adapter128_scale1p75_{ep}_summary.json"
    d = load(p)
    if not d:
        return None
    rec = d.get("record", d)
    scores0 = rec.get("scores") or {}
    scores = {c: float(scores0[c]) for c in CHEAP if scores0.get(c) is not None}
    if len(scores) != len(CHEAP):
        return None
    return {"label": f"adapter128 scale1.75 {exposure}", "scores": scores, "cheap7": cheap7(scores), "path": str(p)}


def delta(ad: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    cd = {c: float(ad["scores"][c]) - float(ref["scores"][c]) for c in CHEAP}
    return {
        "cheap7_delta": float(ad["cheap7"] - ref["cheap7"]),
        "column_deltas": cd,
        "pressure_delta_sum": float(sum(cd[c] for c in PRESSURE)),
        "protected_positive_sum": float(sum(cd[c] for c in PROTECTED_POSITIVE)),
        "negative_columns_lt_minus_0p5": {c: v for c, v in cd.items() if v < -0.5},
        "positive_columns_gt_0p5": {c: v for c, v in cd.items() if v > 0.5},
    }


def overall_arithmetic(research: Dict[str, Any], endpoint_cheap7: Optional[float] = None) -> Dict[str, Any]:
    leader = research["live_leader_overall"]
    sg = float(research["scores"]["SuperGLUE"])
    aoa = float(research["scores"].get("AoA", 0.0))
    cheap = cheap7({c: float(research["scores"][c]) for c in CHEAP})
    required_cheap_if_sg_aoa_flat = (9.0 * leader - sg - aoa) / 7.0
    out = {
        "leader_overall": leader,
        "complete_overall": research["overall"],
        "reference_100M_cheap7": cheap,
        "superglue": sg,
        "aoa": aoa,
        "required_cheap7_if_superglue_and_aoa_match_step35": required_cheap_if_sg_aoa_flat,
        "required_cheap7_gain_over_step35_if_superglue_and_aoa_match_step35": required_cheap_if_sg_aoa_flat - cheap,
    }
    if endpoint_cheap7 is not None:
        projected_overall = (7.0 * endpoint_cheap7 + sg + aoa) / 9.0
        required_sg_plus_aoa = 9.0 * leader - 7.0 * endpoint_cheap7
        out.update({
            "endpoint_cheap7": endpoint_cheap7,
            "projected_overall_if_superglue_and_aoa_match_step35": projected_overall,
            "required_superglue_plus_aoa_for_41p8_at_endpoint_cheap7": required_sg_plus_aoa,
            "required_superglue_gain_if_aoa_zero": required_sg_plus_aoa - sg,
        })
    return out


def interpret(deltas: Dict[str, Any]) -> Dict[str, Any]:
    text = []
    signal = "pending_70_80"
    if "70M" not in deltas or "80M" not in deltas:
        text.append("70M/80M adapter cheap-column scores are not both present yet; no late-route interpretation is made.")
        return {"route_signal": signal, "interpretation": text}

    d70 = deltas["70M"]
    d80 = deltas["80M"]
    ch70, ch80 = d70["cheap7_delta"], d80["cheap7_delta"]
    p70 = d70["pressure_delta_sum"]
    p80 = d80["pressure_delta_sum"]
    neg70 = d70["negative_columns_lt_minus_0p5"]
    neg80 = d80["negative_columns_lt_minus_0p5"]
    pos70 = d70["positive_columns_gt_0p5"]
    pos80 = d80["positive_columns_gt_0p5"]
    text.append(f"70M cheap7 delta {ch70:+.4f}; 80M cheap7 delta {ch80:+.4f}.")
    text.append(f"70M negative columns below -0.5: {neg70}; positive columns above +0.5: {pos70}.")
    text.append(f"80M negative columns below -0.5: {neg80}; positive columns above +0.5: {pos80}.")
    text.append(f"Pressure-axis sum (EWoK+GlobalPIQA+Reading) moves {p70:+.3f} at 70M and {p80:+.3f} at 80M.")

    both_broadly_bad = (ch70 < -0.25 and ch80 < -0.25 and p80 < p70 - 0.5 and len(neg80) >= 3)
    mixed_or_positive = (ch70 > -0.25 or ch80 > -0.25 or len(pos80) >= 2)
    if both_broadly_bad:
        signal = "stop_fixed_scale1p75_maturation"
        text.append("Late points jointly show a widening broad deficit; the remaining 20M is unlikely to convert the fixed-amplitude path into a stronger endpoint.")
    elif mixed_or_positive:
        signal = "finish_exact_100M_endpoint"
        text.append("Late evidence is mixed or positive rather than coherently worsening; because the path is deterministic and only 20M from the official endpoint, finish this exact trajectory to 100M before judging endpoint value.")
    else:
        signal = "analyze_before_100M"
        text.append("Late points are weak but not clearly widening; inspect domain reports and mechanism before launching more training.")
    return {"route_signal": signal, "interpretation": text}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    research = load_step48()
    traj = load_trajectory_20_50()
    rows: Dict[str, Dict[str, Any]] = {"research": {}, "scale1p75": {}}
    deltas: Dict[str, Any] = {}
    for exp in ["20M", "30M", "40M", "50M"]:
        rows["research"][exp] = traj["rows"]["research"][exp]
        rows["scale1p75"][exp] = traj["rows"]["scale1p75"][exp]
        deltas[exp] = traj["deltas"][exp]
        deltas[exp]["pressure_delta_sum"] = float(sum(deltas[exp]["column_deltas"][c] for c in PRESSURE))
        deltas[exp]["protected_positive_sum"] = float(sum(deltas[exp]["column_deltas"][c] for c in PROTECTED_POSITIVE))
        deltas[exp]["negative_columns_lt_minus_0p5"] = {c: v for c, v in deltas[exp]["column_deltas"].items() if v < -0.5}
        deltas[exp]["positive_columns_gt_0p5"] = {c: v for c, v in deltas[exp]["column_deltas"].items() if v > 0.5}
    for exp in ["70M", "80M"]:
        ref = ref(exp)
        ad = adapter_eval(exp)
        if ref:
            rows["research"][exp] = ref
        if ad:
            rows["scale1p75"][exp] = ad
        if ref and ad:
            deltas[exp] = delta(ad, ref)
    interp = interpret(deltas)
    endpoint_cheap = rows["scale1p75"].get("100M", {}).get("cheap7")
    arith = overall_arithmetic(research, endpoint_cheap)
    out = {
        "status": "SCALE1P75_LATE_READOUT",
        "available_exposures": {arm: sorted(v) for arm, v in rows.items()},
        "rows": rows,
        "deltas": deltas,
        "overall_arithmetic": arith,
        **interp,
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_late_readout/scale1p75_late_readout.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_late_readout/scale1p75_late_readout.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research scale1.75 late readout", "", f"Route signal: `{out['route_signal']}`", "", "## Cheap-column trajectory", "", "| exposure | research cheap7 | scale1.75 cheap7 | delta | EWoK Δ | GlobalPIQA Δ | Reading Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for exp in ["20M", "30M", "40M", "50M", "70M", "80M"]:
        ref = rows["research"].get(exp)
        ad = rows["scale1p75"].get(exp)
        dd = deltas.get(exp)
        if ref and ad and dd:
            cd = dd["column_deltas"]
            lines.append(f"| {exp} | {ref['cheap7']:.4f} | {ad['cheap7']:.4f} | {dd['cheap7_delta']:+.4f} | {cd['EWoK']:+.3f} | {cd['GlobalPIQA']:+.3f} | {cd['Reading']:+.3f} |")
        elif ref or ad:
            lines.append(f"| {exp} | {ref['cheap7']:.4f}" if ref else f"| {exp} | pending" + f" | {ad['cheap7']:.4f}" if ad else " | pending" + " | | | | |")
    lines += ["", "## Official-endpoint arithmetic", "", f"- research complete Overall: {arith['complete_overall']:.6f}", f"- research 100M cheap7: {arith['reference_100M_cheap7']:.6f}", f"- Required cheap7 for Overall 41.8 if SuperGLUE/AoA match research: {arith['required_cheap7_if_superglue_and_aoa_match_step35']:.6f}", f"- Required cheap7 gain over research under that assumption: {arith['required_cheap7_gain_over_step35_if_superglue_and_aoa_match_step35']:+.6f}", "", "## Interpretation", ""]
    lines += [f"- {s}" for s in out["interpretation"]]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "route_signal": out["route_signal"], "available": out["available_exposures"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
