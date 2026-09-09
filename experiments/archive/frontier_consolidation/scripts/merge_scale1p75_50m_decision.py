#!/usr/bin/env python3
"""research: merge scale-1.75 adapter 50M maturation evidence with prefix replication.

This script is intentionally score-file tolerant: it can be run before or after the two
research 50M eval tasks finish. Once both scores are present, it computes the matched
50M delta and a route-relevant interpretation.
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50m_decision')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

PATHS = {
    "prefix": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_prefix_repro/scale1p75_prefix_repro.json'),
    "reference_20": _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json'),
    "s1p625_20": _public_path('experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_s1p625/adapter128_scale1p625_h100M20M_seed43022_summary.json'),
    "reference_50": _public_path('experiments/archive/frontier_consolidation/data/50M_eval/legal_chck50M_summary.json'),
    "s1p75_50": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50M_eval/adapter128_scale1p75_h100M50M_seed43022_summary.json'),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_record(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    return payload.get("record", payload)


def cheap7(scores: Dict[str, float]) -> float:
    return float(mean(float(scores[c]) for c in CHEAP))


def score_row(label: str, scores: Dict[str, float], ref: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    row = {"label": label, "scores": scores, "cheap7": cheap7(scores)}
    if ref is not None:
        row["delta_vs_ref"] = {c: float(scores[c]) - float(ref[c]) for c in CHEAP}
        row["cheap7_delta_vs_ref"] = row["cheap7"] - cheap7(ref)
    return row


def reference_20_and_scale_rows() -> Dict[str, Any]:
    payload = load_json(PATHS["reference_20"])
    rows: Dict[str, Any] = {}
    if payload:
        for k, v in payload.get("rows", {}).items():
            rows[k] = v
    p1625 = unwrap_record(load_json(PATHS["s1p625_20"]))
    if p1625 and p1625.get("scores"):
        rows["train_scale1p625_20M"] = {
            "label": "train-time adapter scale1.625 20M",
            "scores": p1625["scores"],
            "cheap7": p1625["cheap7"],
        }
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = load_json(PATHS["prefix"])
    prefix_ok = bool(prefix and prefix.get("conclusion", {}).get("prefix_reproduces"))
    rows20 = reference_20_and_scale_rows()
    r35 = unwrap_record(load_json(PATHS["reference_50"]))
    r175 = unwrap_record(load_json(PATHS["s1p75_50"]))

    rows50: Dict[str, Any] = {}
    matched_delta = None
    if r35 and r35.get("scores"):
        rows50["reference_50M"] = score_row("research legal chck_50M", r35["scores"])
    if r175 and r175.get("scores"):
        ref_scores = r35["scores"] if r35 and r35.get("scores") else None
        rows50["Scale1p75_50M"] = score_row("adapter128 scale1.75 chck_50M", r175["scores"], ref_scores)
    if "reference_50M" in rows50 and "Scale1p75_50M" in rows50:
        matched_delta = rows50["Scale1p75_50M"]["cheap7_delta_vs_ref"]

    interpretation = []
    if prefix_ok:
        interpretation.append("The 50M run embeds the exact 20M scale1.75 trajectory: same normalized training prefix and bitwise-identical chck_20M model tensors.")
    elif prefix is not None:
        interpretation.append("The embedded 20M prefix did not reproduce exactly; resolve this before further exposure.")
    else:
        interpretation.append("Prefix replication evidence is not yet present.")

    if matched_delta is None:
        interpretation.append("Matched 50M score delta is not yet available because one or both evaluation summaries are missing.")
        route = "wait_for_scores"
    elif matched_delta > 0.30:
        bad = rows50["Scale1p75_50M"].get("delta_vs_ref", {})
        damaging = {k: v for k, v in bad.items() if v < -0.5}
        if not damaging:
            interpretation.append(f"Scale1.75 retains a broad 50M cheap7 advantage of {matched_delta:+.4f} without any cheap-column drop below -0.5.")
            route = "continue_exact_trajectory_to_80M"
        else:
            interpretation.append(f"Scale1.75 retains cheap7 advantage {matched_delta:+.4f}, but damaging columns remain: {damaging}.")
            route = "analyze_tradeoff_before_80M"
    elif matched_delta > 0.0:
        interpretation.append(f"Scale1.75 is positive at 50M but small ({matched_delta:+.4f}); this is weaker than the 20M signal and needs anatomy before further exposure.")
        route = "analyze_small_positive"
    else:
        interpretation.append(f"Scale1.75 does not retain a 50M cheap7 advantage ({matched_delta:+.4f}); treat the 20M gain as early/transient unless more precise anatomy says otherwise.")
        route = "stop_fixed_scale_maturation"

    out = {
        "status": "SCALE1P75_50M_DECISION",
        "inputs": {k: str(v) for k, v in PATHS.items()},
        "prefix_ok": prefix_ok,
        "rows20": rows20,
        "rows50": rows50,
        "matched_50m_cheap7_delta_scale1p75_minus_step35": matched_delta,
        "route_signal": route,
        "interpretation": interpretation,
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50m_decision/scale1p75_50m_decision.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_50m_decision/scale1p75_50m_decision.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research scale1.75 50M maturation decision",
        "",
        "## Prefix replication",
        f"- Embedded 20M prefix exact: **{prefix_ok}**",
    ]
    if prefix:
        c = prefix.get("conclusion", {})
        lines += [
            f"- safetensors hash equal: {c.get('hash_model_safetensors_equal')}",
            f"- log prefix equal ignoring elapsed seconds: {c.get('log_exact_equal_ignoring_elapsed')}",
            f"- tensor exact equal: {c.get('tensor_exact_equal')}",
        ]
    lines += ["", "## 20M reference rows", "", "| row | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in rows20.items():
        sc = row["scores"]
        lines.append(f"| {row.get('label', key)} | {float(row['cheap7']):.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    lines += ["", "## Matched 50M rows", "", "| row | cheap7 | Δ cheap7 vs research 50M | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ref50 = rows50.get("reference_50M", {}).get("cheap7")
    for key, row in rows50.items():
        sc = row["scores"]
        delta = "" if ref50 is None else f"{(float(row['cheap7']) - float(ref50)):+.4f}"
        lines.append(f"| {row.get('label', key)} | {float(row['cheap7']):.4f} | {delta} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    if matched_delta is not None and "Scale1p75_50M" in rows50:
        d = rows50["Scale1p75_50M"].get("delta_vs_ref", {})
        lines += ["", "### 50M per-column deltas scale1.75 minus research", "", "| column | delta |", "|---|---:|"]
        for c in CHEAP:
            lines.append(f"| {c} | {d[c]:+.3f} |")
    lines += ["", "## Interpretation", ""]
    for s in interpretation:
        lines.append(f"- {s}")
    lines += ["", f"Route signal: `{route}`", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "prefix_ok": prefix_ok, "matched_delta": matched_delta, "route_signal": route}, indent=2), flush=True)


if __name__ == "__main__":
    main()
