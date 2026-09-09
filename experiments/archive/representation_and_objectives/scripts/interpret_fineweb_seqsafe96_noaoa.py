#!/usr/bin/env python3
"""Interpret the pending FineWeb seqsafe96 no-AoA result.

The raw research summarizer reports treatment-control deltas. This script adds the
research scientific reading: the 1.753M-word FineWeb block is a 17.5% source-
breadth intervention, so decision should use magnitude and component signature,
not merely sign. It also compares the best treatment trajectory to the public
leader and inherited trusted clean-Qwen coordinate.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

SUMMARY = pathlib.Path("experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json")
PUBLIC = pathlib.Path("experiments/archive/representation_and_objectives/data/public_component_tradeoffs/public_component_tradeoffs.json")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_interpretation")
NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_seqsafe96_interpretation.md")

KNOWLEDGE_COLS = ["EWoK", "Entity", "COMPS", "GlobalPIQA"]
PROTECT_COLS = ["Supplement", "Reading", "BLiMP"]
ALL_NOAOA_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def col(row: dict[str, Any], name: str) -> float | None:
    return as_float(row.get(name))


def sum_cols(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [col(row, c) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(sum(v for v in vals if v is not None))


def ck_num(ck: str) -> int:
    try:
        return int(ck.split("_")[1].rstrip("M"))
    except Exception:
        return 10**9


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"pending summary does not exist yet: {SUMMARY}")
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pub = json.loads(PUBLIC.read_text(encoding="utf-8"))
    leader = {"Overall": float(pub["strict_small_leader"]["overall"])}
    for r in pub.get("top15", []):
        if r.get("rank") == 1:
            leader.update({
                "BLiMP": as_float(r.get("BLiMP")),
                "Supplement": as_float(r.get("Supplement")),
                "EWoK": as_float(r.get("EWoK")),
                "Entity": as_float(r.get("Entity")),
                "COMPS": as_float(r.get("COMPS")),
                "GlobalPIQA": as_float(r.get("GlobalPIQA")),
                "SuperGLUE": as_float(r.get("SuperGLUE")),
                "Reading": as_float(r.get("Reading")),
                "AoA": as_float(r.get("AoA")),
            })
            break
    ours = pub.get("ours", {})
    inherited = {
        "Overall": as_float(ours.get("Overall Average")),
        "BLiMP": as_float(ours.get("BLiMP")),
        "Supplement": as_float(ours.get("BLiMP Supplement")),
        "EWoK": as_float(ours.get("EWoK")),
        "Entity": as_float(ours.get("Entity Tracking")),
        "COMPS": as_float(ours.get("COMPS")),
        "GlobalPIQA": as_float(ours.get("GlobalPIQA")),
        "SuperGLUE": as_float(ours.get("(Super)GLUE")),
        "Reading": as_float(ours.get("Reading")),
        "AoA": as_float(ours.get("AoA")),
    }

    matched = s.get("matched_rows", {})
    enriched: dict[str, Any] = {}
    for ck, r in sorted(matched.items(), key=lambda kv: ck_num(kv[0])):
        d = r.get("delta_treatment_minus_control", {})
        tr = r.get("treatment", {})
        cr = r.get("control", {})
        ksum = sum_cols(d, KNOWLEDGE_COLS)
        protect_sum = sum_cols(d, PROTECT_COLS)
        tr_equal7 = col(tr, "equal7_full_eval")
        cr_equal7 = col(cr, "equal7_full_eval")
        tr_vs_leader = {c: None if col(tr, c) is None or leader.get(c) is None else round(col(tr, c) - leader[c], 6) for c in ALL_NOAOA_COLS}
        tr_vs_inherited = {c: None if col(tr, c) is None or inherited.get(c) is None else round(col(tr, c) - inherited[c], 6) for c in ALL_NOAOA_COLS}
        enriched[ck] = {
            "treatment_equal7": tr_equal7,
            "control_equal7": cr_equal7,
            "delta_equal7": col(d, "equal7_full_eval"),
            "delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA": None if ksum is None else round(ksum, 6),
            "delta_protected_sum_Supp_Reading_BLiMP": None if protect_sum is None else round(protect_sum, 6),
            "delta_components": d,
            "treatment_minus_public_leader_noaoa_components": tr_vs_leader,
            "treatment_minus_inherited_cleanqwen_noaoa_components": tr_vs_inherited,
        }

    valid = {ck: r for ck, r in enriched.items() if r["delta_equal7"] is not None}
    best_delta = max(valid.items(), key=lambda kv: kv[1]["delta_equal7"]) if valid else None
    best_ksum = max(valid.items(), key=lambda kv: kv[1]["delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"] if kv[1]["delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"] is not None else -9999) if valid else None
    best_treat = max(valid.items(), key=lambda kv: kv[1]["treatment_equal7"] if kv[1]["treatment_equal7"] is not None else -9999) if valid else None

    # Magnitude logic for a 17.5% source-breadth arm. Thresholds are route-reading
    # guides, not claims of statistical certainty.
    route_reading = "unclassified"
    reason = ""
    if best_ksum:
        ksum = best_ksum[1]["delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"]
        peq = best_delta[1]["delta_equal7"] if best_delta else None
        prot = best_ksum[1]["delta_protected_sum_Supp_Reading_BLiMP"]
        if ksum is not None and ksum >= 2.0 and (prot is None or prot >= -1.0):
            route_reading = "source_breadth_positive_scale_larger_and_combine_with_views"
            reason = "17.5% FineWeb source breadth produced a substantial knowledge-cluster gain without large protected-column loss."
        elif ksum is not None and ksum > 0.0:
            route_reading = "source_breadth_directional_but_underscaled_or_low_quality"
            reason = "Knowledge-cluster delta is positive but small for a 17.5% block; test larger and/or cleaner FineWeb plus generated views before rejecting source breadth."
        elif peq is not None and peq > 0.2:
            route_reading = "broad_noaoa_positive_but_not_knowledge_targeted"
            reason = "Overall/equal7 delta is positive, but the intended knowledge columns do not carry it strongly; inspect tradeoffs before scaling."
        else:
            route_reading = "cached_source_repetition_weak_use_view_or_new_mechanism"
            reason = "Cached FineWeb source repetition at 17.5% did not move the intended deficit cluster; prefer source+view, cleaner live FineWeb selection, or non-data mechanisms."

    payload = {
        "status": "FINEWEB_SEQSAFE96_INTERPRETED",
        "source_summary": str(SUMMARY),
        "public_reference": {"leader": leader, "inherited_cleanqwen": inherited},
        "fineweb_fraction": 0.175328,
        "knowledge_cols": KNOWLEDGE_COLS,
        "protected_cols": PROTECT_COLS,
        "enriched_rows": enriched,
        "best_by_delta_equal7": {"checkpoint": best_delta[0], "row": best_delta[1]} if best_delta else None,
        "best_by_knowledge_sum": {"checkpoint": best_ksum[0], "row": best_ksum[1]} if best_ksum else None,
        "best_by_treatment_equal7": {"checkpoint": best_treat[0], "row": best_treat[1]} if best_treat else None,
        "route_reading": route_reading,
        "reason": reason,
        "next_logic": {
            "large_positive": "train/evaluate larger live FineWeb fraction and joint source+faithful-view arms; consider full eval for best endpoint if no-AoA exceeds inherited clean-Qwen by enough margin",
            "small_positive": "treat as under-scaled source-breadth evidence; build larger live FineWeb source asset and use A02 near/compact view evidence for joint arm",
            "flat_negative": "do not extend cached source repetition alone; use A02 view mechanism, cleaner relation-dense selection, tokenization/masking/architecture routes",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "fineweb_seqsafe96_route_interpretation.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research FineWeb seqsafe96 route interpretation\n\n"]
    lines.append(f"Source summary: `{SUMMARY}`\n\n")
    lines.append(f"FineWeb source fraction: {payload['fineweb_fraction']:.2%}. This is large enough to test direction, not necessarily large enough to match the public leader's full FineWeb-pair phenotype.\n\n")
    if best_delta:
        lines.append(f"Best equal7 delta: `{best_delta[0]}` = {best_delta[1]['delta_equal7']:.4f}.\n")
    if best_ksum:
        lines.append(f"Best knowledge-cluster delta: `{best_ksum[0]}` = {best_ksum[1]['delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA']:.4f} across EWoK+Entity+COMPS+GlobalPIQA.\n")
    if best_treat:
        lines.append(f"Best treatment equal7: `{best_treat[0]}` = {best_treat[1]['treatment_equal7']:.4f}.\n")
    lines.append(f"\nRoute reading: `{route_reading}` — {reason}\n\n")
    lines.append("| checkpoint | treat eq7 | ctrl eq7 | delta eq7 | knowledge Δsum | protected Δsum | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    def fmt(x: Any) -> str:
        return "" if x is None else f"{float(x):.4f}"
    for ck, r in sorted(enriched.items(), key=lambda kv: ck_num(kv[0])):
        d = r["delta_components"]
        lines.append(f"| {ck} | {fmt(r['treatment_equal7'])} | {fmt(r['control_equal7'])} | {fmt(r['delta_equal7'])} | {fmt(r['delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA'])} | {fmt(r['delta_protected_sum_Supp_Reading_BLiMP'])} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA'))} |\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "route_reading": route_reading}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
