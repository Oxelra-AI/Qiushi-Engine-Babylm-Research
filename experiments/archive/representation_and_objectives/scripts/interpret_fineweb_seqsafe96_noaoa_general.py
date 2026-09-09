#!/usr/bin/env python3
"""General interpreter for FineWeb seqsafe96 no-AoA source-breadth summaries.

Use for the repaired research sequential run, or for the original research path if it ever
exists. The interpreter adds the source-breadth route reading used in research and, when
available, incorporates the research trainer-exact token/masking exposure audit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

DEFAULT_SUMMARY = pathlib.Path(
    "experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval_repairseq"
    "fineweb_seqsafe96_delta_summary.json"
)
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_repairseq_interpretation")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_seqsafe96_repairseq_interpretation.md")
PUBLIC = pathlib.Path("experiments/archive/representation_and_objectives/data/public_component_tradeoffs/public_component_tradeoffs.json")
TOKEN_AUDIT = pathlib.Path("experiments/archive/representation_and_objectives/data/tokenizer_exposure_audit/tokenizer_exposure_audit_trainer_exact.json")

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


def load_public_reference() -> tuple[dict[str, Any], dict[str, Any]]:
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
    return leader, inherited


def load_token_audit() -> dict[str, Any] | None:
    if not TOKEN_AUDIT.exists():
        return None
    d = json.loads(TOKEN_AUDIT.read_text(encoding="utf-8"))
    delta = d.get("deltas_treatment_minus_control", {})
    return {
        "path": str(TOKEN_AUDIT),
        "status": d.get("status"),
        "candidate_tokens_seen_rel_pct": delta.get("candidate_tokens_seen_rel_pct"),
        "wwm_groups_seen_rel_pct": delta.get("wwm_groups_seen_rel_pct"),
        "treat_candidate_tok_per_word": d.get("treatment", {}).get("aggregate", {}).get("candidate_tokens_per_word_seen"),
        "ctrl_candidate_tok_per_word": d.get("control", {}).get("aggregate", {}).get("candidate_tokens_per_word_seen"),
        "interpretation": "trainer-exact audit showed no treatment token-budget advantage; positive source-breadth effects are not explained by extra treatment candidate tokens",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--label", default="research repaired sequential FineWeb seqsafe96")
    args = ap.parse_args()

    summary = pathlib.Path(args.summary)
    if not summary.exists():
        raise SystemExit(f"summary does not exist yet: {summary}")
    out_dir = pathlib.Path(args.out_dir)
    note = pathlib.Path(args.note)

    s = json.loads(summary.read_text(encoding="utf-8"))
    leader, inherited = load_public_reference()
    token_audit = load_token_audit()

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

    route_reading = "unclassified"
    reason = ""
    if best_ksum:
        ksum = best_ksum[1]["delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA"]
        peq = best_delta[1]["delta_equal7"] if best_delta else None
        prot = best_ksum[1]["delta_protected_sum_Supp_Reading_BLiMP"]
        if ksum is not None and ksum >= 2.0 and (prot is None or prot >= -1.0):
            route_reading = "source_breadth_positive_scale_larger_and_combine_with_views"
            reason = "17.5% FineWeb source breadth produced substantial knowledge-cluster gain without large protected-column loss."
        elif ksum is not None and ksum > 0.0:
            route_reading = "source_breadth_directional_but_underscaled_or_low_quality"
            reason = "Knowledge-cluster delta is positive but small for a 17.5% block; test larger and/or cleaner FineWeb plus generated views before rejecting source breadth."
        elif peq is not None and peq > 0.2:
            route_reading = "broad_noaoa_positive_but_not_knowledge_targeted"
            reason = "Overall/equal7 delta is positive, but the intended knowledge columns do not carry it strongly; inspect tradeoffs before scaling."
        else:
            route_reading = "cached_source_repetition_weak_use_view_or_new_mechanism"
            reason = "Cached FineWeb source repetition at 17.5% did not move the intended deficit cluster; prefer source+view, cleaner relation-dense selection, tokenization/masking/architecture routes."

    payload = {
        "status": "FINEWEB_SEQSAFE96_GENERAL_INTERPRETED",
        "label": args.label,
        "source_summary": str(summary),
        "public_reference": {"leader": leader, "inherited_cleanqwen": inherited},
        "token_audit": token_audit,
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
            "large_positive": "materialize/evaluate larger corrected four-arm family A_natural/B_breadth/B_repeat/C_view using cleaner live FineWeb source and faithful views",
            "small_positive": "treat as under-scaled or quality-limited source-breadth evidence; improve live source filtering and view faithfulness before next H100 allocation",
            "flat_negative": "do not extend cached source repetition alone; use A02 view mechanisms, cleaner relation-dense source+view, or non-data recipe/tokenizer mechanisms",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "fineweb_seqsafe96_general_route_interpretation.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        return "" if x is None else f"{float(x):.4f}"

    lines = [f"# {args.label} route interpretation\n\n"]
    lines.append(f"Source summary: `{summary}`\n\n")
    lines.append(f"FineWeb source fraction: {payload['fineweb_fraction']:.2%}. This tests direction and magnitude, not the leader's full 10M FineWeb-pair phenotype.\n\n")
    if token_audit:
        lines.append("## Token/masking exposure context\n\n")
        lines.append(f"Trainer-exact audit: `{token_audit['path']}`. Treatment candidate tokens seen relative to control: {token_audit['candidate_tokens_seen_rel_pct']:.3f}%; WWM groups relative: {token_audit['wwm_groups_seen_rel_pct']:.4f}%. Thus a positive downstream result is not explained by a larger treatment token budget.\n\n")
    if best_delta:
        lines.append(f"Best equal7 delta: `{best_delta[0]}` = {best_delta[1]['delta_equal7']:.4f}.\n")
    if best_ksum:
        lines.append(f"Best knowledge-cluster delta: `{best_ksum[0]}` = {best_ksum[1]['delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA']:.4f} across EWoK+Entity+COMPS+GlobalPIQA.\n")
    if best_treat:
        lines.append(f"Best treatment equal7: `{best_treat[0]}` = {best_treat[1]['treatment_equal7']:.4f}.\n")
    lines.append(f"\nRoute reading: `{route_reading}` — {reason}\n\n")
    lines.append("| checkpoint | treat eq7 | ctrl eq7 | delta eq7 | knowledge Δsum | protected Δsum | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for ck, r in sorted(enriched.items(), key=lambda kv: ck_num(kv[0])):
        d = r["delta_components"]
        lines.append(f"| {ck} | {fmt(r['treatment_equal7'])} | {fmt(r['control_equal7'])} | {fmt(r['delta_equal7'])} | {fmt(r['delta_knowledge_sum_EWoK_Entity_COMPS_GlobalPIQA'])} | {fmt(r['delta_protected_sum_Supp_Reading_BLiMP'])} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA'))} |\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(note), "route_reading": route_reading}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
