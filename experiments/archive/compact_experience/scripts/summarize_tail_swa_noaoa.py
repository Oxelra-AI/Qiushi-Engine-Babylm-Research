#!/usr/bin/env python3
"""Summarize research tail-SWA no-AoA evaluation against clean-Qwen and target.

Reads only no-AoA trajectory output for averaged single model artifacts. It does not
combine scores across models and does not inspect SuperGLUE or AoA.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
SUMMARY = _public_path('experiments/archive/compact_experience/data/tail_swa_trajectory/tail_swa_clean_seed43022_trajectory_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/tail_swa_trajectory/tail_swa_noaoa_review.json')
NOTE = _public_path('research/notes/compact_experience/tail_swa_noaoa_review.md')

CLEAN = {
    "label": "clean_qwen_seed43022_chck_100M",
    "equal7": 43.112857142857145,
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "Reading": 7.76,
    "complete_overall9": 41.34429066479573,
}
VISIBLE_LEADER = 41.8
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"missing trajectory summary: {SUMMARY}")
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    table = payload.get("table", {})
    rows = []
    for ckpt, row in table.items():
        if not all(row.get(k) is not None for k in KEYS):
            rows.append({"checkpoint": ckpt, "complete": False, "row": row})
            continue
        equal7 = float(row["equal7_full_eval"])
        deltas = {k: float(row[k]) - float(CLEAN[k]) for k in KEYS}
        required_superglue_plus_aoa_to_reach_41p8 = 9 * VISIBLE_LEADER - 7 * equal7
        rows.append({
            "checkpoint": ckpt,
            "complete": True,
            "equal7": equal7,
            "delta_equal7_vs_clean": equal7 - CLEAN["equal7"],
            "scores": {k: float(row[k]) for k in KEYS},
            "deltas_vs_clean": deltas,
            "recovery_region_delta_mean_EWoK_COMPS_GlobalPIQA": (deltas["EWoK"] + deltas["COMPS"] + deltas["GlobalPIQA"]) / 3.0,
            "retention_region_delta_mean_Supplement_Entity_Reading": (deltas["Supplement"] + deltas["Entity"] + deltas["Reading"]) / 3.0,
            "required_superglue_plus_aoa_to_reach_41p8": required_superglue_plus_aoa_to_reach_41p8,
            "required_superglue_plus_aoa_to_match_clean_overall9": 9 * CLEAN["complete_overall9"] - 7 * equal7,
        })
    ranked = sorted([r for r in rows if r.get("complete")], key=lambda r: r["equal7"], reverse=True)
    best = ranked[0] if ranked else None
    if best:
        if best["delta_equal7_vs_clean"] >= 0.25 and best["retention_region_delta_mean_Supplement_Entity_Reading"] > -0.25:
            action = "consider_full_eval_frozen_single_swa_model_only"
        else:
            action = "close_tail_swa_as_frontier_route_without_full_eval"
    else:
        action = "no_complete_swa_result"
    review = {
        "status": "TAIL_SWA_NOAOA_REVIEW",
        "scope": "BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading only; no SuperGLUE or AoA read or used.",
        "source_summary": str(SUMMARY),
        "clean_reference": CLEAN,
        "visible_leader_overall_reference": VISIBLE_LEADER,
        "rows": rows,
        "best": best,
        "preceptor_action": action,
    }
    _public_path('experiments/archive/compact_experience/data/tail_swa_trajectory').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research tail-SWA no-AoA review", "", review["scope"], "", f"Scientific decision: `{action}`", ""]
    if best:
        lines += [
            f"Best averaged model: `{best['checkpoint']}` equal7={best['equal7']:.6f} (delta vs clean {best['delta_equal7_vs_clean']:+.6f}).",
            f"Recovery-region delta mean (EWoK/COMPS/GlobalPIQA): {best['recovery_region_delta_mean_EWoK_COMPS_GlobalPIQA']:+.6f}.",
            f"Retention-region delta mean (Supplement/Entity/Reading): {best['retention_region_delta_mean_Supplement_Entity_Reading']:+.6f}.",
            f"Required SuperGLUE+AoA to reach Overall 41.8: {best['required_superglue_plus_aoa_to_reach_41p8']:.6f}.",
            "",
            "Column deltas vs clean:",
        ]
        for k, v in best["deltas_vs_clean"].items():
            lines.append(f"- {k}: {v:+.4f}")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "action": action, "best": best}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
