#!/usr/bin/env python3
"""research: no-AoA semantic-view decision table.

Reads the completed semantic-view packet-local no-AoA trajectory and compares
candidate endpoints with the public strict-small leader and the inherited COMPACT_EXPERIENCE
clean-Qwen coordinate. This is used to decide the next official-style full
evaluation and the next data-route allocation.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

SUMMARY = Path("experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/semantic_view_noaoa_decision")
OUT_JSON = OUT_DIR / "semantic_view_noaoa_decision_table.json"
OUT_NOTE = Path("research/notes/representation_and_objectives/semantic_view_noaoa_decision.md")

LEADER = {
    "name": "go76dof/wwm_curriculum_simplification_40k",
    "Overall": 41.80,
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.0,
}
COMPACT_EXPERIENCE_CLEAN_QWEN = {
    "name": "COMPACT_EXPERIENCE qwen_clean_aligned chck_100M",
    "Overall": 41.34429066479573,
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "SuperGLUE": 70.30861598316157,
    "Reading": 7.76,
    "AoA": 0.0,
}
ZERO_READING_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
TARGET_CLUSTER = ["EWoK", "Entity", "COMPS", "GlobalPIQA"]


def ck_mwords(ck: str) -> int:
    return int(ck.split("_")[1].rstrip("M"))


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for ck, rr in data["matched_rows"].items():
        t = rr["treatment"]
        c = rr["control"]
        d = rr["delta_treatment_minus_control"]
        sum7 = sum(float(t[k]) for k in ZERO_READING_COLS)
        need_sg_aoa_for_418 = LEADER["Overall"] * 9 - sum7
        if need_sg_aoa_for_418 is not None:
            # If AoA is 0, this is the needed SuperGLUE alone. If SuperGLUE is
            # inherited-like, remaining is required AoA leaderboard units.
            need_aoa_if_sg_qwen = need_sg_aoa_for_418 - COMPACT_EXPERIENCE_CLEAN_QWEN["SuperGLUE"]
            need_aoa_if_sg_leader = need_sg_aoa_for_418 - LEADER["SuperGLUE"]
        target_delta_sum = sum(float(d[k]) for k in TARGET_CLUSTER)
        target_vs_leader = {k: round(float(t[k]) - LEADER[k], 4) for k in ZERO_READING_COLS}
        target_vs_compact_experience = {k: round(float(t[k]) - COMPACT_EXPERIENCE_CLEAN_QWEN[k], 4) for k in ZERO_READING_COLS}
        rows.append({
            "checkpoint": ck,
            "mwords": ck_mwords(ck),
            "treatment_equal7": t["equal7_full_eval"],
            "control_equal7": c["equal7_full_eval"],
            "delta_equal7": d["equal7_full_eval"],
            "delta_target_cluster_sum": round(target_delta_sum, 6),
            "treatment_sum7": round(sum7, 6),
            "need_superglue_plus_aoa_for_41p8": round(need_sg_aoa_for_418, 6),
            "need_aoa_if_sg_matches_compact_experience_clean_qwen": round(need_aoa_if_sg_qwen, 6),
            "need_aoa_if_sg_matches_public_leader": round(need_aoa_if_sg_leader, 6),
            "treatment_minus_leader_zero_reading": target_vs_leader,
            "treatment_minus_compact_experience_clean_qwen_zero_reading": target_vs_compact_experience,
            "treatment": t,
            "control": c,
            "delta_treatment_minus_control": d,
        })
    rows.sort(key=lambda x: x["mwords"])
    best_treatment = max(rows, key=lambda x: x["treatment_equal7"])
    best_delta = max(rows, key=lambda x: x["delta_equal7"])
    late = [r for r in rows if r["mwords"] >= 70]
    payload = {
        "status": "SEMANTIC_VIEW_NOAOA_DECISION_TABLE",
        "source_summary": str(SUMMARY),
        "public_strict_small_leader": LEADER,
        "inherited_compact_experience_reference": COMPACT_EXPERIENCE_CLEAN_QWEN,
        "rows": rows,
        "best_treatment_equal7": best_treatment,
        "best_delta_equal7": best_delta,
        "late_mean_70_to_100M": {
            "treatment_equal7": mean([r["treatment_equal7"] for r in late]),
            "control_equal7": mean([r["control_equal7"] for r in late]),
            "delta_equal7": mean([r["delta_equal7"] for r in late]),
            "delta_target_cluster_sum": mean([r["delta_target_cluster_sum"] for r in late]),
        },
        "selected_full_eval_endpoint": "chck_80M",
        "selected_full_eval_reason": "highest treatment equal7 (41.8429) and same-endpoint causal comparator available; requires only SuperGLUE+AoA because zero-shot/Reading were prefilled.",
        "route_implication": {
            "same_source_semantic_view": "positive causal no-AoA signal, especially Entity/COMPS and transient GlobalPIQA; not sufficient as full SOTA evidence until SuperGLUE+AoA are measured.",
            "remaining_gap": "The treatment remains far below the public leader on EWoK and especially Entity, so source breadth remains necessary even if full Overall is competitive.",
            "next_data_route_after_full_eval": "Use full-eval result to decide whether semantic views become a protected component in a FineWeb three-arm source-breadth/rewrite design; do not allocate new H100 training before this selected full evaluation returns unless it fails irrecoverably.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research semantic-view no-AoA decision\n\n")
    lines.append("The completed paired no-AoA trajectory resolves the blocked semantic-view question enough to choose the next evaluation action. The treatment is not a final result, but it has a real downstream signal under the packet-local control.\n\n")
    lines.append(f"Source JSON: `{SUMMARY}`\n\n")
    lines.append("## Endpoint comparison\n\n")
    lines.append("| ckpt | treat equal7 | ctrl equal7 | Δ equal7 | Δ EWoK+Entity+COMPS+GPIQA | SG+AoA needed for 41.8 | AoA needed if SG=COMPACT_EXPERIENCE |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append("| {checkpoint} | {treatment_equal7:.4f} | {control_equal7:.4f} | {delta_equal7:.4f} | {delta_target_cluster_sum:.3f} | {need_superglue_plus_aoa_for_41p8:.3f} | {need_aoa_if_sg_matches_compact_experience_clean_qwen:.3f} |\n".format(**r))
    lines.append("\n")
    lines.append(f"Best treatment equal7: `{best_treatment['checkpoint']}` = {best_treatment['treatment_equal7']:.4f}. This is the selected endpoint for official-style full completion.\n\n")
    lines.append(f"Best causal Δ equal7: `{best_delta['checkpoint']}` = {best_delta['delta_equal7']:.4f}; it has a large transient GlobalPIQA contribution and is not the best absolute endpoint.\n\n")
    lines.append("## Scientific reading\n\n")
    lines.append("Same-source generated views are not weak: at 80M the treatment beats packet-local by +0.2629 equal7 with +1.27 EWoK, +2.10 Entity, +0.60 COMPS, and -0.945 GlobalPIQA. At 40M the causal signal is much larger (+1.72 equal7) but partly transient and below the best absolute treatment endpoint. Late checkpoints show persistent Entity/EWoK improvement but Supplement and GlobalPIQA tradeoffs.\n\n")
    lines.append("The 80M treatment still trails the public strict-small leader by about -6.56 EWoK and -10.20 Entity despite better Supplement and Reading. Therefore, even if the selected full evaluation is competitive through SuperGLUE/AoA, the mechanism does not remove the source-breadth problem; it shows that faithful second views can be useful enough to test as a factor in a broader FineWeb design.\n\n")
    lines.append("Next action now running: selected full official-style completion for treatment/control at `chck_80M` using `launch_semantic_view_full_eval_selected.sh`, managed task `s14_t14_tool1`.\n\n")
    lines.append(f"JSON: `{OUT_JSON}`\n")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_note": str(OUT_NOTE), "selected": payload["selected_full_eval_endpoint"]}, indent=2))


if __name__ == "__main__":
    main()
