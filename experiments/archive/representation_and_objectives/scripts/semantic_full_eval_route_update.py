#!/usr/bin/env python3
"""research: synthesize selected semantic-view full eval and next route.

This script reads the completed selected full evaluation, compares it with the
public strict-small leader and the inherited COMPACT_EXPERIENCE clean-Qwen coordinate, and
writes a scientific interpretation note.
"""
from __future__ import annotations

import json
from pathlib import Path

FULL = Path("experiments/archive/representation_and_objectives/data/semantic_view_full_eval/semantic_view_selected_full_eval_summary.json")
NOAOA = Path("experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json")
DECISION = Path("experiments/archive/representation_and_objectives/data/semantic_view_noaoa_decision/semantic_view_noaoa_decision_table.json")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/semantic_full_eval_route_update")
OUT_JSON = OUT_DIR / "semantic_full_eval_route_update.json"
OUT_NOTE = Path("research/notes/representation_and_objectives/semantic_full_eval_route_update.md")

KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall"]
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
COMPACT_EXPERIENCE = {
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


def diff(row: dict, ref: dict) -> dict:
    out = {}
    for k in KEYS:
        if k in row and k in ref and row[k] is not None and ref[k] is not None:
            out[k] = round(float(row[k]) - float(ref[k]), 6)
    return out


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    noaoa = json.loads(NOAOA.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8")) if DECISION.exists() else None
    rows = full["targets"]
    treat = rows["semantic_view_treatment__chck_80M"]
    ctrl = rows["original_packet_local__chck_80M"]
    causal = full["same_endpoint_treatment_minus_packet_local"]["semantic_view_treatment__chck_80M_minus_original_packet_local__chck_80M"]
    noaoa_best_delta = noaoa["best_by_delta_equal7"]
    noaoa_best_treat = noaoa["best_treatment_equal7"]

    payload = {
        "status": "SEMANTIC_FULL_EVAL_ROUTE_UPDATE",
        "full_eval_summary": str(FULL),
        "noaoa_summary": str(NOAOA),
        "decision_table": str(DECISION) if DECISION.exists() else None,
        "selected_endpoint": "chck_80M",
        "semantic_view_treatment": treat,
        "packet_local_control": ctrl,
        "same_endpoint_treatment_minus_packet_local": causal,
        "treatment_minus_public_leader": diff(treat, LEADER),
        "treatment_minus_compact_experience_clean_qwen": diff(treat, COMPACT_EXPERIENCE),
        "control_minus_compact_experience_clean_qwen": diff(ctrl, COMPACT_EXPERIENCE),
        "noaoa_context": {
            "best_treatment_equal7": noaoa_best_treat,
            "best_causal_delta_equal7": noaoa_best_delta,
            "decision_selected_endpoint": decision.get("selected_full_eval_endpoint") if isinstance(decision, dict) else None,
        },
        "scientific_conclusion": {
            "semantic_view_factor": "Same-source simplification/paraphrase views have real causal downstream value over packet-local source repetition: at chck_80M full Overall +1.586 mainly from avoiding the control's negative AoA and with +2.10 Entity, +1.27 EWoK, +0.60 COMPS, +0.44 Reading, but -1.91 Supplement and -0.945 GlobalPIQA.",
            "not_sota_endpoint": "The treatment's submit-ready official-style Overall is 40.1399, far below public strict-small 41.8 and below inherited COMPACT_EXPERIENCE clean-Qwen 41.3443.",
            "remaining_deficit": "At the selected treatment endpoint, deficits vs public leader are -6.56 EWoK, -10.20 Entity, -1.37 COMPS, -2.535 GlobalPIQA, and -1.43 SuperGLUE, partly offset by +5.73 Supplement and +2.385 Reading. The route still lacks broad factual/entity coverage.",
            "next_route": "Proceed to the already-audited FineWeb seqsafe96 source-breadth contrast before any further same-source semantic-view hybrid; then use its component trajectory to decide whether to add relation-preserving rewrites as a third arm.",
        },
        "launched_next_task": {
            "task_ref": "s14_t33_tool1",
            "purpose": "wait for usable GPUs, train cached FineWeb seqsafe96 source-breadth treatment/control, then run no-AoA trajectory evaluation",
            "controller": "experiments/archive/representation_and_objectives/training/scripts/wait_train_eval_fineweb_seqsafe96.sh",
            "expected_summary": "experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research semantic-view full evaluation and source-breadth route update\n\n")
    lines.append("## Selected full evaluation result\n\n")
    lines.append("The `chck_80M` semantic-view endpoint selected by the completed no-AoA trajectory was completed with official-style SuperGLUE and AoA. The evaluation is submit-ready in the local sense: complete zero-shot/Reading, SuperGLUE, and the 19-step strict-small AoA ladder.\n\n")
    lines.append("| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name,row in [("semantic_view_treatment__chck_80M", treat), ("original_packet_local__chck_80M", ctrl)]:
        lines.append(f"| {name} | {row['Overall']:.4f} | {row['BLiMP']:.2f} | {row['Supplement']:.2f} | {row['EWoK']:.2f} | {row['Entity']:.2f} | {row['COMPS']:.2f} | {row['GlobalPIQA']:.3f} | {row['SuperGLUE']:.4f} | {row['Reading']:.3f} | {row['AoA']:.4f} |\n")
    lines.append("\nSame-endpoint treatment minus packet-local: Overall +{Overall:.4f}, BLiMP +{BLiMP:.2f}, Supplement {Supplement:.2f}, EWoK +{EWoK:.2f}, Entity +{Entity:.2f}, COMPS +{COMPS:.2f}, GlobalPIQA {GlobalPIQA:.3f}, SuperGLUE +{SuperGLUE:.4f}, Reading +{Reading:.3f}, AoA +{AoA:.4f}.\n\n".format(**causal))
    lines.append("## Interpretation\n\n")
    lines.append("Same-source generated views are mechanistically real, not noise: they improve Entity/EWoK/COMPS and avoid the packet-local control's negative AoA at the selected endpoint. But they are not a SOTA endpoint. The treatment Overall is 40.1399, below the public strict-small leader 41.8 and below the inherited COMPACT_EXPERIENCE clean-Qwen reference 41.3443.\n\n")
    dl = payload["treatment_minus_public_leader"]
    lines.append(f"Against the public leader, the selected treatment remains far behind on EWoK ({dl['EWoK']:+.2f}), Entity ({dl['Entity']:+.2f}), COMPS ({dl['COMPS']:+.2f}), GlobalPIQA ({dl['GlobalPIQA']:+.3f}), and SuperGLUE ({dl['SuperGLUE']:+.2f}), while better on Supplement ({dl['Supplement']:+.2f}) and Reading ({dl['Reading']:+.3f}). This points back to missing source breadth/entity coverage rather than more same-source rewriting alone.\n\n")
    lines.append("## Next active experiment\n\n")
    lines.append("Launched managed task `s14_t33_tool1` using `experiments/archive/representation_and_objectives/training/scripts/wait_train_eval_fineweb_seqsafe96.sh`. It waits for genuinely usable GPUs, trains the audited cached FineWeb seqsafe96 source-breadth treatment/control pair, and then runs the paired no-AoA trajectory. Expected delta summary: `experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json`.\n\n")
    lines.append("The FineWeb contrast should decide whether broad factual source replacement moves the remaining EWoK/Entity/COMPS/GlobalPIQA deficit. Only after that should relation-preserving rewrites be added as a third arm.\n\n")
    lines.append(f"JSON: `{OUT_JSON}`\n")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_note": str(OUT_NOTE), "treatment_overall": treat["Overall"], "control_overall": ctrl["Overall"], "next_task": "s14_t33_tool1"}, indent=2))


if __name__ == "__main__":
    main()
