#!/usr/bin/env python3
"""Interpret selected FineWeb seqsafe96 full-eval result when it exists.

This research-facing route reader combines the repaired no-AoA trajectory,
selected SuperGLUE+AoA completion, public/inherited references, compact-view
evidence, and research/020 live-source measurements.  It is intentionally strict
about the causal object: source-breadth support is always read from the selected
FineWeb treatment endpoint against its same-endpoint control, never from whichever
selected target happens to have the larger Overall.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

FULL_SUMMARY = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval/fineweb_selected_full_eval_summary.json")
FULL_PLAN = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_full_eval/fineweb_full_eval_endpoint_plan.json")
NOAOA_INTERP = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_seqsafe96_repairseq_interpretation/fineweb_seqsafe96_general_route_interpretation.json")
NOAOA_SUMMARY = pathlib.Path("experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval_repairseq/fineweb_seqsafe96_delta_summary.json")
PUBLIC = pathlib.Path("experiments/archive/representation_and_objectives/data/public_component_tradeoffs/public_component_tradeoffs.json")
A02_VIEW = pathlib.Path("experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json")
V3 = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_source_selector_v3/live_fineweb_source_selector_v3_summary.json")
V5 = pathlib.Path("experiments/archive/representation_and_objectives/data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_summary.json")
BLUEPRINT = pathlib.Path("experiments/archive/representation_and_objectives/data/stratified_live_fineweb_blueprint/stratified_live_fineweb_blueprint.json")
MOTIF = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_source_benchmark_overlap_refined/fineweb_source_benchmark_motif_overlap_refined.json")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/fineweb_full_route_interpretation")
NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_full_result_route_interpretation.md")

TREAT_SOURCE = "fineweb_seqsafe96_treatment_repairseq"
CONTROL_SOURCE = "fineweb_seqsafe96_control_repairseq"
KNOWLEDGE = ["EWoK", "Entity", "COMPS", "GlobalPIQA"]
PROTECT = ["BLiMP", "Supplement", "Reading"]
FULL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall"]


def load_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def subtract_rows(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        av, bv = f(a.get(k)), f(b.get(k))
        out[k] = None if av is None or bv is None else round(av - bv, 6)
    return out


def sum_known(d: dict[str, Any], keys: list[str]) -> float | None:
    vals = [f(d.get(k)) for k in keys]
    if any(v is None for v in vals):
        return None
    return round(sum(v for v in vals if v is not None), 6)


def target_name(source: str, endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return f"{source}__{endpoint}"


def refs() -> dict[str, Any]:
    out: dict[str, Any] = {
        "leader": {"Overall": 41.8, "SuperGLUE": 69.79, "AoA": 0.0},
        "inherited_cleanqwen": {"Overall": 41.34429066479573, "SuperGLUE": 70.30861598316157, "AoA": 0.0},
    }
    p = load_json(PUBLIC)
    if not isinstance(p, dict):
        return out
    for r in p.get("top15", []):
        if r.get("rank") == 1:
            out["leader"].update({
                "model": r.get("model"), "repo": r.get("repo"), "Overall": f(r.get("overall")) or out["leader"]["Overall"],
                "BLiMP": f(r.get("BLiMP")), "Supplement": f(r.get("Supplement")), "EWoK": f(r.get("EWoK")),
                "Entity": f(r.get("Entity")), "COMPS": f(r.get("COMPS")), "GlobalPIQA": f(r.get("GlobalPIQA")),
                "SuperGLUE": f(r.get("SuperGLUE")), "Reading": f(r.get("Reading")), "AoA": f(r.get("AoA")),
            })
            break
    ours = p.get("ours") or {}
    out["inherited_cleanqwen"].update({
        "Overall": f(ours.get("Overall Average")) or out["inherited_cleanqwen"]["Overall"],
        "BLiMP": f(ours.get("BLiMP")), "Supplement": f(ours.get("BLiMP Supplement")), "EWoK": f(ours.get("EWoK")),
        "Entity": f(ours.get("Entity Tracking")), "COMPS": f(ours.get("COMPS")), "GlobalPIQA": f(ours.get("GlobalPIQA")),
        "SuperGLUE": f(ours.get("(Super)GLUE")), "Reading": f(ours.get("Reading")), "AoA": f(ours.get("AoA")),
    })
    return out


def noaoa_for_endpoint(endpoint: str | None) -> dict[str, Any] | None:
    if endpoint is None:
        return None
    s = load_json(NOAOA_SUMMARY)
    if not isinstance(s, dict):
        return None
    return (s.get("matched_rows") or {}).get(endpoint)


def source_yield_summary() -> dict[str, Any]:
    v3 = load_json(V3) or {}
    v5 = load_json(V5) or {}
    bp = load_json(BLUEPRINT) or {}
    selected = bp.get("selected_seed_summary") or {}
    projection = bp.get("projection") or {}
    return {
        "v3_relation_rich_context": {
            "stable_words": v3.get("stable_words"),
            "doccap8_words": v3.get("stable_doccap8_words"),
            "type_counts": v3.get("type_counts_final"),
            "projected_scanned_doc_words_needed": (v3.get("projected_scanned_doc_words_needed") or {}).get("1750000"),
            "path": str(V3),
        },
        "v5_self_contained_facts": {
            "kept_words": v5.get("kept_words"),
            "doccap8_words": v5.get("doccap8_words"),
            "type_counts": v5.get("type_counts"),
            "projected_scanned_doc_words_needed": (v5.get("projected_scanned_doc_words_needed") or {}).get("1750000"),
            "path": str(V5),
        },
        "current_seed": {
            "rows": selected.get("rows"),
            "words": selected.get("words"),
            "docs": selected.get("docs"),
            "words_by_class": selected.get("words_by_class"),
            "words_by_focus": selected.get("words_by_focus"),
            "tokenizer_probe": bp.get("tokenizer_probe_selected_seed"),
            "projection": projection,
            "path": str(BLUEPRINT),
        },
    }


def motif_context() -> dict[str, Any]:
    m = load_json(MOTIF) or {}
    return {
        "path": str(MOTIF),
        "task_term_density_ratio_cached_fineweb_over_control": ((m.get("comparisons") or {}).get("cached_fineweb_vs_official_control") or {}).get("task_term_density_ratios"),
        "relation_density_ratio_cached_fineweb_over_control": ((m.get("comparisons") or {}).get("cached_fineweb_vs_official_control") or {}).get("relation_density_ratios"),
        "exact_field_aware_7gram_overlap": m.get("exact_field_aware_7gram_overlap"),
        "interpretation": "Cached FineWeb is not lexically enriched for all deficit columns; downstream gains would not be reducible to simple task-term density.",
    }


def a02_summary() -> dict[str, Any]:
    a = load_json(A02_VIEW) or {}
    c = ((a.get("contrasts") or {}).get("near_view_minus_near_repeat") or {}) if isinstance(a, dict) else {}
    return {
        "path": str(A02_VIEW),
        "near_view_minus_near_repeat": {k: c.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity"]},
        "interpretation": "A02 shows faithful FineWeb views strongly improve Entity and Reading over repetition at compact scale, while EWoK/COMPS/GlobalPIQA remain weak; source+view is therefore an Entity/state mechanism, not by itself the whole knowledge-cluster answer.",
    }


def selected_endpoint_from_plan(plan: dict[str, Any] | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    ep = plan.get("primary_checkpoint")
    if isinstance(ep, str) and ep:
        return ep
    selected = plan.get("selected_checkpoints") or []
    if selected:
        return selected[0]
    return None


def extract_selected_pair(full: dict[str, Any] | None, plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(full, dict):
        return {"full_exists": False}
    rows = full.get("targets") or {}
    endpoint = selected_endpoint_from_plan(plan)
    if endpoint is None:
        for name, row in rows.items():
            if name.startswith(TREAT_SOURCE + "__"):
                endpoint = row.get("endpoint") or name.rsplit("__", 1)[-1]
                break
    t_name = target_name(TREAT_SOURCE, endpoint)
    c_name = target_name(CONTROL_SOURCE, endpoint)
    t_row = rows.get(t_name or "")
    c_row = rows.get(c_name or "")
    deltas = full.get("same_endpoint_treatment_minus_control") or {}
    delta_key = f"{t_name}_minus_{c_name}" if t_name and c_name else None
    delta_row = deltas.get(delta_key or "")
    return {
        "full_exists": True,
        "selected_endpoint": endpoint,
        "treatment_target": t_name,
        "control_target": c_name,
        "treatment_row": t_row,
        "control_row": c_row,
        "delta_key": delta_key,
        "delta_row": delta_row,
        "best_by_overall": full.get("best_by_overall"),
    }


def route_from_noaoa_only(plan: dict[str, Any] | None, noaoa_interp: dict[str, Any] | None) -> tuple[str, str, list[str]]:
    if not isinstance(plan, dict):
        return "awaiting_noaoa_or_endpoint_plan", "No endpoint plan exists yet; the repaired no-AoA trajectory has not been converted into a route result.", ["Collect the managed result when delivered; do not start a new expensive route from absence of data."]
    if plan.get("status") == "AWAITING_NOAOA_SUMMARY":
        return "awaiting_noaoa_summary", "The endpoint plan itself records that the repaired no-AoA summary is absent.", ["Wait for the managed training/no-AoA task to deliver before choosing any next training route."]
    if not plan.get("should_run_full_eval"):
        nr = (noaoa_interp or {}).get("route_reading") if isinstance(noaoa_interp, dict) else None
        rs = (noaoa_interp or {}).get("reason") if isinstance(noaoa_interp, dict) else None
        return (
            "full_eval_skipped_by_noaoa_pattern",
            f"The endpoint plan did not spend SuperGLUE+AoA because the no-AoA component pattern was not strong enough. no-AoA route={nr}; reason={rs}",
            [
                "Use the no-AoA component table, trainer-token exposure measurement, and source-quality evidence to choose a cheaper repair or a different mechanism rather than evaluating more endpoints.",
                "Do not scale cached FineWeb source repetition alone unless a new result changes the component pattern.",
            ],
        )
    return "awaiting_selected_full_eval", "The endpoint plan requested selected SuperGLUE+AoA completion, but the selected full summary is not present yet.", ["Collect the selected full-eval managed result when delivered before committing a next H100 route."]


def route_from_pattern(endpoint: str | None, treatment: dict[str, Any] | None, control: dict[str, Any] | None, delta: dict[str, Any] | None, noaoa_pair: dict[str, Any] | None, ref: dict[str, Any]) -> tuple[str, str, list[str]]:
    if treatment is None or control is None:
        return "selected_full_pair_incomplete", f"The full summary exists but lacks the same-endpoint treatment/control pair for endpoint {endpoint}.", ["Inspect the per-target JSON files and rerun only the missing target if the no-AoA plan still warrants it."]
    leader = ref["leader"]
    inherited = ref["inherited_cleanqwen"]
    overall = f(treatment.get("Overall"))
    d_overall = f((delta or {}).get("Overall"))
    d_know = sum_known(delta or {}, KNOWLEDGE)
    d_protect = sum_known(delta or {}, PROTECT)
    treat_vs_leader = subtract_rows(treatment, leader, [k for k in FULL_KEYS if k != "Overall"])
    treat_vs_inherited = subtract_rows(treatment, inherited, [k for k in FULL_KEYS if k != "Overall"])
    source_tasks: list[str] = []

    if overall is not None and overall > float(leader.get("Overall", 41.8)):
        return (
            "potential_sota_full_result_needs_reproduction",
            f"The selected FineWeb treatment endpoint {endpoint} has full Overall {overall:.4f}, above the public 41.8 reference; verify arithmetic, AoA, per-target provenance, and reproducibility before final expression.",
            ["Inspect the treatment and control per-target JSON, SuperGLUE logs, AoA helper output, and no-AoA prefill provenance.", "Run an independent scoring reproduction and confirm official submission requirements."],
        )

    if d_know is not None and d_know >= 2.0 and (d_protect is None or d_protect >= -1.5):
        source_tasks.append("Build a stratified live FineWeb four-arm family with both v3 relation-rich context and v5 self-contained facts; do not use v5 alone.")
        source_tasks.append("Carry A02's view mechanism into a C_view arm, especially if Entity remains the column with the clearest view gain.")
        source_tasks.append("Keep a clean-Qwen natural arm to protect Supplement/Reading/BLiMP and measure replacement cost.")
        return "source_breadth_supported_make_stratified_live_family", f"Selected treatment-control full delta={d_overall}, knowledge movement={d_know}; pattern vs leader: {treat_vs_leader}.", source_tasks

    if d_know is not None and d_know > 0.0:
        source_tasks.append("Treat cached source breadth as directionally useful but either under-scaled or quality-limited; prepare a smaller live stratified source+view test before another large replacement.")
        source_tasks.append("Use v3 for relation-rich context and v5 for stable facts in an explicit mixture; inspect benchmark overlap and token exposure before training.")
        return "source_breadth_directional_but_needs_stratified_live_repair", f"Knowledge-cluster delta is positive but not large enough to close the leader gap; selected treatment vs inherited components: {treat_vs_inherited}.", source_tasks

    if d_overall is not None and d_overall > 0.15:
        source_tasks.append("Inspect which non-target columns carry the gain; do not scale cached source repetition unless EWoK/Entity/COMPS/GlobalPIQA also move.")
        source_tasks.append("If Entity is the only reliable movement, combine with A02-style faithful views and relation-rich source contexts rather than stricter single-sentence facts.")
        return "overall_positive_but_not_targeted", "Full Overall moved without enough intended knowledge-cluster support; use the component table to choose a more specific mechanism.", source_tasks

    source_tasks.append("Stop cached FineWeb source repetition as a standalone route.")
    source_tasks.append("Return to source+faithful-view, tokenizer/masking interaction, or architecture/optimizer mechanisms; use A02 Entity-positive view evidence as a possible sub-mechanism only.")
    return "cached_source_repetition_not_supported", "The selected treatment/control result does not show enough useful source-breadth movement to justify extending cached source repetition.", source_tasks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = refs()
    full = load_json(FULL_SUMMARY)
    plan = load_json(FULL_PLAN)
    noaoa_interp = load_json(NOAOA_INTERP)

    pair = extract_selected_pair(full, plan)
    endpoint = pair.get("selected_endpoint")
    treatment_row = pair.get("treatment_row")
    control_row = pair.get("control_row")
    delta_row = pair.get("delta_row")
    noaoa_pair = noaoa_for_endpoint(endpoint)

    if pair.get("full_exists"):
        route, reason, next_actions = route_from_pattern(endpoint, treatment_row, control_row, delta_row, noaoa_pair, ref)
    else:
        route, reason, next_actions = route_from_noaoa_only(plan, noaoa_interp)

    payload = {
        "status": "FINEWEB_FULL_ROUTE_INTERPRETATION",
        "full_summary": str(FULL_SUMMARY),
        "full_summary_exists": FULL_SUMMARY.exists(),
        "full_plan": str(FULL_PLAN),
        "full_plan_exists": FULL_PLAN.exists(),
        "full_plan_should_run": plan.get("should_run_full_eval") if isinstance(plan, dict) else None,
        "selected_endpoint": endpoint,
        "selected_treatment_target": pair.get("treatment_target"),
        "selected_control_target": pair.get("control_target"),
        "selected_treatment_row": treatment_row,
        "selected_control_row": control_row,
        "best_by_overall_in_selected_full_summary": pair.get("best_by_overall"),
        "same_endpoint_full_delta": delta_row,
        "same_endpoint_noaoa_pair": noaoa_pair,
        "noaoa_route_reading": (noaoa_interp or {}).get("route_reading") if isinstance(noaoa_interp, dict) else None,
        "noaoa_reason": (noaoa_interp or {}).get("reason") if isinstance(noaoa_interp, dict) else None,
        "references": ref,
        "a02_view_evidence": a02_summary(),
        "source_yield_summary": source_yield_summary(),
        "motif_context": motif_context(),
        "route_reading": route,
        "reason": reason,
        "next_actions": next_actions,
        "strategist_alignment": "No further selector churn. If FineWeb continues, use the result pattern to choose a stratified natural mixture of self-contained facts and relation-rich context; v5 is useful but not the sole substrate.",
    }
    out_json = OUT_DIR / "fineweb_full_route_interpretation.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# FineWeb full-result route interpretation\n\n"]
    lines.append(f"Full summary exists: `{FULL_SUMMARY.exists()}` at `{FULL_SUMMARY}`\n\n")
    if treatment_row:
        lines.append(f"Selected FineWeb treatment target: `{pair.get('treatment_target')}` Overall {f(treatment_row.get('Overall')):.4f}, endpoint `{treatment_row.get('endpoint')}`.\n\n")
        lines.append(f"Matched control target: `{pair.get('control_target')}` Overall {f(control_row.get('Overall')):.4f}.\n\n" if control_row else "Matched control target is missing.\n\n")
        lines.append("Same-endpoint treatment-control full delta:\n\n")
        lines.append("```json\n" + json.dumps(delta_row, indent=2, ensure_ascii=False) + "\n```\n\n")
        if pair.get("best_by_overall"):
            lines.append("Best selected target by Overall, recorded separately from the causal treatment reading:\n\n")
            lines.append("```json\n" + json.dumps(pair.get("best_by_overall"), indent=2, ensure_ascii=False) + "\n```\n\n")
    elif isinstance(plan, dict) and not plan.get("should_run_full_eval"):
        lines.append("The endpoint plan did not request selected SuperGLUE+AoA completion; this interpretation is therefore based on the no-AoA route result and source evidence.\n\n")
    else:
        lines.append("The selected full-eval summary is not available yet; the already-running conversion should be collected before choosing the next expensive route.\n\n")
    lines.append(f"Route reading: `{route}` — {reason}\n\n")
    lines.append("## Source substrate implication\n\n")
    lines.append("The next FineWeb mixture, if justified by the result, should be stratified rather than increasingly neat: v3-like relation-rich context preserves causal, temporal, discourse and attribution-bearing material; v5-like rows provide safer self-contained facts. A02's compact result shows faithful views help Entity/Reading but did not solve EWoK/COMPS/GlobalPIQA alone.\n\n")
    lines.append("Next actions:\n\n")
    for a in next_actions:
        lines.append(f"- {a}\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "route_reading": route, "selected_endpoint": endpoint, "out_json": str(out_json), "note": str(NOTE)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
