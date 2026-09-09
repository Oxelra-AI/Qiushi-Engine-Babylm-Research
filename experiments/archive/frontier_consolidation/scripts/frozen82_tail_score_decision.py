#!/usr/bin/env python3
"""Merge frozen-82M tail cheap7, SuperGLUE, and source-free probe evidence.

Runs with partial inputs and records pending state.  Once SuperGLUE-only summaries are
available, it computes exact projected Overall assuming AoA=0, tie thresholds versus
the verified chck_82M reference, and separates the intended aligned correspondence
mechanism from a generic shuffled/private-tail endpoint effect.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_score_decision')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_score_decision/frozen82_tail_score_decision.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/frozen82_tail_score_decision/frozen82_tail_score_decision.md')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
PANEL = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_panel_analysis/frozen82_tail_panel_analysis.json')
ITEM = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_item_family_analysis/frozen82_tail_item_family_analysis.json')
PROBE = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/tail4M_adapter_on/tail4M_adapter_on.json')
SG_SUMMARIES = {
    "aligned": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_aligned_superglue_summary/frozen82_tail4M_aligned_superglue_superglue_summary.json'),
    "shuffled": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_superglue_summary/frozen82_tail4M_shuffled_superglue_superglue_summary.json'),
}
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def chck82_ref() -> dict[str, Any]:
    j = load_json(CHCK82_VERIFY)
    scores = {k: float(v) for k, v in j["score_arithmetic"]["scores"].items() if v is not None}
    return {
        "scores": scores,
        "cheap7": mean(scores[c] for c in CHEAP_COLS),
        "superglue": scores["SuperGLUE"],
        "aoa": scores["AoA"],
        "overall": float(j["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def required_superglue_for_overall(cheap_scores: dict[str, float], target_overall: float, aoa: float = 0.0) -> float:
    return 9.0 * target_overall - sum(float(cheap_scores[c]) for c in CHEAP_COLS) - float(aoa)


def projected_overall(cheap_scores: dict[str, float], superglue: float, aoa: float = 0.0) -> float:
    return mean([float(cheap_scores[c]) for c in CHEAP_COLS] + [float(superglue), float(aoa)])


def arm_record(mode: str, panel: dict[str, Any] | None, ref: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "mode": mode,
        "cheap_panel_record": None,
        "superglue_summary_path": rel(SG_SUMMARIES[mode]),
        "superglue_summary_exists": False,
        "superglue": None,
        "projected_overall_with_aoa0": None,
        "deltas_vs_chck82": None,
        "required_superglue_to_tie_chck82": None,
        "required_superglue_to_exceed_41p8": None,
        "score_read": "pending_superglue",
    }
    if panel is None:
        return rec
    p_rec = panel.get("records", {}).get(mode)
    rec["cheap_panel_record"] = p_rec
    if not p_rec or p_rec.get("scores") is None:
        return rec
    scores = {c: float(p_rec["scores"][c]) for c in CHEAP_COLS}
    rec["required_superglue_to_tie_chck82"] = required_superglue_for_overall(scores, ref["overall"], 0.0)
    rec["required_superglue_to_exceed_41p8"] = required_superglue_for_overall(scores, 41.8, 0.0)
    sg = load_json(SG_SUMMARIES[mode])
    if sg is not None:
        rec["superglue_summary_exists"] = True
        rec["superglue"] = float(sg["superglue"])
        rec["projected_overall_with_aoa0"] = float(sg["projected_overall_with_aoa0"])
        rec["deltas_vs_chck82"] = {
            "cheap7": float(p_rec["cheap7"] - ref["cheap7"]),
            "superglue": float(sg["superglue"] - ref["superglue"]),
            "projected_overall_with_aoa0": float(sg["projected_overall_with_aoa0"] - ref["overall"]),
            "margin_vs_41p8": float(sg["projected_overall_with_aoa0"] - 41.8),
        }
        if rec["projected_overall_with_aoa0"] > ref["overall"]:
            rec["score_read"] = "candidate_exceeds_chck82_before_aoa"
        elif rec["projected_overall_with_aoa0"] > 41.8:
            rec["score_read"] = "above_41p8_but_below_chck82_before_aoa"
        else:
            rec["score_read"] = "not_above_41p8_before_aoa"
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = chck82_ref()
    panel = load_json(PANEL)
    item = load_json(ITEM)
    probe = load_json(PROBE)
    arms = {m: arm_record(m, panel, ref) for m in ["aligned", "shuffled"]}
    pending = [m for m, r in arms.items() if not r["superglue_summary_exists"]]
    probe_keys = probe.get("key_deltas") if probe else None
    decisions: dict[str, Any] = {}
    if panel:
        decisions.update(panel.get("decisions", {}))
    if probe_keys:
        decisions["aligned_true_free_nll_delta_vs_shuffled"] = probe_keys.get("aligned_model_minus_shuffled_model_on_true_free_view")
        decisions["aligned_true_cond_nll_delta_vs_shuffled"] = probe_keys.get("aligned_model_minus_shuffled_model_on_true_conditioned_view")
        decisions["intended_correspondence_source_free_supported"] = (probe_keys.get("aligned_model_minus_shuffled_model_on_true_free_view") is not None and probe_keys.get("aligned_model_minus_shuffled_model_on_true_free_view") < 0 and decisions.get("true_correspondence_beats_shuffled_on_tail_cheap7", False))
    route_read = "pending_superglue"
    if not pending:
        a = arms["aligned"]
        s = arms["shuffled"]
        if decisions.get("intended_correspondence_source_free_supported"):
            route_read = "aligned_correspondence_tail_supported"
        elif s["projected_overall_with_aoa0"] is not None and s["projected_overall_with_aoa0"] > ref["overall"]:
            route_read = "generic_shuffled_private_tail_endpoint_candidate"
        elif a["projected_overall_with_aoa0"] is not None and a["projected_overall_with_aoa0"] > ref["overall"]:
            route_read = "aligned_score_candidate_without_correspondence_mechanism"
        else:
            route_read = "tails_do_not_improve_protected_endpoint"
    result = {
        "status": "PENDING" if pending else "COMPLETE",
        "created_utc": now(),
        "protected_chck82": ref,
        "arms": arms,
        "pending": pending,
        "source_free_probe_key_deltas": probe_keys,
        "panel_decisions_plus_probe": decisions,
        "item_analysis_path": rel(ITEM),
        "route_read": route_read,
        "scientific_reading": "Aligned correspondence is supported only if it beats shuffled both on source-free NLL and endpoint score. A shuffled endpoint candidate can improve the score but would be generic private-tail adaptation, not evidence for true correspondence transfer.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research frozen82 tail score decision", "", f"Status: **{result['status']}**", f"Route read: `{route_read}`", "", f"Protected chck82 Overall: `{ref['overall']}`; cheap7 `{ref['cheap7']}`; SuperGLUE `{ref['superglue']}`; AoA `{ref['aoa']}`", "", "## Arms", "", "| arm | cheap7 | required SG to tie chck82 | SuperGLUE | projected Overall AoA0 | delta vs chck82 | score read |", "|---|---:|---:|---:|---:|---:|---|"]
    for m, r in arms.items():
        pr = r.get("cheap_panel_record") or {}
        d = r.get("deltas_vs_chck82") or {}
        lines.append(f"| {m} | {pr.get('cheap7')} | {r.get('required_superglue_to_tie_chck82')} | {r.get('superglue')} | {r.get('projected_overall_with_aoa0')} | {d.get('projected_overall_with_aoa0')} | {r.get('score_read')} |")
    lines += ["", "## Source-free probe", "", "```json", json.dumps(probe_keys, indent=2), "```", "", "## Decisions", ""]
    for k, v in decisions.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", result["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "route_read": route_read, "pending": pending, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
