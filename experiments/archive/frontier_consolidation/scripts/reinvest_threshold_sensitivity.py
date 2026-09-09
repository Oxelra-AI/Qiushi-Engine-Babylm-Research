#!/usr/bin/env python3
"""research reinvest full-surface threshold sensitivity.

Uses already computed no-AoA tables and compact-core full evaluation to estimate
what SuperGLUE+AoA must do for compact_view_reinvest.  This is arithmetic for
route interpretation, not evidence replacing actual full evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from typing import Dict, List

ROOT = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/reinvest_projection')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/reinvest_projection/reinvest_threshold_sensitivity.json')
OUT_NOTE = _public_path('research/notes/frontier_consolidation/reinvest_threshold_sensitivity.md')
COMPACT_FULL = _public_path('experiments/archive/frontier_consolidation/data/density_full_eval/density_full_eval_summary.json')
COMPACT_FAST = _public_path('experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json')
REINVEST_FAST = _public_path('experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json')

OFFICIAL7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FAST_TO_OFFICIAL7 = {
    "BLiMP": "BLiMP",
    "Supplement": "Supplement",
    "EWoK": "EWoK",
    "Entity": "Entity_full",  # official Entity uses full entity_tracking, not fast Entity
    "COMPS": "COMPS",
    "GlobalPIQA": "GlobalPIQA_mean",
    "Reading": "Reading",
}
TARGETS = {"inherited_clean_qwen": 41.34429066479573, "visible_leader_41p8": 41.8, "round_42p0": 42.0}


def official_like_from_fast(table: Dict[str, float]) -> Dict[str, float]:
    return {col: float(table[FAST_TO_OFFICIAL7[col]]) for col in OFFICIAL7}


def seven_sum(scores: Dict[str, float]) -> float:
    return sum(float(scores[k]) for k in OFFICIAL7)


def thresholds(sum7: float) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, overall in TARGETS.items():
        req = 9.0 * overall - sum7
        out[name] = {
            "target_overall": overall,
            "seven_column_sum": sum7,
            "required_superglue_plus_aoa": req,
            "required_aoa_if_superglue_compact_core_full_68p901": req - 68.90115283225359,
            "required_aoa_if_superglue_clean_qwen_70p309": req - 70.30861598316157,
            "required_aoa_if_superglue_68": req - 68.0,
            "required_aoa_if_superglue_69": req - 69.0,
        }
    return out


def diff(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in OFFICIAL7}


def apply_delta(scores: Dict[str, float], delta: Dict[str, float]) -> Dict[str, float]:
    return {k: float(scores[k]) + float(delta[k]) for k in OFFICIAL7}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cf = json.loads(COMPACT_FULL.read_text(encoding="utf-8"))
    cfast = json.loads(COMPACT_FAST.read_text(encoding="utf-8"))["table"]["compact_view_core"]
    rfast = json.loads(REINVEST_FAST.read_text(encoding="utf-8"))["table"]["compact_view_reinvest"]
    compact_full_scores = {k: float(cf["targets"]["compact_view_core"]["scores"][k]) for k in OFFICIAL7}
    compact_fast_official_like = official_like_from_fast(cfast)
    reinvest_fast_official_like = official_like_from_fast(rfast)
    full_minus_fast_delta = diff(compact_full_scores, compact_fast_official_like)
    reinvest_if_same_full_delta = apply_delta(reinvest_fast_official_like, full_minus_fast_delta)
    payload = {
        "status": "REINVEST_THRESHOLD_SENSITIVITY",
        "purpose": "Correct the reinvest projection by using full Entity for the official-like seven-column surface and by showing sensitivity to compact-core full-split deltas.",
        "sources": {
            "compact_full": str(COMPACT_FULL),
            "compact_fast": str(COMPACT_FAST),
            "reinvest_fast": str(REINVEST_FAST),
        },
        "official_like_fast_mapping": FAST_TO_OFFICIAL7,
        "compact_view_core_fast_official_like": compact_fast_official_like,
        "compact_view_core_full_official7": compact_full_scores,
        "compact_full_minus_fast_official_like_delta": full_minus_fast_delta,
        "compact_view_reinvest_fast_official_like": reinvest_fast_official_like,
        "compact_view_reinvest_fast_official_like_sum": seven_sum(reinvest_fast_official_like),
        "compact_view_reinvest_fast_official_like_mean": seven_sum(reinvest_fast_official_like) / 7.0,
        "thresholds_from_reinvest_fast_official_like": thresholds(seven_sum(reinvest_fast_official_like)),
        "compact_view_reinvest_if_same_full_minus_fast_delta": reinvest_if_same_full_delta,
        "compact_view_reinvest_if_same_full_delta_sum": seven_sum(reinvest_if_same_full_delta),
        "compact_view_reinvest_if_same_full_delta_mean": seven_sum(reinvest_if_same_full_delta) / 7.0,
        "thresholds_if_reinvest_receives_compact_core_full_split_delta": thresholds(seven_sum(reinvest_if_same_full_delta)),
        "interpretation": "These are route-arithmetic bounds only. The paired-session full reinvest evaluation is the real measurement.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines: List[str] = ["# research reinvest threshold sensitivity", ""]
    lines.append("This note corrects the reinvest projection by using `Entity_full` from the no-AoA screen as the official-like Entity column. It also applies the observed compact-core full-minus-fast split deltas as a sensitivity case. This is not a substitute for the paired full reinvest evaluation.")
    lines.append("")
    s = payload["compact_view_reinvest_fast_official_like_sum"]
    lines.append(f"Reinvest official-like fast seven-column sum = {s:.3f}, mean = {s/7:.4f}.")
    t = payload["thresholds_from_reinvest_fast_official_like"]
    lines.append(f"From that surface, required SuperGLUE+AoA: {t['inherited_clean_qwen']['required_superglue_plus_aoa']:.3f} for inherited clean-Qwen, {t['visible_leader_41p8']['required_superglue_plus_aoa']:.3f} for 41.8, {t['round_42p0']['required_superglue_plus_aoa']:.3f} for 42.0.")
    lines.append(f"If SuperGLUE equals compact-core full (68.901), the tolerated AoA for 41.8 is {t['visible_leader_41p8']['required_aoa_if_superglue_compact_core_full_68p901']:.3f}.")
    lines.append("")
    d = payload["compact_full_minus_fast_official_like_delta"]
    lines.append("Observed compact-core full-minus-fast official-like deltas: " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items()))
    s2 = payload["compact_view_reinvest_if_same_full_delta_sum"]
    t2 = payload["thresholds_if_reinvest_receives_compact_core_full_split_delta"]
    lines.append(f"If reinvest receives the same full-split deltas, seven-column sum = {s2:.3f}, required SuperGLUE+AoA = {t2['visible_leader_41p8']['required_superglue_plus_aoa']:.3f} for 41.8, tolerated AoA at SG=68.901 is {t2['visible_leader_41p8']['required_aoa_if_superglue_compact_core_full_68p901']:.3f}.")
    lines.append("")
    lines.append(f"JSON: `{OUT_JSON}`")
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(OUT_NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
