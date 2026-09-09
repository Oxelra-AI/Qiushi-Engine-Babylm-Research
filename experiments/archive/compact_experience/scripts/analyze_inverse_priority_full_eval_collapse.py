#!/usr/bin/env python3
"""Analyze why inverse-priority no-AoA gains failed in full official-style scoring.

Uses only aggregate full-eval payloads and no-AoA summaries. It does not inspect
or use AoA/CDI item words or curve internals. The purpose is to decide whether to
replicate, close, or redirect the route based on official-compatible evidence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
FULL = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval/mask_endpoint_full_eval_summary.json')
CONTRAST = _public_path('experiments/archive/compact_experience/data/endpoint_matched_mask_contrast/endpoint_matched_mask_contrast.json')
OUT = _public_path('experiments/archive/compact_experience/data/inverse_priority_collapse/inverse_priority_full_eval_collapse.json')
NOTE = _public_path('research/notes/compact_experience/inverse_priority_full_eval_collapse.md')
VISIBLE = 41.8
CLEAN = {
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "SuperGLUE": 70.30861598316157,
    "Reading": 7.76,
    "AoA": 0.0,
    "Overall": 41.34429066479573,
}
ORDER = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def mean(xs):
    return sum(xs) / len(xs)


def score_with_aoa(row: dict, aoa_value: float) -> float:
    scores = row.get("task_scores") or {}
    return mean([float(scores[c]) if c != "AoA" else aoa_value for c in ORDER])


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    contrast = json.loads(CONTRAST.read_text(encoding="utf-8")) if CONTRAST.exists() else None
    rows = full.get("candidates", [])
    analyzed = []
    for r in rows:
        scores = r.get("task_scores") or {}
        noaoa_equal7 = mean([scores[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]])
        nlp8 = mean([scores[c] for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading"]])
        aoa_zero = score_with_aoa(r, 0.0)
        aoa_actual = float(scores.get("AoA", 0.0))
        analyzed.append({
            "target": r.get("target"),
            "endpoint": r.get("endpoint"),
            "endpoint_frozen": r.get("endpoint_frozen"),
            "submit_ready_overall": r.get("submit_ready_overall"),
            "overall_actual": r.get("overall"),
            "overall_if_aoa_zero": aoa_zero,
            "aoa_penalty_vs_zero_overall_points": float(r.get("overall")) - aoa_zero,
            "aoa_leaderboard_score": aoa_actual,
            "aoa_raw_correlation": r.get("aoa_raw_correlation"),
            "superglue": r.get("superglue_mean"),
            "noaoa_equal7": noaoa_equal7,
            "nlp8_mean_excluding_aoa": nlp8,
            "needed_aoa_to_reach_visible_leader_given_other_columns": 9 * VISIBLE - sum(float(scores[c]) for c in ORDER if c != "AoA"),
            "gap_to_visible_actual": float(r.get("overall")) - VISIBLE,
            "gap_to_visible_if_aoa_zero": aoa_zero - VISIBLE,
            "delta_columns_vs_clean": {c: float(scores[c]) - CLEAN[c] for c in ORDER},
        })
    payload = {
        "status": "INVERSE_PRIORITY_FULL_EVAL_COLLAPSE_ANALYSIS",
        "full_eval_summary": str(FULL),
        "endpoint_matched_noaoa_contrast": str(CONTRAST) if CONTRAST.exists() else None,
        "visible_leader": VISIBLE,
        "clean_qwen_seed43022_reference": CLEAN,
        "analyzed_targets": analyzed,
        "main_interpretation": "Inverse-priority achieved broad no-AoA gains but the complete official-style result failed because AoA became strongly negative. The true chck_100M endpoint would be near the visible leader if AoA were zero, but official scoring includes the negative AoA in leaderboard units; therefore this route should not be replicated as a SOTA candidate until an AoA-safe variant exists.",
        "noaoa_correction_from_step048": contrast.get("interpretation") if isinstance(contrast, dict) else None,
        "non_leakage_statement": "Uses aggregate task scores only; no AoA/CDI item words, child curves, or per-word curve internals are used for training or schedule design.",
    }
    _public_path('experiments/archive/compact_experience/data/inverse_priority_collapse').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research inverse-priority full-eval collapse", "",
        "This note uses aggregate completed full-eval values only; it does not use AoA/CDI item words or curve internals for training design.", "",
        f"Visible leader: {VISIBLE:.3f}; clean-Qwen seed43022 reference Overall {CLEAN['Overall']:.6f}.", "",
        "| target | endpoint | submit-ready | Overall | Overall if AoA=0 | AoA lb | AoA raw | SuperGLUE | no-AoA equal7 | gap actual | gap if AoA=0 | needed AoA lb |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in analyzed:
        lines.append(f"| {a['target']} | {a['endpoint']} | {a['submit_ready_overall']} | {a['overall_actual']:.6f} | {a['overall_if_aoa_zero']:.6f} | {a['aoa_leaderboard_score']:.3f} | {a['aoa_raw_correlation']:.4f} | {a['superglue']:.3f} | {a['noaoa_equal7']:.4f} | {a['gap_to_visible_actual']:+.4f} | {a['gap_to_visible_if_aoa_zero']:+.4f} | {a['needed_aoa_to_reach_visible_leader_given_other_columns']:+.3f} |")
    lines.extend(["", "## Interpretation", "",
                  "- The true `chck_100M` inverse endpoint is submit-ready in the local full-eval wrapper but scores only 39.7142 because AoA is -18.345 leaderboard units.",
                  "- If its AoA were zero, its Overall would be about 41.752, still just below 41.8; the no-AoA gain is real but not enough to survive a large negative AoA.",
                  "- The endpoint-frozen 95M measurement would exceed 41.8 under AoA=0, but it is not submission-facing and its actual AoA is also strongly negative.",
                  "- Seed43122 inverse replication should not be launched merely to reproduce this failed official profile; first determine whether the AoA collapse is inverse-specific or general to tail/masking continuations using existing trained arms (evidence-visible is currently the best candidate).",
                  ""])
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "analyzed": [{"target": a["target"], "overall": a["overall_actual"], "aoa_lb": a["aoa_leaderboard_score"], "overall_if_aoa_zero": a["overall_if_aoa_zero"]} for a in analyzed]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
