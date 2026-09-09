#!/usr/bin/env python3
"""Projection thresholds for research compact-core full evaluation.

Uses only the research no-AoA fast task-family screen as arithmetic context.  It
answers: given the observed seven non-SuperGLUE/non-AoA columns, what combined
(SuperGLUE + AoA leaderboard score) is needed to pass relevant Overall targets?
This is not a replacement for full evaluation because fast/full zero-shot splits
and AoA can change.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
FAST = WORKSPACE / "data" / "density_noaoa_eval_compact_core" / "density_noaoa_eval_summary.json"
OUT_DIR = WORKSPACE / "data" / "projection_thresholds"
OUT_JSON = OUT_DIR / "compact_core_full_eval_thresholds.json"
NOTE = (WORKSPACE / 'notes'.parents[3] / 'research/notes/frontier_consolidation/compact_core_full_eval_projection_thresholds.md')

CLEAN = {
    "Overall": 41.34429066479573,
    "SuperGLUE": 70.30861598316157,
    "AoA": 0.0,
}
LEADER = {
    "Overall": 41.80,
    "SuperGLUE": 69.79,
    "AoA": 0.0,
}
TARGETS = {
    "inherited_clean_qwen_41p3443": CLEAN["Overall"],
    "visible_leader_41p8": LEADER["Overall"],
    "round_42p0": 42.0,
}
SEVEN = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_arm(name: str, scores: dict[str, Any]) -> dict[str, Any]:
    s7 = sum(float(scores[k]) for k in SEVEN)
    thresholds = {}
    for label, overall in TARGETS.items():
        need_sg_aoa = 9.0 * float(overall) - s7
        thresholds[label] = {
            "target_overall": overall,
            "seven_column_sum": s7,
            "required_superglue_plus_aoa": need_sg_aoa,
            "required_superglue_if_aoa_zero": need_sg_aoa,
            "required_aoa_if_superglue_clean_qwen": need_sg_aoa - CLEAN["SuperGLUE"],
            "required_aoa_if_superglue_visible_leader": need_sg_aoa - LEADER["SuperGLUE"],
        }
    scenarios = {}
    for sg_label, sg in [("clean_qwen_superglue", CLEAN["SuperGLUE"]), ("visible_leader_superglue", LEADER["SuperGLUE"]), ("superglue_69", 69.0), ("superglue_68", 68.0)]:
        for aoa in [0.0, -2.0, -5.0, 2.0, 5.0]:
            key = f"{sg_label}_aoa_{aoa:+.1f}"
            scenarios[key] = {
                "SuperGLUE": sg,
                "AoA": aoa,
                "projected_overall_from_fast7": (s7 + sg + aoa) / 9.0,
            }
    return {
        "arm": name,
        "fast_scores": {k: scores[k] for k in SEVEN if k in scores},
        "seven_column_sum": s7,
        "seven_column_mean": s7 / len(SEVEN),
        "thresholds": thresholds,
        "scenarios": scenarios,
    }


def main() -> None:
    fast = read_json(FAST)
    arms = {}
    for name, scores in fast.get("table", {}).items():
        if all(k in scores and scores[k] is not None for k in SEVEN):
            arms[name] = summarize_arm(name, scores)
    payload = {
        "status": "COMPACT_CORE_FULL_EVAL_PROJECTION_THRESHOLDS",
        "scope": "Arithmetic from research no-AoA fast seven-column screen only; full official-compatible evaluation remains decisive.",
        "fast_source": str(FAST),
        "seven_columns": SEVEN,
        "reference": {"clean_qwen": CLEAN, "visible_leader": LEADER, "targets": TARGETS},
        "arms": arms,
        "interpretation": "If compact_view_core keeps its fast seven-column surface, it needs SuperGLUE+AoA >= 69.255 to reach Overall 41.8 and >=71.055 to reach 42.0. Full zero-shot splits and AoA can change these inputs.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research compact-core full-evaluation projection thresholds",
        "",
        f"JSON: `{OUT_JSON}`",
        f"Fast-screen source: `{FAST}`",
        "",
        "This is arithmetic context only. The running full official-compatible evaluation remains the evidence.",
        "",
        "| arm | fast seven-column mean | need SG+AoA for 41.344 | need SG+AoA for 41.8 | need SG+AoA for 42.0 | projected Overall if SG=clean and AoA=0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, rec in arms.items():
        th = rec["thresholds"]
        proj = rec["scenarios"]["clean_qwen_superglue_aoa_+0.0"]["projected_overall_from_fast7"]
        lines.append(
            f"| {name} | {rec['seven_column_mean']:.4f} | "
            f"{th['inherited_clean_qwen_41p3443']['required_superglue_plus_aoa']:.3f} | "
            f"{th['visible_leader_41p8']['required_superglue_plus_aoa']:.3f} | "
            f"{th['round_42p0']['required_superglue_plus_aoa']:.3f} | {proj:.4f} |"
        )
    lines.extend([
        "",
        "For compact_view_core, preserving a clean-Qwen-like SuperGLUE with AoA=0 would project above the visible 41.8 surface from the fast seven-column screen, but a negative AoA or full-split degradation can erase that margin. This is why the full evaluation is decisive before any recipe changes.",
    ])
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE)}, indent=2))


if __name__ == "__main__":
    main()
