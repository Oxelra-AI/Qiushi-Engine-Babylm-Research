#!/usr/bin/env python3
"""research: merge the split U256 chck_80M cheap-column screen.

This consumes the two already-finished split per-target JSON payloads from research,
computes the official cheap7 surface, compares it with research, scale1.75 100M, and
U256 100M, and writes a durable route-decision record.  It performs no model
loading, training, or evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from statistics import mean
from typing import Any

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
PART_A = WORKSPACE / "data/u256_80m_cheap_screen/partA/eval/per_target/U256_80M_cheapA.json"
PART_B = WORKSPACE / "data/u256_80m_cheap_screen/partB/eval/per_target/U256_80M_cheapB.json"
SUMMARY = WORKSPACE / "data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json"
SCALE_SUMMARY = WORKSPACE / "data/scale1p75_100M_repaired_merge/summary/scale1p75_100M_full_eval_hardened_summary.json"
U256_100M_SUMMARY = WORKSPACE / "data/u256_100M_full_eval_hardened/summary/u256_100M_full_eval_hardened_summary.json"
OUT_DIR = WORKSPACE / "data/u256_80m_cheap_merge"
OUT_JSON = OUT_DIR / "u256_80M_cheap_merge_summary.json"
OUT_MD = OUT_DIR / "u256_80M_cheap_merge_summary.md"
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALL_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    scores = payload.get("official_overall", {}).get("scores") or {}
    out: dict[str, float | None] = {}
    for k, v in scores.items():
        out[k] = None if v is None else float(v)
    # Some partial payloads keep GlobalPIQA split columns in task scores but merged score in official_overall.
    if out.get("GlobalPIQA") is None:
        tasks = payload.get("tasks") or {}
        gp_p = tasks.get("GlobalPIQA_parallel", {}).get("score")
        gp_n = tasks.get("GlobalPIQA_nonparallel", {}).get("score")
        if gp_p is not None and gp_n is not None:
            out["GlobalPIQA"] = (float(gp_p) + float(gp_n)) / 2.0
    return out


def get_summary_scores(path: pathlib.Path) -> dict[str, float]:
    obj = load_json(path)
    src = obj.get("scores")
    if not isinstance(src, dict):
        src = (obj.get("score_summary") or {}).get("scores")
    if not isinstance(src, dict):
        src = {}
    scores = {k: float(v) for k, v in src.items() if v is not None}
    # New hardened summaries put cheap7/Overall at top-level; old pristine summaries put them under score_summary.
    for key in ("Overall", "cheap7"):
        if key not in scores:
            if obj.get(key) is not None:
                scores[key] = float(obj[key])
            elif (obj.get("score_summary") or {}).get(key) is not None:
                scores[key] = float((obj.get("score_summary") or {})[key])
    if "cheap7" not in scores and all(c in scores for c in CHEAP_COLS):
        scores["cheap7"] = mean(scores[c] for c in CHEAP_COLS)
    return scores


def delta_table(scores: dict[str, float], ref: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for c in CHEAP_COLS:
        out[c] = scores.get(c) - ref.get(c) if c in scores and c in ref else None
    if "cheap7" in scores and "cheap7" in ref:
        out["cheap7"] = scores["cheap7"] - ref["cheap7"]
    if "Overall" in scores and "Overall" in ref:
        out["Overall"] = scores["Overall"] - ref["Overall"]
    return out


def fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = load_json(PART_A)
    b = load_json(PART_B)
    scores: dict[str, float] = {}
    for payload in (a, b):
        for k, v in extract_scores(payload).items():
            if v is not None:
                if k in scores and abs(scores[k] - v) > 1e-9:
                    raise RuntimeError(f"score conflict for {k}: {scores[k]} vs {v}")
                scores[k] = float(v)
    missing = [c for c in CHEAP_COLS if c not in scores]
    if missing:
        raise RuntimeError(f"missing cheap columns: {missing}")
    scores["cheap7"] = mean(scores[c] for c in CHEAP_COLS)
    # No SuperGLUE/AoA for this checkpoint; only project with known endpoints.
    research = get_summary_scores(SUMMARY)
    scale = get_summary_scores(SCALE_SUMMARY)
    u100 = get_summary_scores(U256_100M_SUMMARY)
    # research summary has scores under top-level in old files; if not, use known nested.
    if not all(c in research for c in CHEAP_COLS):
        obj35 = load_json(SUMMARY)
        for c in CHEAP_COLS + ["SuperGLUE", "AoA"]:
            if c in obj35:
                research[c] = float(obj35[c])
        if "Overall" in obj35:
            research["Overall"] = float(obj35["Overall"])
        if "cheap7" in obj35:
            research["cheap7"] = float(obj35["cheap7"])
    # Projected official if research or U256100M SuperGLUE/AoA are borrowed only for route triage.
    projected_with_step35_sg_aoa = (scores["cheap7"] * 7 + research["SuperGLUE"] + research.get("AoA", 0.0)) / 9.0
    projected_with_u256100_sg_aoa = (scores["cheap7"] * 7 + u100["SuperGLUE"] + u100.get("AoA", 0.0)) / 9.0
    required_superglue_for_41p8_aoa0 = 41.8 * 9 - scores["cheap7"] * 7
    continuation_decision = {
        "full_80m_superglue_aoa_supported": bool(scores["cheap7"] >= 43.70 or projected_with_u256100_sg_aoa >= 41.75),
        "reason": (
            "U256 80M cheap7 is below research and far below scale1.75 100M; with AoA=0 it would require "
            "SuperGLUE above any observed U256/research/scale1.75 value to approach 41.8."
        ),
    }
    result = {
        "status": "U256_80M_CHEAP_MERGE",
        "inputs": {"partA": str(PART_A), "partB": str(PART_B)},
        "scores": scores,
        "references": {
            "research": {c: research.get(c) for c in CHEAP_COLS + ["cheap7", "SuperGLUE", "Overall"]},
            "scale1p75_100M": {c: scale.get(c) for c in CHEAP_COLS + ["cheap7", "SuperGLUE", "Overall"]},
            "u256_100M": {c: u100.get(c) for c in CHEAP_COLS + ["cheap7", "SuperGLUE", "Overall"]},
        },
        "deltas_vs_step35": delta_table(scores, research),
        "deltas_vs_scale1p75_100M": delta_table(scores, scale),
        "deltas_vs_u256_100M": delta_table(scores, u100),
        "projected_overall_if_step35_superglue_aoa": projected_with_step35_sg_aoa,
        "projected_overall_if_u256100_superglue_aoa": projected_with_u256100_sg_aoa,
        "required_superglue_for_overall_41p8_with_aoa0": required_superglue_for_41p8_aoa0,
        "continuation_decision": continuation_decision,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# research — U256 chck_80M cheap-column merge",
        "",
        "This merges the two split cheap-column screens from research. It is not a full official checkpoint evaluation; it is the low-cost test of whether U256 had an intermediate cheap-surface peak worth SuperGLUE/AoA follow-up.",
        "",
        "| Column | U256 80M | research 100M | scale1.75 100M | U256 100M | Δ80M-research | Δ80M-scale | Δ80M-U256100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CHEAP_COLS + ["cheap7"]:
        lines.append(
            f"| {c} | {fmt(scores.get(c))} | {fmt(research.get(c))} | {fmt(scale.get(c))} | {fmt(u100.get(c))} | "
            f"{fmt(result['deltas_vs_step35'].get(c))} | {fmt(result['deltas_vs_scale1p75_100M'].get(c))} | {fmt(result['deltas_vs_u256_100M'].get(c))} |"
        )
    lines += [
        "",
        f"Projected Overall if research SuperGLUE/AoA are borrowed only for triage: `{projected_with_step35_sg_aoa:.6f}`.",
        f"Projected Overall if U256-100M SuperGLUE/AoA are borrowed only for triage: `{projected_with_u256100_sg_aoa:.6f}`.",
        f"Required SuperGLUE at AoA=0 for Overall 41.8: `{required_superglue_for_41p8_aoa0:.6f}`.",
        "",
        f"Decision: full U256-80M SuperGLUE/AoA follow-up supported? `{continuation_decision['full_80m_superglue_aoa_supported']}`.",
        f"Reason: {continuation_decision['reason']}",
        "",
        f"JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "cheap7": scores["cheap7"],
        "cheap7_delta_vs_step35": result["deltas_vs_step35"]["cheap7"],
        "cheap7_delta_vs_scale1p75": result["deltas_vs_scale1p75_100M"]["cheap7"],
        "full_80m_superglue_aoa_supported": continuation_decision["full_80m_superglue_aoa_supported"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
