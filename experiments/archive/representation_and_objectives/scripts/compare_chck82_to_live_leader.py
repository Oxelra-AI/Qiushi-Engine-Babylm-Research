#!/usr/bin/env python3
"""research: compare repeated chck_82M score with refreshed public Strict-Small leader.

This is research evidence for endpoint review.  It reads the live leaderboard parse
refreshed in research and the independent hardened chck_82M evaluation summary, then
writes a compact JSON/Markdown comparison with arithmetic checks.  It does not run
or alter any evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
LEADERBOARD = STUDY / "data/babylm2026_live_surface/strict_small_top30.json"
HARDENED = STUDY / "data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json"
FIRST = STUDY / "data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json"
COMPARE = STUDY / "data/chck82_eval_comparison/chck82_eval_comparison.json"
OUT = STUDY / "data/chck82_public_leader_comparison"

CAND_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
LEADER_MAP = {
    "BLiMP": "BLiMP",
    "Supplement": "BLiMP Supplement",
    "EWoK": "EWoK",
    "Entity": "Entity Tracking",
    "COMPS": "COMPS",
    "SuperGLUE": "(Super)GLUE",
    "GlobalPIQA": "GlobalPIQA",
    "Reading": "Reading",
    "AoA": "AoA",
}
ZERO_READING_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EXPECTED_CANDIDATE_HASH = "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> float:
    if x is None:
        raise ValueError("None is not numeric")
    return float(x)


def hardened_scores(summary: dict[str, Any]) -> dict[str, float]:
    cols = summary.get("scores") or summary.get("official_overall", {}).get("scores")
    if not isinstance(cols, dict):
        # research hardened summary uses top-level keys too.
        cols = {k: summary.get(k) for k in CAND_KEYS if k in summary}
    return {k: fnum(cols[k]) for k in CAND_KEYS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    leaderboard = read_json(LEADERBOARD)
    if not leaderboard:
        raise RuntimeError("no strict-small rows in leaderboard parse")
    leader = leaderboard[0]
    hardened = read_json(HARDENED)
    first = read_json(FIRST)
    eval_compare = read_json(COMPARE)

    cand = hardened_scores(hardened)
    cand_overall_reported = fnum(hardened.get("Overall") or hardened.get("official_overall", {}).get("Overall"))
    cand_cheap7_reported = fnum(hardened.get("cheap7") or sum(cand[k] for k in ZERO_READING_KEYS) / 7.0)
    cand_overall_calc = sum(cand[k] for k in CAND_KEYS) / 9.0
    cand_cheap7_calc = sum(cand[k] for k in ZERO_READING_KEYS) / 7.0

    leader_scores = {k: fnum(leader[LEADER_MAP[k]]) for k in CAND_KEYS}
    leader_overall_reported = fnum(leader["Overall Average"])
    leader_overall_calc = sum(leader_scores[k] for k in CAND_KEYS) / 9.0
    leader_cheap7_calc = sum(leader_scores[k] for k in ZERO_READING_KEYS) / 7.0

    deltas = {k: cand[k] - leader_scores[k] for k in CAND_KEYS}
    out = {
        "status": "PASS" if cand_overall_reported > leader_overall_reported else "BELOW_LEADER",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "leaderboard_top30": rel(LEADERBOARD),
            "hardened_score": rel(HARDENED),
            "first_score": rel(FIRST),
            "measurement_repeatability": rel(COMPARE),
        },
        "public_leader": {
            "model": leader.get("Model_plain"),
            "repo": leader.get("HF Repo"),
            "overall_reported": leader_overall_reported,
            "overall_recomputed_from_displayed_columns": leader_overall_calc,
            "cheap7_recomputed_from_displayed_columns": leader_cheap7_calc,
            "scores": leader_scores,
        },
        "candidate_chck82": {
            "endpoint": hardened.get("endpoint") or "chck_82M",
            "run_dir": hardened.get("run_dir") or hardened.get("endpoint_ready", {}).get("run_dir"),
            "model_sha256_expected": EXPECTED_CANDIDATE_HASH,
            "overall_reported": cand_overall_reported,
            "overall_recomputed": cand_overall_calc,
            "cheap7_reported": cand_cheap7_reported,
            "cheap7_recomputed": cand_cheap7_calc,
            "scores": cand,
        },
        "candidate_minus_public_leader": {
            "Overall": cand_overall_reported - leader_overall_reported,
            "Overall_recomputed_minus_displayed": cand_overall_calc - leader_overall_calc,
            "cheap7_recomputed": cand_cheap7_calc - leader_cheap7_calc,
            "scores": deltas,
        },
        "arithmetic_checks": {
            "candidate_overall_reported_matches_recomputed": math.isclose(cand_overall_reported, cand_overall_calc, rel_tol=0, abs_tol=1e-9),
            "candidate_cheap7_reported_matches_recomputed": math.isclose(cand_cheap7_reported, cand_cheap7_calc, rel_tol=0, abs_tol=1e-9),
            "measurement_repeatability_status": eval_compare.get("status"),
            "measurement_repeatability_delta_overall": eval_compare.get("deltas_hardened_minus_first", {}).get("Overall"),
            "measurement_repeatability_max_abs_column_delta": eval_compare.get("deltas_hardened_minus_first", {}).get("max_abs_column_delta"),
        },
        "scientific_reading": {
            "score_source": "Use the hardened repeated evaluation as the current score-bearing vector; research first score agrees to leaderboard rounding.",
            "where_candidate_wins": [k for k, v in deltas.items() if v > 0],
            "where_candidate_trails": [k for k, v in deltas.items() if v < 0],
            "interpretation": "The candidate clears the visible Overall target through large Supplement and Reading gains plus a smaller BLiMP gain, while it is slightly below the public leader on Entity and SuperGLUE and clearly below on EWoK, COMPS, and GlobalPIQA. This supports endpoint preservation but not a claim that the context-conditioned alternative-binding deficit has been solved.",
        },
    }
    out_json = OUT / "chck82_public_leader_comparison.json"
    out_md = OUT / "chck82_public_leader_comparison.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research chck_82M vs refreshed public Strict-Small leader",
        "",
        f"Status: **{out['status']}**",
        f"Public leader: `{leader.get('Model_plain')}` (`{leader.get('HF Repo')}`), Overall {leader_overall_reported:.6f}",
        f"Candidate: `chck_82M`, hardened repeated Overall {cand_overall_reported:.6f}, margin {cand_overall_reported - leader_overall_reported:+.6f}",
        "",
        "| column | candidate | public leader | delta |",
        "|---|---:|---:|---:|",
    ]
    for k in CAND_KEYS:
        lines.append(f"| {k} | {cand[k]:.6f} | {leader_scores[k]:.6f} | {deltas[k]:+.6f} |")
    lines += [
        f"| **Overall** | **{cand_overall_reported:.6f}** | **{leader_overall_reported:.6f}** | **{cand_overall_reported - leader_overall_reported:+.6f}** |",
        "",
        f"Candidate recomputed Overall: `{cand_overall_calc:.12f}`; cheap7: `{cand_cheap7_calc:.12f}`.",
        f"Measurement repeatability Overall delta: `{eval_compare.get('deltas_hardened_minus_first', {}).get('Overall')}`; max column delta: `{eval_compare.get('deltas_hardened_minus_first', {}).get('max_abs_column_delta')}`.",
        "",
        "Scientific reading: the endpoint is above the visible public Overall target, but still trails on EWoK/COMPS/GlobalPIQA; endpoint preservation and mechanism interpretation must remain separate.",
        "",
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "overall_candidate": cand_overall_reported,
        "overall_leader": leader_overall_reported,
        "margin": cand_overall_reported - leader_overall_reported,
        "json": rel(out_json),
        "md": rel(out_md),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
