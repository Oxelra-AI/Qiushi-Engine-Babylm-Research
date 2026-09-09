#!/usr/bin/env python3
"""Compute corrected nine-column Overall for research full-eval candidate per-target JSONs.

Uses babylm_official_scoring.compute_overall_from_tasks so AoA is in leaderboard units (100*raw).
Prints each candidate's nine columns and Overall, and compares to the current clean-Qwen best 41.3443
and the visible 41.8 leader.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
from babylm_official_scoring import compute_overall_from_tasks, normalize_aoa_record  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
PER = _public_path('experiments/archive/compact_experience/data/full_eval_candidates/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/full_eval_candidates/candidate_overall_summary.json')

BASELINE = 41.34429066479573
LEADER = 41.8


def score_target(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    sg = tasks.get("SuperGLUE", {})
    # Prefer the finalized official_overall stored by the corrected research runner; recompute as fallback.
    oo = payload.get("official_overall")
    if isinstance(oo, dict) and oo.get("Overall") is not None:
        fields = oo
        overall = oo.get("Overall")
        scores = oo.get("scores")
    else:
        if isinstance(tasks.get("AoA"), dict):
            tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
        fields = compute_overall_from_tasks(tasks)
        overall = fields.get("Overall")
        scores = fields.get("scores")
    return {
        "target": payload.get("target"),
        "endpoint": payload.get("endpoint"),
        "model_path": payload.get("model_path"),
        "overall": overall,
        "complete": fields.get("complete_for_provisional_overall"),
        "task_scores": scores,
        "aoa_raw_correlation": fields.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": fields.get("aoa_leaderboard_score"),
        "superglue_mean": sg.get("superglue_mean") if isinstance(sg, dict) else None,
        "fields": fields,
    }


def main() -> None:
    rows = []
    for p in sorted(PER.glob("*.json")):
        rows.append(score_target(p))
    for r in rows:
        r["delta_vs_clean_best"] = (r["overall"] - BASELINE) if r["overall"] is not None else None
        r["delta_vs_leader"] = (r["overall"] - LEADER) if r["overall"] is not None else None
    payload = {
        "status": "CANDIDATE_OVERALL_SUMMARY",
        "baseline_clean_qwen_100M": BASELINE,
        "visible_leader": LEADER,
        "candidates": rows,
        "note": "AoA in leaderboard units (100*raw). SuperGLUE mean from official glue_filtered finetune. Overall is mean of nine leaderboard-unit columns.",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(json.dumps({
            "target": r["target"], "endpoint": r["endpoint"], "overall": r["overall"],
            "superglue_mean": r["superglue_mean"], "aoa_leaderboard_score": r["aoa_leaderboard_score"],
            "delta_vs_clean_best": r["delta_vs_clean_best"], "delta_vs_leader": r["delta_vs_leader"],
            "task_scores": r["task_scores"],
        }, ensure_ascii=False))
    print(json.dumps({"summary": str(OUT)}))


if __name__ == "__main__":
    main()
