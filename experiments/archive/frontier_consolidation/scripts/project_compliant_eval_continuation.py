#!/usr/bin/env python3
"""Project whether partial compliant evaluation warrants remaining columns.

This is a CPU-only arithmetic helper.  It reads the per-target JSON produced by
`evaluate_compliant_endpoint.py` and reports what partial scores imply.
It deliberately does *not* substitute for the full official result of the
submission-relevant reinvest endpoint.

Evaluation-policy correction:
- partial columns may conserve compute;
- but the reinvest endpoint is the sole current submission-relevant compliant
  endpoint;
- remaining evaluation should be stopped only when the completed columns prove
  that even maximally favorable unfinished columns cannot reach the chosen
  target;
- otherwise complete SuperGLUE, the 8,005-row AoA trajectory, and pristine
  nine-column collation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
VISIBLE_LEADER = 41.8
FROZEN_OLDTOKENIZER_REFERENCE = 42.0331347900748
DEFAULT_MISSING_UPPER = 100.0


def target_scores(tasks: dict[str, Any]) -> dict[str, float | None]:
    """Extract leaderboard-unit scores without inventing missing official columns."""
    out: dict[str, float | None] = {k: None for k in OVERALL_KEYS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            out[col] = float(rec["score"])

    gp = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp.append(float(rec["score"]))
    if len(gp) == 2:
        out["GlobalPIQA"] = sum(gp) / 2.0

    rec = tasks.get("Reading")
    if isinstance(rec, dict):
        scores = rec.get("scores") or {}
        if scores.get("Reading") is not None:
            out["Reading"] = float(scores["Reading"])

    rec = tasks.get("SuperGLUE")
    if isinstance(rec, dict) and rec.get("superglue_mean") is not None:
        out["SuperGLUE"] = float(rec["superglue_mean"])

    rec = tasks.get("AoA")
    if isinstance(rec, dict):
        # Count AoA only when the repaired official trajectory actually ran.
        # A missing-checkpoint/non-official placeholder must remain missing so it
        # cannot be used as a fake completed zero.  A genuine official AoA can be
        # exactly 0.0, as in prior valid runs.
        status = str(rec.get("status", ""))
        row_values = rec.get("row_count_values")
        finite = rec.get("finite_surprisals")
        if (
            status == "official_aoa_done"
            and row_values == [8005]
            and finite is True
            and rec.get("aoa_leaderboard_score") is not None
        ):
            out["AoA"] = float(rec["aoa_leaderboard_score"])
    return out


def bound_payload(scores: dict[str, float | None], target: float, missing_upper: float) -> dict[str, Any]:
    known = {k: v for k, v in scores.items() if v is not None}
    missing = [k for k, v in scores.items() if v is None]
    known_sum = sum(float(v) for v in known.values())
    target_sum = 9.0 * target
    missing_upper_sum = len(missing) * missing_upper
    hard_upper_overall = (known_sum + missing_upper_sum) / 9.0
    lower_overall_missing_zero = known_sum / 9.0
    needed_sum = target_sum - known_sum
    needed_mean_missing = needed_sum / len(missing) if missing else None
    complete_overall = known_sum / 9.0 if not missing else None
    return {
        "target_overall": target,
        "known_sum": known_sum,
        "target_sum_to_exceed_or_equal": target_sum,
        "missing_keys": missing,
        "missing_upper_each": missing_upper,
        "missing_upper_sum": missing_upper_sum,
        "hard_upper_overall_if_missing_at_upper": hard_upper_overall,
        "lower_overall_if_missing_zero": lower_overall_missing_zero,
        "needed_sum_from_missing_to_equal_target": needed_sum,
        "needed_mean_missing_to_equal_target": needed_mean_missing,
        "mathematically_can_reach_target_with_missing_upper": hard_upper_overall >= target if missing else (complete_overall is not None and complete_overall >= target),
        "complete_overall_if_no_missing": complete_overall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-target-json", required=True)
    ap.add_argument("--leader", type=float, default=VISIBLE_LEADER)
    ap.add_argument("--reference-overall", type=float, default=FROZEN_OLDTOKENIZER_REFERENCE,
                    help="Frozen old-tokenizer complete reference for context; not a submission-valid target.")
    ap.add_argument("--decision-target", type=float, default=VISIBLE_LEADER,
                    help="Remaining evaluation may be stopped only if even favorable missing columns cannot reach this target. For the submission-relevant reinvest endpoint the default is the visible leaderboard target, not a projection of unfinished columns.")
    ap.add_argument("--missing-upper", type=float, default=DEFAULT_MISSING_UPPER,
                    help="Favorable upper bound used for unfinished leaderboard columns; 100 is the hard percentage/correlation-scale bound.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = pathlib.Path(args.per_target_json)
    payload = json.loads(p.read_text(encoding="utf-8"))
    scores = target_scores(payload.get("tasks", {}))
    known = {k: v for k, v in scores.items() if v is not None}
    missing = [k for k, v in scores.items() if v is None]

    bounds = {
        "leader": bound_payload(scores, args.leader, args.missing_upper),
        "frozen_oldtokenizer_reference": bound_payload(scores, args.reference_overall, args.missing_upper),
        "decision_target": bound_payload(scores, args.decision_target, args.missing_upper),
    }

    is_reinvest_endpoint = payload.get("family") == "end_to_end_compliant_tokenizer_density_reinvestment" or "reinvest" in str(payload.get("target", "")).lower()
    decision_can_reach = bounds["decision_target"]["mathematically_can_reach_target_with_missing_upper"]
    hard_stop_remaining = bool(missing) and not decision_can_reach
    complete_now = not missing
    if complete_now:
        action = "complete_columns_present_run_pristine_collation_or_read_collated_score"
    elif hard_stop_remaining:
        action = "stop_remaining_evaluation_and_route_interpretation_before_more_expensive_work"
    elif is_reinvest_endpoint:
        action = "continue_remaining_official_evaluation_for_submission_relevant_endpoint"
    else:
        action = "continue_remaining_evaluation_if_scientific_control_value_justifies_compute"

    if set(missing) == {"SuperGLUE", "AoA"}:
        bounds["leader"]["required_superglue_plus_aoa_to_equal_target"] = bounds["leader"]["needed_sum_from_missing_to_equal_target"]
        bounds["leader"]["old_reference_superglue_plus_aoa"] = 71.03604952825312 + 0.0
        bounds["frozen_oldtokenizer_reference"]["required_superglue_plus_aoa_to_equal_target"] = bounds["frozen_oldtokenizer_reference"]["needed_sum_from_missing_to_equal_target"]
        bounds["frozen_oldtokenizer_reference"]["old_reference_superglue_plus_aoa"] = 71.03604952825312 + 0.0

    out = {
        "status": "COMPLIANT_EVAL_CONTINUATION_POLICY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(p),
        "target": payload.get("target"),
        "family": payload.get("family"),
        "description": payload.get("description"),
        "budget_semantics": payload.get("budget_semantics"),
        "scores": scores,
        "known_keys": list(known),
        "missing_keys": missing,
        "policy": {
            "strategist_rule": "For the submission-relevant reinvest endpoint, partial columns conserve compute but must not substitute for the full official result. Stop SuperGLUE/AoA/pristine collation only if completed columns plus favorable upper-bound unfinished columns cannot reach the decision target; otherwise complete the official surface.",
            "decision_target_overall": args.decision_target,
            "missing_upper_each": args.missing_upper,
            "hard_stop_remaining_evaluation": hard_stop_remaining,
            "recommended_action": action,
            "reinvest_endpoint_detected": is_reinvest_endpoint,
            "aoa_counting_rule": "AoA is counted only after official_aoa_done with row_count_values=[8005] and finite_surprisals=true; missing-checkpoint placeholders remain missing, not zero.",
        },
        "bounds": bounds,
        "notes": [
            "The frozen old-tokenizer 42.0331347900748 reference is scientifically important but not an end-to-end compliant Strict-Small submission because of tokenizer provenance.",
            "A complete compliant reinvest endpoint above the visible leader can answer the submission question even if it does not improve the old-tokenizer reference.",
            "Clean-Qwen fixed-tokenizer projections are scientific controls only, not independent submission-valid scores, because tokenizer fitting used the reinvest 10M pool.",
        ],
    }
    out_path = pathlib.Path(args.out) if args.out else p.with_name(p.stem + "_continuation_policy.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out": str(out_path),
        "target": out["target"],
        "known_keys": out["known_keys"],
        "missing_keys": missing,
        "decision_target": args.decision_target,
        "hard_upper_overall_decision_target": bounds["decision_target"]["hard_upper_overall_if_missing_at_upper"],
        "hard_stop_remaining_evaluation": hard_stop_remaining,
        "recommended_action": action,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
