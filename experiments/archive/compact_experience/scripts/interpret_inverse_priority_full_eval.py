#!/usr/bin/env python3
"""Interpret the pending seed43022 inverse-priority full SuperGLUE+AoA result.

This reader keeps official Overall (nine-column, leaderboard units) separate from
no-AoA equal7/equal6 screening metrics. It classifies the true chck_100M endpoint
and the endpoint-frozen chck_95M scientific measurement, then prints the concrete
next execution action. It does not train or evaluate models.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
SUMMARY = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval/mask_endpoint_full_eval_summary.json')
STAGED_CONTRAST = _public_path('data/external/endpoint_matched_mask_contrast.json')
OUT = _public_path('experiments/archive/compact_experience/data/inverse_priority_full_eval_interpretation/inverse_priority_full_eval_interpretation.json')
NOTE = _public_path('research/notes/compact_experience/inverse_priority_full_eval_interpretation.md')
VISIBLE_LEADER = 41.8
CLEAN43022 = 41.34429066479573
CLEAN43122 = 40.65005195633467
NEAR_LEADER = 41.65


def endpoint_m(row: dict[str, Any]) -> int | None:
    ep = row.get("endpoint") or ""
    if isinstance(ep, str) and ep.startswith("chck_") and ep.endswith("M"):
        try:
            return int(ep[5:-1])
        except Exception:
            return None
    return None


def pick(rows: list[dict[str, Any]], ep: int) -> dict[str, Any] | None:
    for r in rows:
        if endpoint_m(r) == ep:
            return r
    return None


def status_for(row: dict[str, Any] | None, *, true_endpoint: bool) -> str:
    if not row or row.get("overall") is None:
        return "missing"
    ov = float(row["overall"])
    if true_endpoint:
        if ov >= VISIBLE_LEADER:
            return "true_100M_crosses_visible_leader"
        if ov >= NEAR_LEADER:
            return "true_100M_near_visible_leader"
        return "true_100M_below_visible_leader"
    if ov >= VISIBLE_LEADER:
        return "endpoint_frozen_crosses_visible_leader_but_not_submission_facing"
    if ov >= NEAR_LEADER:
        return "endpoint_frozen_near_visible_leader"
    return "endpoint_frozen_below_visible_leader"


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Full-eval summary not found yet: {SUMMARY}")
    obj = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = obj.get("candidates", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Malformed summary: {SUMMARY}")
    true100 = pick(rows, 100)
    frozen95 = pick(rows, 95)
    staged = json.loads(STAGED_CONTRAST.read_text(encoding="utf-8")) if STAGED_CONTRAST.exists() else None
    true_status = status_for(true100, true_endpoint=True)
    frozen_status = status_for(frozen95, true_endpoint=False)

    if true_status in {"true_100M_crosses_visible_leader", "true_100M_near_visible_leader"}:
        next_action = "launch_seed43122_inverse_uniform_replication_then_noaoa_and_true100_full_eval"
    elif frozen_status in {"endpoint_frozen_crosses_visible_leader_but_not_submission_facing", "endpoint_frozen_near_visible_leader"}:
        next_action = "launch_seed43122_replication_and_measure_true100_plus_selected_endpoint_if_replicated"
    else:
        next_action = "inspect_column_collapse_before_spending_gpu_on_replication"

    payload = {
        "status": "INVERSE_PRIORITY_FULL_EVAL_INTERPRETATION",
        "summary_path": str(SUMMARY),
        "visible_leader": VISIBLE_LEADER,
        "clean43022_overall": CLEAN43022,
        "clean43122_overall": CLEAN43122,
        "true100": true100,
        "frozen95": frozen95,
        "true100_status": true_status,
        "frozen95_status": frozen_status,
        "next_action": next_action,
        "staged_endpoint_matched_noaoa_contrast": staged.get("interpretation") if isinstance(staged, dict) else None,
        "separation_statement": "Official Overall includes SuperGLUE and AoA in leaderboard units; equal7/equal6 are only no-AoA internal screening metrics.",
    }
    _public_path('experiments/archive/compact_experience/data/inverse_priority_full_eval_interpretation').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research inverse-priority full-evaluation interpretation", "",
        f"Visible leader: {VISIBLE_LEADER:.3f}; clean seed43022: {CLEAN43022:.6f}; clean seed43122: {CLEAN43122:.6f}.", "",
        "Official Overall includes SuperGLUE and AoA; no-AoA equal7/equal6 are separate internal measurements.", "",
    ]
    for label, row, st in [("true 100M", true100, true_status), ("frozen 95M", frozen95, frozen_status)]:
        if row:
            s = row.get("task_scores") or {}
            lines.append(f"## {label}: `{st}`")
            lines.append("")
            lines.append(f"- target: `{row.get('target')}`; endpoint: `{row.get('endpoint')}`; endpoint_frozen: `{row.get('endpoint_frozen')}`")
            lines.append(f"- Overall: {row.get('overall')}; Δvisible leader: {row.get('delta_vs_visible_leader')}; SuperGLUE: {row.get('superglue_mean')}; AoA raw: {row.get('aoa_raw_correlation')}; AoA lb: {row.get('aoa_leaderboard_score')}; submit-ready: {row.get('submit_ready_overall')}")
            lines.append(f"- no-AoA columns: BLiMP {s.get('BLiMP')}, Supplement {s.get('Supplement')}, EWoK {s.get('EWoK')}, Entity {s.get('Entity')}, COMPS {s.get('COMPS')}, GlobalPIQA {s.get('GlobalPIQA')}, Reading {s.get('Reading')}")
            lines.append("")
        else:
            lines.append(f"## {label}: missing")
            lines.append("")
    lines.append(f"Next action: `{next_action}`.")
    lines.append("")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "true100_status": true_status, "frozen95_status": frozen_status, "next_action": next_action}, indent=2), flush=True)


if __name__ == "__main__":
    main()
