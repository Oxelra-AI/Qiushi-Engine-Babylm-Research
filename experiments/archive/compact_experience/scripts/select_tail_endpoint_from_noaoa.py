#!/usr/bin/env python3
"""Select the strongest research clean-tail restart endpoint from no-AoA columns."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time

WORKSPACE = _public_path('experiments/archive/compact_experience')
NOAOA_SUMMARY = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/tail_restart_noaoa_summary.json')
TRAJ = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/clean_tail_restart_ladder_trajectory_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/tail_endpoint_selection/selected_tail_endpoint.json')
NOTE = _public_path('research/notes/compact_experience/selected_tail_endpoint.md')
ENDPOINTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CLEAN_EQUAL7 = 43.112857142857145
VISIBLE_LEADER = 41.8


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_rows() -> list[dict]:
    if NOAOA_SUMMARY.exists():
        payload = json.loads(NOAOA_SUMMARY.read_text(encoding="utf-8"))
        rows = payload.get("ranked_results") or []
        if rows:
            return rows
    obj = json.loads(TRAJ.read_text(encoding="utf-8"))
    rows = []
    for ck in ENDPOINTS:
        r = (obj.get("table") or {}).get(ck)
        if not isinstance(r, dict) or r.get("equal7_full_eval") is None:
            continue
        rows.append({
            "checkpoint": ck,
            "equal7": float(r["equal7_full_eval"]),
            "delta_vs_clean_equal7": float(r["equal7_full_eval"]) - CLEAN_EQUAL7,
            "scores": {c: float(r.get(c, 0.0)) for c in COLUMNS},
        })
    rows.sort(key=lambda x: x["equal7"], reverse=True)
    return rows


def main() -> None:
    rows = [r for r in load_rows() if r.get("checkpoint") in ENDPOINTS]
    if not rows:
        raise RuntimeError("No eligible restart endpoint rows found")
    rows.sort(key=lambda x: float(x["equal7"]), reverse=True)
    selected = rows[0]
    # Overall = (7*equal7 + SuperGLUE + AoA_leaderboard) / 9.
    required_sg_plus_aoa_for_visible_leader = 9 * VISIBLE_LEADER - 7 * float(selected["equal7"])
    payload = {
        "status": "TAIL_ENDPOINT_SELECTED_FROM_NOAOA",
        "created_utc": now(),
        "selection_rule": "Choose the highest equal7 score among the four eligible fresh-tail endpoints chck_85M, chck_90M, chck_95M, chck_100M before measuring SuperGLUE or AoA.",
        "selected_endpoint": selected["checkpoint"],
        "selected_equal7": selected["equal7"],
        "selected_delta_vs_clean_equal7": selected.get("delta_vs_clean_equal7"),
        "selected_scores": selected.get("scores"),
        "required_superglue_plus_aoa_for_visible_leader": required_sg_plus_aoa_for_visible_leader,
        "ranked_results": rows,
        "noaoa_summary": str(NOAOA_SUMMARY),
        "trajectory_summary": str(TRAJ),
        "non_leakage_statement": "Endpoint choice uses only post-training no-AoA columns for these frozen restart endpoints; AoA and SuperGLUE are measured only after the endpoint is fixed.",
    }
    _public_path('experiments/archive/compact_experience/data/tail_endpoint_selection').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research selected clean-tail restart endpoint",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        f"Selected endpoint: `{payload['selected_endpoint']}` with equal7 {payload['selected_equal7']:.6f} (Δ vs clean {payload['selected_delta_vs_clean_equal7']:+.6f}).",
        f"To exceed visible Overall {VISIBLE_LEADER:.3f} with this equal7, SuperGLUE + AoA_leaderboard must exceed {required_sg_plus_aoa_for_visible_leader:.6f}.",
        "",
        "| endpoint | equal7 | Δ vs clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r.get("scores") or {}
        lines.append(f"| {r['checkpoint']} | {float(r['equal7']):.4f} | {float(r.get('delta_vs_clean_equal7', 0.0)):+.4f} | {s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} |")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "selected_endpoint": selected["checkpoint"], "selected_equal7": selected["equal7"], "required_sg_plus_aoa_for_visible_leader": required_sg_plus_aoa_for_visible_leader}, indent=2), flush=True)


if __name__ == "__main__":
    main()
