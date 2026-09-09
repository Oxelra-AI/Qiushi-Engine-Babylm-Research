#!/usr/bin/env python3
"""Select the strongest research mask arm/endpoint from no-AoA trajectories.

This script is intentionally post-training and no-AoA-only.  It compares the four
matched mask-target distributions against the uniform restart reference and records
whether any target redistribution adds beyond the common optimizer/LR rephasing.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any

WORKSPACE = _public_path('experiments/archive/compact_experience')
STUDY = _public_path('experiments/archive/compact_experience')
DEFAULT_ROOTS = [
    _public_path('data/external/mask_noaoa_eval'),
    _public_path('experiments/archive/compact_experience/data/mask_eval'),
]
OUT = _public_path('experiments/archive/compact_experience/data/mask_endpoint_selection/selected_mask_endpoint.json')
NOTE = _public_path('research/notes/compact_experience/selected_mask_endpoint.md')
ARMS = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]
ENDPOINTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
TARGET_RECOVERY = ["EWoK", "COMPS", "GlobalPIQA"]
PRESERVE = ["BLiMP", "Supplement", "Entity", "Reading"]
CLEAN = {"BLiMP":66.84,"Supplement":62.84,"EWoK":50.19,"Entity":25.76,"COMPS":51.78,"GlobalPIQA":36.62,"Reading":7.76,"equal7":43.112857142857145}
VISIBLE_LEADER = 41.8


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def find_traj(root: pathlib.Path, arm: str) -> pathlib.Path | None:
    target = f"mask_{arm}"
    for p in [root / f"{target}_trajectory_summary.json", root / target / f"{target}_trajectory_summary.json"]:
        if p.exists():
            return p
    return None


def load_metrics(arm: str) -> dict[str, Any] | None:
    p = _public_path('experiments/archive/compact_experience/training/runs') / f"mask_{arm}" / "scientific_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {k: m.get(k) for k in ["mask_mode", "masked_token_budget_ratio", "high_priority_fraction_among_selected_words", "loss_last", "continuation_word_exposure", "actual_total_word_exposure", "train_rng_seed"]}


def load_rows(roots: list[pathlib.Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARMS:
        traj = None
        for root in roots:
            traj = find_traj(root, arm)
            if traj is not None:
                break
        if traj is None:
            missing.append(arm)
            continue
        obj = json.loads(traj.read_text(encoding="utf-8"))
        table = obj.get("table") or {}
        for ep in ENDPOINTS:
            r = table.get(ep)
            if not isinstance(r, dict) or r.get("equal7_full_eval") is None:
                continue
            scores = {c: float(r.get(c, 0.0)) for c in COLUMNS}
            rows.append({
                "arm": arm,
                "checkpoint": ep,
                "equal7": float(r["equal7_full_eval"]),
                "delta_vs_clean_equal7": float(r["equal7_full_eval"]) - CLEAN["equal7"],
                "scores": scores,
                "target_recovery_sum_delta_vs_clean": sum(scores[c] - CLEAN[c] for c in TARGET_RECOVERY),
                "preserve_sum_delta_vs_clean": sum(scores[c] - CLEAN[c] for c in PRESERVE),
                "trajectory_summary": str(traj),
                "metrics": load_metrics(arm),
            })
    rows.sort(key=lambda x: x["equal7"], reverse=True)
    return rows, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noaoa_root", action="append", default=[], help="May be repeated; defaults to staging then data.")
    args = ap.parse_args()
    roots = [pathlib.Path(x) for x in args.noaoa_root] if args.noaoa_root else DEFAULT_ROOTS
    rows, missing = load_rows(roots)
    if not rows:
        raise RuntimeError("No mask no-AoA rows found")
    best = rows[0]
    uniform_rows = [r for r in rows if r["arm"] == "uniform_control"]
    best_uniform = max(uniform_rows, key=lambda r: r["equal7"]) if uniform_rows else None
    evidence_rows = [r for r in rows if r["arm"] == "evidence_visible"]
    best_evidence = max(evidence_rows, key=lambda r: r["equal7"]) if evidence_rows else None
    contrasts = {}
    if best_uniform is not None:
        for r in rows:
            if r["arm"] == "uniform_control":
                continue
            contrasts[f"{r['arm']}_{r['checkpoint']}_minus_uniform_best"] = {
                "equal7": r["equal7"] - best_uniform["equal7"],
                "target_recovery_sum": r["target_recovery_sum_delta_vs_clean"] - best_uniform["target_recovery_sum_delta_vs_clean"],
                "preserve_sum": r["preserve_sum_delta_vs_clean"] - best_uniform["preserve_sum_delta_vs_clean"],
                "per_column": {c: r["scores"][c] - best_uniform["scores"][c] for c in COLUMNS},
            }
    selected_for_full_eval = best
    interpretation = "best_noaoa_arm_endpoint"
    if best_uniform and best["arm"] == "uniform_control":
        interpretation = "mask_targeting_did_not_beat_uniform_restart_on_noaoa"
    elif best_uniform and best["equal7"] <= best_uniform["equal7"]:
        interpretation = "mask_targeting_not_additive_beyond_uniform_restart"
    elif best_uniform:
        interpretation = "mask_targeting_candidate_adds_beyond_uniform_restart_on_noaoa"
    if best_evidence and best_uniform:
        evidence_minus_uniform = best_evidence["equal7"] - best_uniform["equal7"]
    else:
        evidence_minus_uniform = None
    required_sg_plus_aoa_for_visible_leader = 9 * VISIBLE_LEADER - 7 * selected_for_full_eval["equal7"]
    payload = {
        "status": "MASK_ENDPOINT_SELECTED_FROM_NOAOA",
        "created_utc": now(),
        "selection_rule": "Rank all four mask arms and four eligible endpoints by no-AoA equal7; compare the best target-redistribution arm against the uniform restart reference before any SuperGLUE/AoA measurement.",
        "roots_searched": [str(r) for r in roots],
        "missing_arms": missing,
        "selected_arm": selected_for_full_eval["arm"],
        "selected_endpoint": selected_for_full_eval["checkpoint"],
        "selected_equal7": selected_for_full_eval["equal7"],
        "selected_delta_vs_clean_equal7": selected_for_full_eval["delta_vs_clean_equal7"],
        "selected_scores": selected_for_full_eval["scores"],
        "required_superglue_plus_aoa_for_visible_leader": required_sg_plus_aoa_for_visible_leader,
        "best_uniform": best_uniform,
        "best_evidence_visible": best_evidence,
        "best_evidence_minus_best_uniform_equal7": evidence_minus_uniform,
        "interpretation": interpretation,
        "ranked_results": rows,
        "contrasts_vs_uniform_best": contrasts,
        "non_leakage_statement": "Selection uses post-training no-AoA columns only; AoA and SuperGLUE are not used to choose the arm/endpoint.",
    }
    _public_path('experiments/archive/compact_experience/data/mask_endpoint_selection').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research selected mask endpoint", "", f"Created UTC: {payload['created_utc']}", "",
        f"Selected: `{payload['selected_arm']}` `{payload['selected_endpoint']}` equal7 {payload['selected_equal7']:.6f}; interpretation `{interpretation}`.",
        f"Best evidence-visible minus best uniform equal7: {evidence_minus_uniform}.",
        "", "| arm | endpoint | equal7 | Δclean | Δtarget recovery | Δpreserve | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["scores"]
        lines.append(f"| {r['arm']} | {r['checkpoint']} | {r['equal7']:.4f} | {r['delta_vs_clean_equal7']:+.4f} | {r['target_recovery_sum_delta_vs_clean']:+.4f} | {r['preserve_sum_delta_vs_clean']:+.4f} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Reading']:.3f} |")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "selected_arm": payload["selected_arm"], "selected_endpoint": payload["selected_endpoint"], "selected_equal7": payload["selected_equal7"], "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
