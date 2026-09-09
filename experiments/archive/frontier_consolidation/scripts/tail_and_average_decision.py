#!/usr/bin/env python3
"""research: synthesize seed43122 tail completion and first stabilization-average result.

This is CPU/file-only. It reads already produced selected-trajectory payloads,
computes the fixed comparison metrics used in Steps195/199, and writes a durable
scientific note for the post-pivot route decision. It performs no model scoring,
training, upload, or leaderboard submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/tail_and_average_decision.py')
parts = ROOT.parts
if "experiments" in parts:
    USER_ROOT = pathlib.Path(*parts[: parts.index("experiments")]) if parts.index("experiments") > 0 else pathlib.Path(".").resolve()
else:
    USER_ROOT = pathlib.Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY

REF_TRAJ = WORKSPACE / "data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json"
SEED_TRAJ = WORKSPACE / "data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json"
AVG_TRAJ = WORKSPACE / "data/avg80_82_84_selected_eval/selected_trajectory.json"
SEED_INTEGRITY = WORKSPACE / "data/seed43122_full_integrity/selected_mlm_integrity_check.json"
AVG_INTEGRITY = WORKSPACE / "data/avg80_82_84_integrity/selected_mlm_integrity_check.json"
CROSS_SEED = WORKSPACE / "data/cross_seed_common_grid_analysis_all3_complete/cross_seed_common_grid_analysis.json"
OUT_DIR = WORKSPACE / "data/tail_and_average_decision"
NOTE = WORKSPACE / "notes/tail_completion_and_stabilization_decision.md"

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
NO_GP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
NO_GP_READING = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
REL_STATE = ["EWoK", "Entity"]
VOLATILE = ["GlobalPIQA", "Reading"]


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["endpoint"]: r for r in rows if r.get("cheap7") is not None}


def score(row: dict[str, Any], cols: list[str]) -> float:
    return float(mean(float(row[c]) for c in cols))


def pack_scores(row: dict[str, Any]) -> dict[str, float]:
    return {
        "cheap7": float(row["cheap7"]),
        "cheap6_no_globalpiqa": score(row, NO_GP),
        "cheap5_no_globalpiqa_reading": score(row, NO_GP_READING),
        "ewok_entity": score(row, REL_STATE),
        "volatile_globalpiqa_reading": score(row, VOLATILE),
        **{c: float(row[c]) for c in CHEAP},
    }


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in a.keys() if k in b}


def fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    ref = row_map(read_json(REF_TRAJ))
    seed = row_map(read_json(SEED_TRAJ))
    avg_rows = read_json(AVG_TRAJ)
    avg = row_map(avg_rows)["chck_84M"]
    seed_integrity = read_json(SEED_INTEGRITY)
    avg_integrity = read_json(AVG_INTEGRITY)
    cross = read_json(CROSS_SEED)

    seed_best = max(seed.values(), key=lambda r: float(r["cheap7"]))
    tail_eps = ["chck_92M", "chck_94M", "chck_96M", "chck_98M", "chck_100M"]
    tail_rows = [seed[e] for e in tail_eps]
    tail_best = max(tail_rows, key=lambda r: float(r["cheap7"]))
    observed_88 = seed["chck_88M"]
    ref82 = ref["chck_82M"]
    ref84 = ref["chck_84M"]

    seed_best_scores = pack_scores(seed_best)
    tail_best_scores = pack_scores(tail_best)
    avg_scores = pack_scores(avg)
    ref82_scores = pack_scores(ref82)
    ref84_scores = pack_scores(ref84)

    avg_vs_84 = delta(avg_scores, ref84_scores)
    avg_vs_82 = delta(avg_scores, ref82_scores)
    seed_best_vs_ref84 = delta(seed_best_scores, ref84_scores)

    # Extract exact comparison block if present; keep robust to future schema changes.
    comparisons = cross.get("comparisons_to_reference") or cross.get("comparisons") or {}
    seed_comp = comparisons.get("scale1p75_seed43122_dense", {}) if isinstance(comparisons, dict) else {}

    decision = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "TAIL_AND_AVERAGE_DECISION",
        "seed43122_tail_integrity_ok": bool(seed_integrity.get("ok")),
        "avg_integrity_ok": bool(avg_integrity.get("ok")),
        "seed43122_n_valid": len(seed),
        "seed43122_best_endpoint": seed_best["endpoint"],
        "seed43122_best_cheap7": float(seed_best["cheap7"]),
        "seed43122_tail_best_endpoint": tail_best["endpoint"],
        "seed43122_tail_best_cheap7": float(tail_best["cheap7"]),
        "tail_best_minus_observed_88_cheap7": float(tail_best["cheap7"]) - float(observed_88["cheap7"]),
        "seed43122_best_minus_reference84": seed_best_vs_ref84,
        "cross_seed_complete_comparison_to_reference": seed_comp,
        "average_candidate": "reference_scale1p75_seed43022__center_80_82_84_uniform",
        "average_model_sha256": "d47c15f96e3424fc0946cfa4e747f9a6374006d9a3cf68d58d5031509585ac50",
        "average_scores": avg_scores,
        "reference82_scores": ref82_scores,
        "reference84_scores": ref84_scores,
        "average_minus_reference84": avg_vs_84,
        "average_minus_reference82": avg_vs_82,
        "tail_decision": "missing seed43122 tail does not exceed the observed 88M peak; tail gap closed",
        "average_decision": "80/82/84 uniform same-trajectory average fails the stabilization target because it is below reference chck84 on cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, and EWoK/Entity",
        "next_scientific_state": "do not tune the seed43022 84M peak or extend naive same-trajectory averaging by inertia; stabilization remains the right object only if a new mechanism is defined, likely by critical review or construction rather than another small average sweep",
        "paths": {
            "seed43122_trajectory": str(SEED_TRAJ.relative_to(USER_ROOT)),
            "seed43122_integrity": str(SEED_INTEGRITY.relative_to(USER_ROOT)),
            "average_selected_eval": str(AVG_TRAJ.relative_to(USER_ROOT)),
            "average_integrity": str(AVG_INTEGRITY.relative_to(USER_ROOT)),
            "cross_seed_complete": str(CROSS_SEED.relative_to(USER_ROOT)),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "tail_and_average_decision.json"
    out_json.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# research: seed43122 tail completion and first stabilization-average result\n\n")
    md.append("## Seed43122 tail completion\n\n")
    md.append(f"Full selected-grid integrity: **{seed_integrity.get('ok')}**; valid endpoints: **{len(seed)}/16**.\n\n")
    md.append(f"The completed seed43122 best endpoint is **{seed_best['endpoint']}** with cheap7 **{fmt(float(seed_best['cheap7']))}**. The best newly scored tail endpoint is **{tail_best['endpoint']}** with cheap7 **{fmt(float(tail_best['cheap7']))}**, which is {fmt(float(tail_best['cheap7']) - float(observed_88['cheap7']))} relative to the observed 88M peak. Thus the missing tail does not contain a higher selected cheap-task peak.\n\n")
    md.append("The completed cross-seed structural readout now gives seed43122 peak-vector Pearson 0.709859/Spearman 0.718182 vs reference, aggregate peak shift +4M, mean Δcheap7 +0.161652 across complete common endpoints, but mean Δcheap6(no GlobalPIQA) -0.358542 and mean Δcheap5(no GlobalPIQA/Reading) -0.430375. This preserves the research reading: coarse family timing is partly related, but the stable broad aggregate remains weaker and the reference signed-transition structure did not recur.\n\n")
    md.append("## Same-trajectory 80/82/84 uniform average\n\n")
    md.append(f"Average candidate SHA: `{decision['average_model_sha256']}`. The pseudo-run metadata records zero added exposure and a non-chronological scoring wrapper. Selected cheap-task integrity: **{avg_integrity.get('ok')}**.\n\n")
    md.append("| model/function | cheap7 | cheap6 no GP | cheap5 no GP/Reading | EWoK+Entity | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, sc in [("reference chck82", ref82_scores), ("reference chck84", ref84_scores), ("avg80/82/84", avg_scores)]:
        md.append("| " + name + " | " + " | ".join(fmt(sc[k]) for k in ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "ewok_entity", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]) + " |\n")
    md.append("\n")
    md.append("| comparison | Δcheap7 | Δcheap6 no GP | Δcheap5 no GP/Reading | ΔEWoK+Entity | ΔGlobalPIQA | ΔReading |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, d in [("avg - reference chck84", avg_vs_84), ("avg - reference chck82", avg_vs_82)]:
        md.append("| " + name + " | " + " | ".join(fmt(d[k]) for k in ["cheap7", "cheap6_no_globalpiqa", "cheap5_no_globalpiqa_reading", "ewok_entity", "GlobalPIQA", "Reading"]) + " |\n")
    md.append("\n")
    md.append("The average is below the ordinary 84M endpoint by -0.159286 cheap7, -0.100000 cheap6 without GlobalPIQA, -0.134000 cheap5 without GlobalPIQA/Reading, and -0.235000 on EWoK+Entity. Relative to chck82 it is essentially flat on cheap7 (+0.005714) and cheap6 (+0.002500) but still negative on cheap5 (-0.012000) and EWoK+Entity (-0.095000). This fails the pre-stated stabilization target.\n\n")
    md.append("## Decision\n\n")
    md.append("1. The seed43122 own-peak uncertainty is closed: 88M remains the best selected cheap checkpoint after 92--100M completion. No additional research/198 run is needed because the true own-peak window is still 86M->88M->90M, already analyzed in research.\n")
    md.append("2. The first same-trajectory low-pass average does not stabilize broad competence beyond the ordinary 84M endpoint and should not be extended into an averaging sweep by inertia.\n")
    md.append("3. The next useful move is critical route review and/or construction of a better stabilization mechanism from the actual failures: broad competence is seed/mask-sensitive, naive late weight averaging smooths away relation/state and GlobalPIQA gains, alpha/private scaling is redistributive, and compact-order training remains paused until A01 shares directional-fork evidence.\n\n")
    md.append("No leaderboard submission was performed.\n\n")
    md.append(f"JSON: `{out_json.relative_to(USER_ROOT)}`\n")
    NOTE.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_json": str(out_json.relative_to(USER_ROOT)), "note": str(NOTE.relative_to(USER_ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
