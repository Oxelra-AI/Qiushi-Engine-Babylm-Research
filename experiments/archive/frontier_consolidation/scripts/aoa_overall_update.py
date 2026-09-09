#!/usr/bin/env python3
"""research update: official-rowcount AoA and Overall arithmetic for reinvest seeds."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
OUT_DIR = ROOT / "experiments/archive/frontier_consolidation/data/aoa_overall_update"
OUT_JSON = OUT_DIR / "aoa_overall_update.json"
OUT_MD = ROOT / "research/notes/frontier_consolidation/aoa_overall_update.md"
FULL_43022 = ROOT / "experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json"
AOA_SUMMARY = ROOT / "experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds/official_rowcount_aoa_run_summary.json"
PROJ_43122 = ROOT / "experiments/archive/frontier_consolidation/data/seed43122_fast_projection/seed43122_fast_projection.json"
LIVE_PROBE = ROOT / "experiments/archive/frontier_consolidation/data/live_leaderboard_scoring_probe/live_leaderboard_scoring_probe.json"
NINE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = jload(FULL_43022)
    aoa = jload(AOA_SUMMARY)
    proj = jload(PROJ_43122)
    live = jload(LIVE_PROBE)
    recs = {r["target"]: r for r in aoa["records"]}
    scores = dict(full["targets"]["compact_view_reinvest"]["scores"])
    scores["AoA"] = recs["reinvest_seed43022"]["aoa"]
    overall = sum(float(scores[k]) for k in NINE) / 9.0
    leader = 41.8
    result = {
        "status": "AOA_OVERALL_UPDATE",
        "inputs": {"seed43022_full": str(FULL_43022), "aoa_summary": str(AOA_SUMMARY), "seed43122_projection": str(PROJ_43122), "live_probe": str(LIVE_PROBE)},
        "seed43022_scores_with_official_rowcount_aoa": scores,
        "seed43022_overall": overall,
        "delta_vs_visible_41p8_leader": overall - leader,
        "aoa_records": recs,
        "rowcount_verification": {k: {"row_count_values": v["row_count_values"], "num_rows": v["num_rows"], "num_steps": None, "aoa": v["aoa"], "curve_fitness_record": v["curve_fitness_record"]} for k, v in recs.items()},
        "seed43122_fast_projection_key_numbers": {
            "seven_sum_excluding_superglue_aoa": proj["scenarios"]["optimistic_no_fast_to_full_penalty"]["seven_sum_excluding_superglue_aoa"],
            "required_superglue_if_aoa0_optimistic": proj["scenarios"]["optimistic_no_fast_to_full_penalty"]["required_SuperGLUE_to_reach_41p8_if_AoA_0"],
            "required_superglue_if_aoa0_calibrated": proj["scenarios"]["seed43022_fast_to_full_calibrated"]["required_SuperGLUE_to_reach_41p8_if_AoA_0"],
            "required_aoa_if_superglue_seed43022_optimistic": proj["scenarios"]["optimistic_no_fast_to_full_penalty"]["required_AoA_leaderboard_units_to_reach_41p8_if_SuperGLUE_equals_seed43022"],
            "required_aoa_if_superglue_seed43022_calibrated": proj["scenarios"]["seed43022_fast_to_full_calibrated"]["required_AoA_leaderboard_units_to_reach_41p8_if_SuperGLUE_equals_seed43022"],
        },
        "live_leader_top": live.get("strict_small_top_rows", [None])[0] if isinstance(live.get("strict_small_top_rows"), list) else None,
        "interpretation": "Seed43022 compact_view_reinvest remains above the visible 41.8 leader under the current live 8005-row AoA convention because recomputed AoA maps to 0. Seed43122 also has 8005-row AoA=0, but research fast evidence means this does not by itself make seed43122 a likely above-41.8 replicate; its non-AoA surface would need an unusually high SuperGLUE score or positive AoA, neither currently observed.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — official-rowcount AoA and reinvest Overall update\n\n"]
    lines.append("The repaired AoA run used min_context=0, exactly 8005 rows per checkpoint, and 19 checkpoint steps for both compact_view_reinvest seeds.\n\n")
    lines.append("## Seed43022 submission-facing arithmetic\n")
    for k in NINE:
        lines.append(f"- {k}: {scores[k]:.12g}\n")
    lines.append(f"- Overall: {overall:.12g}\n")
    lines.append(f"- Delta versus visible 41.8 leader: {overall - leader:+.12g}\n\n")
    lines.append("## AoA rowcount result\n")
    for target, r in recs.items():
        lines.append(f"- {target}: AoA={r['aoa']}, row_count_values={r['row_count_values']}, num_rows={r['num_rows']}, curve_fitness={r['curve_fitness_record']}\n")
    lines.append("\n## Consequence for seed43122\n")
    nums = result["seed43122_fast_projection_key_numbers"]
    lines.append(f"Seed43122 also has official-rowcount AoA=0, but its fast seven-column sum excluding SuperGLUE/AoA is {nums['seven_sum_excluding_superglue_aoa']:.6f}. With AoA=0 it would need SuperGLUE {nums['required_superglue_if_aoa0_optimistic']:.6f} under the optimistic fast projection, or {nums['required_superglue_if_aoa0_calibrated']:.6f} after applying seed43022 fast-to-full shifts. Thus AoA=0 removes the catastrophic failure mode but does not restore seed43122 as a likely above-41.8 replicate.\n\n")
    lines.append("## Scientific interpretation\n")
    lines.append("The endpoint is now stronger as a single submission-facing model: its load-bearing AoA convention has been recomputed under the live 8005-row convention and remains 0. The broader recipe, however, is not yet stable: the second seed's no-AoA surface is broadly weaker, so the active research question shifts from whether seed43022 is valid to why the density mechanism produces a high upper-tail seed and how to make it reliable.\n")
    lines.append(f"\nMachine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
