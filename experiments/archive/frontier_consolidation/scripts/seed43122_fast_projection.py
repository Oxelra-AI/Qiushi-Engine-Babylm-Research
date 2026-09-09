#!/usr/bin/env python3
"""research: project seed43122 compact_view_reinvest viability from cheap fast evidence.

This is CPU-only arithmetic. It does not replace official full evaluation and does not
use AoA results before the managed official-rowcount AoA task returns. Its purpose is
to decide what full seed43122 evaluation could still distinguish after the fast no-AoA
screen became readable.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from statistics import mean

ROOT = _public_path('experiments/archive/frontier_consolidation')
# parents[1] = workspace for experiments/archive/frontier_consolidation/scripts/...
USER_ROOT = Path.cwd()

P_SEED43122_FAST = USER_ROOT / "experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/per_target/compact_view_reinvest_seed43122.json"
P_SEED43022_FAST = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json"
P_SEED43022_FULL = USER_ROOT / "experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json"
OUT_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/seed43122_fast_projection"
OUT_JSON = OUT_DIR / "seed43122_fast_projection.json"
OUT_MD = USER_ROOT / "research/notes/frontier_consolidation/seed43122_fast_projection.md"

LEADER = {
    "Overall": 41.8,
    "BLiMP": 67.2,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.0,
}
CLEAN = {
    "Overall": 41.34429066479573,
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "SuperGLUE": 70.30861598316156,
    "Reading": 7.76,
    "AoA": 0.0,
}
NEARBY_SUPERGLUE = {
    "compact_view_core": 68.9012,
    "visible_leader": 69.79,
    "compact_experience_clean_qwen": 70.30861598316156,
    "compact_repeat_core": 70.62444886622056,
    "compact_view_reinvest_seed43022": 71.38107147224343,
}

NINE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
SEVEN_NO_SG_AOA = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def overall(scores: dict[str, float]) -> float:
    return sum(float(scores[k]) for k in NINE) / 9.0

def threshold_for_missing_sum(target_overall: float, known_sum: float) -> float:
    return target_overall * 9.0 - known_sum


def main() -> None:
    f43122 = jload(P_SEED43122_FAST)
    f43022 = jload(P_SEED43022_FAST)
    full43022 = jload(P_SEED43022_FULL)

    t43122 = f43122["tasks"]
    fast43122 = {
        "BLiMP": t43122["BLiMP"]["score"],
        "Supplement": t43122["Supplement"]["score"],
        "EWoK": t43122["EWoK"]["score"],
    # official full Entity column uses the full entity_tracking set; seed43122 fast screen
        # already ran Entity_full for this purpose.
        "Entity": t43122["Entity_full"]["score"],
        "COMPS": t43122["COMPS"]["score"],
        "GlobalPIQA": mean([t43122["GlobalPIQA_parallel"]["score"], t43122["GlobalPIQA_nonparallel"]["score"]]),
        "Reading": t43122["Reading"]["scores"]["Reading"],
    }

    fast43022_table = f43022["table"]["compact_view_reinvest"]
    fast43022 = {
        "BLiMP": fast43022_table["BLiMP"],
        "Supplement": fast43022_table["Supplement"],
        "EWoK": fast43022_table["EWoK"],
        "Entity": fast43022_table["Entity_full"],
        "COMPS": fast43022_table["COMPS"],
        "GlobalPIQA": fast43022_table["GlobalPIQA_mean"],
        "Reading": fast43022_table["Reading"],
    }
    full43022_scores = full43022["targets"]["compact_view_reinvest"]["scores"]
    full43022_7 = {k: full43022_scores[k] for k in SEVEN_NO_SG_AOA}
    fast_to_full_delta_seed43022 = {k: full43022_7[k] - fast43022[k] for k in SEVEN_NO_SG_AOA}

    # Two cheap projections: a deliberately optimistic one that treats fast values as full;
    # and a calibrated one that applies the seed43022 fast->full deltas task-wise.
    optimistic_7 = dict(fast43122)
    calibrated_7 = {k: fast43122[k] + fast_to_full_delta_seed43022[k] for k in SEVEN_NO_SG_AOA}

    scenarios = {}
    for name, seven in [("optimistic_no_fast_to_full_penalty", optimistic_7), ("seed43022_fast_to_full_calibrated", calibrated_7)]:
        seven_sum = sum(seven.values())
        required_sg_if_aoa0 = threshold_for_missing_sum(LEADER["Overall"], seven_sum + 0.0)
        required_aoa_if_sg_seed43022 = threshold_for_missing_sum(
            LEADER["Overall"], seven_sum + NEARBY_SUPERGLUE["compact_view_reinvest_seed43022"]
        )
        required_aoa_if_sg_best_nearby = threshold_for_missing_sum(
            LEADER["Overall"], seven_sum + max(NEARBY_SUPERGLUE.values())
        )
        required_sg_vs_clean_if_aoa0 = threshold_for_missing_sum(CLEAN["Overall"], seven_sum + 0.0)
        required_aoa_vs_clean_if_sg_seed43022 = threshold_for_missing_sum(
            CLEAN["Overall"], seven_sum + NEARBY_SUPERGLUE["compact_view_reinvest_seed43022"]
        )
        table = []
        for sg_label, sg in NEARBY_SUPERGLUE.items():
            for aoa in [-10, -5, 0, 5, 10, 15]:
                scores = dict(seven)
                scores.update({"SuperGLUE": sg, "AoA": aoa})
                ov = overall(scores)
                table.append({
                    "superglue_source": sg_label,
                    "SuperGLUE": sg,
                    "AoA": aoa,
                    "Overall": ov,
                    "delta_vs_41p8": ov - LEADER["Overall"],
                    "delta_vs_clean": ov - CLEAN["Overall"],
                })
        scenarios[name] = {
            "seven_column_scores_excluding_superglue_aoa": seven,
            "seven_sum_excluding_superglue_aoa": seven_sum,
            "seven_delta_vs_seed43022_full": seven_sum - sum(full43022_7.values()),
            "required_SuperGLUE_to_reach_41p8_if_AoA_0": required_sg_if_aoa0,
            "required_AoA_leaderboard_units_to_reach_41p8_if_SuperGLUE_equals_seed43022": required_aoa_if_sg_seed43022,
            "required_AoA_leaderboard_units_to_reach_41p8_if_SuperGLUE_equals_best_nearby": required_aoa_if_sg_best_nearby,
            "required_SuperGLUE_to_reach_clean_if_AoA_0": required_sg_vs_clean_if_aoa0,
            "required_AoA_leaderboard_units_to_reach_clean_if_SuperGLUE_equals_seed43022": required_aoa_vs_clean_if_sg_seed43022,
            "scenario_table": table,
        }

    seed43122_fast_vs_refs = {
        "minus_seed43022_fast_or_full_where_available": {
            k: fast43122[k] - fast43022[k] for k in SEVEN_NO_SG_AOA
        },
        "minus_seed43022_full": {
            k: fast43122[k] - full43022_7[k] for k in SEVEN_NO_SG_AOA
        },
        "minus_visible_leader": {
            k: fast43122[k] - LEADER[k] for k in SEVEN_NO_SG_AOA
        },
        "minus_clean_qwen": {
            k: fast43122[k] - CLEAN[k] for k in SEVEN_NO_SG_AOA
        },
    }

    result = {
        "status": "SEED43122_FAST_PROJECTION",
        "purpose": "Use newly-readable seed43122 fast/no-AoA evidence to decide whether full second-seed surface evaluation could still be a minimum reliable expensive action after official-rowcount AoA returns.",
        "inputs": {
            "seed43122_fast": str(P_SEED43122_FAST),
            "seed43022_fast": str(P_SEED43022_FAST),
            "seed43022_full": str(P_SEED43022_FULL),
        },
        "seed43122_fast_official_relevant_7cols": fast43122,
        "seed43022_fast_official_relevant_7cols": fast43022,
        "seed43022_full_official_relevant_7cols": full43022_7,
        "seed43022_fast_to_full_delta": fast_to_full_delta_seed43022,
        "seed43122_fast_vs_refs": seed43122_fast_vs_refs,
        "nearby_superglue_values": NEARBY_SUPERGLUE,
        "scenarios": scenarios,
        "interpretation": {
            "aoa_not_measured_here": True,
            "main_result": "The seed43122 cheap screen is materially weaker than seed43022 on the official-relevant non-SuperGLUE/non-AoA surface. If AoA maps to 0 and SuperGLUE is in the observed nearby range (~68.9-71.4), seed43122 does not plausibly clear the 41.8 leader; it would need either an unusually high SuperGLUE score or a positive AoA contribution.",
            "expensive_work_implication": "After the 8005-row AoA result becomes available, full seed43122 evaluation should be launched only if the measured 8005-row AoA is positive enough, or if a scientific replication question still justifies the cost despite low SOTA probability. AoA merely non-significant at 0 is no longer enough by itself to make full seed43122 surface evaluation a likely SOTA-confirming action.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — seed43122 fast-screen projection before full-surface spending\n")
    lines.append("This is CPU-only arithmetic from the newly readable A01 seed43122 fast/no-AoA file. It does **not** use or infer the running official-rowcount AoA task.\n")
    lines.append("## Seed43122 official-relevant fast columns (SuperGLUE and AoA missing)\n")
    for k in SEVEN_NO_SG_AOA:
        lines.append(f"- {k}: {fast43122[k]:.6g}\n")
    lines.append("\n## Change vs seed43022 compact_view_reinvest\n")
    lines.append("Against seed43022 fast columns (using full Entity where available):\n")
    for k, v in seed43122_fast_vs_refs["minus_seed43022_fast_or_full_where_available"].items():
        lines.append(f"- {k}: {v:+.6g}\n")
    lines.append("\nAgainst seed43022 full columns (where direct full values exist):\n")
    for k, v in seed43122_fast_vs_refs["minus_seed43022_full"].items():
        lines.append(f"- {k}: {v:+.6g}\n")
    lines.append("\n## What would seed43122 need to beat 41.8?\n")
    for name, sc in scenarios.items():
        lines.append(f"\n### {name}\n")
        lines.append(f"7-column sum excluding SuperGLUE/AoA: {sc['seven_sum_excluding_superglue_aoa']:.6f}\n")
        lines.append(f"Delta of those 7 columns vs seed43022 full: {sc['seven_delta_vs_seed43022_full']:+.6f}\n")
        lines.append(f"Required SuperGLUE if AoA=0: {sc['required_SuperGLUE_to_reach_41p8_if_AoA_0']:.6f}\n")
        lines.append(f"Required AoA leaderboard units if SuperGLUE equals seed43022 full ({NEARBY_SUPERGLUE['compact_view_reinvest_seed43022']:.6f}): {sc['required_AoA_leaderboard_units_to_reach_41p8_if_SuperGLUE_equals_seed43022']:.6f}\n")
        lines.append(f"Required SuperGLUE to reach clean-Qwen if AoA=0: {sc['required_SuperGLUE_to_reach_clean_if_AoA_0']:.6f}\n")
        lines.append(f"Required AoA to reach clean-Qwen if SuperGLUE equals seed43022: {sc['required_AoA_leaderboard_units_to_reach_clean_if_SuperGLUE_equals_seed43022']:.6f}\n")
    lines.append("\n## Interpretation\n")
    lines.append("Seed43122 is much weaker than seed43022 on the cheap no-AoA surface: BLiMP −0.88, Supplement −3.2, EWoK −3.73, full Entity −1.46, COMPS −0.43, GlobalPIQA −0.485, Reading +0.625 relative to seed43022's fast screen. Applying seed43022's fast→full deltas makes the SOTA threshold stricter, not easier, especially because Supplement full was lower than fast by 3.12.\n")
    lines.append("If 8005-row AoA maps to 0 and SuperGLUE lands anywhere near the observed neighboring values (compact_view_core 68.90, visible leader 69.79, clean-Qwen 70.31, compact_repeat_core 70.62, seed43022 reinvest 71.38), seed43122 is not likely to clear 41.8. It would need SuperGLUE about 76.06 even under the no-penalty optimistic projection, or about 78.36 under the seed43022-calibrated projection. Equivalently, if SuperGLUE equaled the seed43022 reinvest value, it would need positive AoA of about +4.68 to +6.98 leaderboard units.\n")
    lines.append("After the official-row-count AoA result is available, AoA merely staying non-significant for seed43122 is not by itself enough to justify full second-seed official evaluation as a likely SOTA-confirming action. Full seed43122 evaluation becomes a minimum reliable expensive action only if the 8005-row AoA is positive enough to keep an above-41.8 path plausible, or if the research chooses to spend the evaluation cost for scientific replication despite the cheap evidence that the second seed's non-AoA surface is much weaker.\n")
    lines.append(f"\nMachine-readable arithmetic: `{OUT_JSON}`.\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))

if __name__ == "__main__":
    main()
