#!/usr/bin/env python3
"""research: U256 score leverage and data-route reading.

CPU-only measurement over already-produced experiment artifacts. It does not train,
evaluate a checkpoint or create a training corpus. Its purpose
is to decide whether the launch-ready U256 experience-utilization arm is scientifically
large and aligned enough to be the first expensive route if SGCR is weak.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
WS = ROOT
OUT = WS / "data/u256_leverage_and_data_route_reading"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/u256_leverage_and_data_route_reading.md')

U256_JSON = WS / "data/experience_utilization_prelaunch_measurement/experience_utilization_prelaunch_measurement.json"
U256_QUALITY_JSON = WS / "data/recovered_suffix_quality_measurement/recovered_suffix_quality_measurement.json"
LEGAL40_JSON = WS / "data/legal40k_two_seed_comparison/legal40k_two_seed_comparison.json"
DEPTH_JSON = WS / "data/depth_vector_decision/depth_vector_decision.json"
SOURCE_BREAKDOWN = WS / "data/corpus_lineage_multiplicity_audit/source_class_word_breakdown_10M.csv"

LEADER = {
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "SuperGLUE": 69.79,
    "GlobalPIQA": 39.67,
    "Reading": 5.42,
    "AoA": 0.0,
    "Overall": 41.80,
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_source_breakdown(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["source_class"]] = {
                "rows_10M": int(row["rows_10M"]),
                "words_10M": int(row["words_10M"]),
                "fraction_words_10M": float(row["fraction_words_10M"]),
            }
    return out


def vector_reading(name: str, scores: dict[str, float]) -> dict[str, Any]:
    deltas_vs_leader = {c: float(scores.get(c, 0.0)) - LEADER[c] for c in COLUMNS}
    positive_sum = sum(v for v in deltas_vs_leader.values() if v > 0)
    deficit_sum = sum(-v for v in deltas_vs_leader.values() if v < 0)
    required_colsum_to_418 = max(0.0, (LEADER["Overall"] - float(scores.get("Overall", 0.0))) * len(COLUMNS))
    below = sorted(
        [(c, -deltas_vs_leader[c]) for c in COLUMNS if deltas_vs_leader[c] < 0],
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "name": name,
        "overall": float(scores["Overall"]),
        "margin_over_leader_overall": float(scores["Overall"]) - LEADER["Overall"],
        "required_positive_column_sum_to_reach_41p80": required_colsum_to_418,
        "positive_column_sum_vs_leader": positive_sum,
        "below_leader_column_deficit_sum": deficit_sum,
        "deltas_vs_leader": deltas_vs_leader,
        "below_leader_ranked_deficits": below,
        "deficit_share_EWoK_GlobalPIQA": sum(x for c, x in below if c in {"EWoK", "GlobalPIQA"}) / deficit_sum if deficit_sum else 0.0,
        "deficit_share_EWoK_GlobalPIQA_COMPS": sum(x for c, x in below if c in {"EWoK", "GlobalPIQA", "COMPS"}) / deficit_sum if deficit_sum else 0.0,
        "deficit_share_child_dialogue_plausible_columns_entity_reading_supplement": sum(x for c, x in below if c in {"Entity", "Reading", "Supplement"}) / deficit_sum if deficit_sum else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    u256 = read_json(U256_JSON)
    quality = read_json(U256_QUALITY_JSON) if U256_QUALITY_JSON.exists() else {}
    legal40 = read_json(LEGAL40_JSON)
    depth = read_json(DEPTH_JSON)
    source = read_source_breakdown(SOURCE_BREAKDOWN)

    legal40_8x480_43022 = legal40["by_seed"]["43022"]["legal40k"]
    legal40_mean = legal40["legal40k_mean"]
    depth_scores = dict(depth["depth_scores"])
    depth_scores["Overall"] = depth["depth_overall"]

    rowprof = u256["row256_recovered_suffix_profile"]
    source_table = {r["source"]: r for r in rowprof["source_table"]}
    update = u256["baseline_vs_chunked"]

    # Source-class comparison to the public leader's model-card data size.
    leader_words = 9_999_969
    fineweb_words = source["fineweb_source_qwen_compact_rewrite_pair_row"]["words_10M"]
    qwen_words = source["inherited_official_source_qwen_paraphrase_pair_row"]["words_10M"]
    official_words = source["official_babylm_source_row"]["words_10M"]
    generated_pair_words = fineweb_words + qwen_words

    hidden_total = rowprof["hidden_full_words_total"]
    hidden_fineweb = source_table["cleanqwen_fineweb_compact_view_reinvest"]["hidden_words"]
    hidden_childes = source_table["childes"]["hidden_words"]
    hidden_dialogue_sources = hidden_childes + source_table["open_subtitles"]["hidden_words"] + source_table["bnc_spoken"]["hidden_words"]

    content_categories = quality.get("category_table", []) or quality.get("content_categories", [])
    if isinstance(content_categories, dict):
        category_map = content_categories
    else:
        category_map = {r.get("category", r.get("label", "unknown")): r for r in content_categories if isinstance(r, dict)}

    vectors = [
        vector_reading("legal40_8x480_seed43022", legal40_8x480_43022),
        vector_reading("legal40_mean_two_seed", legal40_mean),
        vector_reading("legal40_depth_12x384_seed43022", depth_scores),
    ]

    u256_mass = {
        "hidden_full_words_per_10M_pass": hidden_total,
        "hidden_full_word_fraction_per_pass": rowprof["hidden_full_word_fraction"],
        "u256_recovered_tokens_per_10M_pass": rowprof["u256_recovered_tokens_total"],
        "u256_active_token_ratio_vs_row256": rowprof["u256_active_token_ratio_vs_row256"],
        "u256_masked_target_ratio_vs_row256": update["u256_vs_row256"]["masked_target_ratio"],
        "row256_masked_targets_100M": update["row256_baseline_log_order"]["total_masked_tokens"],
        "u256_masked_targets_100M": update["U256_mask015"]["total_masked_tokens"],
        "extra_masked_targets_100M": update["U256_mask015"]["total_masked_tokens"] - update["row256_baseline_log_order"]["total_masked_tokens"],
        "hidden_childes_words": hidden_childes,
        "hidden_childes_fraction_of_hidden": hidden_childes / hidden_total,
        "hidden_dialogue_source_words_childes_open_subtitles_bnc": hidden_dialogue_sources,
        "hidden_dialogue_source_fraction_of_hidden": hidden_dialogue_sources / hidden_total,
        "hidden_fineweb_compact_words": hidden_fineweb,
        "hidden_fineweb_compact_fraction_of_hidden": hidden_fineweb / hidden_total,
        "hidden_words_exposure_10_epochs": hidden_total * 10,
        "hidden_fineweb_words_exposure_10_epochs": hidden_fineweb * 10,
    }

    source_comparison = {
        "leader_model_card_fineweb_simplification_pair_words": leader_words,
        "current_official_babylm_source_words": official_words,
        "current_official_source_qwen_pair_words": qwen_words,
        "current_fineweb_source_compact_pair_words": fineweb_words,
        "current_all_generated_pair_words_qwen_plus_fineweb": generated_pair_words,
        "current_fineweb_pair_words_as_fraction_of_leader_fineweb_pair_words": fineweb_words / leader_words,
        "current_all_generated_pair_words_as_fraction_of_leader_pair_words": generated_pair_words / leader_words,
        "current_official_babylm_source_fraction": official_words / 10_000_000,
        "u256_hidden_fineweb_words_per_pass_as_fraction_of_leader_pair_words": hidden_fineweb / leader_words,
        "u256_hidden_all_words_per_pass_as_fraction_of_leader_pair_words": hidden_total / leader_words,
    }

    interpretation = []
    interpretation.append(
        "U256 is a small visibility repair: +1.73% active tokens and +1.69% masked targets over 100M, with only 128,664 fully hidden words per 10M pass."
    )
    interpretation.append(
        "The recovered words are overwhelmingly not the leader-like FineWeb simplification-pair substrate: 78.59% CHILDES and only 17 FineWeb compact words per 10M pass."
    )
    interpretation.append(
        "The best completed compliant legal40 8x480 seed needs +5.94 summed column-points to reach 41.80; depth needs +6.95. Most below-leader deficit is EWoK+GlobalPIQA, while U256 mainly changes dialogue/transcript and name-heavy tails."
    )
    interpretation.append(
        "Therefore U256 should not be the automatic first expensive route after a weak SGCR endpoint. It remains useful only if a delivered vector specifically points to child/dialogue/entity-state tail exposure as the limiting factor, or as a compact, fixed-length comparison after a stronger data route is unavailable."
    )
    interpretation.append(
        "If SGCR is weak, the bigger unresolved difference from the public leader is data substrate: our corpus has 0.4235M FineWeb compact-pair words (4.2% of the leader's FineWeb pair word budget), while the leader model card reports 9.999969M FineWeb simplification-pair words."
    )

    result = {
        "status": "U256_LEVERAGE_AND_DATA_ROUTE_READING",
        "no_training_no_model_eval_no_managed_task_query": True,
        "inputs": {
            "u256_profile": str(U256_JSON),
            "u256_quality": str(U256_QUALITY_JSON),
            "legal40_vectors": str(LEGAL40_JSON),
            "depth_vector": str(DEPTH_JSON),
            "source_breakdown": str(SOURCE_BREAKDOWN),
            "leader_model_card_source": "data/external/go76dof-wwm-curriculum-simplification-40k-Hugging-Face.md",
        },
        "u256_mass": u256_mass,
        "source_comparison": source_comparison,
        "vectors_vs_leader": vectors,
        "suffix_content_categories_raw": category_map,
        "scientific_reading": interpretation,
    }
    (OUT / "u256_leverage_and_data_route_reading.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compact CSV for future route comparisons.
    with (OUT / "vector_leverage_vs_leader.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "name", "overall", "margin_over_leader_overall", "required_positive_column_sum_to_reach_41p80",
            "below_leader_column_deficit_sum", "deficit_share_EWoK_GlobalPIQA", "deficit_share_EWoK_GlobalPIQA_COMPS",
            "deficit_share_child_dialogue_plausible_columns_entity_reading_supplement",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in vectors:
            w.writerow({k: v[k] for k in fieldnames})

    lines = []
    lines.append("# research — U256 leverage and data-route reading\n\n")
    lines.append("CPU-only reading of existing artifacts. No model was trained or evaluated, and no training corpus was created.\n\n")
    lines.append("## U256 mass\n\n")
    lines.append(f"- Hidden full words recovered by U256: {hidden_total:,} per 10M pass ({rowprof['hidden_full_word_fraction']:.4%}).\n")
    lines.append(f"- Active tokens: {rowprof['prefix_active_tokens_total']:,} -> {rowprof['raw_tokens_total']:,} per pass (+{(rowprof['u256_active_token_ratio_vs_row256']-1)*100:.3f}%).\n")
    lines.append(f"- Realized masked targets over 100M: {update['row256_baseline_log_order']['total_masked_tokens']:,} -> {update['U256_mask015']['total_masked_tokens']:,} (+{(update['u256_vs_row256']['masked_target_ratio']-1)*100:.3f}%).\n\n")
    lines.append("## What content U256 actually changes\n\n")
    lines.append(f"- CHILDES hidden words: {hidden_childes:,} ({hidden_childes/hidden_total:.2%} of hidden).\n")
    lines.append(f"- CHILDES+OpenSubtitles+BNC hidden words: {hidden_dialogue_sources:,} ({hidden_dialogue_sources/hidden_total:.2%} of hidden).\n")
    lines.append(f"- FineWeb compact hidden words: {hidden_fineweb:,} ({hidden_fineweb/hidden_total:.4%} of hidden).\n\n")
    lines.append("U256 is therefore not a compact-view or leader-like FineWeb amplifier. It mainly exposes row tails from child/dialogue/transcript material and a smaller amount of heterogeneous long-row residue.\n\n")
    lines.append("## Score leverage\n\n")
    for v in vectors:
        deficits = ", ".join(f"{c} {x:.2f}" for c, x in v["below_leader_ranked_deficits"][:5])
        lines.append(f"- {v['name']}: Overall {v['overall']:.4f}, needs +{v['required_positive_column_sum_to_reach_41p80']:.2f} summed column-points to reach 41.80; largest below-leader deficits: {deficits}. EWoK+GlobalPIQA account for {v['deficit_share_EWoK_GlobalPIQA']:.1%} of its below-leader deficit.\n")
    lines.append("\nA fixed-length suffix-visibility intervention would have to create a surprisingly large and broad score movement to cross from the best compliant completed endpoint. The necessary gains sit mostly in EWoK and GlobalPIQA, while the material U256 newly exposes is mostly dialogue/transcript tails rather than factual FineWeb simplification pairs.\n\n")
    lines.append("## Data substrate contrast\n\n")
    lines.append(f"- Public leader model card: {leader_words:,} FineWeb simplification-pair words.\n")
    lines.append(f"- Current compact-view corpus: {fineweb_words:,} FineWeb source+compact words ({fineweb_words/leader_words:.2%} of leader pair budget); {generated_pair_words:,} all generated-pair words including official-source Qwen ({generated_pair_words/leader_words:.2%}).\n")
    lines.append(f"- U256 recovers only {hidden_fineweb:,} FineWeb compact words per 10M pass.\n\n")
    lines.append("## Scientific reading\n\n")
    for s in interpretation:
        lines.append(f"- {s}\n")
    lines.append("\nFiles:\n")
    lines.append(f"- JSON: `{OUT / 'u256_leverage_and_data_route_reading.json'}`\n")
    lines.append(f"- CSV: `{OUT / 'vector_leverage_vs_leader.csv'}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(OUT / "u256_leverage_and_data_route_reading.json"),
        "note": str(NOTE),
        "u256_hidden_words": hidden_total,
        "hidden_fineweb_words": hidden_fineweb,
        "legal40_8x480_required_colsum": vectors[0]["required_positive_column_sum_to_reach_41p80"],
        "depth_required_colsum": vectors[2]["required_positive_column_sum_to_reach_41p80"],
    }, indent=2))


if __name__ == "__main__":
    main()
