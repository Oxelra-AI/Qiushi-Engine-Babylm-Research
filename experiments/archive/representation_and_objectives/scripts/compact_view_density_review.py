#!/usr/bin/env python3
"""research compact-view density scientific review.

CPU-only synthesis of existing compact-view evidence for subsequent experiments.
It computes the natural-coordinate and repeat-coordinate component movements and
records how to read the pending full evaluation and seed replication without
letting a scalar Overall alone replace component-level scientific judgment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path("experiments/archive/representation_and_objectives")
A02 = Path("experiments/archive/frontier_consolidation")
CORE = A02 / "data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json"
REINVEST = A02 / "data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json"
EXPOSURE = A02 / "data/trainer_exact_token_exposure_measurement/trainer_exact_token_exposure_summary.json"
META = A02 / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
OUT_DIR = ROOT / "data/compact_view_density_review"
OUT_JSON = OUT_DIR / "compact_view_density_review.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/compact_view_density_review.md')

SEVEN = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
HARD = ["EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(table: Dict[str, Any], arm: str) -> Dict[str, float]:
    row = table[arm]
    return {k: float(row[k]) for k in SEVEN}


def delta(b: Dict[str, float], a: Dict[str, float]) -> Dict[str, float]:
    out = {k: round(b[k] - a[k], 6) for k in SEVEN}
    out["seven_sum"] = round(sum(b[k] for k in SEVEN) - sum(a[k] for k in SEVEN), 6)
    out["seven_mean"] = round(out["seven_sum"] / 7.0, 6)
    out["hard_sum"] = round(sum(b[k] for k in HARD) - sum(a[k] for k in HARD), 6)
    return out


def seven_summary(row: Dict[str, float]) -> Dict[str, float]:
    return {
        "seven_sum": round(sum(row[k] for k in SEVEN), 6),
        "seven_mean": round(sum(row[k] for k in SEVEN) / 7.0, 6),
        "hard_sum": round(sum(row[k] for k in HARD), 6),
        "sg_plus_aoa_needed_for_overall_41p8": round(41.8 * 9 - sum(row[k] for k in SEVEN), 6),
        "sg_plus_aoa_needed_for_overall_42p0": round(42.0 * 9 - sum(row[k] for k in SEVEN), 6),
    }


def scenario(row: Dict[str, float], sg: float, aoa: float) -> float:
    return round((sum(row[k] for k in SEVEN) + sg + aoa) / 9.0, 6)


def main() -> None:
    core = load(CORE)
    reinvest = load(REINVEST)
    exposure = load(EXPOSURE)
    meta = load(META)

    clean = {k: float(core["reference"]["compact_experience_clean_qwen_full"][k if k != "GlobalPIQA_mean" else "GlobalPIQA_mean"]) for k in SEVEN}
    leader = {k: float(core["reference"]["leader_card"][k if k != "GlobalPIQA_mean" else "GlobalPIQA_mean"]) for k in SEVEN}
    compact_repeat = pick(core["table"], "compact_repeat_core")
    compact_view = pick(core["table"], "compact_view_core")
    reinvest_view = pick(reinvest["table"], "compact_view_reinvest")

    c = exposure["contrasts"]
    exposure_compact = c["compact_view_core_minus_compact_repeat_core"]
    exposure_reinvest = c["compact_view_reinvest_minus_compact_view_core"]

    pair_core = meta["pair_summaries"]["compact_core_neutral"]
    pair_reinvest = meta["pair_summaries"]["compact_reinvest"]

    review = {
        "status": "COMPACT_VIEW_DENSITY_PRECEPTOR_REVIEW",
        "sources": {
            "core_fast_summary": str(CORE),
            "reinvest_fast_summary": str(REINVEST),
            "trainer_exact_exposure": str(EXPOSURE),
            "corpus_metadata": str(META),
        },
        "component_rows": {
            "compact_experience_clean_qwen_reference": clean,
            "visible_leader_reference": leader,
            "compact_repeat_core_seed43022": compact_repeat,
            "compact_view_core_seed43022": compact_view,
            "compact_view_reinvest_seed43022": reinvest_view,
        },
        "seven_column_summaries": {
            "compact_experience_clean_qwen_reference": seven_summary(clean),
            "compact_repeat_core": seven_summary(compact_repeat),
            "compact_view_core": seven_summary(compact_view),
            "compact_view_reinvest": seven_summary(reinvest_view),
        },
        "component_movements": {
            "compact_view_core_minus_repeat_core": delta(compact_view, compact_repeat),
            "compact_view_core_minus_compact_experience_clean_qwen": delta(compact_view, clean),
            "compact_view_reinvest_minus_compact_view_core": delta(reinvest_view, compact_view),
            "compact_view_reinvest_minus_compact_experience_clean_qwen": delta(reinvest_view, clean),
            "compact_view_reinvest_minus_visible_leader_fast_columns": delta(reinvest_view, leader),
        },
        "overall_scenarios_from_fast7": {
            "compact_view_core_sg_clean_aoa0": scenario(compact_view, 70.30861598316157, 0.0),
            "compact_view_core_sg68_aoa0": scenario(compact_view, 68.0, 0.0),
            "compact_view_core_sg68_aoa_minus2": scenario(compact_view, 68.0, -2.0),
            "compact_view_reinvest_sg_clean_aoa0": scenario(reinvest_view, 70.30861598316157, 0.0),
            "compact_view_reinvest_sg68_aoa0": scenario(reinvest_view, 68.0, 0.0),
            "compact_view_reinvest_sg68_aoa_minus2": scenario(reinvest_view, 68.0, -2.0),
        },
        "corpus_density_facts": {
            "compact_core_pairs": pair_core["pairs"],
            "compact_core_source_words": pair_core["source_words"],
            "compact_core_rewrite_words": pair_core["rewrite_words"],
            "compact_core_pair_words": pair_core["pair_words"],
            "compact_core_neutral_topup_words": pair_core["neutral_cleanqwen_topup_words_inside_changed_block"],
            "compact_core_rewrite_to_source_ratio_weighted": pair_core["rewrite_to_source_ratio_weighted"],
            "compact_core_content_recall_mean": pair_core["content_recall_stats"]["mean"],
            "compact_core_entity_recall_mean": pair_core["entity_recall_stats"]["mean"],
            "compact_core_number_recall_mean": pair_core["number_recall_stats"]["mean"],
            "compact_reinvest_pairs": pair_reinvest["pairs"],
            "compact_reinvest_source_multiplier_vs_core": round(pair_reinvest["pairs"] / pair_core["pairs"], 6),
            "compact_reinvest_rewrite_to_source_ratio_weighted": pair_reinvest["rewrite_to_source_ratio_weighted"],
            "compact_reinvest_content_recall_mean": pair_reinvest["content_recall_stats"]["mean"],
            "compact_reinvest_entity_recall_mean": pair_reinvest["entity_recall_stats"]["mean"],
            "compact_reinvest_number_recall_mean": pair_reinvest["number_recall_stats"]["mean"],
        },
        "trainer_exact_exposure_reading": {
            "compact_view_core_vs_repeat_candidate_token_relative": exposure_compact["candidate_tokens_visible"]["relative_to_a"],
            "compact_view_core_vs_repeat_wwm_group_relative": exposure_compact["wwm_groups_visible"]["relative_to_a"],
            "compact_view_core_vs_repeat_over_seq256_rows_delta": exposure_compact["over_seq256_rows"]["b_minus_a"],
            "reinvest_vs_core_candidate_token_relative": exposure_reinvest["candidate_tokens_visible"]["relative_to_a"],
            "reinvest_vs_core_wwm_group_relative": exposure_reinvest["wwm_groups_visible"]["relative_to_a"],
            "reinvest_vs_core_over_seq256_rows_delta": exposure_reinvest["over_seq256_rows"]["b_minus_a"],
        },
        "scientific_judgment": {
            "strongest_current_route": "compact generated same-source views as information-density/consolidation, with reinvested saved words as the most promising extension after core full evaluation",
            "why_it_survives_review": "It improves compact_view_core over same-source repetition across six of seven fast columns and also improves the inherited clean-Qwen natural coordinate in the seven-column surface, while trainer-exact WWM groups are not higher in the view arm.",
            "why_it_is_not_settled": "Full official-compatible compact_view_core scores, SuperGLUE, AoA, and seed43122 endpoint evidence are still unresolved; same-seed repeat43122 and a true natural/lengthmatched comparison remain needed for a fully defensible mechanism.",
            "controller_condition_warning": "The running A01 seed43122 controller uses scalar Overall >=41.6 from A02 full evaluation as a start condition. If it exits because SuperGLUE or AoA depress Overall while the hard fast/full component pattern survives, later agents should still consider seed43122 view and matched repeat43122 scientifically justified.",
            "failure_pattern_that_changes_direction": "If EWoK/Entity/COMPS/GlobalPIQA movement disappears in full evaluation or seed43122, repeating the same compact corpus is not useful; inspect which component collapsed and rebuild the data mechanism rather than extending it unchanged."
        }
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    cm = review["component_movements"]
    ss = review["seven_column_summaries"]
    scenarios = review["overall_scenarios_from_fast7"]
    facts = review["corpus_density_facts"]
    exp = review["trainer_exact_exposure_reading"]
    note = f"""# Compact-view density analysis

## Review outcome

The compact-view density line should remain the group focus unless the pending full evaluation or the seed43122 endpoint result removes the hard-component movement. The older cached-FineWeb branch may still be useful orthogonal evidence, but it should not reclaim attention merely because it was started earlier.

## Component reading

`compact_view_core` vs same-seed `compact_repeat_core`: seven-column sum {cm['compact_view_core_minus_repeat_core']['seven_sum']:+.3f} ({cm['compact_view_core_minus_repeat_core']['seven_mean']:+.3f} mean), hard cluster {cm['compact_view_core_minus_repeat_core']['hard_sum']:+.3f}. Column deltas: BLiMP {cm['compact_view_core_minus_repeat_core']['BLiMP']:+.2f}, Supplement {cm['compact_view_core_minus_repeat_core']['Supplement']:+.2f}, EWoK {cm['compact_view_core_minus_repeat_core']['EWoK']:+.2f}, Entity {cm['compact_view_core_minus_repeat_core']['Entity']:+.2f}, COMPS {cm['compact_view_core_minus_repeat_core']['COMPS']:+.2f}, GlobalPIQA {cm['compact_view_core_minus_repeat_core']['GlobalPIQA_mean']:+.3f}, Reading {cm['compact_view_core_minus_repeat_core']['Reading']:+.2f}.

`compact_view_core` vs the inherited clean-Qwen natural coordinate: seven-column sum {cm['compact_view_core_minus_compact_experience_clean_qwen']['seven_sum']:+.3f} ({cm['compact_view_core_minus_compact_experience_clean_qwen']['seven_mean']:+.3f} mean), hard cluster {cm['compact_view_core_minus_compact_experience_clean_qwen']['hard_sum']:+.3f}. This matters because the same-source repeat arm is a deliberately harsh comparator; progress toward the user goal must also survive against the strong natural allocation.

`compact_view_reinvest` is the strongest fast surface now available: seven-column mean {ss['compact_view_reinvest']['seven_mean']:.4f}, with SuperGLUE+AoA needed for 41.8 equal to {ss['compact_view_reinvest']['sg_plus_aoa_needed_for_overall_41p8']:.3f}. Against clean-Qwen it is {cm['compact_view_reinvest_minus_compact_experience_clean_qwen']['seven_mean']:+.3f} in seven-column mean and {cm['compact_view_reinvest_minus_compact_experience_clean_qwen']['hard_sum']:+.3f} in the hard cluster. It is a promising extension, not a substitute for the unresolved core full evaluation and seed evidence.

Fast-seven scenarios: compact_view_core with clean-Qwen SuperGLUE and AoA 0 gives Overall {scenarios['compact_view_core_sg_clean_aoa0']:.4f}; with SuperGLUE 68 and AoA 0 gives {scenarios['compact_view_core_sg68_aoa0']:.4f}. Reinvest with SuperGLUE 68 and AoA 0 already gives {scenarios['compact_view_reinvest_sg68_aoa0']:.4f}; with SuperGLUE 68 and AoA -2 gives {scenarios['compact_view_reinvest_sg68_aoa_minus2']:.4f}. These are arithmetic projections from fast columns, not results.

## Mechanism reading

The best current hypothesis is information-density and consolidation, not generic paraphrase. Compact views keep anchors and numerical facts while deleting redundant wording and some secondary content: core rewrite/source ratio {facts['compact_core_rewrite_to_source_ratio_weighted']:.3f}, content recall {facts['compact_core_content_recall_mean']:.3f}, entity recall {facts['compact_core_entity_recall_mean']:.3f}, number recall {facts['compact_core_number_recall_mean']:.3f}. The model receives the original source proposition and a shorter generated view in close proximity, which can stabilize entity/state and predicate abstractions under a fixed word budget. Reinvestment then uses the saved words to add more source-view packets, increasing source count by {facts['compact_reinvest_source_multiplier_vs_core']:.3f}x with similar compression and retention.

Trainer-exact exposure does not explain the main contrast: compact_view_core has candidate tokens {exp['compact_view_core_vs_repeat_candidate_token_relative']:+.6f} relative to repeat and WWM groups {exp['compact_view_core_vs_repeat_wwm_group_relative']:+.6f}; reinvest vs core has candidate tokens {exp['reinvest_vs_core_candidate_token_relative']:+.6f} and WWM groups {exp['reinvest_vs_core_wwm_group_relative']:+.6f}. These differences are too small and not aligned enough to replace the data-mechanism explanation, though they should remain in the interpretation.

## Vulnerabilities that matter

The automatic semantic numbers are not proof of perfect meaning preservation. Content recall around 0.66 means many generated views omit propositions, and source samples include extraction artifacts and occasional mixed-topic continuations. The current result may be robust because compression removes noisy surface burden, or it may be partly selected by fast tasks. The next evidence must therefore read the component vector, not just a scalar Overall.

A low full Overall caused mainly by SuperGLUE or AoA would not by itself refute the data mechanism; it would indicate that the data mechanism needs combination or repair. Loss of the EWoK/Entity/COMPS/GlobalPIQA pattern in full evaluation or seed43122 would be much more damaging and should stop unchanged extension.

## Next scientific work

When the managed results arrive, read the actual files before acting. If A02 core full evaluation preserves the hard-component surface but the running A01 seed43122 controller exits because its scalar start condition is not met, the next agent should not treat that exit as scientific evidence against replication; it should decide from the full component vector whether seed43122 view and then matched repeat43122 are still warranted. If seed43122 view remains strong, the next expensive same-seed comparison is `compact_repeat_core` seed43122. If core full is healthy, evaluate `compact_view_reinvest` fully because its existing trained endpoint has the stronger fast surface and requires much less SuperGLUE+AoA to cross 41.8.

JSON: `{OUT_JSON}`
"""
    OUT_NOTE.write_text(note, encoding="utf-8")
    print(json.dumps({"status": review["status"], "out_json": str(OUT_JSON), "note": str(OUT_NOTE)}, indent=2))


if __name__ == "__main__":
    main()
