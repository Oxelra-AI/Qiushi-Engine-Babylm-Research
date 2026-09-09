#!/usr/bin/env python3
"""research experience-utilization experiment design from research measurements.

CPU-only.  No model training, no official evaluation text, no GPU use.

The purpose is to turn the sequence-curriculum finding into a clean set of
training comparisons.  The central object is not imitation of a public schedule;
it is whether every charged word becomes usable model experience, and whether a
64->256 order adds anything after full word visibility is restored.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
CURRICULUM_MEASUREMENTS = WORKSPACE / "data" / "sequence_curriculum_loop_measurement" / "sequence_curriculum_loop_measurement.json"
OUT_DIR = WORKSPACE / "data" / "experience_utilization_design"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/experience_utilization_experiment_design.md')

ROWS_PER_10M = 64_740
WORDS_PER_EPOCH = 10_000_000
EPOCHS_TOTAL = 10
BASE_ROW_BATCH = 256
CURRENT_100M_ROW_STEPS = math.ceil((ROWS_PER_10M * EPOCHS_TOTAL) / BASE_ROW_BATCH)  # actual continuous-stream trainer scale
CURRENT_PER_EPOCH_ROW_STEPS = math.ceil(ROWS_PER_10M / BASE_ROW_BATCH)             # stage-reset scale for clean staged comparisons
BASE_MASK_PROB = 0.15
MICRO_BATCH_CHUNKS = 64
PRIMARY_SCHEDULE = {64: 3, 128: 4, 256: 3}
LEADER_STYLE_LONG_SHORT = {64: 7, 128: 0, 256: 3}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def batch_distribution(n_items: int, n_steps: int) -> dict[str, Any]:
    """Exact variable-batch grouping: each item appears once in n_steps updates."""
    base = n_items // n_steps
    rem = n_items % n_steps
    return {
        "items": n_items,
        "steps": n_steps,
        "base_items_per_step": base,
        "steps_with_base_plus_one": rem,
        "steps_with_base": n_steps - rem,
        "max_items_per_step": base + (1 if rem else 0),
        "mean_items_per_step": n_items / n_steps,
        "microbatch_size": MICRO_BATCH_CHUNKS,
        "max_accumulation_microbatches": math.ceil((base + (1 if rem else 0)) / MICRO_BATCH_CHUNKS),
    }


def schedule_plan(rec: dict[str, Any], schedule: dict[int, int], steps_per_epoch: int) -> dict[str, Any]:
    per_stage: dict[str, Any] = {}
    chunks_total = 0
    active_total = 0
    steps_total = 0
    nominal_slot_total = 0
    max_nominal_slots_per_update = 0
    for L, epochs in schedule.items():
        if epochs <= 0:
            continue
        r = rec["by_length"][str(L)]
        chunks = int(r["faithful_chunks_per_epoch"])
        active = int(r["faithful_active_tokens_per_epoch"])
        dist = batch_distribution(chunks, steps_per_epoch)
        max_slots = int(dist["max_items_per_step"] * L)
        mean_active_per_update = active / steps_per_epoch
        stage = {
            "length": L,
            "epochs": epochs,
            "chunks_per_epoch": chunks,
            "active_tokens_per_epoch": active,
            "words_charged_per_epoch": int(r["faithful_words_charged_per_epoch"]),
            "target_steps_per_epoch": steps_per_epoch,
            "batch_distribution_per_epoch": dist,
            "mean_active_tokens_per_update": mean_active_per_update,
            "max_nominal_token_slots_per_update": max_slots,
            "microbatch_nominal_slots": MICRO_BATCH_CHUNKS * L,
        }
        per_stage[str(L)] = stage
        chunks_total += epochs * chunks
        active_total += epochs * active
        steps_total += epochs * steps_per_epoch
        nominal_slot_total += epochs * chunks * L
        max_nominal_slots_per_update = max(max_nominal_slots_per_update, max_slots)
    return {
        "schedule": {str(k): v for k, v in schedule.items()},
        "charged_words_total": WORDS_PER_EPOCH * sum(schedule.values()),
        "chunks_total": chunks_total,
        "active_tokens_total": active_total,
        "steps_total_stage_reset": steps_total,
        "mean_active_tokens_per_update": active_total / max(1, steps_total),
        "nominal_token_slots_total": nominal_slot_total,
        "mean_nominal_token_slots_per_update": nominal_slot_total / max(1, steps_total),
        "max_nominal_token_slots_per_update": max_nominal_slots_per_update,
        "per_stage": per_stage,
    }


def make_arm(label: str, description: str, tokenizer: str, plan: dict[str, Any], baseline_active: int, mask_prob: float, comparison_role: str) -> dict[str, Any]:
    expected_targets = plan["active_tokens_total"] * mask_prob
    baseline_expected_targets = baseline_active * BASE_MASK_PROB
    return {
        "label": label,
        "tokenizer": tokenizer,
        "description": description,
        "comparison_role": comparison_role,
        "charged_words_total": plan["charged_words_total"],
        "active_tokens_total": plan["active_tokens_total"],
        "active_token_ratio_vs_current_row256": plan["active_tokens_total"] / baseline_active,
        "mask_prob": mask_prob,
        "expected_masked_target_tokens": expected_targets,
        "expected_target_ratio_vs_current_row256_mask015": expected_targets / baseline_expected_targets,
        "steps_total_stage_reset": plan["steps_total_stage_reset"],
        "step_ratio_vs_current_row_stage_reset": plan["steps_total_stage_reset"] / (CURRENT_PER_EPOCH_ROW_STEPS * EPOCHS_TOTAL),
        "mean_active_tokens_per_update": plan["mean_active_tokens_per_update"],
        "max_nominal_token_slots_per_update": plan["max_nominal_token_slots_per_update"],
        "nominal_token_slots_total": plan["nominal_token_slots_total"],
        "schedule_plan": plan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    curriculum_measurements = json.loads(CURRICULUM_MEASUREMENTS.read_text(encoding="utf-8"))

    design: dict[str, Any] = {
        "status": "EXPERIENCE_UTILIZATION_EXPERIMENT_DESIGN",
        "purpose": "Separate recovered charged-word visibility from 64->256 ordering after research showed the current prefix schedule debits unseen words.",
        "input_measurement": rel(CURRICULUM_MEASUREMENTS),
        "baseline_step_scale": {
            "rows_per_10M": ROWS_PER_10M,
            "words_per_epoch": WORDS_PER_EPOCH,
            "epochs_total": EPOCHS_TOTAL,
            "base_row_batch": BASE_ROW_BATCH,
            "current_100M_continuous_row_steps": CURRENT_100M_ROW_STEPS,
            "current_per_epoch_row_steps": CURRENT_PER_EPOCH_ROW_STEPS,
            "current_10_epoch_stage_reset_steps": CURRENT_PER_EPOCH_ROW_STEPS * EPOCHS_TOTAL,
            "base_mask_prob": BASE_MASK_PROB,
        },
        "tokenizers": {},
        "comparison_logic": {
            "row256_current_to_U256": "fixed length 256; restores hidden suffix words by word-boundary chunking while keeping the same 10M words per epoch. This measures charged-word experience utilization without a short-to-long order.",
            "U256_to_U64_128_256": "both arms expose every tokenized word once per epoch and use the same expected target tokens when mask_prob is shared. This isolates context/order/locality after visibility is restored.",
            "target_matched_mask": "lower the chunked-arm mask probability to match the expected masked-target count of the current row256 baseline, so the comparison is not explained by a larger target count.",
            "broken_prefix_schedule": "kept only as an ablation of the existing prefix-slice path: it charges the same words but hides large fractions of them during short-length epochs, so it should not be used as the leader-style factor.",
        },
        "expensive_work_position": "Do not launch these arms while depth training s74_t5_tool1 is pending. If depth leaves a gap, a paired U256 vs U64_128_256 wave is the cleanest way to test the experience-utilization principle and the extra value of length ordering.",
    }

    csv_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []

    for tok_label, rec in curriculum_measurements["faithful_streaming_chunking"].items():
        r256 = rec["by_length"]["256"]
        baseline_active = int(r256["prefix_active_tokens_per_epoch"]) * EPOCHS_TOTAL
        baseline_visible_words = int(r256["prefix_visible_whitespace_words_per_epoch"]) * EPOCHS_TOTAL
        baseline_hidden_words = int(r256["prefix_hidden_whitespace_words_but_debited"]) * EPOCHS_TOTAL
        faithful_active_10 = int(r256["faithful_active_tokens_per_epoch"]) * EPOCHS_TOTAL
        mask_prob_match_current_targets = BASE_MASK_PROB * baseline_active / faithful_active_10

        current_row256 = {
            "label": "current_row256_truncate_mask015",
            "description": "existing fixed-256 row-truncating route: one original row per example, full row word debit, suffix words beyond max_seq_length hidden",
            "charged_words_total": WORDS_PER_EPOCH * EPOCHS_TOTAL,
            "active_tokens_total": baseline_active,
            "visible_words_total": baseline_visible_words,
            "hidden_words_but_debited_total": baseline_hidden_words,
            "hidden_word_fraction": baseline_hidden_words / (WORDS_PER_EPOCH * EPOCHS_TOTAL),
            "examples_total": ROWS_PER_10M * EPOCHS_TOTAL,
            "steps_total_continuous_stream": CURRENT_100M_ROW_STEPS,
            "steps_total_stage_reset": CURRENT_PER_EPOCH_ROW_STEPS * EPOCHS_TOTAL,
            "mask_prob": BASE_MASK_PROB,
            "expected_masked_target_tokens": baseline_active * BASE_MASK_PROB,
            "mean_active_tokens_per_continuous_update": baseline_active / CURRENT_100M_ROW_STEPS,
            "mean_active_tokens_per_stage_reset_update": baseline_active / (CURRENT_PER_EPOCH_ROW_STEPS * EPOCHS_TOTAL),
        }

        U256_plan = schedule_plan(rec, {256: EPOCHS_TOTAL}, CURRENT_PER_EPOCH_ROW_STEPS)
        U_order_plan = schedule_plan(rec, PRIMARY_SCHEDULE, CURRENT_PER_EPOCH_ROW_STEPS)
        U_64_7_plan = schedule_plan(rec, LEADER_STYLE_LONG_SHORT, CURRENT_PER_EPOCH_ROW_STEPS)

        prefix_sched = rec["schedules"]["64x3_128x4_256x3"]
        broken_prefix = {
            "label": "current_prefix_64x3_128x4_256x3_do_not_use_as_leader_factor",
            "description": "what the existing seq_len_schedule path would do: slice original rows to L64/L128/L256 while still charging all row words",
            "charged_words_total": int(prefix_sched["charged_words_total"]),
            "active_tokens_total": int(prefix_sched["prefix_active_tokens_total"]),
            "active_token_ratio_vs_current_row256": int(prefix_sched["prefix_active_tokens_total"]) / baseline_active,
            "hidden_words_weighted_fraction": (
                3 * rec["by_length"]["64"]["prefix_hidden_whitespace_words_but_debited"]
                + 4 * rec["by_length"]["128"]["prefix_hidden_whitespace_words_but_debited"]
                + 3 * rec["by_length"]["256"]["prefix_hidden_whitespace_words_but_debited"]
            ) / (WORDS_PER_EPOCH * EPOCHS_TOTAL),
            "mask_prob": BASE_MASK_PROB,
            "expected_masked_target_tokens": int(prefix_sched["prefix_active_tokens_total"]) * BASE_MASK_PROB,
            "expected_target_ratio_vs_current_row256_mask015": int(prefix_sched["prefix_active_tokens_total"]) / baseline_active,
            "steps_total_stage_reset": CURRENT_PER_EPOCH_ROW_STEPS * EPOCHS_TOTAL,
        }

        arms = [
            make_arm(
                "U256_chunked_visibility_mask015",
                "fixed-256 word-boundary chunks, every charged word visible once per epoch, no short-to-long order; mask probability unchanged",
                tok_label,
                U256_plan,
                baseline_active,
                BASE_MASK_PROB,
                "experience utilization at fixed sequence length",
            ),
            make_arm(
                "U256_chunked_visibility_targetmatched",
                "same as U256 but mask probability is reduced so expected masked-target count equals the current row256 baseline",
                tok_label,
                U256_plan,
                baseline_active,
                mask_prob_match_current_targets,
                "experience utilization without extra expected target count",
            ),
            make_arm(
                "U64_128_256_chunked_order_mask015",
                "word-boundary chunks with 3 epochs at 64, 4 at 128, 3 at 256; every charged word visible in every epoch; same mask probability as U256",
                tok_label,
                U_order_plan,
                baseline_active,
                BASE_MASK_PROB,
                "length ordering after recovered visibility",
            ),
            make_arm(
                "U64_128_256_chunked_order_targetmatched",
                "same schedule as U64_128_256 but with mask probability matched to current row256 expected target count",
                tok_label,
                U_order_plan,
                baseline_active,
                mask_prob_match_current_targets,
                "length ordering after recovered visibility without extra expected target count",
            ),
            make_arm(
                "U64x7_256x3_chunked_order_mask015",
                "more aggressive short-context version: 7 epochs at 64 and 3 at 256, all words visible each epoch",
                tok_label,
                U_64_7_plan,
                baseline_active,
                BASE_MASK_PROB,
                "longer early local-context order after recovered visibility",
            ),
        ]

        tok_design = {
            "raw_tokens_per_word": rec["raw_tokens_per_word"],
            "current_row256_baseline": current_row256,
            "mask_prob_to_match_current_row256_targets": mask_prob_match_current_targets,
            "broken_prefix_schedule_reference": broken_prefix,
            "arms": {arm["label"]: arm for arm in arms},
            "recommended_first_wave_if_route_selected": {
                "paired_arms": ["U256_chunked_visibility_mask015", "U64_128_256_chunked_order_mask015"],
                "reason": "They share charged words, active tokens, mask probability, and stage-reset update count; their difference is short-to-long context/order after word visibility is restored. U256 vs current row256 reads the visibility-utilization effect.",
                "stronger_target_control": ["U256_chunked_visibility_targetmatched", "U64_128_256_chunked_order_targetmatched"],
                "when_to_use_target_control": "Use if the first wave improves but the interpretation hinges on whether the 1.7-1.9 percent extra visible target-token count, rather than recovered suffix visibility and contexts, accounts for the gain.",
            },
        }
        design["tokenizers"][tok_label] = tok_design

        # Arm summary CSV rows.
        csv_rows.append({
            "tokenizer": tok_label,
            "arm": current_row256["label"],
            "charged_words_total": current_row256["charged_words_total"],
            "active_tokens_total": current_row256["active_tokens_total"],
            "active_ratio_vs_current_row256": 1.0,
            "mask_prob": current_row256["mask_prob"],
            "expected_target_ratio_vs_current_row256": 1.0,
            "hidden_word_fraction": current_row256["hidden_word_fraction"],
            "steps_total_stage_reset": current_row256["steps_total_stage_reset"],
            "mean_active_tokens_per_update": current_row256["mean_active_tokens_per_stage_reset_update"],
            "role": "measured current baseline",
        })
        csv_rows.append({
            "tokenizer": tok_label,
            "arm": broken_prefix["label"],
            "charged_words_total": broken_prefix["charged_words_total"],
            "active_tokens_total": broken_prefix["active_tokens_total"],
            "active_ratio_vs_current_row256": broken_prefix["active_token_ratio_vs_current_row256"],
            "mask_prob": broken_prefix["mask_prob"],
            "expected_target_ratio_vs_current_row256": broken_prefix["expected_target_ratio_vs_current_row256_mask015"],
            "hidden_word_fraction": broken_prefix["hidden_words_weighted_fraction"],
            "steps_total_stage_reset": broken_prefix["steps_total_stage_reset"],
            "mean_active_tokens_per_update": broken_prefix["active_tokens_total"] / broken_prefix["steps_total_stage_reset"],
            "role": "existing prefix path; not a clean leader-factor test",
        })
        for arm in arms:
            csv_rows.append({
                "tokenizer": tok_label,
                "arm": arm["label"],
                "charged_words_total": arm["charged_words_total"],
                "active_tokens_total": arm["active_tokens_total"],
                "active_ratio_vs_current_row256": arm["active_token_ratio_vs_current_row256"],
                "mask_prob": arm["mask_prob"],
                "expected_target_ratio_vs_current_row256": arm["expected_target_ratio_vs_current_row256_mask015"],
                "hidden_word_fraction": 0.0,
                "steps_total_stage_reset": arm["steps_total_stage_reset"],
                "mean_active_tokens_per_update": arm["mean_active_tokens_per_update"],
                "role": arm["comparison_role"],
            })
            for L, stage in arm["schedule_plan"]["per_stage"].items():
                dist = stage["batch_distribution_per_epoch"]
                batch_rows.append({
                    "tokenizer": tok_label,
                    "arm": arm["label"],
                    "length": L,
                    "epochs": stage["epochs"],
                    "chunks_per_epoch": stage["chunks_per_epoch"],
                    "target_steps_per_epoch": stage["target_steps_per_epoch"],
                    "base_chunks_per_step": dist["base_items_per_step"],
                    "steps_with_base_plus_one": dist["steps_with_base_plus_one"],
                    "steps_with_base": dist["steps_with_base"],
                    "max_chunks_per_step": dist["max_items_per_step"],
                    "max_accumulation_microbatches": dist["max_accumulation_microbatches"],
                    "mean_active_tokens_per_update": stage["mean_active_tokens_per_update"],
                    "max_nominal_token_slots_per_update": stage["max_nominal_token_slots_per_update"],
                    "microbatch_nominal_slots": stage["microbatch_nominal_slots"],
                })

    out_json = OUT_DIR / "experience_utilization_experiment_design.json"
    out_json.write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_csv = OUT_DIR / "experience_utilization_arm_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "arm", "charged_words_total", "active_tokens_total", "active_ratio_vs_current_row256",
            "mask_prob", "expected_target_ratio_vs_current_row256", "hidden_word_fraction", "steps_total_stage_reset",
            "mean_active_tokens_per_update", "role",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    batch_csv = OUT_DIR / "experience_utilization_batch_plan.csv"
    with batch_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "arm", "length", "epochs", "chunks_per_epoch", "target_steps_per_epoch",
            "base_chunks_per_step", "steps_with_base_plus_one", "steps_with_base", "max_chunks_per_step",
            "max_accumulation_microbatches", "mean_active_tokens_per_update", "max_nominal_token_slots_per_update",
            "microbatch_nominal_slots",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(batch_rows)

    lines: list[str] = []
    lines.append("# research — Experience-utilization experiment design")
    lines.append("")
    lines.append("CPU-only design derived from the research direct loop measurement. No model was trained or evaluated.")
    lines.append("")
    lines.append("## Mechanism")
    lines.append("")
    lines.append("The current prefix length path charges words that the model never receives. A useful next experiment should therefore read *experience utilization*: whether the model benefits when every charged word becomes visible and eligible for prediction. A short-to-long sequence order is a second factor, not the same factor.")
    lines.append("")
    lines.append("## Clean comparisons")
    lines.append("")
    lines.append("1. Current row256 -> U256: fixed 256-token length in both; U256 uses word-boundary chunks to reveal suffix words that current row truncation hides. This reads recovered charged-word visibility without short-to-long order.")
    lines.append("2. U256 -> U64_128_256: both expose every tokenized word once per epoch. With shared mask probability and matched stage-reset updates, total active tokens and expected target tokens are the same; the remaining change is local context/order/packing.")
    lines.append("3. Target-matched variants reduce the chunked-arm mask probability so expected masked-target count equals the current row256 baseline; these are interpretation controls if a first wave improves.")
    lines.append("")
    for tok_label, tok_design in design["tokenizers"].items():
        baseline = tok_design["current_row256_baseline"]
        p_match = tok_design["mask_prob_to_match_current_row256_targets"]
        U256 = tok_design["arms"]["U256_chunked_visibility_mask015"]
        Uord = tok_design["arms"]["U64_128_256_chunked_order_mask015"]
        broken = tok_design["broken_prefix_schedule_reference"]
        lines.append(f"## {tok_label}")
        lines.append("")
        lines.append(f"- Current row256 hides {baseline['hidden_words_but_debited_total']:,} of 100M charged words ({baseline['hidden_word_fraction']:.4%}) and has {baseline['active_tokens_total']:,} active tokens at mask 0.15.")
        lines.append(f"- U256 chunking exposes all charged words, has {U256['active_tokens_total']:,} active tokens ({U256['active_token_ratio_vs_current_row256']:.4f}x current), and can match current expected targets with mask probability {p_match:.6f}.")
        lines.append(f"- U64_128_256 after chunking has the same active-token total as U256 ({Uord['active_tokens_total']:,}) and the same 2,530 stage-reset updates; this separates ordering from visibility recovery.")
        lines.append(f"- The broken prefix 64x3_128x4_256x3 path would use only {broken['active_tokens_total']:,} active tokens ({broken['active_token_ratio_vs_current_row256']:.4f}x current row256) and hide {broken['hidden_words_weighted_fraction']:.2%} of charged word opportunities across the schedule.")
        lines.append("")
    lines.append("## Expensive-run use")
    lines.append("")
    lines.append("Do not launch these arms while the legal40k 12x384 depth training is pending. If depth and A02 support-floor evidence still leave the SOTA gap, the clean first wave is U256_chunked_visibility_mask015 versus U64_128_256_chunked_order_mask015 on the same tokenizer/backbone seed: the pair distinguishes recovered experience utilization from length ordering while preserving charged words and target-token totals between the two chunked arms.")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    lines.append(f"CSV: `{rel(summary_csv)}`, `{rel(batch_csv)}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": design["status"],
        "out_json": rel(out_json),
        "summary_csv": rel(summary_csv),
        "batch_csv": rel(batch_csv),
        "note": rel(NOTE),
        "legal40k_U256_active_ratio": design["tokenizers"]["legal40k"]["arms"]["U256_chunked_visibility_mask015"]["active_token_ratio_vs_current_row256"],
        "legal40k_targetmatched_mask_prob": design["tokenizers"]["legal40k"]["mask_prob_to_match_current_row256_targets"],
        "legal40k_broken_prefix_hidden_fraction": design["tokenizers"]["legal40k"]["broken_prefix_schedule_reference"]["hidden_words_weighted_fraction"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
