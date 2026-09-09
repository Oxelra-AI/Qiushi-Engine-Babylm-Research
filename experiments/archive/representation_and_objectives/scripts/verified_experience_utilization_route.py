#!/usr/bin/env python3
"""research verified route synthesis for experience-utilization training.

CPU-only.  Integrates research construction preflight and independent review verification into a
corrected route artifact.  No model training and no official evaluation text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
PREFLIGHT = WORKSPACE / "data" / "chunk_stream_preflight" / "chunk_stream_preflight.json"
DESIGN = WORKSPACE / "data" / "experience_utilization_design" / "experience_utilization_experiment_design.json"
independent_review = STUDY / "trace" / "independent_review" / "results" / "independent_review01_verifier1_integration.md"
OUT_DIR = WORKSPACE / "data" / "verified_experience_utilization_route"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/verified_experience_utilization_route.md')


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def tokenizer_summary(label: str, pre: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    p = pre["tokenizers"][label]
    d = design["tokenizers"][label]
    current = d["current_row256_baseline"]
    u256 = p["schedule_U256"]
    uord = p["schedule_U64_128_256"]
    return {
        "tokenizer": label,
        "tokenizer_json_sha256": p["tokenizer_json_sha256"],
        "raw_tokens_per_epoch": p["raw_tokens_per_epoch"],
        "raw_tokens_per_word": p["raw_tokens_per_word"],
        "current_row256": {
            "charged_words_total": current["charged_words_total"],
            "active_tokens_total": current["active_tokens_total"],
            "hidden_words_but_debited_total": current["hidden_words_but_debited_total"],
            "hidden_word_fraction": current["hidden_word_fraction"],
            "mask_prob": current["mask_prob"],
            "steps_total_stage_reset": current["steps_total_stage_reset"],
        },
        "U256_chunked": {
            "charged_words_total": u256["charged_words_total"],
            "active_tokens_total": u256["active_tokens_total"],
            "steps_total_stage_reset": u256["steps_total_stage_reset"],
            "chunks_total_with_overlong_split": u256["chunks_total_with_overlong_split"],
            "active_token_ratio_vs_current_row256": u256["active_tokens_total"] / current["active_tokens_total"],
            "expected_masked_targets_mask015": u256["expected_masked_targets_mask015"],
            "targetmatched_mask_prob_vs_current": d["mask_prob_to_match_current_row256_targets"],
        },
        "U64_128_256_chunked": {
            "charged_words_total": uord["charged_words_total"],
            "active_tokens_total": uord["active_tokens_total"],
            "steps_total_stage_reset": uord["steps_total_stage_reset"],
            "chunks_total_with_overlong_split": uord["chunks_total_with_overlong_split"],
            "expected_masked_targets_mask015": uord["expected_masked_targets_mask015"],
            "per_length_epochs": uord["per_length_epochs"],
        },
        "matched_between_U256_and_U64_128_256": p["schedules_match_on_charged_words_active_tokens_steps"],
        "first_update_microbatches_by_length": {
            L: p["by_length"][L]["first_update"]["accumulation_microbatches"] for L in ["64", "128", "256"]
        },
        "L64_overlong_words": p["by_length"]["64"]["overlong_words"],
        "L64_overlong_continuation_chunks_per_epoch": p["by_length"]["64"]["overlong_continuation_chunks"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    pre = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    independent_review_exists = independent_review.exists()
    summaries = {label: tokenizer_summary(label, pre, design) for label in pre["tokenizers"]}
    route = {
        "status": "VERIFIED_EXPERIENCE_UTILIZATION_ROUTE",
        "purpose": "Correct the research design after construction and independent_review verification so any later H100 run tests experience utilization rather than a misdescribed sequence schedule.",
        "source_files": {
            "design": rel(DESIGN),
            "chunk_stream_preflight": rel(PREFLIGHT),
            "independent_review_verification": rel(independent_review) if independent_review_exists else None,
        },
        "tokenizers": summaries,
        "scientific_interpretation": {
            "current_row256_to_U256": "Reads the value of a chunk-stream intervention that restores most row-truncated charged words at fixed maximum length 256. It is not pure visibility, because row-to-chunk segmentation, update composition, boundaries, padding, and microbatch accumulation also change.",
            "U256_to_U64_128_256": "Reads a short-to-long context/order/packing bundle after all words are visible. U256 and U64_128_256 match on charged words, active tokens, expected targets at shared mask probability, and stage-reset update count. It is not pure chronological order unless later compared with a length-mixture-matched interleaved or permuted stream.",
            "targetmatched_variants": "Lower mask probability to match the current row256 expected masked-token count; useful if a gain needs separation from the extra 1.7-1.9 percent active target-token count. This still does not match target context distribution or actual realized WWM samples.",
            "general_learning_principle": "Under a strict word budget, accounting should align with the experience actually visible to the model and eligible for learning. A positive U256 target-matched result would support this principle more directly than copying any public schedule.",
        },
        "implementation_requirements_before_training": [
            "Build stage-specific chunk streams from the exact 10M corpus and tokenizer SHA, with every whitespace word charged once per epoch and every tokenizer token visible once per epoch.",
            "Use the preflight-corrected overlong split counts: one L64 overlong continuation chunk per epoch for both legal40k and minfreq25, charged_words=0 on continuation.",
            "Form exactly 253 optimizer updates per 10M-word epoch for both U256 and U64_128_256 by distributing chunks across updates; do not add a tail update at stage boundaries.",
            "Apply WWM once on the full effective batch before microbatch forwards; keep one mask generator stream and record realized masked tokens, selected groups, active tokens, charged words, and accumulation depth per update.",
            "Weight microbatch losses by masked-token count so the update objective is the masked-token mean over the effective batch, not an average of microbatch means.",
            "Keep optimizer, LR scheduler, warmup, clipping, global step, initialization seed, train RNG seed, and data order continuous across stage boundaries unless intentionally changed in a paired arm.",
            "Verify a dry run over a full 10M stage epoch before H100 launch: exact 10M charged words, exact preflight active-token totals, no duplicate or missing chunks, no special-token insertion drift, and all checkpoint word thresholds reachable.",
        ],
        "recommended_first_wave_after_depth_if_selected": {
            "wait_for": "legal40k 12x384 depth vector and A02 support-floor vector",
            "primary_pair": ["U256_chunked_visibility_mask015", "U64_128_256_chunked_order_mask015"],
            "same_seed_first": 43022,
            "why": "This pair separates restored charged-word experience from the short-to-long context/order/packing bundle, because both chunked arms match on words, tokens, expected targets, and update count.",
            "if_U256_improves_row256": "experience utilization becomes a load-bearing mechanism; if target-matched U256 also improves, the explanation is not simply extra expected targets.",
            "if_U64_128_256_improves_U256": "test a length-mixture matched interleaved or permuted stream before attributing the gain to chronological short-to-long order.",
        },
    }
    out_json = OUT_DIR / "verified_experience_utilization_route.json"
    out_json.write_text(json.dumps(route, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research — Verified experience-utilization route")
    lines.append("")
    lines.append("This note integrates the research construction preflight with independent_review verification. It does not report any new training or evaluation result.")
    lines.append("")
    lines.append("## Corrected mechanism wording")
    lines.append("")
    lines.append("The central mechanism is charged-experience utilization: the training pipeline should not debit words that cannot enter the model or the prediction candidate set. Current row256 to U256 reads a fixed-length chunk-stream intervention that restores row-truncated suffix material; U256 to U64_128_256 reads a short-to-long context/order/packing bundle after full visibility is restored.")
    lines.append("")
    for label, s in summaries.items():
        cur = s["current_row256"]
        u256 = s["U256_chunked"]
        uord = s["U64_128_256_chunked"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Current row256: {cur['active_tokens_total']:,} active tokens, {cur['hidden_words_but_debited_total']:,} hidden-but-debited words ({cur['hidden_word_fraction']:.4%}).")
        lines.append(f"- U256: {u256['active_tokens_total']:,} active tokens ({u256['active_token_ratio_vs_current_row256']:.4f}x current), {u256['chunks_total_with_overlong_split']:,} chunks, mask probability {u256['targetmatched_mask_prob_vs_current']:.6f} to match current expected targets.")
        lines.append(f"- U64_128_256: {uord['active_tokens_total']:,} active tokens, {uord['chunks_total_with_overlong_split']:,} chunks, same 100M charged words and 2,530 stage-reset updates as U256; L64 has {s['L64_overlong_continuation_chunks_per_epoch']} zero-charge continuation chunk per epoch.")
        lines.append(f"- First-update accumulation microbatches: L64 {s['first_update_microbatches_by_length']['64']}, L128 {s['first_update_microbatches_by_length']['128']}, L256 {s['first_update_microbatches_by_length']['256']}.")
        lines.append("")
    lines.append("## Before any H100 run")
    lines.append("")
    for item in route["implementation_requirements_before_training"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Route use")
    lines.append("")
    lines.append("Do not launch this while depth training is pending. If depth and A02 support-floor results leave the gap, use the U256 vs U64_128_256 pair as the first experience-utilization wave. If U64_128_256 beats U256, add a length-mixture matched interleaved or permuted stream before attributing the effect to chronological order.")
    lines.append("")
    lines.append(f"JSON: `{rel(out_json)}`")
    lines.append(f"independent_review: `{rel(independent_review) if independent_review_exists else 'missing'}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": route["status"],
        "out_json": rel(out_json),
        "note": rel(NOTE),
        "legal40k_U64_chunks_corrected": summaries["legal40k"]["U64_128_256_chunked"]["chunks_total_with_overlong_split"],
        "minfreq25_U64_chunks_corrected": summaries["minfreq25"]["U64_128_256_chunked"]["chunks_total_with_overlong_split"],
        "independent_review_exists": independent_review_exists,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
