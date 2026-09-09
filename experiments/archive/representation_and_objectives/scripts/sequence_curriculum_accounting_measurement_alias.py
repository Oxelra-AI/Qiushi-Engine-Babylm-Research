#!/usr/bin/env python3
"""Create neutral research sequence-curriculum accounting measurement summary.

This reads the already produced research sequence accounting JSON and writes a compact,
neutral-named decision summary for future route work.
"""
import json
import pathlib

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SRC = ROOT / "data/sequence_curriculum_accounting_audit/sequence_curriculum_accounting_audit.json"
OUT_DIR = ROOT / "data/sequence_curriculum_accounting_measurement"
OUT_JSON = OUT_DIR / "sequence_curriculum_accounting_measurement.json"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/sequence_curriculum_accounting_measurement.md')


def main():
    data = json.loads(SRC.read_text())
    summaries = {}
    for label, s in data["tokenizer_summaries"].items():
        lengths = s["lengths"]
        schedules = s["schedules"]
        summaries[label] = {
            "vocab_size": s["vocab_size"],
            "tokens_per_word": s["tokens_per_word"],
            "prefix_visible_group_fraction_by_length": {
                L: lengths[L]["current_prefix_slicing_fixed_row_batch"]["visible_group_fraction_vs_full"]
                for L in ["64", "128", "256"]
            },
            "prefix_missing_group_fraction_by_length": {
                L: 1.0 - lengths[L]["current_prefix_slicing_fixed_row_batch"]["visible_group_fraction_vs_full"]
                for L in ["64", "128", "256"]
            },
            "three_stage_64x3_128x4_256x3": {
                "prefix_missing_group_fraction": schedules["three_stage_64x3_128x4_256x3"]["current_prefix_missing_group_fraction_over_10_epochs"],
                "faithful_chunking_target_token_ratio_vs_prefix": schedules["three_stage_64x3_128x4_256x3"]["target_token_ratio_faithful_vs_current_prefix"],
                "faithful_chunking_optimizer_step_ratio_vs_prefix": schedules["three_stage_64x3_128x4_256x3"]["optimizer_step_ratio_faithful_vs_current_prefix"],
            },
            "two_stage_64x7_256x3": {
                "prefix_missing_group_fraction": schedules["two_stage_64x7_256x3"]["current_prefix_missing_group_fraction_over_10_epochs"],
                "faithful_chunking_target_token_ratio_vs_prefix": schedules["two_stage_64x7_256x3"]["target_token_ratio_faithful_vs_current_prefix"],
                "faithful_chunking_optimizer_step_ratio_vs_prefix": schedules["two_stage_64x7_256x3"]["optimizer_step_ratio_faithful_vs_current_prefix"],
            },
        }
    payload = {
        "status": "SEQUENCE_CURRICULUM_ACCOUNTING_MEASUREMENT",
        "pool_rows": data["pool_rows"],
        "pool_words": data["pool_words"],
        "leader_factor": data["published_leader_factor"],
        "current_schedule_path_behavior": data["current_trainer_finding"],
        "summary": summaries,
        "route_implication": "A future 64->256 route should use stage-length word-boundary chunking or streaming with inverse row-batch scaling. The existing short-prefix slicing path should not be treated as the leader-style sequence factor because it hides large fractions of word-groups while counting their words.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# research — Sequence curriculum accounting measurement\n",
        "Neutral summary of the CPU-only sequence-curriculum accounting measurement on the exact compact_view_reinvest 10M pool. No model was trained or evaluated.\n",
        f"- Pool: {payload['pool_rows']} rows / {payload['pool_words']} words.",
        "- Public leader factor: 64→256 sequence length with inverse batch scaling.",
        "- Existing local schedule behavior: tokenizes rows at 256, slices short prefixes, and still counts full row words.\n",
    ]
    for label, rec in summaries.items():
        tri = rec["three_stage_64x3_128x4_256x3"]
        two = rec["two_stage_64x7_256x3"]
        lines.append(f"## {label}")
        lines.append(f"- vocab {rec['vocab_size']}; tokens/word {rec['tokens_per_word']:.4f}.")
        lines.append(f"- Prefix-visible group fractions by length: L64 {rec['prefix_visible_group_fraction_by_length']['64']:.3f}, L128 {rec['prefix_visible_group_fraction_by_length']['128']:.3f}, L256 {rec['prefix_visible_group_fraction_by_length']['256']:.3f}.")
        lines.append(f"- For 64×3/128×4/256×3, prefix slicing misses {tri['prefix_missing_group_fraction']:.3f} of word-groups; faithful chunking provides {tri['faithful_chunking_target_token_ratio_vs_prefix']:.3f}× target tokens at {tri['faithful_chunking_optimizer_step_ratio_vs_prefix']:.3f}× optimizer steps.")
        lines.append(f"- For 64×7/256×3, prefix slicing misses {two['prefix_missing_group_fraction']:.3f} of word-groups; faithful chunking provides {two['faithful_chunking_target_token_ratio_vs_prefix']:.3f}× target tokens at {two['faithful_chunking_optimizer_step_ratio_vs_prefix']:.3f}× optimizer steps.\n")
    lines.append("## Route implication\n")
    lines.append(payload["route_implication"] + "\n")
    lines.append(f"JSON: `{OUT_JSON}`")
    NOTE.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(OUT_JSON),
        "note": str(NOTE),
        "legal40k_three_stage_missing": round(summaries["legal40k"]["three_stage_64x3_128x4_256x3"]["prefix_missing_group_fraction"], 4),
        "legal40k_three_stage_target_ratio": round(summaries["legal40k"]["three_stage_64x3_128x4_256x3"]["faithful_chunking_target_token_ratio_vs_prefix"], 4),
    }, indent=2))


if __name__ == "__main__":
    main()
