# full ewok interaction synthesis — FW compact-view preservation standard and identical source-set comparison

The FW mechanism-scale route should test broad faithful compact consolidation, not simply pair volume. I froze a single preservation standard and applied it to both already trusted companion analysis/Qwen3.5 rewrites and the new Qwen3.5 pilot outputs. The same usable source hashes define compact_view and source_repeat arms; source_repeat uses the first `rewrite_words` words of the identical source sentence as the companion text, so pair word totals match exactly.

## Overall available measurement

- Rows measured: 12,408
- Usable rows: 8,475 (0.683)
- Usable source words: 173,727
- Usable rewrite words: 109,391
- Usable pair words: 283,118
- Usable rewrite/source ratio: 0.630

## By source kind

- existing_a02_qwen35: 8,386/12,152 usable (0.690), pair words 279,128, ratio 0.629
- new_qwen35_pilot: 89/256 usable (0.348), pair words 3,990, ratio 0.676

## Most common hard reasons

- modality: 1682
- comparison_direction: 1241
- causal_relation: 1046
- polarity: 501
- missing_entities: 251
- missing_numbers: 11
- new_numbers: 8
- low_content_recall: 4
- bad_start: 1

## Identical source-set arm comparison

- Sources retained: 8,475
- compact_view pair words: 283,118
- source_repeat pair words: 283,118
- Pair-word totals match: True

## Files

- Standard: `experiments/archive/representation_and_objectives/data/fw_preservation_standard/preservation_standard_v1.json`
- Summary: `experiments/archive/representation_and_objectives/data/fw_preservation_standard/standard_summary.json`
- Usable pair set: `experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_usable_pairs_for_materializer.jsonl`
- Review samples: `experiments/archive/representation_and_objectives/data/fw_preservation_standard/manual_review_samples.json`
