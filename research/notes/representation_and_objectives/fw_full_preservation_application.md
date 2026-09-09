# full ewok interaction synthesis full preservation-standard application

This applies the frozen full ewok interaction synthesis semantic-preservation standard to the old Qwen3.5 rewrites and the new Qwen3.5 outputs. The usable pair set is the only input that the preservation-aware arm materializer should use.

- Outputs: `experiments/archive/representation_and_objectives/training/runs/qwen35_fw_compact_full_26015/outputs.jsonl` (26,015 rows for 26,015 prompts)
- Overall usable: 22,820/38,167 (0.598)
- Usable pair words: 813,005
- compact_view/source_repeat pair totals match on retained source set: False

## By source kind

- existing_a02_qwen35: 8,386/12,152 usable (0.690), pair words 279,128
- new_qwen35_full26k: 14,434/26,015 usable (0.555), pair words 533,877

## Files

- Summary: `experiments/archive/representation_and_objectives/data/fw_full_preservation/full26k_standard_summary.json`
- Usable pair set: `experiments/archive/representation_and_objectives/data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl`
- Review samples: `experiments/archive/representation_and_objectives/data/fw_full_preservation/full26k_review_samples.json`
