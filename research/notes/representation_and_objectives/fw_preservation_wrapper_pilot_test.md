# full ewok interaction synthesis full preservation-standard application

This applies the frozen full ewok interaction synthesis semantic-preservation standard to the old Qwen3.5 rewrites and the new Qwen3.5 outputs. The usable pair set is the only input that the preservation-aware arm materializer should use.

- Outputs: `experiments/archive/representation_and_objectives/training/runs/qwen35_recovery_pilot256/outputs.jsonl` (256 rows for 256 prompts)
- Overall usable: 8,475/12,408 (0.683)
- Usable pair words: 283,118
- compact_view/source_repeat pair totals match on retained source set: True

## By source kind

- existing_a02_qwen35: 8,386/12,152 usable (0.690), pair words 279,128
- new_qwen35_pilot: 89/256 usable (0.348), pair words 3,990

## Files

- Summary: `experiments/archive/representation_and_objectives/data/fw_preservation_wrapper_pilot_test/pilot_test_standard_summary.json`
- Usable pair set: `experiments/archive/representation_and_objectives/data/fw_preservation_wrapper_pilot_test/pilot_test_usable_pairs_for_materializer.jsonl`
- Review samples: `experiments/archive/representation_and_objectives/data/fw_preservation_wrapper_pilot_test/pilot_test_review_samples.json`
