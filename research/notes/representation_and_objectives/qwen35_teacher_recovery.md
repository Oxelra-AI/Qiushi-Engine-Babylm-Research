# qwen35 teacher recovery — Qwen3.5-9B teacher recovery quality

This analysis checks the original Qwen3.5-9B compact-view generation process on the FW mechanism-route sources before any BabyLM pretraining is launched.

Prompts: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_recovery_pilot_prompts.jsonl`

Outputs: `experiments/archive/representation_and_objectives/training/runs/qwen35_recovery_pilot256/outputs.jsonl`

Rows merged: 256; core accepted: 157 (0.613); structure-strict accepted: 37 (0.145).

Accepted structure-strict rewrite words: 644.

## Main measurements

- Compression ratio median/all: 0.641941391941392 (mean 0.6463450108281191).
- Content recall median/all: 0.6 (mean 0.5972312415670227).
- Copy-like fraction: 0.008.
- Entity loss when source has entities: 0.352; number loss when source has numbers: 0.023.
- Negation loss: 0.371; modality loss: 0.426; causal low-recall: 0.355; comparison low-recall: 0.312.

## Files

- Summary JSON: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_generation_quality_summary.json`
- Merged rows: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_generation_merged_quality.jsonl`
- Accepted structure-strict rows: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_generation_accepted_structure_strict.jsonl`
