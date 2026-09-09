# Historical rewrite-generation configuration

## Generation configuration
- **Model and requested sample count**: Qwen3.5-9B, 45,000 rewrites, batch-64.
- **Expected output**: `experiments/archive/compact_experience/training/runs/qwen_rewrites_full/outputs.jsonl`

These are historical configuration values, not a completed-generation count or an accepted-pair count.

## Construction and evaluation sequence

The proposed sequence was validation, materialization, training and full nine-column evaluation. The historical training estimate was ~90 min on 2 H100s; this note does not establish a measured completion time. The initial materialization procedure was superseded because pair truncation and unrelated-fragment padding confounded the alignment test; complete-pair construction and its measured acceptance counts must be used for interpretation.

## Unvalidated reduced-data alternatives
The original plan considered partial `outputs.jsonl` with > 5,000 lines or the first N lines of `data/qwen_aligned/rewrite_prompts.jsonl`. Neither threshold establishes semantic validity or a clean accepted-pair dose, and this note does not report an experiment using either alternative.

## Key Configuration
- Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`
- Seeds: extra_init_seed=43022, train_rng_seed=43023, seed=43
- Checkpoints: every 1M words (chck_1M...chck_100M for AoA)
- Trainer requires EXACT 160 words per JSONL row (strict validation)
