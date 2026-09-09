# b256 fixedseq pair eval raw — earlier analysis alignment comparison

JSON: `experiments/archive/compact_experience/data/b256_pair_eval/alignment_comparison.json`

This compares the old INITIAL_MODEL_STUDIES earlier analysis fixed-WWM run with the COMPACT_EXPERIENCE b256/fixed-seq fixed-WWM rerun after the earlier analysis-style direct recheck showed an 8-point Supplement mismatch.

- Source-word accounting equal: `True`.
- Selection-epoch summaries equal: `True`.
- Checkpoint tokenizer SHA equal: `True`.
- INITIAL_MODEL_STUDIES manifest stores exact consumed IDs: `True`; COMPACT_EXPERIENCE manifest stores exact consumed IDs: `False`.

## Core metric differences

```json
{
  "deberta_pos_att_type": {
    "INITIAL_MODEL_STUDIES": "p2c,c2p",
    "COMPACT_EXPERIENCE": null
  },
  "deberta_relative_attention": {
    "INITIAL_MODEL_STUDIES": "true",
    "COMPACT_EXPERIENCE": null
  },
  "embedding_parameter_count": {
    "INITIAL_MODEL_STUDIES": 7864320,
    "COMPACT_EXPERIENCE": null
  },
  "example_pool_words_actual": {
    "INITIAL_MODEL_STUDIES": 10000000,
    "COMPACT_EXPERIENCE": null
  },
  "extra_init_seed": {
    "INITIAL_MODEL_STUDIES": -1,
    "COMPACT_EXPERIENCE": null
  },
  "intermediate_size": {
    "INITIAL_MODEL_STUDIES": 1920,
    "COMPACT_EXPERIENCE": null
  },
  "loss_first": {
    "INITIAL_MODEL_STUDIES": 9.814163208007812,
    "COMPACT_EXPERIENCE": 9.801941871643066
  },
  "loss_last": {
    "INITIAL_MODEL_STUDIES": 2.608774185180664,
    "COMPACT_EXPERIENCE": 2.5929057598114014
  },
  "lr_schedule_total_steps": {
    "INITIAL_MODEL_STUDIES": 2442,
    "COMPACT_EXPERIENCE": null
  },
  "mask_mode": {
    "INITIAL_MODEL_STUDIES": "wwm",
    "COMPACT_EXPERIENCE": null
  },
  "mask_prob": {
    "INITIAL_MODEL_STUDIES": 0.15,
    "COMPACT_EXPERIENCE": null
  },
  "masked_tokens_mean_per_step": {
    "INITIAL_MODEL_STUDIES": 8871.17076167076,
    "COMPACT_EXPERIENCE": null
  },
  "masked_tokens_per_whitespace_word": {
    "INITIAL_MODEL_STUDIES": 0.21663399,
    "COMPACT_EXPERIENCE": null
  },
  "masked_tokens_total": {
    "INITIAL_MODEL_STUDIES": 21663399,
    "COMPACT_EXPERIENCE": null
  },
  "max_relative_positions": {
    "INITIAL_MODEL_STUDIES": 256,
    "COMPACT_EXPERIENCE": null
  },
  "model_type": {
    "INITIAL_MODEL_STUDIES": "deberta_v2",
    "COMPACT_EXPERIENCE": null
  },
  "non_embedding_parameter_count": {
    "INITIAL_MODEL_STUDIES": 26603104,
    "COMPACT_EXPERIENCE": null
  },
  "position_buckets": {
    "INITIAL_MODEL_STUDIES": 256,
    "COMPACT_EXPERIENCE": null
  },
  "selected_for_training_words": {
    "INITIAL_MODEL_STUDIES": 100000000,
    "COMPACT_EXPERIENCE": null
  },
  "tokenizer_path": {
    "INITIAL_MODEL_STUDIES": "",
    "COMPACT_EXPERIENCE": null
  },
  "train_rng_seed": {
    "INITIAL_MODEL_STUDIES": -1,
    "COMPACT_EXPERIENCE": null
  },
  "variant": {
    "INITIAL_MODEL_STUDIES": "masked_wwm",
    "COMPACT_EXPERIENCE": "masking_curriculum_wwm_fixed"
  }
}
```

## Core config differences

```json
{}
```

## Interpretation

The two runs match the high-level data source word counts and selection-epoch summary, and use the same checkpoint tokenizer. They do not have the same final weights, and the COMPACT_EXPERIENCE manifest lacks the full consumed-example ID list, so exact data order identity cannot be verified from COMPACT_EXPERIENCE artifacts. The old INITIAL_MODEL_STUDIES trainer and COMPACT_EXPERIENCE curriculum trainer must be compared at implementation/RNG level before treating the b256 rerun as the earlier analysis coordinate.
