# fineweb random quality 3m profile FineWeb relation_explicit 3M profile

Evidence JSON: `experiments/archive/initial_model_studies/data/fineweb_relation_explicit_3m_profile.json`

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading_mean |
|---|---:|---:|---:|---:|---:|---:|
| chck_1M | 53.1600 | 46.4000 | 50.9100 | 17.8700 | 50.1800 | 6.8400 |
| chck_2M | 53.1700 | 47.2000 | 51.3600 | 17.8700 | 50.0200 | 6.8400 |
| chck_3M | 53.4400 | 46.8000 | 49.4500 | 17.8700 | 50.3100 | 6.8400 |

## Training/tokenization summary

{
  "parameter_count": 34467424,
  "word_exposure": 3000000,
  "steps": 147,
  "loss_first": 9.824013710021973,
  "loss_last": 7.410793781280518,
  "truncated_fraction": 0.19552,
  "kept_tokens_per_word": 1.4518386666666667,
  "masked_tokens_per_word": 0.21729633333333334,
  "saved_checkpoints": [
    {
      "name": "chck_1M",
      "target_word_exposure": 1000000,
      "actual_cumulative_word_exposure": 1003520,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_relation_explicit_debertav2_8x480_3M_b128_seed42/hf_model/chck_1M"
    },
    {
      "name": "chck_2M",
      "target_word_exposure": 2000000,
      "actual_cumulative_word_exposure": 2007040,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_relation_explicit_debertav2_8x480_3M_b128_seed42/hf_model/chck_2M"
    },
    {
      "name": "chck_3M",
      "target_word_exposure": 3000000,
      "actual_cumulative_word_exposure": 3000000,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_relation_explicit_debertav2_8x480_3M_b128_seed42/hf_model/chck_3M"
    }
  ]
}
