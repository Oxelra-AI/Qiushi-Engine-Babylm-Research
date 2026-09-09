# fineweb random quality 3m profile FineWeb random_quality 3M profile

Evidence JSON: `experiments/archive/initial_model_studies/data/fineweb_random_quality_3m_profile.json`

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading_mean |
|---|---:|---:|---:|---:|---:|---:|
| chck_1M | 54.1600 | 44.8000 | 49.0900 | 17.5700 | 49.9600 | 6.7550 |
| chck_2M | 54.1900 | 44.8000 | 49.6400 | 17.5700 | 50.0000 | 6.7550 |
| chck_3M | 54.2200 | 44.4000 | 49.3600 | 17.5700 | 49.9400 | 6.7550 |

## Training/tokenization summary

{
  "parameter_count": 34467424,
  "word_exposure": 3000000,
  "steps": 147,
  "loss_first": 9.836081504821777,
  "loss_last": 7.287203788757324,
  "truncated_fraction": 0.19722666666666666,
  "kept_tokens_per_word": 1.449911,
  "masked_tokens_per_word": 0.21762733333333334,
  "saved_checkpoints": [
    {
      "name": "chck_1M",
      "target_word_exposure": 1000000,
      "actual_cumulative_word_exposure": 1003520,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_random_quality_debertav2_8x480_3M_b128_seed42/hf_model/chck_1M"
    },
    {
      "name": "chck_2M",
      "target_word_exposure": 2000000,
      "actual_cumulative_word_exposure": 2007040,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_random_quality_debertav2_8x480_3M_b128_seed42/hf_model/chck_2M"
    },
    {
      "name": "chck_3M",
      "target_word_exposure": 3000000,
      "actual_cumulative_word_exposure": 3000000,
      "path": "experiments/archive/initial_model_studies/training/runs/fineweb_random_quality_debertav2_8x480_3M_b128_seed42/hf_model/chck_3M"
    }
  ]
}
