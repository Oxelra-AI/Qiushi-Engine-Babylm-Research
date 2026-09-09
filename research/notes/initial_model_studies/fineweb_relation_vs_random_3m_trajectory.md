# fineweb random quality 3m profile FineWeb relation-explicit vs random-quality 3M trajectory

Evidence JSON: `experiments/archive/initial_model_studies/data/fineweb_relation_vs_random_3m_trajectory.json`

| checkpoint | ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading |
|---|---:|---:|---:|---:|---:|---:|
| chck_1M | -1.0000 | +1.6000 | +1.8200 | +0.3000 | +0.2200 | +0.0850 |
| chck_2M | -1.0200 | +2.4000 | +1.7200 | +0.3000 | +0.0200 | +0.0850 |
| chck_3M | -0.7800 | +2.4000 | +0.0900 | +0.3000 | +0.3700 | +0.0850 |

## Absolute scores

### chck_1M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 54.1600 | 44.8000 | 49.0900 | 17.5700 | 49.9600 | 6.7550 |
| relation_explicit | 53.1600 | 46.4000 | 50.9100 | 17.8700 | 50.1800 | 6.8400 |

### chck_2M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 54.1900 | 44.8000 | 49.6400 | 17.5700 | 50.0000 | 6.7550 |
| relation_explicit | 53.1700 | 47.2000 | 51.3600 | 17.8700 | 50.0200 | 6.8400 |

### chck_3M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 54.2200 | 44.4000 | 49.3600 | 17.5700 | 49.9400 | 6.7550 |
| relation_explicit | 53.4400 | 46.8000 | 49.4500 | 17.8700 | 50.3100 | 6.8400 |

## Training/tokenization summaries

```json
{
  "random": {
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
  },
  "relation": {
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
}
```
