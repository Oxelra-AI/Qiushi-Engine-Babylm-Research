# legal bsm 1m eval — legal BSM 1M three-arm evaluation

Evidence JSON: `experiments/archive/initial_model_studies/data/legal_bsm_1m_eval.json`

**Attribution warning:** coherent−official is not a pure binding effect because BSM arms have about 3× optimizer updates and different row packing/targeted masking. Coherent−swapped is the cleaner current comparison; if it separates, a matched-update official-experience reference is needed before scaling.

## Official fast scores

| model | BLiMP | Supplement | Entity | EWoK | GPIQA-par | GPIQA-non | GPIQA-mean | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official_control | 54.31 | 48.8 | 17.99 | 48.73 | 18.45 | 56.0 | 37.225 | 6.470000000000001 |
| bsm_coherent_20pct | 55.29 | 47.6 | 19.85 | 51.0 | 15.53 | 51.0 | 33.265 | 6.13 |
| bsm_swapped_20pct | 53.35 | 50.4 | 19.12 | 50.45 | 19.42 | 54.0 | 36.71 | 6.8100000000000005 |

## Binding probes: both-correct fraction

| model | train | heldout_entities | heldout_values | heldout_templates | order_flip | natural |
|---|---:|---:|---:|---:|---:|---:|
| official_control | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bsm_coherent_20pct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bsm_swapped_20pct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Coherent minus swapped deltas

```json
{
  "official_fast": {
    "blimp_fast": 1.9399999999999977,
    "supplement_fast": -2.799999999999997,
    "entity_tracking_fast": 0.7300000000000004,
    "ewok_fast": 0.5499999999999972,
    "global_piqa_parallel": -3.8900000000000023,
    "global_piqa_nonparallel": -3.0,
    "reading_eye_tracking": -1.6300000000000008,
    "reading_self_paced": 0.27,
    "reading_mean": -0.6800000000000006
  },
  "binding_probes": {
    "train_both_correct": 0.0,
    "heldout_entities_both_correct": 0.0,
    "heldout_values_both_correct": 0.0,
    "heldout_templates_both_correct": 0.0,
    "heldout_order_flip_both_correct": 0.0,
    "natural_templates_both_correct": 0.0
  }
}
```
