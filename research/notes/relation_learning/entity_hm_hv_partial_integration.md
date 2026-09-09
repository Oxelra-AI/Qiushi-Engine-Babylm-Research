# entity hm hv partial integration HM/HV Entity relevant-update integration

This note repairs the corrected two component frame Entity parser by joining official prediction blocks keyed by UID to `entity_item_metadata.csv` with `(uid,item_index)`, following the price tradeoff parser. It treats HM/HV as a test of identity-practice increments in Entity behavior, not as evidence for a single latent trust component.

## Missing checkpoint run

The previously missing HV chck_100M evaluation completed successfully in entity hm hv partial integration: official score 25.44; predictions and report are under `experiments/archive/relation_learning/data/entity_hm_hv/eval_output/HV_chck_100M`.

## Available official predictions

| arm | checkpoint | available | score |
|---|---|---:|---:|
| HM | chck_80M | True | 27.91 |
| HM | chck_90M | True | 28.71 |
| HM | chck_100M | True | 28.35 |
| HV | chck_80M | True | 24.65 |
| HV | chck_90M | True | 25.28 |
| HV | chck_100M | True | 25.44 |

## Late accuracy by relevant-update group

| group | HM acc | HV acc | HM−HV | C acc | R acc | V acc |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 28.19 | 25.35 | +2.83 | 25.04 | 25.32 | 27.80 |
| rel_eq0 | 39.78 | 41.73 | -1.95 | 39.45 | 48.24 | 38.55 |
| rel_ge1 | 24.78 | 20.54 | +4.24 | 20.81 | 18.58 | 24.64 |
| rel_ge2 | 27.50 | 22.81 | +4.69 | 22.61 | 20.11 | 27.46 |
| rel_ge3 | 29.69 | 23.79 | +5.90 | 24.34 | 20.73 | 29.62 |
| rel_updates_0 | 39.78 | 41.73 | -1.95 | 39.45 | 48.24 | 38.55 |
| rel_updates_1 | 16.70 | 13.81 | +2.90 | 15.47 | 14.08 | 16.28 |
| rel_updates_2 | 22.93 | 20.77 | +2.16 | 18.98 | 18.80 | 22.93 |
| rel_updates_3 | 26.75 | 21.25 | +5.50 | 23.63 | 17.80 | 26.26 |
| rel_updates_4 | 32.25 | 25.15 | +7.10 | 25.30 | 23.71 | 32.85 |
| rel_updates_5 | 32.14 | 29.00 | +3.14 | 23.70 | 21.65 | 31.39 |

## Scientific reading

The targeted zero-update discriminator goes against the identity-increment interpretation: HV exceeds HM at rel_eq0 by 1.95 points (41.73 vs 39.78). HM is essentially at CLEAN on rel_eq0 (C 39.45, HM 39.78) and far below REPEAT (48.24). Thus the exact-copy half in HM does not add the same zero-update Entity increment that full REPEAT adds.

HM is instead much better than HV on updated states, tracking VIEW at rel_ge3 (HM 29.69, V 29.62, HV 23.79). Together with corrected composition and trust compact/natural evidence, this means HM/HV is not a clean decomposition of a generic pairing component. HM can acquire REPEAT-level copy gain and full natural substitution liability while preserving VIEW-like updated-state behavior, and HV's neutral-filling construction changes the Entity profile in its own way.

Use this result in the report as a reason to move any broad trust interpretation out of the headline. The stable scientific object remains relation-typed readout and reach asymmetry.

## Files

- `experiments/archive/relation_learning/data/entity_hm_hv_integration/prediction_inventory.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_prediction_rows.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_accuracy_by_checkpoint_group.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_late_accuracy_by_group.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_hm_hv_shared_checkpoint_contrasts_by_group.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/entity_with_baseline_partial_contrasts_by_group.csv`
- `experiments/archive/relation_learning/data/entity_hm_hv_integration/result.json`
