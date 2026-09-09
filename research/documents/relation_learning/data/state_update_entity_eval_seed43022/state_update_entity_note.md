# earlier analysis state-update Entity deployment readout

Created UTC: 2026-09-06T21:56:55Z

## Official Entity scores

| seed | arm | checkpoint | score | elapsed s | predictions |
|---:|---|---|---:|---:|---|
| 43022 | base | chck_86M | +28.58 | 363.8 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43022/outputs/seed43022_base_chck_86M/Entity/chck_86M/seed43022_base_chck_86M/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43022 | intervention | chck_86M | +25.70 | 312.1 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43022/outputs/seed43022_intervention_chck_86M/Entity/chck_86M/seed43022_intervention_chck_86M/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43022 | base | final | +27.46 | 326.7 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43022/outputs/seed43022_base_final/Entity/hf_model/seed43022_base_final/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43022 | intervention | final | +25.72 | 294.7 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43022/outputs/seed43022_intervention_final/Entity/hf_model/seed43022_intervention_final/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |

## Within-seed intervention minus base by Entity operation structure

Positive delta means the state-update intervention is more accurate than its matched base for the same seed/checkpoint/item group.

| seed | ck | group | n | base | int | delta |
|---:|---|---|---:|---:|---:|---:|
| 43022 | chck_86M | ALL | 6780 | 28.20 | 25.84 | -2.36 |
| 43022 | chck_86M | rel_eq0 | 1541 | 37.51 | 39.07 | +1.56 |
| 43022 | chck_86M | rel_eq0_total_ops0 | 304 | 35.53 | 39.47 | +3.95 |
| 43022 | chck_86M | rel_eq0_irrelevant_ops_gt0 | 1237 | 38.00 | 38.97 | +0.97 |
| 43022 | chck_86M | rel_eq0_irrelevant_ops_1to3 | 641 | 38.07 | 38.85 | +0.78 |
| 43022 | chck_86M | rel_eq0_irrelevant_ops_4to6 | 323 | 41.18 | 41.49 | +0.31 |
| 43022 | chck_86M | rel_eq0_irrelevant_ops_ge7 | 273 | 34.07 | 36.26 | +2.20 |
| 43022 | chck_86M | rel_ge1 | 5239 | 25.46 | 21.95 | -3.51 |
| 43022 | chck_86M | rel_ge1_postrel_ops0 | 1850 | 30.43 | 25.24 | -5.19 |
| 43022 | chck_86M | rel_ge1_postrel_ops_gt0 | 3389 | 22.75 | 20.15 | -2.60 |
| 43022 | chck_86M | rel_updates_1 | 1323 | 16.93 | 16.10 | -0.83 |
| 43022 | chck_86M | rel_updates_2 | 1266 | 23.70 | 20.22 | -3.48 |
| 43022 | chck_86M | rel_updates_3 | 1230 | 26.99 | 24.88 | -2.11 |
| 43022 | chck_86M | rel_updates_4 | 1112 | 33.99 | 26.17 | -7.82 |
| 43022 | chck_86M | rel_updates_5 | 308 | 32.47 | 27.27 | -5.19 |
| 43022 | chck_86M | rel_ge3 | 2650 | 30.57 | 25.70 | -4.87 |
| 43022 | chck_86M | rel_ge3_postrel_ops0 | 1114 | 35.37 | 28.99 | -6.37 |
| 43022 | chck_86M | rel_ge3_postrel_ops_gt0 | 1536 | 27.08 | 23.31 | -3.78 |
| 43022 | chck_86M | stale_available_not_gold | 1221 | 27.76 | 23.01 | -4.75 |
| 43022 | final | ALL | 6780 | 27.35 | 25.94 | -1.40 |
| 43022 | final | rel_eq0 | 1541 | 38.68 | 40.62 | +1.95 |
| 43022 | final | rel_eq0_total_ops0 | 304 | 36.18 | 40.46 | +4.28 |
| 43022 | final | rel_eq0_irrelevant_ops_gt0 | 1237 | 39.29 | 40.66 | +1.37 |
| 43022 | final | rel_eq0_irrelevant_ops_1to3 | 641 | 40.41 | 39.47 | -0.94 |
| 43022 | final | rel_eq0_irrelevant_ops_4to6 | 323 | 43.03 | 43.65 | +0.62 |
| 43022 | final | rel_eq0_irrelevant_ops_ge7 | 273 | 32.23 | 39.93 | +7.69 |
| 43022 | final | rel_ge1 | 5239 | 24.01 | 21.63 | -2.39 |
| 43022 | final | rel_ge1_postrel_ops0 | 1850 | 28.22 | 24.76 | -3.46 |
| 43022 | final | rel_ge1_postrel_ops_gt0 | 3389 | 21.72 | 19.92 | -1.80 |
| 43022 | final | rel_updates_1 | 1323 | 16.40 | 15.19 | -1.21 |
| 43022 | final | rel_updates_2 | 1266 | 22.59 | 19.75 | -2.84 |
| 43022 | final | rel_updates_3 | 1230 | 26.10 | 24.72 | -1.38 |
| 43022 | final | rel_updates_4 | 1112 | 30.40 | 26.44 | -3.96 |
| 43022 | final | rel_updates_5 | 308 | 31.17 | 27.27 | -3.90 |
| 43022 | final | rel_ge3 | 2650 | 28.49 | 25.74 | -2.75 |
| 43022 | final | rel_ge3_postrel_ops0 | 1114 | 32.59 | 28.64 | -3.95 |
| 43022 | final | rel_ge3_postrel_ops_gt0 | 1536 | 25.52 | 23.63 | -1.89 |
| 43022 | final | stale_available_not_gold | 1221 | 25.88 | 23.59 | -2.29 |

## Across-seed / late compact rows

| checkpoint | group | rows | mean base | mean int | mean delta | se |
|---|---|---:|---:|---:|---:|---:|
| chck_86M | ALL | 1 | 28.20 | 25.84 | -2.36 | NA |
| chck_86M | rel_eq0 | 1 | 37.51 | 39.07 | +1.56 | NA |
| chck_86M | rel_eq0_total_ops0 | 1 | 35.53 | 39.47 | +3.95 | NA |
| chck_86M | rel_eq0_irrelevant_ops_gt0 | 1 | 38.00 | 38.97 | +0.97 | NA |
| chck_86M | rel_eq0_irrelevant_ops_1to3 | 1 | 38.07 | 38.85 | +0.78 | NA |
| chck_86M | rel_eq0_irrelevant_ops_4to6 | 1 | 41.18 | 41.49 | +0.31 | NA |
| chck_86M | rel_eq0_irrelevant_ops_ge7 | 1 | 34.07 | 36.26 | +2.20 | NA |
| chck_86M | rel_ge1 | 1 | 25.46 | 21.95 | -3.51 | NA |
| chck_86M | rel_ge1_postrel_ops0 | 1 | 30.43 | 25.24 | -5.19 | NA |
| chck_86M | rel_ge1_postrel_ops_gt0 | 1 | 22.75 | 20.15 | -2.60 | NA |
| chck_86M | rel_updates_1 | 1 | 16.93 | 16.10 | -0.83 | NA |
| chck_86M | rel_updates_2 | 1 | 23.70 | 20.22 | -3.48 | NA |
| chck_86M | rel_updates_3 | 1 | 26.99 | 24.88 | -2.11 | NA |
| chck_86M | rel_updates_4 | 1 | 33.99 | 26.17 | -7.82 | NA |
| chck_86M | rel_updates_5 | 1 | 32.47 | 27.27 | -5.19 | NA |
| chck_86M | rel_ge3 | 1 | 30.57 | 25.70 | -4.87 | NA |
| chck_86M | rel_ge3_postrel_ops0 | 1 | 35.37 | 28.99 | -6.37 | NA |
| chck_86M | rel_ge3_postrel_ops_gt0 | 1 | 27.08 | 23.31 | -3.78 | NA |
| chck_86M | stale_available_not_gold | 1 | 27.76 | 23.01 | -4.75 | NA |
| final | ALL | 1 | 27.35 | 25.94 | -1.40 | NA |
| final | rel_eq0 | 1 | 38.68 | 40.62 | +1.95 | NA |
| final | rel_eq0_total_ops0 | 1 | 36.18 | 40.46 | +4.28 | NA |
| final | rel_eq0_irrelevant_ops_gt0 | 1 | 39.29 | 40.66 | +1.37 | NA |
| final | rel_eq0_irrelevant_ops_1to3 | 1 | 40.41 | 39.47 | -0.94 | NA |
| final | rel_eq0_irrelevant_ops_4to6 | 1 | 43.03 | 43.65 | +0.62 | NA |
| final | rel_eq0_irrelevant_ops_ge7 | 1 | 32.23 | 39.93 | +7.69 | NA |
| final | rel_ge1 | 1 | 24.01 | 21.63 | -2.39 | NA |
| final | rel_ge1_postrel_ops0 | 1 | 28.22 | 24.76 | -3.46 | NA |
| final | rel_ge1_postrel_ops_gt0 | 1 | 21.72 | 19.92 | -1.80 | NA |
| final | rel_updates_1 | 1 | 16.40 | 15.19 | -1.21 | NA |
| final | rel_updates_2 | 1 | 22.59 | 19.75 | -2.84 | NA |
| final | rel_updates_3 | 1 | 26.10 | 24.72 | -1.38 | NA |
| final | rel_updates_4 | 1 | 30.40 | 26.44 | -3.96 | NA |
| final | rel_updates_5 | 1 | 31.17 | 27.27 | -3.90 | NA |
| final | rel_ge3 | 1 | 28.49 | 25.74 | -2.75 | NA |
| final | rel_ge3_postrel_ops0 | 1 | 32.59 | 28.64 | -3.95 | NA |
| final | rel_ge3_postrel_ops_gt0 | 1 | 25.52 | 23.63 | -1.89 | NA |
| final | stale_available_not_gold | 1 | 25.88 | 23.59 | -2.29 | NA |
| late_mean_over_checkpoints | ALL | 2 | 27.77 | 25.89 | -1.88 | +0.48 |
| late_mean_over_checkpoints | rel_eq0 | 2 | 38.09 | 39.84 | +1.75 | +0.19 |
| late_mean_over_checkpoints | rel_eq0_total_ops0 | 2 | 35.86 | 39.97 | +4.11 | +0.16 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_gt0 | 2 | 38.64 | 39.81 | +1.17 | +0.20 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_1to3 | 2 | 39.24 | 39.16 | -0.08 | +0.86 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_4to6 | 2 | 42.11 | 42.57 | +0.46 | +0.15 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_ge7 | 2 | 33.15 | 38.10 | +4.95 | +2.75 |
| late_mean_over_checkpoints | rel_ge1 | 2 | 24.74 | 21.79 | -2.95 | +0.56 |
| late_mean_over_checkpoints | rel_ge1_postrel_ops0 | 2 | 29.32 | 25.00 | -4.32 | +0.86 |
| late_mean_over_checkpoints | rel_ge1_postrel_ops_gt0 | 2 | 22.23 | 20.04 | -2.20 | +0.40 |
| late_mean_over_checkpoints | rel_updates_1 | 2 | 16.67 | 15.65 | -1.02 | +0.19 |
| late_mean_over_checkpoints | rel_updates_2 | 2 | 23.14 | 19.98 | -3.16 | +0.32 |
| late_mean_over_checkpoints | rel_updates_3 | 2 | 26.54 | 24.80 | -1.75 | +0.37 |
| late_mean_over_checkpoints | rel_updates_4 | 2 | 32.19 | 26.30 | -5.89 | +1.93 |
| late_mean_over_checkpoints | rel_updates_5 | 2 | 31.82 | 27.27 | -4.55 | +0.65 |
| late_mean_over_checkpoints | rel_ge3 | 2 | 29.53 | 25.72 | -3.81 | +1.06 |
| late_mean_over_checkpoints | rel_ge3_postrel_ops0 | 2 | 33.98 | 28.82 | -5.16 | +1.21 |
| late_mean_over_checkpoints | rel_ge3_postrel_ops_gt0 | 2 | 26.30 | 23.47 | -2.83 | +0.94 |
| late_mean_over_checkpoints | stale_available_not_gold | 2 | 26.82 | 23.30 | -3.52 | +1.23 |

## Interpretation reminders

The pre-scored prediction was gain on `rel_eq0_irrelevant_ops_gt0` and cost/flat behavior on relevant-update items if the arm learned a coarse source-retention/anti-recency relation rather than entity-gated state binding. A rel_ge3 gain would contradict that margin-based reading. Keep the 99,909,920-word exposure and replacement of inherited ALN rows visible; this script does not run cheap7, ordinary held-out, SuperGLUE, or AoA.
