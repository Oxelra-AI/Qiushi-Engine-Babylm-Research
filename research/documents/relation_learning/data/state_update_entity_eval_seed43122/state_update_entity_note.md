# earlier analysis state-update Entity deployment readout

Created UTC: 2026-09-06T21:50:02Z

## Official Entity scores

| seed | arm | checkpoint | score | elapsed s | predictions |
|---:|---|---|---:|---:|---|
| 43122 | base | chck_86M | +25.78 | 220.8 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43122/outputs/seed43122_base_chck_86M/Entity/chck_86M/seed43122_base_chck_86M/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43122 | intervention | chck_86M | +25.86 | 220.5 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43122/outputs/seed43122_intervention_chck_86M/Entity/chck_86M/seed43122_intervention_chck_86M/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43122 | base | final | +25.77 | 220.8 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43122/outputs/seed43122_base_final/Entity/hf_model/seed43122_base_final/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| 43122 | intervention | final | +26.23 | 221.2 | `experiments/archive/relation_learning/data/state_update_entity_eval_seed43122/outputs/seed43122_intervention_final/Entity/hf_model/seed43122_intervention_final/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |

## Within-seed intervention minus base by Entity operation structure

Positive delta means the state-update intervention is more accurate than its matched base for the same seed/checkpoint/item group.

| seed | ck | group | n | base | int | delta |
|---:|---|---|---:|---:|---:|---:|
| 43122 | chck_86M | ALL | 6780 | 26.05 | 26.31 | +0.27 |
| 43122 | chck_86M | rel_eq0 | 1541 | 40.69 | 38.55 | -2.14 |
| 43122 | chck_86M | rel_eq0_total_ops0 | 304 | 41.12 | 39.47 | -1.64 |
| 43122 | chck_86M | rel_eq0_irrelevant_ops_gt0 | 1237 | 40.58 | 38.32 | -2.26 |
| 43122 | chck_86M | rel_eq0_irrelevant_ops_1to3 | 641 | 39.16 | 39.47 | +0.31 |
| 43122 | chck_86M | rel_eq0_irrelevant_ops_4to6 | 323 | 43.65 | 38.70 | -4.95 |
| 43122 | chck_86M | rel_eq0_irrelevant_ops_ge7 | 273 | 40.29 | 35.16 | -5.13 |
| 43122 | chck_86M | rel_ge1 | 5239 | 21.74 | 22.71 | +0.97 |
| 43122 | chck_86M | rel_ge1_postrel_ops0 | 1850 | 25.46 | 24.81 | -0.65 |
| 43122 | chck_86M | rel_ge1_postrel_ops_gt0 | 3389 | 19.71 | 21.57 | +1.86 |
| 43122 | chck_86M | rel_updates_1 | 1323 | 15.27 | 17.38 | +2.12 |
| 43122 | chck_86M | rel_updates_2 | 1266 | 19.91 | 21.72 | +1.82 |
| 43122 | chck_86M | rel_updates_3 | 1230 | 23.50 | 23.50 | +0.00 |
| 43122 | chck_86M | rel_updates_4 | 1112 | 27.88 | 28.42 | +0.54 |
| 43122 | chck_86M | rel_updates_5 | 308 | 27.92 | 25.97 | -1.95 |
| 43122 | chck_86M | rel_ge3 | 2650 | 25.85 | 25.85 | +0.00 |
| 43122 | chck_86M | rel_ge3_postrel_ops0 | 1114 | 28.99 | 27.38 | -1.62 |
| 43122 | chck_86M | rel_ge3_postrel_ops_gt0 | 1536 | 23.57 | 24.74 | +1.17 |
| 43122 | chck_86M | stale_available_not_gold | 1221 | 23.42 | 24.24 | +0.82 |
| 43122 | final | ALL | 6780 | 26.18 | 26.45 | +0.27 |
| 43122 | final | rel_eq0 | 1541 | 40.30 | 38.81 | -1.49 |
| 43122 | final | rel_eq0_total_ops0 | 304 | 40.13 | 39.80 | -0.33 |
| 43122 | final | rel_eq0_irrelevant_ops_gt0 | 1237 | 40.34 | 38.56 | -1.78 |
| 43122 | final | rel_eq0_irrelevant_ops_1to3 | 641 | 39.63 | 40.09 | +0.47 |
| 43122 | final | rel_eq0_irrelevant_ops_4to6 | 323 | 43.03 | 39.63 | -3.41 |
| 43122 | final | rel_eq0_irrelevant_ops_ge7 | 273 | 38.83 | 33.70 | -5.13 |
| 43122 | final | rel_ge1 | 5239 | 22.03 | 22.81 | +0.78 |
| 43122 | final | rel_ge1_postrel_ops0 | 1850 | 25.30 | 24.97 | -0.32 |
| 43122 | final | rel_ge1_postrel_ops_gt0 | 3389 | 20.24 | 21.63 | +1.39 |
| 43122 | final | rel_updates_1 | 1323 | 15.65 | 17.23 | +1.59 |
| 43122 | final | rel_updates_2 | 1266 | 19.43 | 21.25 | +1.82 |
| 43122 | final | rel_updates_3 | 1230 | 24.15 | 24.31 | +0.16 |
| 43122 | final | rel_updates_4 | 1112 | 28.96 | 28.24 | -0.72 |
| 43122 | final | rel_updates_5 | 308 | 26.62 | 27.60 | +0.97 |
| 43122 | final | rel_ge3 | 2650 | 26.45 | 26.34 | -0.11 |
| 43122 | final | rel_ge3_postrel_ops0 | 1114 | 29.35 | 28.01 | -1.35 |
| 43122 | final | rel_ge3_postrel_ops_gt0 | 1536 | 24.35 | 25.13 | +0.78 |
| 43122 | final | stale_available_not_gold | 1221 | 22.93 | 24.32 | +1.39 |

## Across-seed / late compact rows

| checkpoint | group | rows | mean base | mean int | mean delta | se |
|---|---|---:|---:|---:|---:|---:|
| chck_86M | ALL | 1 | 26.05 | 26.31 | +0.27 | NA |
| chck_86M | rel_eq0 | 1 | 40.69 | 38.55 | -2.14 | NA |
| chck_86M | rel_eq0_total_ops0 | 1 | 41.12 | 39.47 | -1.64 | NA |
| chck_86M | rel_eq0_irrelevant_ops_gt0 | 1 | 40.58 | 38.32 | -2.26 | NA |
| chck_86M | rel_eq0_irrelevant_ops_1to3 | 1 | 39.16 | 39.47 | +0.31 | NA |
| chck_86M | rel_eq0_irrelevant_ops_4to6 | 1 | 43.65 | 38.70 | -4.95 | NA |
| chck_86M | rel_eq0_irrelevant_ops_ge7 | 1 | 40.29 | 35.16 | -5.13 | NA |
| chck_86M | rel_ge1 | 1 | 21.74 | 22.71 | +0.97 | NA |
| chck_86M | rel_ge1_postrel_ops0 | 1 | 25.46 | 24.81 | -0.65 | NA |
| chck_86M | rel_ge1_postrel_ops_gt0 | 1 | 19.71 | 21.57 | +1.86 | NA |
| chck_86M | rel_updates_1 | 1 | 15.27 | 17.38 | +2.12 | NA |
| chck_86M | rel_updates_2 | 1 | 19.91 | 21.72 | +1.82 | NA |
| chck_86M | rel_updates_3 | 1 | 23.50 | 23.50 | +0.00 | NA |
| chck_86M | rel_updates_4 | 1 | 27.88 | 28.42 | +0.54 | NA |
| chck_86M | rel_updates_5 | 1 | 27.92 | 25.97 | -1.95 | NA |
| chck_86M | rel_ge3 | 1 | 25.85 | 25.85 | +0.00 | NA |
| chck_86M | rel_ge3_postrel_ops0 | 1 | 28.99 | 27.38 | -1.62 | NA |
| chck_86M | rel_ge3_postrel_ops_gt0 | 1 | 23.57 | 24.74 | +1.17 | NA |
| chck_86M | stale_available_not_gold | 1 | 23.42 | 24.24 | +0.82 | NA |
| final | ALL | 1 | 26.18 | 26.45 | +0.27 | NA |
| final | rel_eq0 | 1 | 40.30 | 38.81 | -1.49 | NA |
| final | rel_eq0_total_ops0 | 1 | 40.13 | 39.80 | -0.33 | NA |
| final | rel_eq0_irrelevant_ops_gt0 | 1 | 40.34 | 38.56 | -1.78 | NA |
| final | rel_eq0_irrelevant_ops_1to3 | 1 | 39.63 | 40.09 | +0.47 | NA |
| final | rel_eq0_irrelevant_ops_4to6 | 1 | 43.03 | 39.63 | -3.41 | NA |
| final | rel_eq0_irrelevant_ops_ge7 | 1 | 38.83 | 33.70 | -5.13 | NA |
| final | rel_ge1 | 1 | 22.03 | 22.81 | +0.78 | NA |
| final | rel_ge1_postrel_ops0 | 1 | 25.30 | 24.97 | -0.32 | NA |
| final | rel_ge1_postrel_ops_gt0 | 1 | 20.24 | 21.63 | +1.39 | NA |
| final | rel_updates_1 | 1 | 15.65 | 17.23 | +1.59 | NA |
| final | rel_updates_2 | 1 | 19.43 | 21.25 | +1.82 | NA |
| final | rel_updates_3 | 1 | 24.15 | 24.31 | +0.16 | NA |
| final | rel_updates_4 | 1 | 28.96 | 28.24 | -0.72 | NA |
| final | rel_updates_5 | 1 | 26.62 | 27.60 | +0.97 | NA |
| final | rel_ge3 | 1 | 26.45 | 26.34 | -0.11 | NA |
| final | rel_ge3_postrel_ops0 | 1 | 29.35 | 28.01 | -1.35 | NA |
| final | rel_ge3_postrel_ops_gt0 | 1 | 24.35 | 25.13 | +0.78 | NA |
| final | stale_available_not_gold | 1 | 22.93 | 24.32 | +1.39 | NA |
| late_mean_over_checkpoints | ALL | 2 | 26.11 | 26.38 | +0.27 | +0.00 |
| late_mean_over_checkpoints | rel_eq0 | 2 | 40.49 | 38.68 | -1.82 | +0.32 |
| late_mean_over_checkpoints | rel_eq0_total_ops0 | 2 | 40.62 | 39.64 | -0.99 | +0.66 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_gt0 | 2 | 40.46 | 38.44 | -2.02 | +0.24 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_1to3 | 2 | 39.39 | 39.78 | +0.39 | +0.08 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_4to6 | 2 | 43.34 | 39.16 | -4.18 | +0.77 |
| late_mean_over_checkpoints | rel_eq0_irrelevant_ops_ge7 | 2 | 39.56 | 34.43 | -5.13 | +0.00 |
| late_mean_over_checkpoints | rel_ge1 | 2 | 21.88 | 22.76 | +0.88 | +0.10 |
| late_mean_over_checkpoints | rel_ge1_postrel_ops0 | 2 | 25.38 | 24.89 | -0.49 | +0.16 |
| late_mean_over_checkpoints | rel_ge1_postrel_ops_gt0 | 2 | 19.98 | 21.60 | +1.62 | +0.24 |
| late_mean_over_checkpoints | rel_updates_1 | 2 | 15.46 | 17.31 | +1.85 | +0.26 |
| late_mean_over_checkpoints | rel_updates_2 | 2 | 19.67 | 21.48 | +1.82 | +0.00 |
| late_mean_over_checkpoints | rel_updates_3 | 2 | 23.82 | 23.90 | +0.08 | +0.08 |
| late_mean_over_checkpoints | rel_updates_4 | 2 | 28.42 | 28.33 | -0.09 | +0.63 |
| late_mean_over_checkpoints | rel_updates_5 | 2 | 27.27 | 26.79 | -0.49 | +1.46 |
| late_mean_over_checkpoints | rel_ge3 | 2 | 26.15 | 26.09 | -0.06 | +0.06 |
| late_mean_over_checkpoints | rel_ge3_postrel_ops0 | 2 | 29.17 | 27.69 | -1.48 | +0.13 |
| late_mean_over_checkpoints | rel_ge3_postrel_ops_gt0 | 2 | 23.96 | 24.93 | +0.98 | +0.20 |
| late_mean_over_checkpoints | stale_available_not_gold | 2 | 23.18 | 24.28 | +1.11 | +0.29 |

## Interpretation reminders

The pre-scored prediction was gain on `rel_eq0_irrelevant_ops_gt0` and cost/flat behavior on relevant-update items if the arm learned a coarse source-retention/anti-recency relation rather than entity-gated state binding. A rel_ge3 gain would contradict that margin-based reading. Keep the 99,909,920-word exposure and replacement of inherited ALN rows visible; this script does not run cheap7, ordinary held-out, SuperGLUE, or AoA.
