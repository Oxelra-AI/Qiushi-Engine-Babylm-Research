# babylm scoreboard BabyLM Strict-Small 9/9 scoreboard

Purpose: keep official Overall progress visible while mechanism work proceeds. Partial coordinates are recorded but not treated as SOTA.

## Current complete coordinates

| model | type | BLiMP | Supp | EWoK | Entity | COMPS | SG | GPIQA | Reading | AoA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| public_leader_go76dof_wwm_curriculum_simplification_40k | external_public_leader | 67.200 | 56.040 | 56.070 | 28.450 | 53.570 | 69.790 | 39.665 | 5.425 | 0.000 | 41.801 |
| protected_internal_debertav2_8x480_wwm_100M | protected_internal_complete_9of9 | 66.760 | 59.880 | 52.190 | 22.620 | 52.190 | 68.022 | 35.635 | 7.620 | -0.174 | 40.527 |
| S1_leader_shape_12x384_baseline16k_100M | internal_partial_7of9 | 66.840 | 60.310 | 52.020 | 20.240 | 52.260 |  | 37.605 | 7.250 |  |  |
| S2_true_curriculum_12x384_baseline16k_100M | internal_partial_7of9 | 64.240 | 59.090 | 51.640 | 18.470 | 50.610 |  | 38.635 | 7.520 |  |  |

## Protected internal best

- Protected internal complete 9/9: **protected_internal_debertav2_8x480_wwm_100M**, Overall **40.5269**.

- Public leader Overall: **41.8011**. Gap: **1.2742 Overall** (11.468 score-sum).

- Column gaps protected-minus-leader:

  - BLiMP: -0.440
  - BLiMP Supplement: +3.840
  - EWoK: -3.880
  - Entity Tracking: -5.830
  - COMPS: -1.380
  - (Super)GLUE: -1.768
  - GlobalPIQA: -4.030
  - Reading: +2.195
  - AoA: -0.174

## Partial-coordinate warning

- S1_leader_shape_12x384_baseline16k_100M: known sum 296.525, missing ['(Super)GLUE', 'AoA']; needs missing-column sum > 79.685 to exceed leader.
  - Equivalent SG+AoA need: > 79.685, which is +11.838 above protected SG+AoA.
- S2_true_curriculum_12x384_baseline16k_100M: known sum 290.205, missing ['(Super)GLUE', 'AoA']; needs missing-column sum > 86.005 to exceed leader.
  - Equivalent SG+AoA need: > 86.005, which is +18.158 above protected SG+AoA.

## Operational rule

Every BabyLM-scale candidate must be added to this scoreboard with all nine columns, or with an explicit missing-column requirement. Mechanism experiments are valuable only if they lead to a candidate whose official 9/9 Overall is evaluated.

