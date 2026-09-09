# static decoy contingency validation frozen-82M tail item-family analysis

Status: **COMPLETE**
Route read: `tail_branch_not_supported_by_item_surface`

Available: `['chck82', 'aligned', 'shuffled']`; pending: `[]`

## Payload cheap scores

| arm | payload | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82 | `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 |
| aligned | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_aligned_eval/per_target/frozen82_tail4M_aligned.json` | 43.91285714285714 | 69.19 | 59.27 | 51.81 | 28.95 | 52.36 | 37.09 | 8.72 |
| shuffled | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json` | 44.012857142857136 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 |

## aligned_minus_chck82

Aggregate: `{'total_gain_items': 15640, 'total_loss_items': 14894, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.14957259098336095, 'discrete_reconstructed_mean_delta': -0.15032077083697004, 'total_gain_minus_loss': 746, 'total_gain_minus_loss_pct': 0.4369677018779068}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | +0.699 | +0.698 | 3483 | 3067 | +416 | adjunct_island -145/928, wh_vs_that_with_gap -120/919, matrix_question_npi_licensor_present -93/929, intransitive -70/868 |
| Supplement | -3.668 | -3.668 | 158 | 261 | -103 | qa_congruence_tricky -19/165, qa_congruence_easy -3/64, subject_aux_inversion -78/3867, hypernym -4/842 |
| EWoK | +1.755 | +1.756 | 753 | 685 | +68 | material-properties -8/170, agent-properties -32/2210, physical-relations -6/818, social-properties -2/328 |
| Entity | +0.636 | +0.631 | 268 | 236 | +32 | ambiref_3_ops -4/409, ambiref_4_ops -3/434, ambiref_2_ops -2/413, regular_3_ops -2/425 |
| COMPS | +0.169 | +0.165 | 10967 | 10633 | +334 | wugs_dist_in_between -1662/13896, wugs +85/13896, base +337/49340, wugs_dist_before +1574/13896 |
| GlobalPIQA | -0.488 | -0.485 | 11 | 12 | -1 | GlobalPIQA_parallel -1/103, GlobalPIQA_nonparallel +0/100 |

### Fragile-family readouts

```json
{
  "EWoK_tagged_fragile_groups": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 18,
      "loss": 6,
      "both_correct": 58,
      "both_wrong": 38,
      "net_gain_minus_loss": 12,
      "net_pct": 10.0,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 46,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 114,
      "net_gain_minus_loss": 16,
      "net_pct": 5.095541401273885,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 54.14012738853503
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 61,
      "loss": 42,
      "both_correct": 228,
      "both_wrong": 225,
      "net_gain_minus_loss": 19,
      "net_pct": 3.41726618705036,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 51.97841726618705
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 83,
      "loss": 58,
      "both_correct": 332,
      "both_wrong": 297,
      "net_gain_minus_loss": 25,
      "net_pct": 3.2467532467532467,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 53.896103896103895
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 179,
      "loss": 146,
      "both_correct": 629,
      "both_wrong": 594,
      "net_gain_minus_loss": 33,
      "net_pct": 2.131782945736434,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.19638242894057
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 39,
      "loss": 34,
      "both_correct": 123,
      "both_wrong": 98,
      "net_gain_minus_loss": 5,
      "net_pct": 1.7006802721088434,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 55.10204081632652
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 45,
      "loss": 39,
      "both_correct": 191,
      "both_wrong": 215,
      "net_gain_minus_loss": 6,
      "net_pct": 1.2244897959183674,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 48.16326530612245
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 41,
      "loss": 43,
      "both_correct": 114,
      "both_wrong": 130,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6097560975609756,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 47.256097560975604
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 63,
      "loss": 69,
      "both_correct": 338,
      "both_wrong": 348,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7334963325183375,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 49.02200488997555
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 14,
      "loss": 22,
      "both_correct": 64,
      "both_wrong": 70,
      "net_gain_minus_loss": -8,
      "net_pct": -4.705882352941177,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 45.88235294117647
    }
  ],
  "EWoK_worst_10": [
    {
      "group": "material-properties",
      "n": 170,
      "gain": 14,
      "loss": 22,
      "both_correct": 64,
      "both_wrong": 70,
      "net_gain_minus_loss": -8,
      "net_pct": -4.705882352941177,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 45.88235294117647
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 164,
      "loss": 196,
      "both_correct": 918,
      "both_wrong": 932,
      "net_gain_minus_loss": -32,
      "net_pct": -1.4479638009049773,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 48.959276018099544
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 63,
      "loss": 69,
      "both_correct": 338,
      "both_wrong": 348,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7334963325183375,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 49.02200488997555
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 41,
      "loss": 43,
      "both_correct": 114,
      "both_wrong": 130,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6097560975609756,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 47.256097560975604
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 45,
      "loss": 39,
      "both_correct": 191,
      "both_wrong": 215,
      "net_gain_minus_loss": 6,
      "net_pct": 1.2244897959183674,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 48.16326530612245
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 39,
      "loss": 34,
      "both_correct": 123,
      "both_wrong": 98,
      "net_gain_minus_loss": 5,
      "net_pct": 1.7006802721088434,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 55.10204081632652
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 179,
      "loss": 146,
      "both_correct": 629,
      "both_wrong": 594,
      "net_gain_minus_loss": 33,
      "net_pct": 2.131782945736434,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.19638242894057
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 83,
      "loss": 58,
      "both_correct": 332,
      "both_wrong": 297,
      "net_gain_minus_loss": 25,
      "net_pct": 3.2467532467532467,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 53.896103896103895
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 61,
      "loss": 42,
      "both_correct": 228,
      "both_wrong": 225,
      "net_gain_minus_loss": 19,
      "net_pct": 3.41726618705036,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 51.97841726618705
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 46,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 114,
      "net_gain_minus_loss": 16,
      "net_pct": 5.095541401273885,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 54.14012738853503
    }
  ],
  "EWoK_best_10": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 18,
      "loss": 6,
      "both_correct": 58,
      "both_wrong": 38,
      "net_gain_minus_loss": 12,
      "net_pct": 10.0,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 46,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 114,
      "net_gain_minus_loss": 16,
      "net_pct": 5.095541401273885,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 54.14012738853503
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 61,
      "loss": 42,
      "both_correct": 228,
      "both_wrong": 225,
      "net_gain_minus_loss": 19,
      "net_pct": 3.41726618705036,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 51.97841726618705
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 83,
      "loss": 58,
      "both_correct": 332,
      "both_wrong": 297,
      "net_gain_minus_loss": 25,
      "net_pct": 3.2467532467532467,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 53.896103896103895
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 179,
      "loss": 146,
      "both_correct": 629,
      "both_wrong": 594,
      "net_gain_minus_loss": 33,
      "net_pct": 2.131782945736434,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.19638242894057
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 39,
      "loss": 34,
      "both_correct": 123,
      "both_wrong": 98,
      "net_gain_minus_loss": 5,
      "net_pct": 1.7006802721088434,
      "base_item_pct": 53.401
```

## shuffled_minus_chck82

Aggregate: `{'total_gain_items': 15345, 'total_loss_items': 14347, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.017072590983361852, 'discrete_reconstructed_mean_delta': -0.015450000106201974, 'total_gain_minus_loss': 998, 'total_gain_minus_loss_pct': 0.5845760944693713}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | +0.739 | +0.745 | 3488 | 3037 | +451 | adjunct_island -159/928, matrix_question_npi_licensor_present -87/929, existential_there_quantifiers_2 -71/911, wh_vs_that_with_gap -69/919 |
| Supplement | -3.128 | -3.126 | 154 | 232 | -78 | qa_congruence_tricky -17/165, qa_congruence_easy -2/64, hypernym -18/842, subject_aux_inversion -44/3867 |
| EWoK | +1.385 | +1.385 | 709 | 677 | +32 | physical-relations -19/818, agent-properties -26/2210, material-dynamics -6/770, social-interactions -1/294 |
| Entity | -1.544 | -1.542 | 218 | 291 | -73 | regular_5_ops -7/94, move_contents_5_ops -8/116, ambiref_2_ops -15/413, ambiref_4_ops -15/434 |
| COMPS | +0.459 | +0.461 | 10762 | 10100 | +662 | wugs_dist_in_between -1443/13896, wugs +100/13896, base +565/49340, wugs_dist_before +1440/13896 |
| GlobalPIQA | +1.987 | +1.985 | 14 | 10 | +4 | GlobalPIQA_parallel +1/103, GlobalPIQA_nonparallel +3/100 |

### Fragile-family readouts

```json
{
  "EWoK_tagged_fragile_groups": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 19,
      "loss": 7,
      "both_correct": 57,
      "both_wrong": 37,
      "net_gain_minus_loss": 12,
      "net_pct": 10.0,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 59,
      "loss": 39,
      "both_correct": 231,
      "both_wrong": 227,
      "net_gain_minus_loss": 20,
      "net_pct": 3.597122302158273,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 52.15827338129496
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 174,
      "loss": 136,
      "both_correct": 639,
      "both_wrong": 599,
      "net_gain_minus_loss": 38,
      "net_pct": 2.454780361757106,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.51937984496124
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 41,
      "loss": 33,
      "both_correct": 197,
      "both_wrong": 219,
      "net_gain_minus_loss": 8,
      "net_pct": 1.6326530612244898,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 48.57142857142857
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 34,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 126,
      "net_gain_minus_loss": 4,
      "net_pct": 1.2738853503184713,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 50.318471337579616
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 16,
      "loss": 15,
      "both_correct": 71,
      "both_wrong": 68,
      "net_gain_minus_loss": 1,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 51.17647058823529
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 43,
      "loss": 42,
      "both_correct": 115,
      "both_wrong": 128,
      "net_gain_minus_loss": 1,
      "net_pct": 0.3048780487804878,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 48.170731707317074
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 36,
      "loss": 37,
      "both_correct": 120,
      "both_wrong": 101,
      "net_gain_minus_loss": -1,
      "net_pct": -0.3401360544217687,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 53.06122448979592
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 80,
      "loss": 86,
      "both_correct": 304,
      "both_wrong": 300,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7792207792207793,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 49.87012987012987
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 49,
      "loss": 68,
      "both_correct": 339,
      "both_wrong": 362,
      "net_gain_minus_loss": -19,
      "net_pct": -2.3227383863080684,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 47.43276283618582
    }
  ],
  "EWoK_worst_10": [
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 49,
      "loss": 68,
      "both_correct": 339,
      "both_wrong": 362,
      "net_gain_minus_loss": -19,
      "net_pct": -2.3227383863080684,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 47.43276283618582
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 158,
      "loss": 184,
      "both_correct": 930,
      "both_wrong": 938,
      "net_gain_minus_loss": -26,
      "net_pct": -1.1764705882352942,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 49.23076923076923
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 80,
      "loss": 86,
      "both_correct": 304,
      "both_wrong": 300,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7792207792207793,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 49.87012987012987
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 36,
      "loss": 37,
      "both_correct": 120,
      "both_wrong": 101,
      "net_gain_minus_loss": -1,
      "net_pct": -0.3401360544217687,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 53.06122448979592
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 43,
      "loss": 42,
      "both_correct": 115,
      "both_wrong": 128,
      "net_gain_minus_loss": 1,
      "net_pct": 0.3048780487804878,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 48.170731707317074
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 16,
      "loss": 15,
      "both_correct": 71,
      "both_wrong": 68,
      "net_gain_minus_loss": 1,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 51.17647058823529
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 34,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 126,
      "net_gain_minus_loss": 4,
      "net_pct": 1.2738853503184713,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 50.318471337579616
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 41,
      "loss": 33,
      "both_correct": 197,
      "both_wrong": 219,
      "net_gain_minus_loss": 8,
      "net_pct": 1.6326530612244898,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 48.57142857142857
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 174,
      "loss": 136,
      "both_correct": 639,
      "both_wrong": 599,
      "net_gain_minus_loss": 38,
      "net_pct": 2.454780361757106,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.51937984496124
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 59,
      "loss": 39,
      "both_correct": 231,
      "both_wrong": 227,
      "net_gain_minus_loss": 20,
      "net_pct": 3.597122302158273,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 52.15827338129496
    }
  ],
  "EWoK_best_10": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 19,
      "loss": 7,
      "both_correct": 57,
      "both_wrong": 37,
      "net_gain_minus_loss": 12,
      "net_pct": 10.0,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 59,
      "loss": 39,
      "both_correct": 231,
      "both_wrong": 227,
      "net_gain_minus_loss": 20,
      "net_pct": 3.597122302158273,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 52.15827338129496
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 174,
      "loss": 136,
      "both_correct": 639,
      "both_wrong": 599,
      "net_gain_minus_loss": 38,
      "net_pct": 2.454780361757106,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 52.51937984496124
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 41,
      "loss": 33,
      "both_correct": 197,
      "both_wrong": 219,
      "net_gain_minus_loss": 8,
      "net_pct": 1.6326530612244898,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 48.57142857142857
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 34,
      "loss": 30,
      "both_correct": 124,
      "both_wrong": 126,
      "net_gain_minus_loss": 4,
      "net_pct": 1.2738853503184713,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 50.318471337579616
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 16,
      "loss": 15,
      "both_correct": 71,
      "both_wrong": 68,
      "net_gain_minus_loss": 1,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 50.
```

## aligned_minus_shuffled

Aggregate: `{'total_gain_items': 7469, 'total_loss_items': 7721, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.1324999999999991, 'discrete_reconstructed_mean_delta': -0.13487077073076806, 'total_gain_minus_loss': -252, 'total_gain_minus_loss_pct': -0.1476083925914645}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | -0.040 | -0.047 | 1591 | 1626 | -35 | wh_vs_that_with_gap -51/919, wh_vs_that_with_gap_long_distance -35/910, principle_A_domain_2 -31/915, only_npi_scope -25/837 |
| Supplement | -0.540 | -0.541 | 139 | 164 | -25 | qa_congruence_easy -1/64, qa_congruence_tricky -2/165, subject_aux_inversion -34/3867, turn_taking -2/280 |
| EWoK | +0.370 | +0.371 | 419 | 383 | +36 | material-properties -9/170, social-properties -3/328, spatial-relations -2/490, social-relations -5/1548 |
| Entity | +2.180 | +2.173 | 316 | 211 | +105 | regular_3_ops -3/425, move_contents_3_ops +0/406, move_contents_1_ops +1/437, regular_1_ops +1/409 |
| COMPS | -0.290 | -0.295 | 5001 | 5329 | -328 | wugs_dist_in_between -219/13896, base -228/49340, wugs -15/13896, wugs_dist_before +134/13896 |
| GlobalPIQA | -2.475 | -2.471 | 3 | 8 | -5 | GlobalPIQA_nonparallel -3/100, GlobalPIQA_parallel -2/103 |

### Fragile-family readouts

```json
{
  "EWoK_tagged_fragile_groups": [
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 46,
      "loss": 15,
      "both_correct": 369,
      "both_wrong": 340,
      "net_gain_minus_loss": 31,
      "net_pct": 4.025974025974026,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 53.896103896103895
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 32,
      "loss": 20,
      "both_correct": 138,
      "both_wrong": 124,
      "net_gain_minus_loss": 12,
      "net_pct": 3.821656050955414,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 54.14012738853503
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 20,
      "loss": 14,
      "both_correct": 142,
      "both_wrong": 118,
      "net_gain_minus_loss": 6,
      "net_pct": 2.0408163265306123,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 55.10204081632652
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 66,
      "loss": 53,
      "both_correct": 335,
      "both_wrong": 364,
      "net_gain_minus_loss": 13,
      "net_pct": 1.5892420537897312,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 49.02200488997555
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 3,
      "loss": 3,
      "both_correct": 73,
      "both_wrong": 41,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 42,
      "loss": 43,
      "both_correct": 247,
      "both_wrong": 224,
      "net_gain_minus_loss": -1,
      "net_pct": -0.17985611510791366,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 51.97841726618705
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 69,
      "loss": 74,
      "both_correct": 739,
      "both_wrong": 666,
      "net_gain_minus_loss": -5,
      "net_pct": -0.32299741602067183,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 52.19638242894057
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 21,
      "loss": 23,
      "both_correct": 215,
      "both_wrong": 231,
      "net_gain_minus_loss": -2,
      "net_pct": -0.40816326530612246,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 48.16326530612245
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 19,
      "loss": 22,
      "both_correct": 136,
      "both_wrong": 151,
      "net_gain_minus_loss": -3,
      "net_pct": -0.9146341463414634,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 47.256097560975604
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 2,
      "loss": 11,
      "both_correct": 76,
      "both_wrong": 81,
      "net_gain_minus_loss": -9,
      "net_pct": -5.294117647058823,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 45.88235294117647
    }
  ],
  "EWoK_worst_10": [
    {
      "group": "material-properties",
      "n": 170,
      "gain": 2,
      "loss": 11,
      "both_correct": 76,
      "both_wrong": 81,
      "net_gain_minus_loss": -9,
      "net_pct": -5.294117647058823,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 45.88235294117647
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 19,
      "loss": 22,
      "both_correct": 136,
      "both_wrong": 151,
      "net_gain_minus_loss": -3,
      "net_pct": -0.9146341463414634,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 47.256097560975604
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 21,
      "loss": 23,
      "both_correct": 215,
      "both_wrong": 231,
      "net_gain_minus_loss": -2,
      "net_pct": -0.40816326530612246,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 48.16326530612245
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 69,
      "loss": 74,
      "both_correct": 739,
      "both_wrong": 666,
      "net_gain_minus_loss": -5,
      "net_pct": -0.32299741602067183,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 52.19638242894057
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 99,
      "loss": 105,
      "both_correct": 983,
      "both_wrong": 1023,
      "net_gain_minus_loss": -6,
      "net_pct": -0.27149321266968324,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 48.959276018099544
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 42,
      "loss": 43,
      "both_correct": 247,
      "both_wrong": 224,
      "net_gain_minus_loss": -1,
      "net_pct": -0.17985611510791366,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 51.97841726618705
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 3,
      "loss": 3,
      "both_correct": 73,
      "both_wrong": 41,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 66,
      "loss": 53,
      "both_correct": 335,
      "both_wrong": 364,
      "net_gain_minus_loss": 13,
      "net_pct": 1.5892420537897312,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 49.02200488997555
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 20,
      "loss": 14,
      "both_correct": 142,
      "both_wrong": 118,
      "net_gain_minus_loss": 6,
      "net_pct": 2.0408163265306123,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 55.10204081632652
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 32,
      "loss": 20,
      "both_correct": 138,
      "both_wrong": 124,
      "net_gain_minus_loss": 12,
      "net_pct": 3.821656050955414,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 54.14012738853503
    }
  ],
  "EWoK_best_10": [
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 46,
      "loss": 15,
      "both_correct": 369,
      "both_wrong": 340,
      "net_gain_minus_loss": 31,
      "net_pct": 4.025974025974026,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 53.896103896103895
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 32,
      "loss": 20,
      "both_correct": 138,
      "both_wrong": 124,
      "net_gain_minus_loss": 12,
      "net_pct": 3.821656050955414,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 54.14012738853503
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 20,
      "loss": 14,
      "both_correct": 142,
      "both_wrong": 118,
      "net_gain_minus_loss": 6,
      "net_pct": 2.0408163265306123,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 55.10204081632652
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 66,
      "loss": 53,
      "both_correct": 335,
      "both_wrong": 364,
      "net_gain_minus_loss": 13,
      "net_pct": 1.5892420537897312,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 49.02200488997555
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 3,
      "loss": 3,
      "both_correct": 73,
      "both_wrong": 41,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 63.33333333333333
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 42,
      "loss": 43,
      "both_correct": 247,
      "both_wrong": 224,
      "net_gain_minus_loss": -1,
      "net_pct": -0.17985611510791366,
      "base_item_pct": 52.15827338129496,
      "c
```

This is a saved-prediction item-family analysis. It is intended to show whether a private tail changes broad families by adding competence to the protected chck_82M function or by redistributing within fragile Supplement/EWoK/Entity/COMPS/GlobalPIQA families.

JSON: `experiments/archive/frontier_consolidation/data/frozen82_tail_item_family_analysis/frozen82_tail_item_family_analysis.json`
