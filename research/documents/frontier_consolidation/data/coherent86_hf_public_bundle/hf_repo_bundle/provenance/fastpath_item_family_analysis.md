# frozen anchor fastpath disruption design fast-path item-family analysis

Status: **COMPLETE**
Route read: `needs_scientific_interpretation`

Available: `['chck82', 'coherent', 'spanbreak', 'shuffled86']`; pending: `[]`

## Cheap scores

| arm | payload | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck82 | `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 |
| shuffled86 | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json` | 44.012857142857136 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 |
| coherent | `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json` | 44.10642857142857 | 68.52 | 63.65 | 49.91 | 28.44 | 51.99 | 38.065 | 8.17 |
| spanbreak | `experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json` | 43.121428571428574 | 68.18 | 62.36 | 49.37 | 24.24 | 52.04 | 37.565 | 8.095 |

## coherent_minus_chck82

Aggregate: `{'total_gain_items': 3116, 'total_loss_items': 3231, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.16792740901663686, 'discrete_reconstructed_mean_delta': 0.16843829406280797, 'total_gain_minus_loss': -115, 'loss_to_gain_ratio': 1.0369062901155328}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | +0.029 | +0.038 | 761 | 731 | +30 | npi_present_2 -57/914, npi_present_1 -31/909, ellipsis_n_bar_1 -13/802, left_branch_island_simple_question -15/951 |
| Supplement | +0.712 | +0.708 | 37 | 51 | -14 | subject_aux_inversion -17/3867, turn_taking -1/280, hypernym +0/842, qa_congruence_tricky +2/165 |
| EWoK | -0.145 | -0.149 | 147 | 149 | -2 | physical-dynamics -2/120, social-properties -4/328, quantitative-properties -3/314, spatial-relations -3/490 |
| Entity | +0.126 | +0.126 | 73 | 66 | +7 | move_contents_3_ops -5/406, regular_5_ops -1/94, ambiref_5_ops -1/123, regular_2_ops -3/405 |
| COMPS | -0.201 | -0.197 | 2095 | 2232 | -137 | wugs_dist_before -85/13896, wugs -26/13896, base -38/49340, wugs_dist_in_between +12/13896 |
| GlobalPIQA | +0.487 | +0.485 | 3 | 2 | +1 | GlobalPIQA_nonparallel +0/100, GlobalPIQA_parallel +1/103 |

### Fragile-family readouts

```json
{
  "EWoK_fragile_tagged": {
    "tagged_groups": [
      {
        "group": "material-properties",
        "n": 170,
        "gain": 3,
        "loss": 1,
        "both_correct": 85,
        "both_wrong": 81,
        "net_gain_minus_loss": 2,
        "net_pct": 1.1764705882352942,
        "base_item_pct": 50.588235294117645,
        "cand_item_pct": 51.76470588235295
      },
      {
        "group": "physical-interactions",
        "n": 556,
        "gain": 9,
        "loss": 5,
        "both_correct": 265,
        "both_wrong": 277,
        "net_gain_minus_loss": 4,
        "net_pct": 0.7194244604316546,
        "base_item_pct": 48.561151079136685,
        "cand_item_pct": 49.280575539568346
      },
      {
        "group": "social-relations",
        "n": 1548,
        "gain": 40,
        "loss": 29,
        "both_correct": 746,
        "both_wrong": 733,
        "net_gain_minus_loss": 11,
        "net_pct": 0.710594315245478,
        "base_item_pct": 50.064599483204134,
        "cand_item_pct": 50.775193798449614
      },
      {
        "group": "social-interactions",
        "n": 294,
        "gain": 10,
        "loss": 8,
        "both_correct": 149,
        "both_wrong": 127,
        "net_gain_minus_loss": 2,
        "net_pct": 0.6802721088435374,
        "base_item_pct": 53.40136054421769,
        "cand_item_pct": 54.08163265306123
      },
      {
        "group": "material-dynamics",
        "n": 770,
        "gain": 23,
        "loss": 22,
        "both_correct": 368,
        "both_wrong": 357,
        "net_gain_minus_loss": 1,
        "net_pct": 0.12987012987012986,
        "base_item_pct": 50.649350649350644,
        "cand_item_pct": 50.77922077922078
      },
      {
        "group": "physical-relations",
        "n": 818,
        "gain": 6,
        "loss": 8,
        "both_correct": 399,
        "both_wrong": 405,
        "net_gain_minus_loss": -2,
        "net_pct": -0.24449877750611246,
        "base_item_pct": 49.75550122249388,
        "cand_item_pct": 49.511002444987774
      },
      {
        "group": "spatial-relations",
        "n": 490,
        "gain": 6,
        "loss": 9,
        "both_correct": 221,
        "both_wrong": 254,
        "net_gain_minus_loss": -3,
        "net_pct": -0.6122448979591837,
        "base_item_pct": 46.93877551020408,
        "cand_item_pct": 46.326530612244895
      },
      {
        "group": "quantitative-properties",
        "n": 314,
        "gain": 9,
        "loss": 12,
        "both_correct": 142,
        "both_wrong": 151,
        "net_gain_minus_loss": -3,
        "net_pct": -0.9554140127388535,
        "base_item_pct": 49.044585987261144,
        "cand_item_pct": 48.089171974522294
      },
      {
        "group": "social-properties",
        "n": 328,
        "gain": 5,
        "loss": 9,
        "both_correct": 148,
        "both_wrong": 166,
        "net_gain_minus_loss": -4,
        "net_pct": -1.2195121951219512,
        "base_item_pct": 47.86585365853659,
        "cand_item_pct": 46.646341463414636
      },
      {
        "group": "physical-dynamics",
        "n": 120,
        "gain": 2,
        "loss": 4,
        "both_correct": 60,
        "both_wrong": 54,
        "net_gain_minus_loss": -2,
        "net_pct": -1.6666666666666667,
        "base_item_pct": 53.333333333333336,
        "cand_item_pct": 51.66666666666667
      }
    ],
    "net_gain_minus_loss": 6,
    "n": 5408,
    "gains_reported": 113,
    "losses_reported": 107
  },
  "EWoK_worst_12": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 2,
      "loss": 4,
      "both_correct": 60,
      "both_wrong": 54,
      "net_gain_minus_loss": -2,
      "net_pct": -1.6666666666666667,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 51.66666666666667
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 5,
      "loss": 9,
      "both_correct": 148,
      "both_wrong": 166,
      "net_gain_minus_loss": -4,
      "net_pct": -1.2195121951219512,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 46.646341463414636
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 9,
      "loss": 12,
      "both_correct": 142,
      "both_wrong": 151,
      "net_gain_minus_loss": -3,
      "net_pct": -0.9554140127388535,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 6,
      "loss": 9,
      "both_correct": 221,
      "both_wrong": 254,
      "net_gain_minus_loss": -3,
      "net_pct": -0.6122448979591837,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 34,
      "loss": 42,
      "both_correct": 1072,
      "both_wrong": 1062,
      "net_gain_minus_loss": -8,
      "net_pct": -0.36199095022624433,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 6,
      "loss": 8,
      "both_correct": 399,
      "both_wrong": 405,
      "net_gain_minus_loss": -2,
      "net_pct": -0.24449877750611246,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 23,
      "loss": 22,
      "both_correct": 368,
      "both_wrong": 357,
      "net_gain_minus_loss": 1,
      "net_pct": 0.12987012987012986,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 10,
      "loss": 8,
      "both_correct": 149,
      "both_wrong": 127,
      "net_gain_minus_loss": 2,
      "net_pct": 0.6802721088435374,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 40,
      "loss": 29,
      "both_correct": 746,
      "both_wrong": 733,
      "net_gain_minus_loss": 11,
      "net_pct": 0.710594315245478,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 9,
      "loss": 5,
      "both_correct": 265,
      "both_wrong": 277,
      "net_gain_minus_loss": 4,
      "net_pct": 0.7194244604316546,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 3,
      "loss": 1,
      "both_correct": 85,
      "both_wrong": 81,
      "net_gain_minus_loss": 2,
      "net_pct": 1.1764705882352942,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 51.76470588235295
    }
  ],
  "EWoK_best_12": [
    {
      "group": "material-properties",
      "n": 170,
      "gain": 3,
      "loss": 1,
      "both_correct": 85,
      "both_wrong": 81,
      "net_gain_minus_loss": 2,
      "net_pct": 1.1764705882352942,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 51.76470588235295
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 9,
      "loss": 5,
      "both_correct": 265,
      "both_wrong": 277,
      "net_gain_minus_loss": 4,
      "net_pct": 0.7194244604316546,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 40,
      "loss": 29,
      "both_correct": 746,
      "both_wrong": 733,
      "net_gain_minus_loss": 11,
      "net_pct": 0.710594315245478,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 10,
      "loss": 8,
      "both_correct": 149,
      "both_wrong": 127,
      "net_gain_minus_loss": 2,
      "net_pct": 0.6802721088435374,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 23,
      "loss": 22,
      "both_correct": 368,
      "both_wrong": 357,
      "net_gain_minus_loss": 1,
      "net_pct": 0.12987012987012986,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 6,
      "loss": 8,
      "both_correct": 399,
      "both_wrong": 405,
      "net_gain_minus_loss": -2,
      "net_pct": -0.24449877750611246,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 34,
      "loss": 42,
      "both_correct": 1072,
      "both_wrong": 1062,
      "net_gain_minus_loss": -8,
      "net_pct": -0.36199095022624433,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 6,
      "loss": 9,
      "both_correct": 221,
      "both_wrong": 254,
      "net_gain_minus_loss": -3,
      "net_pct": -0.6122448979591837,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 9,
      "loss": 12,
      "both_correct": 142,
      "both_wrong": 151,
      "net_gain_minus_loss": -3,
      "net_pct": -0.9554140127388535,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 5,
      "loss": 9,
      "both_correct": 148,
      "both_wrong": 166,
      "net_gain_minus_loss": -4,
      "net_pct": -1.2195121951219512,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 46.646341463414636
    },
    {
      "group": "ph
```

## spanbreak_minus_chck82

Aggregate: `{'total_gain_items': 8608, 'total_loss_items': 9099, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.9687392576500287, 'discrete_reconstructed_mean_delta': -0.9652352263579308, 'total_gain_minus_loss': -491, 'loss_to_gain_ratio': 1.057039962825279}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | -0.311 | -0.301 | 1771 | 1953 | -182 | matrix_question_npi_licensor_present -68/929, npi_present_2 -61/914, only_npi_scope -35/837, left_branch_island_simple_question -34/951 |
| Supplement | -0.578 | -0.573 | 101 | 144 | -43 | qa_congruence_tricky -4/165, hypernym -14/842, subject_aux_inversion -27/3867, turn_taking +1/280 |
| EWoK | -0.685 | -0.683 | 427 | 439 | -12 | physical-dynamics -9/120, material-properties -4/170, social-interactions -3/294, agent-properties -12/2210 |
| Entity | -4.074 | -4.073 | 486 | 635 | -149 | move_contents_5_ops -28/116, move_contents_4_ops -45/353, ambiref_4_ops -47/434, regular_4_ops -40/388 |
| COMPS | -0.151 | -0.146 | 5818 | 5923 | -105 | wugs_dist_in_between -486/13896, wugs -13/13896, base -33/49340, wugs_dist_before +427/13896 |
| GlobalPIQA | -0.013 | -0.015 | 5 | 5 | +0 | GlobalPIQA_nonparallel -1/100, GlobalPIQA_parallel +1/103 |

### Fragile-family readouts

```json
{
  "EWoK_fragile_tagged": {
    "tagged_groups": [
      {
        "group": "social-properties",
        "n": 328,
        "gain": 26,
        "loss": 21,
        "both_correct": 136,
        "both_wrong": 145,
        "net_gain_minus_loss": 5,
        "net_pct": 1.524390243902439,
        "base_item_pct": 47.86585365853659,
        "cand_item_pct": 49.390243902439025
      },
      {
        "group": "physical-interactions",
        "n": 556,
        "gain": 32,
        "loss": 28,
        "both_correct": 242,
        "both_wrong": 254,
        "net_gain_minus_loss": 4,
        "net_pct": 0.7194244604316546,
        "base_item_pct": 48.561151079136685,
        "cand_item_pct": 49.280575539568346
      },
      {
        "group": "material-dynamics",
        "n": 770,
        "gain": 54,
        "loss": 49,
        "both_correct": 341,
        "both_wrong": 326,
        "net_gain_minus_loss": 5,
        "net_pct": 0.6493506493506493,
        "base_item_pct": 50.649350649350644,
        "cand_item_pct": 51.298701298701296
      },
      {
        "group": "quantitative-properties",
        "n": 314,
        "gain": 17,
        "loss": 15,
        "both_correct": 139,
        "both_wrong": 143,
        "net_gain_minus_loss": 2,
        "net_pct": 0.6369426751592356,
        "base_item_pct": 49.044585987261144,
        "cand_item_pct": 49.681528662420384
      },
      {
        "group": "physical-relations",
        "n": 818,
        "gain": 37,
        "loss": 33,
        "both_correct": 374,
        "both_wrong": 374,
        "net_gain_minus_loss": 4,
        "net_pct": 0.4889975550122249,
        "base_item_pct": 49.75550122249388,
        "cand_item_pct": 50.24449877750611
      },
      {
        "group": "spatial-relations",
        "n": 490,
        "gain": 21,
        "loss": 20,
        "both_correct": 210,
        "both_wrong": 239,
        "net_gain_minus_loss": 1,
        "net_pct": 0.20408163265306123,
        "base_item_pct": 46.93877551020408,
        "cand_item_pct": 47.14285714285714
      },
      {
        "group": "social-relations",
        "n": 1548,
        "gain": 111,
        "loss": 116,
        "both_correct": 659,
        "both_wrong": 662,
        "net_gain_minus_loss": -5,
        "net_pct": -0.32299741602067183,
        "base_item_pct": 50.064599483204134,
        "cand_item_pct": 49.74160206718346
      },
      {
        "group": "social-interactions",
        "n": 294,
        "gain": 14,
        "loss": 17,
        "both_correct": 140,
        "both_wrong": 123,
        "net_gain_minus_loss": -3,
        "net_pct": -1.0204081632653061,
        "base_item_pct": 53.40136054421769,
        "cand_item_pct": 52.38095238095239
      },
      {
        "group": "material-properties",
        "n": 170,
        "gain": 8,
        "loss": 12,
        "both_correct": 74,
        "both_wrong": 76,
        "net_gain_minus_loss": -4,
        "net_pct": -2.3529411764705883,
        "base_item_pct": 50.588235294117645,
        "cand_item_pct": 48.23529411764706
      },
      {
        "group": "physical-dynamics",
        "n": 120,
        "gain": 4,
        "loss": 13,
        "both_correct": 51,
        "both_wrong": 52,
        "net_gain_minus_loss": -9,
        "net_pct": -7.5,
        "base_item_pct": 53.333333333333336,
        "cand_item_pct": 45.83333333333333
      }
    ],
    "net_gain_minus_loss": 0,
    "n": 5408,
    "gains_reported": 324,
    "losses_reported": 324
  },
  "EWoK_worst_12": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 4,
      "loss": 13,
      "both_correct": 51,
      "both_wrong": 52,
      "net_gain_minus_loss": -9,
      "net_pct": -7.5,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 45.83333333333333
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 8,
      "loss": 12,
      "both_correct": 74,
      "both_wrong": 76,
      "net_gain_minus_loss": -4,
      "net_pct": -2.3529411764705883,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 48.23529411764706
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 14,
      "loss": 17,
      "both_correct": 140,
      "both_wrong": 123,
      "net_gain_minus_loss": -3,
      "net_pct": -1.0204081632653061,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 52.38095238095239
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 103,
      "loss": 115,
      "both_correct": 999,
      "both_wrong": 993,
      "net_gain_minus_loss": -12,
      "net_pct": -0.5429864253393665,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 49.86425339366516
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 111,
      "loss": 116,
      "both_correct": 659,
      "both_wrong": 662,
      "net_gain_minus_loss": -5,
      "net_pct": -0.32299741602067183,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 49.74160206718346
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 21,
      "loss": 20,
      "both_correct": 210,
      "both_wrong": 239,
      "net_gain_minus_loss": 1,
      "net_pct": 0.20408163265306123,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 47.14285714285714
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 37,
      "loss": 33,
      "both_correct": 374,
      "both_wrong": 374,
      "net_gain_minus_loss": 4,
      "net_pct": 0.4889975550122249,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 50.24449877750611
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 17,
      "loss": 15,
      "both_correct": 139,
      "both_wrong": 143,
      "net_gain_minus_loss": 2,
      "net_pct": 0.6369426751592356,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 49.681528662420384
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 54,
      "loss": 49,
      "both_correct": 341,
      "both_wrong": 326,
      "net_gain_minus_loss": 5,
      "net_pct": 0.6493506493506493,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 51.298701298701296
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 32,
      "loss": 28,
      "both_correct": 242,
      "both_wrong": 254,
      "net_gain_minus_loss": 4,
      "net_pct": 0.7194244604316546,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 26,
      "loss": 21,
      "both_correct": 136,
      "both_wrong": 145,
      "net_gain_minus_loss": 5,
      "net_pct": 1.524390243902439,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 49.390243902439025
    }
  ],
  "EWoK_best_12": [
    {
      "group": "social-properties",
      "n": 328,
      "gain": 26,
      "loss": 21,
      "both_correct": 136,
      "both_wrong": 145,
      "net_gain_minus_loss": 5,
      "net_pct": 1.524390243902439,
      "base_item_pct": 47.86585365853659,
      "cand_item_pct": 49.390243902439025
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 32,
      "loss": 28,
      "both_correct": 242,
      "both_wrong": 254,
      "net_gain_minus_loss": 4,
      "net_pct": 0.7194244604316546,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 54,
      "loss": 49,
      "both_correct": 341,
      "both_wrong": 326,
      "net_gain_minus_loss": 5,
      "net_pct": 0.6493506493506493,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 51.298701298701296
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 17,
      "loss": 15,
      "both_correct": 139,
      "both_wrong": 143,
      "net_gain_minus_loss": 2,
      "net_pct": 0.6369426751592356,
      "base_item_pct": 49.044585987261144,
      "cand_item_pct": 49.681528662420384
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 37,
      "loss": 33,
      "both_correct": 374,
      "both_wrong": 374,
      "net_gain_minus_loss": 4,
      "net_pct": 0.4889975550122249,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 50.24449877750611
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 21,
      "loss": 20,
      "both_correct": 210,
      "both_wrong": 239,
      "net_gain_minus_loss": 1,
      "net_pct": 0.20408163265306123,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 47.14285714285714
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 111,
      "loss": 116,
      "both_correct": 659,
      "both_wrong": 662,
      "net_gain_minus_loss": -5,
      "net_pct": -0.32299741602067183,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 49.74160206718346
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 103,
      "loss": 115,
      "both_correct": 999,
      "both_wrong": 993,
      "net_gain_minus_loss": -12,
      "net_pct": -0.5429864253393665,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 49.86425339366516
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 14,
      "loss": 17,
      "both_correct": 140,
      "both_wrong": 123,
      "net_gain_minus_loss": -3,
      "net_pct": -1.0204081632653061,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 52.38095238095239
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 8,
      "loss": 12,
      "both_correct": 74,
      "both_wrong": 76,
      "net_gain_minus_loss": -4,
      "net_pct": -2.3529411764705883,
      "base_item_pct": 50.588235294117645,
      "cand_item_pct": 48.23529411764706
    },
    {
      "group": "physic
```

## shuffled86_minus_chck82

Aggregate: `{'total_gain_items': 15345, 'total_loss_items': 14347, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.017072590983361852, 'discrete_reconstructed_mean_delta': -0.015450000106201974, 'total_gain_minus_loss': 998, 'loss_to_gain_ratio': 0.9349625285109157}`

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
  "EWoK_fragile_tagged": {
    "tagged_groups": [
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
    "net_gain_minus_loss": 58,
    "n": 5408,
    "gains_reported": 551,
    "losses_reported": 493
  },
  "EWoK_worst_12": [
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
    },
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
    }
  ],
  "EWoK_best_12": [
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
      "group": "physical-relations",
      
```

## coherent_minus_spanbreak

Aggregate: `{'total_gain_items': 8768, 'total_loss_items': 8392, 'total_common_items': 170722, 'discrete_payload_mean_delta': 1.1366666666666656, 'discrete_reconstructed_mean_delta': 1.1336735204207387, 'total_gain_minus_loss': 376, 'loss_to_gain_ratio': 0.9571167883211679}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | +0.340 | +0.339 | 1882 | 1670 | +212 | only_npi_licensor_present -63/882, existential_there_quantifiers_2 -51/911, left_branch_island_echo_question -39/947, wh_island -28/960 |
| Supplement | +1.290 | +1.281 | 135 | 106 | +29 | turn_taking -2/280, subject_aux_inversion +10/3867, qa_congruence_easy +1/64, hypernym +14/842 |
| EWoK | +0.540 | +0.534 | 424 | 414 | +10 | social-properties -9/328, quantitative-properties -5/314, spatial-relations -4/490, physical-relations -6/818 |
| Entity | +4.200 | +4.199 | 636 | 480 | +156 | regular_0_ops -75/517, move_contents_0_ops -57/516, ambiref_0_ops -51/508, regular_1_ops -2/409 |
| COMPS | -0.050 | -0.051 | 5686 | 5718 | -32 | wugs_dist_before -512/13896, wugs -13/13896, base -5/49340, wugs_dist_in_between +498/13896 |
| GlobalPIQA | +0.500 | +0.500 | 5 | 4 | +1 | GlobalPIQA_parallel +0/103, GlobalPIQA_nonparallel +1/100 |

### Fragile-family readouts

```json
{
  "EWoK_fragile_tagged": {
    "tagged_groups": [
      {
        "group": "physical-dynamics",
        "n": 120,
        "gain": 11,
        "loss": 4,
        "both_correct": 51,
        "both_wrong": 54,
        "net_gain_minus_loss": 7,
        "net_pct": 5.833333333333333,
        "base_item_pct": 45.83333333333333,
        "cand_item_pct": 51.66666666666667
      },
      {
        "group": "material-properties",
        "n": 170,
        "gain": 14,
        "loss": 8,
        "both_correct": 74,
        "both_wrong": 74,
        "net_gain_minus_loss": 6,
        "net_pct": 3.5294117647058822,
        "base_item_pct": 48.23529411764706,
        "cand_item_pct": 51.76470588235295
      },
      {
        "group": "social-interactions",
        "n": 294,
        "gain": 20,
        "loss": 15,
        "both_correct": 139,
        "both_wrong": 120,
        "net_gain_minus_loss": 5,
        "net_pct": 1.7006802721088434,
        "base_item_pct": 52.38095238095239,
        "cand_item_pct": 54.08163265306123
      },
      {
        "group": "social-relations",
        "n": 1548,
        "gain": 109,
        "loss": 93,
        "both_correct": 677,
        "both_wrong": 669,
        "net_gain_minus_loss": 16,
        "net_pct": 1.0335917312661498,
        "base_item_pct": 49.74160206718346,
        "cand_item_pct": 50.775193798449614
      },
      {
        "group": "physical-interactions",
        "n": 556,
        "gain": 29,
        "loss": 29,
        "both_correct": 245,
        "both_wrong": 253,
        "net_gain_minus_loss": 0,
        "net_pct": 0.0,
        "base_item_pct": 49.280575539568346,
        "cand_item_pct": 49.280575539568346
      },
      {
        "group": "material-dynamics",
        "n": 770,
        "gain": 52,
        "loss": 56,
        "both_correct": 339,
        "both_wrong": 323,
        "net_gain_minus_loss": -4,
        "net_pct": -0.5194805194805194,
        "base_item_pct": 51.298701298701296,
        "cand_item_pct": 50.77922077922078
      },
      {
        "group": "physical-relations",
        "n": 818,
        "gain": 30,
        "loss": 36,
        "both_correct": 375,
        "both_wrong": 377,
        "net_gain_minus_loss": -6,
        "net_pct": -0.7334963325183375,
        "base_item_pct": 50.24449877750611,
        "cand_item_pct": 49.511002444987774
      },
      {
        "group": "spatial-relations",
        "n": 490,
        "gain": 16,
        "loss": 20,
        "both_correct": 211,
        "both_wrong": 243,
        "net_gain_minus_loss": -4,
        "net_pct": -0.8163265306122449,
        "base_item_pct": 47.14285714285714,
        "cand_item_pct": 46.326530612244895
      },
      {
        "group": "quantitative-properties",
        "n": 314,
        "gain": 16,
        "loss": 21,
        "both_correct": 135,
        "both_wrong": 142,
        "net_gain_minus_loss": -5,
        "net_pct": -1.5923566878980893,
        "base_item_pct": 49.681528662420384,
        "cand_item_pct": 48.089171974522294
      },
      {
        "group": "social-properties",
        "n": 328,
        "gain": 17,
        "loss": 26,
        "both_correct": 136,
        "both_wrong": 149,
        "net_gain_minus_loss": -9,
        "net_pct": -2.7439024390243905,
        "base_item_pct": 49.390243902439025,
        "cand_item_pct": 46.646341463414636
      }
    ],
    "net_gain_minus_loss": 6,
    "n": 5408,
    "gains_reported": 314,
    "losses_reported": 308
  },
  "EWoK_worst_12": [
    {
      "group": "social-properties",
      "n": 328,
      "gain": 17,
      "loss": 26,
      "both_correct": 136,
      "both_wrong": 149,
      "net_gain_minus_loss": -9,
      "net_pct": -2.7439024390243905,
      "base_item_pct": 49.390243902439025,
      "cand_item_pct": 46.646341463414636
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 16,
      "loss": 21,
      "both_correct": 135,
      "both_wrong": 142,
      "net_gain_minus_loss": -5,
      "net_pct": -1.5923566878980893,
      "base_item_pct": 49.681528662420384,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 16,
      "loss": 20,
      "both_correct": 211,
      "both_wrong": 243,
      "net_gain_minus_loss": -4,
      "net_pct": -0.8163265306122449,
      "base_item_pct": 47.14285714285714,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 30,
      "loss": 36,
      "both_correct": 375,
      "both_wrong": 377,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7334963325183375,
      "base_item_pct": 50.24449877750611,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 52,
      "loss": 56,
      "both_correct": 339,
      "both_wrong": 323,
      "net_gain_minus_loss": -4,
      "net_pct": -0.5194805194805194,
      "base_item_pct": 51.298701298701296,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 29,
      "loss": 29,
      "both_correct": 245,
      "both_wrong": 253,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 49.280575539568346,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 110,
      "loss": 106,
      "both_correct": 996,
      "both_wrong": 998,
      "net_gain_minus_loss": 4,
      "net_pct": 0.18099547511312217,
      "base_item_pct": 49.86425339366516,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 109,
      "loss": 93,
      "both_correct": 677,
      "both_wrong": 669,
      "net_gain_minus_loss": 16,
      "net_pct": 1.0335917312661498,
      "base_item_pct": 49.74160206718346,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 20,
      "loss": 15,
      "both_correct": 139,
      "both_wrong": 120,
      "net_gain_minus_loss": 5,
      "net_pct": 1.7006802721088434,
      "base_item_pct": 52.38095238095239,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 14,
      "loss": 8,
      "both_correct": 74,
      "both_wrong": 74,
      "net_gain_minus_loss": 6,
      "net_pct": 3.5294117647058822,
      "base_item_pct": 48.23529411764706,
      "cand_item_pct": 51.76470588235295
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 11,
      "loss": 4,
      "both_correct": 51,
      "both_wrong": 54,
      "net_gain_minus_loss": 7,
      "net_pct": 5.833333333333333,
      "base_item_pct": 45.83333333333333,
      "cand_item_pct": 51.66666666666667
    }
  ],
  "EWoK_best_12": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 11,
      "loss": 4,
      "both_correct": 51,
      "both_wrong": 54,
      "net_gain_minus_loss": 7,
      "net_pct": 5.833333333333333,
      "base_item_pct": 45.83333333333333,
      "cand_item_pct": 51.66666666666667
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 14,
      "loss": 8,
      "both_correct": 74,
      "both_wrong": 74,
      "net_gain_minus_loss": 6,
      "net_pct": 3.5294117647058822,
      "base_item_pct": 48.23529411764706,
      "cand_item_pct": 51.76470588235295
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 20,
      "loss": 15,
      "both_correct": 139,
      "both_wrong": 120,
      "net_gain_minus_loss": 5,
      "net_pct": 1.7006802721088434,
      "base_item_pct": 52.38095238095239,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 109,
      "loss": 93,
      "both_correct": 677,
      "both_wrong": 669,
      "net_gain_minus_loss": 16,
      "net_pct": 1.0335917312661498,
      "base_item_pct": 49.74160206718346,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 110,
      "loss": 106,
      "both_correct": 996,
      "both_wrong": 998,
      "net_gain_minus_loss": 4,
      "net_pct": 0.18099547511312217,
      "base_item_pct": 49.86425339366516,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 29,
      "loss": 29,
      "both_correct": 245,
      "both_wrong": 253,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 49.280575539568346,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 52,
      "loss": 56,
      "both_correct": 339,
      "both_wrong": 323,
      "net_gain_minus_loss": -4,
      "net_pct": -0.5194805194805194,
      "base_item_pct": 51.298701298701296,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 30,
      "loss": 36,
      "both_correct": 375,
      "both_wrong": 377,
      "net_gain_minus_loss": -6,
      "net_pct": -0.7334963325183375,
      "base_item_pct": 50.24449877750611,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 16,
      "loss": 20,
      "both_correct": 211,
      "both_wrong": 243,
      "net_gain_minus_loss": -4,
      "net_pct": -0.8163265306122449,
      "base_item_pct": 47.14285714285714,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 16,
      "loss": 21,
      "both_correct": 135,
      "both_wrong": 142,
      "net_gain_minus_loss": -5,
      "net_pct": -1.5923566878980893,
      "base_item_pct": 49.681528662420384,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "social-properties",
      "n"
```

## coherent_minus_shuffled86

Aggregate: `{'total_gain_items': 14551, 'total_loss_items': 15664, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.18499999999999872, 'discrete_reconstructed_mean_delta': 0.18388829416900995, 'total_gain_minus_loss': -1113, 'loss_to_gain_ratio': 1.0764895883444436}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | -0.710 | -0.707 | 3148 | 3569 | -421 | npi_present_2 -137/914, only_npi_scope -125/837, wh_questions_object_gap -109/859, npi_present_1 -114/909 |
| Supplement | +3.840 | +3.835 | 227 | 163 | +64 | turn_taking -4/280, subject_aux_inversion +27/3867, hypernym +18/842, qa_congruence_easy +4/64 |
| EWoK | -1.530 | -1.534 | 692 | 726 | -34 | physical-dynamics -14/120, physical-interactions -16/556, spatial-relations -11/490, quantitative-properties -7/314 |
| Entity | +1.670 | +1.668 | 292 | 212 | +80 | ambiref_5_ops -2/123, move_contents_3_ops -4/406, regular_0_ops -2/517, move_contents_0_ops -1/516 |
| COMPS | -0.660 | -0.658 | 10181 | 10980 | -799 | wugs_dist_before -1525/13896, base -603/49340, wugs -126/13896, wugs_dist_in_between +1455/13896 |
| GlobalPIQA | -1.500 | -1.500 | 11 | 14 | -3 | GlobalPIQA_nonparallel -3/100, GlobalPIQA_parallel +0/103 |

### Fragile-family readouts

```json
{
  "EWoK_fragile_tagged": {
    "tagged_groups": [
      {
        "group": "physical-relations",
        "n": 818,
        "gain": 64,
        "loss": 47,
        "both_correct": 341,
        "both_wrong": 366,
        "net_gain_minus_loss": 17,
        "net_pct": 2.078239608801956,
        "base_item_pct": 47.43276283618582,
        "cand_item_pct": 49.511002444987774
      },
      {
        "group": "social-interactions",
        "n": 294,
        "gain": 41,
        "loss": 38,
        "both_correct": 118,
        "both_wrong": 97,
        "net_gain_minus_loss": 3,
        "net_pct": 1.0204081632653061,
        "base_item_pct": 53.06122448979592,
        "cand_item_pct": 54.08163265306123
      },
      {
        "group": "material-dynamics",
        "n": 770,
        "gain": 92,
        "loss": 85,
        "both_correct": 299,
        "both_wrong": 294,
        "net_gain_minus_loss": 7,
        "net_pct": 0.9090909090909091,
        "base_item_pct": 49.87012987012987,
        "cand_item_pct": 50.77922077922078
      },
      {
        "group": "material-properties",
        "n": 170,
        "gain": 16,
        "loss": 15,
        "both_correct": 72,
        "both_wrong": 67,
        "net_gain_minus_loss": 1,
        "net_pct": 0.5882352941176471,
        "base_item_pct": 51.17647058823529,
        "cand_item_pct": 51.76470588235295
      },
      {
        "group": "social-properties",
        "n": 328,
        "gain": 36,
        "loss": 41,
        "both_correct": 117,
        "both_wrong": 134,
        "net_gain_minus_loss": -5,
        "net_pct": -1.524390243902439,
        "base_item_pct": 48.170731707317074,
        "cand_item_pct": 46.646341463414636
      },
      {
        "group": "social-relations",
        "n": 1548,
        "gain": 154,
        "loss": 181,
        "both_correct": 632,
        "both_wrong": 581,
        "net_gain_minus_loss": -27,
        "net_pct": -1.744186046511628,
        "base_item_pct": 52.51937984496124,
        "cand_item_pct": 50.775193798449614
      },
      {
        "group": "quantitative-properties",
        "n": 314,
        "gain": 28,
        "loss": 35,
        "both_correct": 123,
        "both_wrong": 128,
        "net_gain_minus_loss": -7,
        "net_pct": -2.229299363057325,
        "base_item_pct": 50.318471337579616,
        "cand_item_pct": 48.089171974522294
      },
      {
        "group": "spatial-relations",
        "n": 490,
        "gain": 32,
        "loss": 43,
        "both_correct": 195,
        "both_wrong": 220,
        "net_gain_minus_loss": -11,
        "net_pct": -2.2448979591836733,
        "base_item_pct": 48.57142857142857,
        "cand_item_pct": 46.326530612244895
      },
      {
        "group": "physical-interactions",
        "n": 556,
        "gain": 40,
        "loss": 56,
        "both_correct": 234,
        "both_wrong": 226,
        "net_gain_minus_loss": -16,
        "net_pct": -2.8776978417266186,
        "base_item_pct": 52.15827338129496,
        "cand_item_pct": 49.280575539568346
      },
      {
        "group": "physical-dynamics",
        "n": 120,
        "gain": 7,
        "loss": 21,
        "both_correct": 55,
        "both_wrong": 37,
        "net_gain_minus_loss": -14,
        "net_pct": -11.666666666666666,
        "base_item_pct": 63.33333333333333,
        "cand_item_pct": 51.66666666666667
      }
    ],
    "net_gain_minus_loss": -52,
    "n": 5408,
    "gains_reported": 510,
    "losses_reported": 562
  },
  "EWoK_worst_12": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 7,
      "loss": 21,
      "both_correct": 55,
      "both_wrong": 37,
      "net_gain_minus_loss": -14,
      "net_pct": -11.666666666666666,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 51.66666666666667
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 40,
      "loss": 56,
      "both_correct": 234,
      "both_wrong": 226,
      "net_gain_minus_loss": -16,
      "net_pct": -2.8776978417266186,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 32,
      "loss": 43,
      "both_correct": 195,
      "both_wrong": 220,
      "net_gain_minus_loss": -11,
      "net_pct": -2.2448979591836733,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 28,
      "loss": 35,
      "both_correct": 123,
      "both_wrong": 128,
      "net_gain_minus_loss": -7,
      "net_pct": -2.229299363057325,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 154,
      "loss": 181,
      "both_correct": 632,
      "both_wrong": 581,
      "net_gain_minus_loss": -27,
      "net_pct": -1.744186046511628,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 36,
      "loss": 41,
      "both_correct": 117,
      "both_wrong": 134,
      "net_gain_minus_loss": -5,
      "net_pct": -1.524390243902439,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 46.646341463414636
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 16,
      "loss": 15,
      "both_correct": 72,
      "both_wrong": 67,
      "net_gain_minus_loss": 1,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 51.76470588235295
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 182,
      "loss": 164,
      "both_correct": 924,
      "both_wrong": 940,
      "net_gain_minus_loss": 18,
      "net_pct": 0.8144796380090498,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 92,
      "loss": 85,
      "both_correct": 299,
      "both_wrong": 294,
      "net_gain_minus_loss": 7,
      "net_pct": 0.9090909090909091,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 41,
      "loss": 38,
      "both_correct": 118,
      "both_wrong": 97,
      "net_gain_minus_loss": 3,
      "net_pct": 1.0204081632653061,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 64,
      "loss": 47,
      "both_correct": 341,
      "both_wrong": 366,
      "net_gain_minus_loss": 17,
      "net_pct": 2.078239608801956,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 49.511002444987774
    }
  ],
  "EWoK_best_12": [
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 64,
      "loss": 47,
      "both_correct": 341,
      "both_wrong": 366,
      "net_gain_minus_loss": 17,
      "net_pct": 2.078239608801956,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 41,
      "loss": 38,
      "both_correct": 118,
      "both_wrong": 97,
      "net_gain_minus_loss": 3,
      "net_pct": 1.0204081632653061,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 92,
      "loss": 85,
      "both_correct": 299,
      "both_wrong": 294,
      "net_gain_minus_loss": 7,
      "net_pct": 0.9090909090909091,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 182,
      "loss": 164,
      "both_correct": 924,
      "both_wrong": 940,
      "net_gain_minus_loss": 18,
      "net_pct": 0.8144796380090498,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 16,
      "loss": 15,
      "both_correct": 72,
      "both_wrong": 67,
      "net_gain_minus_loss": 1,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 51.76470588235295
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 36,
      "loss": 41,
      "both_correct": 117,
      "both_wrong": 134,
      "net_gain_minus_loss": -5,
      "net_pct": -1.524390243902439,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 46.646341463414636
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 154,
      "loss": 181,
      "both_correct": 632,
      "both_wrong": 581,
      "net_gain_minus_loss": -27,
      "net_pct": -1.744186046511628,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 28,
      "loss": 35,
      "both_correct": 123,
      "both_wrong": 128,
      "net_gain_minus_loss": -7,
      "net_pct": -2.229299363057325,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 48.089171974522294
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 32,
      "loss": 43,
      "both_correct": 195,
      "both_wrong": 220,
      "net_gain_minus_loss": -11,
      "net_pct": -2.2448979591836733,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 46.326530612244895
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 40,
      "loss": 56,
      "both_correct": 234,
      "both_wrong": 226,
      "net_gain_minus_loss": -16,
      "net_pct": -2.8776978417266186,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 49.280575539568346
 
```

## spanbreak_minus_shuffled86

Aggregate: `{'total_gain_items': 15289, 'total_loss_items': 16778, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.9516666666666668, 'discrete_reconstructed_mean_delta': -0.9497852262517288, 'total_gain_minus_loss': -1489, 'loss_to_gain_ratio': 1.097390280593891}`

| column | payload Δ | reconstructed Δ | gains | losses | net | worst groups |
|---|---:|---:|---:|---:|---:|---|
| BLiMP | -1.050 | -1.046 | 3222 | 3855 | -633 | only_npi_scope -159/837, wh_questions_object_gap -133/859, npi_present_2 -141/914, left_branch_island_simple_question -116/951 |
| Supplement | +2.550 | +2.553 | 224 | 189 | +35 | turn_taking -2/280, subject_aux_inversion +17/3867, hypernym +4/842, qa_congruence_easy +3/64 |
| EWoK | -2.070 | -2.068 | 753 | 797 | -44 | physical-dynamics -21/120, material-properties -5/170, physical-interactions -16/556, social-relations -43/1548 |
| Entity | -2.530 | -2.531 | 536 | 612 | -76 | move_contents_5_ops -20/116, move_contents_4_ops -40/353, regular_4_ops -42/388, regular_3_ops -43/425 |
| COMPS | -0.610 | -0.607 | 10546 | 11313 | -767 | wugs_dist_before -1013/13896, base -598/49340, wugs -113/13896, wugs_dist_in_between +957/13896 |
| GlobalPIQA | -2.000 | -2.000 | 8 | 12 | -4 | GlobalPIQA_nonparallel -4/100, GlobalPIQA_parallel +0/103 |

### Fragile-family readouts

```json
{
  "EWoK_fragile_tagged": {
    "tagged_groups": [
      {
        "group": "physical-relations",
        "n": 818,
        "gain": 70,
        "loss": 47,
        "both_correct": 341,
        "both_wrong": 360,
        "net_gain_minus_loss": 23,
        "net_pct": 2.8117359413202934,
        "base_item_pct": 47.43276283618582,
        "cand_item_pct": 50.24449877750611
      },
      {
        "group": "material-dynamics",
        "n": 770,
        "gain": 100,
        "loss": 89,
        "both_correct": 295,
        "both_wrong": 286,
        "net_gain_minus_loss": 11,
        "net_pct": 1.4285714285714286,
        "base_item_pct": 49.87012987012987,
        "cand_item_pct": 51.298701298701296
      },
      {
        "group": "social-properties",
        "n": 328,
        "gain": 45,
        "loss": 41,
        "both_correct": 117,
        "both_wrong": 125,
        "net_gain_minus_loss": 4,
        "net_pct": 1.2195121951219512,
        "base_item_pct": 48.170731707317074,
        "cand_item_pct": 49.390243902439025
      },
      {
        "group": "quantitative-properties",
        "n": 314,
        "gain": 30,
        "loss": 32,
        "both_correct": 126,
        "both_wrong": 126,
        "net_gain_minus_loss": -2,
        "net_pct": -0.6369426751592356,
        "base_item_pct": 50.318471337579616,
        "cand_item_pct": 49.681528662420384
      },
      {
        "group": "social-interactions",
        "n": 294,
        "gain": 35,
        "loss": 37,
        "both_correct": 119,
        "both_wrong": 103,
        "net_gain_minus_loss": -2,
        "net_pct": -0.6802721088435374,
        "base_item_pct": 53.06122448979592,
        "cand_item_pct": 52.38095238095239
      },
      {
        "group": "spatial-relations",
        "n": 490,
        "gain": 39,
        "loss": 46,
        "both_correct": 192,
        "both_wrong": 213,
        "net_gain_minus_loss": -7,
        "net_pct": -1.4285714285714286,
        "base_item_pct": 48.57142857142857,
        "cand_item_pct": 47.14285714285714
      },
      {
        "group": "social-relations",
        "n": 1548,
        "gain": 171,
        "loss": 214,
        "both_correct": 599,
        "both_wrong": 564,
        "net_gain_minus_loss": -43,
        "net_pct": -2.7777777777777777,
        "base_item_pct": 52.51937984496124,
        "cand_item_pct": 49.74160206718346
      },
      {
        "group": "physical-interactions",
        "n": 556,
        "gain": 49,
        "loss": 65,
        "both_correct": 225,
        "both_wrong": 217,
        "net_gain_minus_loss": -16,
        "net_pct": -2.8776978417266186,
        "base_item_pct": 52.15827338129496,
        "cand_item_pct": 49.280575539568346
      },
      {
        "group": "material-properties",
        "n": 170,
        "gain": 10,
        "loss": 15,
        "both_correct": 72,
        "both_wrong": 73,
        "net_gain_minus_loss": -5,
        "net_pct": -2.9411764705882355,
        "base_item_pct": 51.17647058823529,
        "cand_item_pct": 48.23529411764706
      },
      {
        "group": "physical-dynamics",
        "n": 120,
        "gain": 2,
        "loss": 23,
        "both_correct": 53,
        "both_wrong": 42,
        "net_gain_minus_loss": -21,
        "net_pct": -17.5,
        "base_item_pct": 63.33333333333333,
        "cand_item_pct": 45.83333333333333
      }
    ],
    "net_gain_minus_loss": -58,
    "n": 5408,
    "gains_reported": 551,
    "losses_reported": 609
  },
  "EWoK_worst_12": [
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 2,
      "loss": 23,
      "both_correct": 53,
      "both_wrong": 42,
      "net_gain_minus_loss": -21,
      "net_pct": -17.5,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 45.83333333333333
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 10,
      "loss": 15,
      "both_correct": 72,
      "both_wrong": 73,
      "net_gain_minus_loss": -5,
      "net_pct": -2.9411764705882355,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 48.23529411764706
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 49,
      "loss": 65,
      "both_correct": 225,
      "both_wrong": 217,
      "net_gain_minus_loss": -16,
      "net_pct": -2.8776978417266186,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 171,
      "loss": 214,
      "both_correct": 599,
      "both_wrong": 564,
      "net_gain_minus_loss": -43,
      "net_pct": -2.7777777777777777,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 49.74160206718346
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 39,
      "loss": 46,
      "both_correct": 192,
      "both_wrong": 213,
      "net_gain_minus_loss": -7,
      "net_pct": -1.4285714285714286,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 47.14285714285714
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 35,
      "loss": 37,
      "both_correct": 119,
      "both_wrong": 103,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6802721088435374,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 52.38095238095239
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 30,
      "loss": 32,
      "both_correct": 126,
      "both_wrong": 126,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6369426751592356,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 49.681528662420384
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 202,
      "loss": 188,
      "both_correct": 900,
      "both_wrong": 920,
      "net_gain_minus_loss": 14,
      "net_pct": 0.6334841628959276,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 49.86425339366516
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 45,
      "loss": 41,
      "both_correct": 117,
      "both_wrong": 125,
      "net_gain_minus_loss": 4,
      "net_pct": 1.2195121951219512,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 49.390243902439025
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 100,
      "loss": 89,
      "both_correct": 295,
      "both_wrong": 286,
      "net_gain_minus_loss": 11,
      "net_pct": 1.4285714285714286,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 51.298701298701296
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 70,
      "loss": 47,
      "both_correct": 341,
      "both_wrong": 360,
      "net_gain_minus_loss": 23,
      "net_pct": 2.8117359413202934,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 50.24449877750611
    }
  ],
  "EWoK_best_12": [
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 70,
      "loss": 47,
      "both_correct": 341,
      "both_wrong": 360,
      "net_gain_minus_loss": 23,
      "net_pct": 2.8117359413202934,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 50.24449877750611
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 100,
      "loss": 89,
      "both_correct": 295,
      "both_wrong": 286,
      "net_gain_minus_loss": 11,
      "net_pct": 1.4285714285714286,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 51.298701298701296
    },
    {
      "group": "social-properties",
      "n": 328,
      "gain": 45,
      "loss": 41,
      "both_correct": 117,
      "both_wrong": 125,
      "net_gain_minus_loss": 4,
      "net_pct": 1.2195121951219512,
      "base_item_pct": 48.170731707317074,
      "cand_item_pct": 49.390243902439025
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 202,
      "loss": 188,
      "both_correct": 900,
      "both_wrong": 920,
      "net_gain_minus_loss": 14,
      "net_pct": 0.6334841628959276,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 49.86425339366516
    },
    {
      "group": "quantitative-properties",
      "n": 314,
      "gain": 30,
      "loss": 32,
      "both_correct": 126,
      "both_wrong": 126,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6369426751592356,
      "base_item_pct": 50.318471337579616,
      "cand_item_pct": 49.681528662420384
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 35,
      "loss": 37,
      "both_correct": 119,
      "both_wrong": 103,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6802721088435374,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 52.38095238095239
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 39,
      "loss": 46,
      "both_correct": 192,
      "both_wrong": 213,
      "net_gain_minus_loss": -7,
      "net_pct": -1.4285714285714286,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 47.14285714285714
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 171,
      "loss": 214,
      "both_correct": 599,
      "both_wrong": 564,
      "net_gain_minus_loss": -43,
      "net_pct": -2.7777777777777777,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 49.74160206718346
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 49,
      "loss": 65,
      "both_correct": 225,
      "both_wrong": 217,
      "net_gain_minus_loss": -16,
      "net_pct": -2.8776978417266186,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "material-properties",
      "n": 170,
      "gain": 10,
      "loss": 15,
      "both_correct": 72,
      "both_wrong": 73,
      "net_gain_minus_loss": -5,
      "net_pct": -2.9411764705882355,
      "base_item_pct": 51.17647058823529,
      "cand_item_pct": 48.23529411764706
    },
    
```

Success requires private-ON coherent replay to retain anchor-correct fragile relation/state decisions while adding reproducible new decisions. Private-OFF equality and aggregate gains alone are not sufficient.

JSON: `experiments/archive/frontier_consolidation/data/fastpath_item_family_analysis/fastpath_item_family_analysis.json`
