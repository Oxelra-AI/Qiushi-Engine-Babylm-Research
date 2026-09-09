# fastpath entity focus fast-path vs ordinary86 item review

Status: **COMPLETE**
Route read: `coherent_beats_ordinary86_score_but_item_net_nonpositive`

## Payload readiness and cheap7

| arm | ready | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chck82 | True | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 | `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` |
| ordinary86 | True | 43.770714285714284 | 68.47 | 62.68 | 50.14 | 28.58 | 52.29 | 36.135 | 8.1 | `experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json` |
| shuffled_private86 | True | 44.01285714285714 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json` |
| coherent4M | True | 44.10642857142857 | 68.52 | 63.65 | 49.91 | 28.44 | 51.99 | 38.065 | 8.17 | `experiments/archive/representation_and_objectives/data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json` |
| spanbreak4M | True | 43.121428571428574 | 68.18 | 62.36 | 49.37 | 24.24 | 52.04 | 37.565 | 8.095 | `experiments/archive/representation_and_objectives/data/fastpath4M_spanbreak_cheap_eval/per_target/fastpath4M_spanbreak.json` |

## Comparisons

### coherent_minus_ordinary86

Aggregate: `{'total_gain_items': 4796, 'total_loss_items': 5029, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.3799999999999996, 'discrete_reconstructed_mean_delta': 0.3796021811646971, 'total_gain_minus_loss': -233, 'loss_to_gain_ratio': 1.048582151793161}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | +0.050 | +0.056 | 1131 | 1089 | +42 |
| Supplement | +0.970 | +0.970 | 65 | 86 | -21 |
| EWoK | -0.230 | -0.234 | 218 | 245 | -27 |
| Entity | -0.140 | -0.143 | 131 | 145 | -14 |
| COMPS | -0.300 | -0.299 | 3242 | 3459 | -217 |
| GlobalPIQA | +1.930 | +1.927 | 9 | 5 | +4 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 46,
    "loss": 59,
    "net_gain_minus_loss": -13
  },
  "Entity_high_operation_groups": [
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 7,
      "loss": 2,
      "both_correct": 45,
      "both_wrong": 62,
      "net_gain_minus_loss": 5,
      "net_pct": 4.310344827586207,
      "base_item_pct": 40.51724137931034,
      "cand_item_pct": 44.827586206896555
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 9,
      "loss": 6,
      "both_correct": 101,
      "both_wrong": 293,
      "net_gain_minus_loss": 3,
      "net_pct": 0.7334963325183375,
      "base_item_pct": 26.161369193154034,
      "cand_item_pct": 26.894865525672373
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 13,
      "loss": 12,
      "both_correct": 95,
      "both_wrong": 286,
      "net_gain_minus_loss": 1,
      "net_pct": 0.24630541871921183,
      "base_item_pct": 26.354679802955665,
      "cand_item_pct": 26.60098522167488
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 2,
      "loss": 3,
      "both_correct": 24,
      "both_wrong": 65,
      "net_gain_minus_loss": -1,
      "net_pct": -1.0638297872340425,
      "base_item_pct": 28.723404255319153,
      "cand_item_pct": 27.659574468085108
    },
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 3,
      "loss": 9,
      "both_correct": 148,
      "both_wrong": 274,
      "net_gain_minus_loss": -6,
      "net_pct": -1.3824884792626728,
      "base_item_pct": 36.175115207373274,
      "cand_item_pct": 34.7926267281106
    },
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 6,
      "loss": 12,
      "both_correct": 103,
      "both_wrong": 232,
      "net_gain_minus_loss": -6,
      "net_pct": -1.6997167138810199,
      "base_item_pct": 32.577903682719544,
      "cand_item_pct": 30.878186968838527
    },
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 6,
      "loss": 15,
      "both_correct": 104,
      "both_wrong": 263,
      "net_gain_minus_loss": -9,
      "net_pct": -2.3195876288659796,
      "base_item_pct": 30.670103092783506,
      "cand_item_pct": 28.350515463917525
    }
  ],
  "EWoK_relation_groups": [
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 11,
      "loss": 7,
      "both_correct": 148,
      "both_wrong": 128,
      "net_gain_minus_loss": 4,
      "net_pct": 1.3605442176870748,
      "base_item_pct": 52.721088435374156,
      "cand_item_pct": 54.08163265306123
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 22,
      "loss": 17,
      "both_correct": 252,
      "both_wrong": 265,
      "net_gain_minus_loss": 5,
      "net_pct": 0.8992805755395683,
      "base_item_pct": 48.381294964028775,
      "cand_item_pct": 49.280575539568346
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 68,
      "loss": 63,
      "both_correct": 1038,
      "both_wrong": 1041,
      "net_gain_minus_loss": 5,
      "net_pct": 0.22624434389140272,
      "base_item_pct": 49.81900452488688,
      "cand_item_pct": 50.04524886877828
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 4,
      "loss": 5,
      "both_correct": 58,
      "both_wrong": 53,
      "net_gain_minus_loss": -1,
      "net_pct": -0.8333333333333334,
      "base_item_pct": 52.5,
      "cand_item_pct": 51.66666666666667
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 13,
      "loss": 20,
      "both_correct": 392,
      "both_wrong": 393,
      "net_gain_minus_loss": -7,
      "net_pct": -0.8557457212713936,
      "base_item_pct": 50.36674816625917,
      "cand_item_pct": 49.511002444987774
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 48,
      "loss": 63,
      "both_correct": 738,
      "both_wrong": 699,
      "net_gain_minus_loss": -15,
      "net_pct": -0.9689922480620154,
      "base_item_pct": 51.74418604651163,
      "cand_item_pct": 50.775193798449614
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 13,
      "loss": 23,
      "both_correct": 378,
      "both_wrong": 356,
      "net_gain_minus_loss": -10,
      "net_pct": -1.2987012987012987,
      "base_item_pct": 52.07792207792208,
      "cand_item_pct": 50.77922077922078
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 11,
      "loss": 19,
      "both_correct": 216,
      "both_wrong": 244,
      "net_gain_minus_loss": -8,
      "net_pct": -1.6326530612244898,
      "base_item_pct": 47.95918367346938,
      "cand_item_pct": 46.326530612244895
    }
  ],
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 6,
      "loss": 1,
      "both_correct": 24,
      "both_wrong": 72,
      "net_gain_minus_loss": 5,
      "net_pct": 4.854368932038835,
      "base_item_pct": 24.271844660194176,
      "cand_item_pct": 29.126213592233007
    },
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 3,
      "loss": 4,
      "both_correct": 44,
      "both_wrong": 49,
      "net_gain_minus_loss": -1,
      "net_pct": -1.0,
      "base_item_pct": 48.0,
      "cand_item_pct": 47.0
    }
  ]
}
```

### ordinary86_minus_chck82

Aggregate: `{'total_gain_items': 5366, 'total_loss_items': 5248, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.21207259098336273, 'discrete_reconstructed_mean_delta': -0.21116388710188913, 'total_gain_minus_loss': 118, 'loss_to_gain_ratio': 0.9780096906448006}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | -0.021 | -0.018 | 1172 | 1184 | -12 |
| Supplement | -0.258 | -0.262 | 71 | 64 | +7 |
| EWoK | +0.085 | +0.085 | 275 | 250 | +25 |
| Entity | +0.266 | +0.268 | 145 | 124 | +21 |
| COMPS | +0.099 | +0.101 | 3697 | 3617 | +80 |
| GlobalPIQA | -1.443 | -1.442 | 6 | 9 | -3 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 54,
    "loss": 42,
    "net_gain_minus_loss": 12
  },
  "Entity_high_operation_groups": [
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 14,
      "loss": 3,
      "both_correct": 105,
      "both_wrong": 266,
      "net_gain_minus_loss": 11,
      "net_pct": 2.8350515463917527,
      "base_item_pct": 27.835051546391753,
      "cand_item_pct": 30.670103092783506
    },
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 14,
      "loss": 6,
      "both_correct": 101,
      "both_wrong": 232,
      "net_gain_minus_loss": 8,
      "net_pct": 2.26628895184136,
      "base_item_pct": 30.31161473087819,
      "cand_item_pct": 32.577903682719544
    },
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 8,
      "loss": 2,
      "both_correct": 149,
      "both_wrong": 275,
      "net_gain_minus_loss": 6,
      "net_pct": 1.3824884792626728,
      "base_item_pct": 34.7926267281106,
      "cand_item_pct": 36.175115207373274
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 3,
      "loss": 3,
      "both_correct": 24,
      "both_wrong": 64,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 28.723404255319153,
      "cand_item_pct": 28.723404255319153
    },
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 2,
      "loss": 4,
      "both_correct": 45,
      "both_wrong": 65,
      "net_gain_minus_loss": -2,
      "net_pct": -1.7241379310344827,
      "base_item_pct": 42.241379310344826,
      "cand_item_pct": 40.51724137931034
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 6,
      "loss": 11,
      "both_correct": 101,
      "both_wrong": 291,
      "net_gain_minus_loss": -5,
      "net_pct": -1.2224938875305624,
      "base_item_pct": 27.383863080684595,
      "cand_item_pct": 26.161369193154034
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 7,
      "loss": 13,
      "both_correct": 100,
      "both_wrong": 286,
      "net_gain_minus_loss": -6,
      "net_pct": -1.477832512315271,
      "base_item_pct": 27.832512315270936,
      "cand_item_pct": 26.354679802955665
    }
  ],
  "EWoK_relation_groups": [
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 76,
      "loss": 50,
      "both_correct": 725,
      "both_wrong": 697,
      "net_gain_minus_loss": 26,
      "net_pct": 1.6795865633074936,
      "base_item_pct": 50.064599483204134,
      "cand_item_pct": 51.74418604651163
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 34,
      "loss": 23,
      "both_correct": 367,
      "both_wrong": 346,
      "net_gain_minus_loss": 11,
      "net_pct": 1.4285714285714286,
      "base_item_pct": 50.649350649350644,
      "cand_item_pct": 52.07792207792208
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 19,
      "loss": 14,
      "both_correct": 216,
      "both_wrong": 241,
      "net_gain_minus_loss": 5,
      "net_pct": 1.0204081632653061,
      "base_item_pct": 46.93877551020408,
      "cand_item_pct": 47.95918367346938
    },
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 22,
      "loss": 17,
      "both_correct": 390,
      "both_wrong": 389,
      "net_gain_minus_loss": 5,
      "net_pct": 0.6112469437652812,
      "base_item_pct": 49.75550122249388,
      "cand_item_pct": 50.36674816625917
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 18,
      "loss": 19,
      "both_correct": 251,
      "both_wrong": 268,
      "net_gain_minus_loss": -1,
      "net_pct": -0.17985611510791366,
      "base_item_pct": 48.561151079136685,
      "cand_item_pct": 48.381294964028775
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 64,
      "loss": 77,
      "both_correct": 1037,
      "both_wrong": 1032,
      "net_gain_minus_loss": -13,
      "net_pct": -0.5882352941176471,
      "base_item_pct": 50.40723981900452,
      "cand_item_pct": 49.81900452488688
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 10,
      "loss": 12,
      "both_correct": 145,
      "both_wrong": 127,
      "net_gain_minus_loss": -2,
      "net_pct": -0.6802721088435374,
      "base_item_pct": 53.40136054421769,
      "cand_item_pct": 52.721088435374156
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 3,
      "loss": 4,
      "both_correct": 60,
      "both_wrong": 53,
      "net_gain_minus_loss": -1,
      "net_pct": -0.8333333333333334,
      "base_item_pct": 53.333333333333336,
      "cand_item_pct": 52.5
    }
  ],
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 4,
      "loss": 3,
      "both_correct": 44,
      "both_wrong": 49,
      "net_gain_minus_loss": 1,
      "net_pct": 1.0,
      "base_item_pct": 47.0,
      "cand_item_pct": 48.0
    },
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 2,
      "loss": 6,
      "both_correct": 23,
      "both_wrong": 72,
      "net_gain_minus_loss": -4,
      "net_pct": -3.883495145631068,
      "base_item_pct": 28.155339805825243,
      "cand_item_pct": 24.271844660194176
    }
  ]
}
```

### coherent_minus_chck82

Aggregate: `{'total_gain_items': 3116, 'total_loss_items': 3231, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.16792740901663686, 'discrete_reconstructed_mean_delta': 0.16843829406280797, 'total_gain_minus_loss': -115, 'loss_to_gain_ratio': 1.0369062901155328}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | +0.029 | +0.038 | 761 | 731 | +30 |
| Supplement | +0.712 | +0.708 | 37 | 51 | -14 |
| EWoK | -0.145 | -0.149 | 147 | 149 | -2 |
| Entity | +0.126 | +0.126 | 73 | 66 | +7 |
| COMPS | -0.201 | -0.197 | 2095 | 2232 | -137 |
| GlobalPIQA | +0.487 | +0.485 | 3 | 2 | +1 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 23,
    "loss": 24,
    "net_gain_minus_loss": -1
  },
  "Entity_high_operation_groups": [
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 3,
      "loss": 0,
      "both_correct": 49,
      "both_wrong": 64,
      "net_gain_minus_loss": 3,
      "net_pct": 2.586206896551724,
      "base_item_pct": 42.241379310344826,
      "cand_item_pct": 44.827586206896555
    },
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 5,
      "loss": 3,
      "both_correct": 104,
      "both_wrong": 241,
      "net_gain_minus_loss": 2,
      "net_pct": 0.56657223796034,
      "base_item_pct": 30.31161473087819,
      "cand_item_pct": 30.878186968838527
    },
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 4,
      "loss": 2,
      "both_correct": 106,
      "both_wrong": 276,
      "net_gain_minus_loss": 2,
      "net_pct": 0.5154639175257731,
      "base_item_pct": 27.835051546391753,
      "cand_item_pct": 28.350515463917525
    },
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 4,
      "loss": 4,
      "both_correct": 147,
      "both_wrong": 279,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 34.7926267281106,
      "cand_item_pct": 34.7926267281106
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 1,
      "loss": 2,
      "both_correct": 25,
      "both_wrong": 66,
      "net_gain_minus_loss": -1,
      "net_pct": -1.0638297872340425,
      "base_item_pct": 28.723404255319153,
      "cand_item_pct": 27.659574468085108
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 3,
      "loss": 5,
      "both_correct": 107,
      "both_wrong": 294,
      "net_gain_minus_loss": -2,
      "net_pct": -0.4889975550122249,
      "base_item_pct": 27.383863080684595,
      "cand_item_pct": 26.894865525672373
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 3,
      "loss": 8,
      "both_correct": 105,
      "both_wrong": 290,
      "net_gain_minus_loss": -5,
      "net_pct": -1.2315270935960592,
      "base_item_pct": 27.832512315270936,
      "cand_item_pct": 26.60098522167488
    }
  ],
  "EWoK_relation_groups": [
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
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 2,
      "loss": 1,
      "both_correct": 28,
      "both_wrong": 72,
      "net_gain_minus_loss": 1,
      "net_pct": 0.970873786407767,
      "base_item_pct": 28.155339805825243,
      "cand_item_pct": 29.126213592233007
    },
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 1,
      "loss": 1,
      "both_correct": 46,
      "both_wrong": 52,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 47.0,
      "cand_item_pct": 47.0
    }
  ]
}
```

### coherent_minus_spanbreak

Aggregate: `{'total_gain_items': 8768, 'total_loss_items': 8392, 'total_common_items': 170722, 'discrete_payload_mean_delta': 1.1366666666666656, 'discrete_reconstructed_mean_delta': 1.1336735204207387, 'total_gain_minus_loss': 376, 'loss_to_gain_ratio': 0.9571167883211679}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | +0.340 | +0.339 | 1882 | 1670 | +212 |
| Supplement | +1.290 | +1.281 | 135 | 106 | +29 |
| EWoK | +0.540 | +0.534 | 424 | 414 | +10 |
| Entity | +4.200 | +4.199 | 636 | 480 | +156 |
| COMPS | -0.050 | -0.051 | 5686 | 5718 | -32 |
| GlobalPIQA | +0.500 | +0.500 | 5 | 4 | +1 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 301,
    "loss": 88,
    "net_gain_minus_loss": 213
  },
  "Entity_high_operation_groups": [
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 60,
      "loss": 13,
      "both_correct": 49,
      "both_wrong": 231,
      "net_gain_minus_loss": 47,
      "net_pct": 13.314447592067989,
      "base_item_pct": 17.56373937677054,
      "cand_item_pct": 30.878186968838527
    },
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 61,
      "loss": 14,
      "both_correct": 90,
      "both_wrong": 269,
      "net_gain_minus_loss": 47,
      "net_pct": 10.829493087557603,
      "base_item_pct": 23.963133640552993,
      "cand_item_pct": 34.7926267281106
    },
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 54,
      "loss": 12,
      "both_correct": 56,
      "both_wrong": 266,
      "net_gain_minus_loss": 42,
      "net_pct": 10.824742268041238,
      "base_item_pct": 17.525773195876287,
      "cand_item_pct": 28.350515463917525
    },
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 33,
      "loss": 2,
      "both_correct": 19,
      "both_wrong": 62,
      "net_gain_minus_loss": 31,
      "net_pct": 26.724137931034484,
      "base_item_pct": 18.103448275862068,
      "cand_item_pct": 44.827586206896555
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 41,
      "loss": 17,
      "both_correct": 67,
      "both_wrong": 281,
      "net_gain_minus_loss": 24,
      "net_pct": 5.911330049261084,
      "base_item_pct": 20.689655172413794,
      "cand_item_pct": 26.60098522167488
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 39,
      "loss": 21,
      "both_correct": 71,
      "both_wrong": 278,
      "net_gain_minus_loss": 18,
      "net_pct": 4.400977995110025,
      "base_item_pct": 22.493887530562347,
      "cand_item_pct": 26.894865525672373
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 13,
      "loss": 9,
      "both_correct": 13,
      "both_wrong": 59,
      "net_gain_minus_loss": 4,
      "net_pct": 4.25531914893617,
      "base_item_pct": 23.404255319148938,
      "cand_item_pct": 27.659574468085108
    }
  ],
  "EWoK_relation_groups": [
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
    }
  ],
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 3,
      "loss": 2,
      "both_correct": 44,
      "both_wrong": 51,
      "net_gain_minus_loss": 1,
      "net_pct": 1.0,
      "base_item_pct": 46.0,
      "cand_item_pct": 47.0
    },
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 2,
      "loss": 2,
      "both_correct": 28,
      "both_wrong": 71,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 29.126213592233007,
      "cand_item_pct": 29.126213592233007
    }
  ]
}
```

### coherent_minus_shuffled_private86

Aggregate: `{'total_gain_items': 14551, 'total_loss_items': 15664, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.18499999999999872, 'discrete_reconstructed_mean_delta': 0.18388829416900995, 'total_gain_minus_loss': -1113, 'loss_to_gain_ratio': 1.0764895883444436}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | -0.710 | -0.707 | 3148 | 3569 | -421 |
| Supplement | +3.840 | +3.835 | 227 | 163 | +64 |
| EWoK | -1.530 | -1.534 | 692 | 726 | -34 |
| Entity | +1.670 | +1.668 | 292 | 212 | +80 |
| COMPS | -0.660 | -0.658 | 10181 | 10980 | -799 |
| GlobalPIQA | -1.500 | -1.500 | 11 | 14 | -3 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 102,
    "loss": 56,
    "net_gain_minus_loss": 46
  },
  "Entity_high_operation_groups": [
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 23,
      "loss": 8,
      "both_correct": 128,
      "both_wrong": 275,
      "net_gain_minus_loss": 15,
      "net_pct": 3.456221198156682,
      "base_item_pct": 31.336405529953915,
      "cand_item_pct": 34.7926267281106
    },
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 14,
      "loss": 3,
      "both_correct": 38,
      "both_wrong": 61,
      "net_gain_minus_loss": 11,
      "net_pct": 9.482758620689655,
      "base_item_pct": 35.3448275862069,
      "cand_item_pct": 44.827586206896555
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 21,
      "loss": 10,
      "both_correct": 89,
      "both_wrong": 289,
      "net_gain_minus_loss": 11,
      "net_pct": 2.6894865525672373,
      "base_item_pct": 24.205378973105134,
      "cand_item_pct": 26.894865525672373
    },
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 15,
      "loss": 8,
      "both_correct": 94,
      "both_wrong": 236,
      "net_gain_minus_loss": 7,
      "net_pct": 1.9830028328611897,
      "base_item_pct": 28.89518413597734,
      "cand_item_pct": 30.878186968838527
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 6,
      "loss": 0,
      "both_correct": 20,
      "both_wrong": 68,
      "net_gain_minus_loss": 6,
      "net_pct": 6.382978723404255,
      "base_item_pct": 21.27659574468085,
      "cand_item_pct": 27.659574468085108
    },
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 11,
      "loss": 11,
      "both_correct": 99,
      "both_wrong": 267,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 28.350515463917525,
      "cand_item_pct": 28.350515463917525
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 12,
      "loss": 16,
      "both_correct": 96,
      "both_wrong": 282,
      "net_gain_minus_loss": -4,
      "net_pct": -0.9852216748768473,
      "base_item_pct": 27.586206896551722,
      "cand_item_pct": 26.60098522167488
    }
  ],
  "EWoK_relation_groups": [
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
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 7,
      "loss": 7,
      "both_correct": 23,
      "both_wrong": 66,
      "net_gain_minus_loss": 0,
      "net_pct": 0.0,
      "base_item_pct": 29.126213592233007,
      "cand_item_pct": 29.126213592233007
    },
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 4,
      "loss": 7,
      "both_correct": 43,
      "both_wrong": 46,
      "net_gain_minus_loss": -3,
      "net_pct": -3.0,
      "base_item_pct": 50.0,
      "cand_item_pct": 47.0
    }
  ]
}
```

### ordinary86_minus_shuffled_private86

Aggregate: `{'total_gain_items': 15251, 'total_loss_items': 16131, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.19500000000000087, 'discrete_reconstructed_mean_delta': -0.19571388699568715, 'total_gain_minus_loss': -880, 'loss_to_gain_ratio': 1.0577011343518459}`

| column | payload Δ | reconstructed Δ | gains | losses | net |
|---|---:|---:|---:|---:|---:|
| BLiMP | -0.760 | -0.763 | 3209 | 3672 | -463 |
| Supplement | +2.870 | +2.864 | 241 | 156 | +85 |
| EWoK | -1.300 | -1.300 | 716 | 723 | -7 |
| Entity | +1.810 | +1.811 | 328 | 234 | +94 |
| COMPS | -0.360 | -0.360 | 10748 | 11330 | -582 |
| GlobalPIQA | -3.430 | -3.427 | 9 | 16 | -7 |

Focused readouts:

```json
{
  "Entity_high_operation_total": {
    "n": 2200,
    "gain": 122,
    "loss": 63,
    "net_gain_minus_loss": 59
  },
  "Entity_high_operation_groups": [
    {
      "group": "ambiref_4_ops",
      "n": 434,
      "gain": 27,
      "loss": 6,
      "both_correct": 130,
      "both_wrong": 271,
      "net_gain_minus_loss": 21,
      "net_pct": 4.838709677419355,
      "base_item_pct": 31.336405529953915,
      "cand_item_pct": 36.175115207373274
    },
    {
      "group": "move_contents_4_ops",
      "n": 353,
      "gain": 23,
      "loss": 10,
      "both_correct": 92,
      "both_wrong": 228,
      "net_gain_minus_loss": 13,
      "net_pct": 3.68271954674221,
      "base_item_pct": 28.89518413597734,
      "cand_item_pct": 32.577903682719544
    },
    {
      "group": "regular_4_ops",
      "n": 388,
      "gain": 18,
      "loss": 9,
      "both_correct": 101,
      "both_wrong": 260,
      "net_gain_minus_loss": 9,
      "net_pct": 2.3195876288659796,
      "base_item_pct": 28.350515463917525,
      "cand_item_pct": 30.670103092783506
    },
    {
      "group": "ambiref_3_ops",
      "n": 409,
      "gain": 22,
      "loss": 14,
      "both_correct": 85,
      "both_wrong": 288,
      "net_gain_minus_loss": 8,
      "net_pct": 1.9559902200488997,
      "base_item_pct": 24.205378973105134,
      "cand_item_pct": 26.161369193154034
    },
    {
      "group": "regular_5_ops",
      "n": 94,
      "gain": 7,
      "loss": 0,
      "both_correct": 20,
      "both_wrong": 67,
      "net_gain_minus_loss": 7,
      "net_pct": 7.446808510638298,
      "base_item_pct": 21.27659574468085,
      "cand_item_pct": 28.723404255319153
    },
    {
      "group": "move_contents_5_ops",
      "n": 116,
      "gain": 11,
      "loss": 5,
      "both_correct": 36,
      "both_wrong": 64,
      "net_gain_minus_loss": 6,
      "net_pct": 5.172413793103448,
      "base_item_pct": 35.3448275862069,
      "cand_item_pct": 40.51724137931034
    },
    {
      "group": "move_contents_3_ops",
      "n": 406,
      "gain": 14,
      "loss": 19,
      "both_correct": 93,
      "both_wrong": 280,
      "net_gain_minus_loss": -5,
      "net_pct": -1.2315270935960592,
      "base_item_pct": 27.586206896551722,
      "cand_item_pct": 26.354679802955665
    }
  ],
  "EWoK_relation_groups": [
    {
      "group": "physical-relations",
      "n": 818,
      "gain": 64,
      "loss": 40,
      "both_correct": 348,
      "both_wrong": 366,
      "net_gain_minus_loss": 24,
      "net_pct": 2.93398533007335,
      "base_item_pct": 47.43276283618582,
      "cand_item_pct": 50.36674816625917
    },
    {
      "group": "material-dynamics",
      "n": 770,
      "gain": 113,
      "loss": 96,
      "both_correct": 288,
      "both_wrong": 273,
      "net_gain_minus_loss": 17,
      "net_pct": 2.207792207792208,
      "base_item_pct": 49.87012987012987,
      "cand_item_pct": 52.07792207792208
    },
    {
      "group": "agent-properties",
      "n": 2210,
      "gain": 177,
      "loss": 164,
      "both_correct": 924,
      "both_wrong": 945,
      "net_gain_minus_loss": 13,
      "net_pct": 0.5882352941176471,
      "base_item_pct": 49.23076923076923,
      "cand_item_pct": 49.81900452488688
    },
    {
      "group": "social-interactions",
      "n": 294,
      "gain": 35,
      "loss": 36,
      "both_correct": 120,
      "both_wrong": 103,
      "net_gain_minus_loss": -1,
      "net_pct": -0.3401360544217687,
      "base_item_pct": 53.06122448979592,
      "cand_item_pct": 52.721088435374156
    },
    {
      "group": "spatial-relations",
      "n": 490,
      "gain": 35,
      "loss": 38,
      "both_correct": 200,
      "both_wrong": 217,
      "net_gain_minus_loss": -3,
      "net_pct": -0.6122448979591837,
      "base_item_pct": 48.57142857142857,
      "cand_item_pct": 47.95918367346938
    },
    {
      "group": "social-relations",
      "n": 1548,
      "gain": 152,
      "loss": 164,
      "both_correct": 649,
      "both_wrong": 583,
      "net_gain_minus_loss": -12,
      "net_pct": -0.7751937984496124,
      "base_item_pct": 52.51937984496124,
      "cand_item_pct": 51.74418604651163
    },
    {
      "group": "physical-interactions",
      "n": 556,
      "gain": 42,
      "loss": 63,
      "both_correct": 227,
      "both_wrong": 224,
      "net_gain_minus_loss": -21,
      "net_pct": -3.776978417266187,
      "base_item_pct": 52.15827338129496,
      "cand_item_pct": 48.381294964028775
    },
    {
      "group": "physical-dynamics",
      "n": 120,
      "gain": 8,
      "loss": 21,
      "both_correct": 55,
      "both_wrong": 36,
      "net_gain_minus_loss": -13,
      "net_pct": -10.833333333333334,
      "base_item_pct": 63.33333333333333,
      "cand_item_pct": 52.5
    }
  ],
  "GlobalPIQA_groups": [
    {
      "group": "GlobalPIQA_nonparallel",
      "n": 100,
      "gain": 4,
      "loss": 6,
      "both_correct": 44,
      "both_wrong": 46,
      "net_gain_minus_loss": -2,
      "net_pct": -2.0,
      "base_item_pct": 50.0,
      "cand_item_pct": 48.0
    },
    {
      "group": "GlobalPIQA_parallel",
      "n": 103,
      "gain": 5,
      "loss": 10,
      "both_correct": 20,
      "both_wrong": 68,
      "net_gain_minus_loss": -5,
      "net_pct": -4.854368932038835,
      "base_item_pct": 29.126213592233007,
      "cand_item_pct": 24.271844660194176
    }
  ]
}
```

## Scientific use

The fast-path route should continue only if coherent4M beats ordinary86, not just spanbreak, and the item/family evidence shows added high-operation Entity tracking without compensating losses in EWoK/GlobalPIQA/COMPS/Supplement. If ordinary86 matches the gains, then coherent replay is ordinary extra exposure/private-tail redistribution rather than a distinct protected learning mechanism.

JSON: `experiments/archive/representation_and_objectives/data/fastpath_vs_ordinary86_item_review/fastpath_vs_ordinary86_item_review.json`
