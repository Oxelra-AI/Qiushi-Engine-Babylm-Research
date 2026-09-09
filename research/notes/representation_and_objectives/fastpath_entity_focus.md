# fastpath entity focus fast-path Entity focus from fastpath pair item family review item evidence

Status: **FASTPATH_ENTITY_FOCUS_EXTRACTED**

This note compresses the existing fastpath pair item family review item/family review before the ordinary scale1.75 chck_86M comparator exists. `shuffled86` here is the frozen-82M shuffled private-tail baseline, not ordinary continuation.

## Cheap7 reference

| arm | cheap7 |
|---|---:|
| chck82 | 43.95944987645173 |
| shuffled86 | 44.01285714285714 |
| coherent4M | 44.10642857142857 |
| spanbreak4M | 43.121428571428574 |

## coherent_minus_spanbreak

Base `spanbreak4M` -> candidate `coherent4M`

Aggregate: `{'total_gain_items': 8768, 'total_loss_items': 8392, 'total_common_items': 170722, 'discrete_payload_mean_delta': 1.1366666666666656, 'discrete_reconstructed_mean_delta': 1.1336735204207387, 'total_gain_minus_loss': 376, 'loss_to_gain_ratio': 0.9571167883211679}`

| column | score Δ | gains | losses | net items |
|---|---:|---:|---:|---:|
| BLiMP | 0.3399999999999892 | 1882 | 1670 | 212 |
| Supplement | 1.2899999999999991 | 135 | 106 | 29 |
| EWoK | 0.5399999999999991 | 424 | 414 | 10 |
| Entity | 4.200000000000003 | 636 | 480 | 156 |
| COMPS | -0.04999999999999716 | 5686 | 5718 | -32 |
| GlobalPIQA | 0.5 | 5 | 4 | 1 |

Entity high-operation total: `{'n': 2200, 'gain': 301, 'loss': 88, 'net_gain_minus_loss': 213, 'net_pct': 9.681818181818182}`

Entity high-operation groups:

```json
[
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
]
```

EWoK relation-group total: `{'n': 6806, 'gain': 377, 'loss': 359, 'net_gain_minus_loss': 18, 'net_pct': 0.26447252424331474}`

GlobalPIQA groups:

```json
[
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
```

## coherent_minus_chck82

Base `chck82` -> candidate `coherent4M`

Aggregate: `{'total_gain_items': 3116, 'total_loss_items': 3231, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.16792740901663686, 'discrete_reconstructed_mean_delta': 0.16843829406280797, 'total_gain_minus_loss': -115, 'loss_to_gain_ratio': 1.0369062901155328}`

| column | score Δ | gains | losses | net items |
|---|---:|---:|---:|---:|
| BLiMP | 0.028715963480138385 | 761 | 731 | 30 |
| Supplement | 0.7121887437997998 | 37 | 51 | -14 |
| EWoK | -0.14545332553276324 | 147 | 149 | -2 |
| Entity | 0.1259580697012268 | 73 | 66 | 7 |
| COMPS | -0.2011750944359605 | 2095 | 2232 | -137 |
| GlobalPIQA | 0.48733009708737995 | 3 | 2 | 1 |

Entity high-operation total: `{'n': 2200, 'gain': 23, 'loss': 24, 'net_gain_minus_loss': -1, 'net_pct': -0.045454545454545456}`

Entity high-operation groups:

```json
[
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
]
```

EWoK relation-group total: `{'n': 6806, 'gain': 130, 'loss': 127, 'net_gain_minus_loss': 3, 'net_pct': 0.044078754040552455}`

GlobalPIQA groups:

```json
[
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
```

## coherent_minus_shuffled86

Base `shuffled86` -> candidate `coherent4M`

Aggregate: `{'total_gain_items': 14551, 'total_loss_items': 15664, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.18499999999999872, 'discrete_reconstructed_mean_delta': 0.18388829416900995, 'total_gain_minus_loss': -1113, 'loss_to_gain_ratio': 1.0764895883444436}`

| column | score Δ | gains | losses | net items |
|---|---:|---:|---:|---:|
| BLiMP | -0.710000000000008 | 3148 | 3569 | -421 |
| Supplement | 3.8399999999999963 | 227 | 163 | 64 |
| EWoK | -1.5300000000000011 | 692 | 726 | -34 |
| Entity | 1.6700000000000017 | 292 | 212 | 80 |
| COMPS | -0.6599999999999966 | 10181 | 10980 | -799 |
| GlobalPIQA | -1.5 | 11 | 14 | -3 |

Entity high-operation total: `{'n': 2200, 'gain': 102, 'loss': 56, 'net_gain_minus_loss': 46, 'net_pct': 2.090909090909091}`

Entity high-operation groups:

```json
[
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
]
```

EWoK relation-group total: `{'n': 6806, 'gain': 612, 'loss': 635, 'net_gain_minus_loss': -23, 'net_pct': -0.3379371143109021}`

GlobalPIQA groups:

```json
[
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
```

## shuffled86_minus_chck82

Base `chck82` -> candidate `shuffled86`

Aggregate: `{'total_gain_items': 15345, 'total_loss_items': 14347, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.017072590983361852, 'discrete_reconstructed_mean_delta': -0.015450000106201974, 'total_gain_minus_loss': 998, 'loss_to_gain_ratio': 0.9349625285109157}`

| column | score Δ | gains | losses | net items |
|---|---:|---:|---:|---:|
| BLiMP | 0.7387159634801463 | 3488 | 3037 | 451 |
| Supplement | -3.1278112562001965 | 154 | 232 | -78 |
| EWoK | 1.384546674467238 | 709 | 677 | 32 |
| Entity | -1.544041930298775 | 218 | 291 | -73 |
| COMPS | 0.4588249055640361 | 10762 | 10100 | 662 |
| GlobalPIQA | 1.98733009708738 | 14 | 10 | 4 |

Entity high-operation total: `{'n': 2200, 'gain': 52, 'loss': 99, 'net_gain_minus_loss': -47, 'net_pct': -2.1363636363636362}`

Entity high-operation groups:

```json
[
  {
    "group": "regular_4_ops",
    "n": 388,
    "gain": 11,
    "loss": 9,
    "both_correct": 99,
    "both_wrong": 269,
    "net_gain_minus_loss": 2,
    "net_pct": 0.5154639175257731,
    "base_item_pct": 27.835051546391753,
    "cand_item_pct": 28.350515463917525
  },
  {
    "group": "move_contents_3_ops",
    "n": 406,
    "gain": 15,
    "loss": 16,
    "both_correct": 97,
    "both_wrong": 278,
    "net_gain_minus_loss": -1,
    "net_pct": -0.24630541871921183,
    "base_item_pct": 27.832512315270936,
    "cand_item_pct": 27.586206896551722
  },
  {
    "group": "move_contents_4_ops",
    "n": 353,
    "gain": 9,
    "loss": 14,
    "both_correct": 93,
    "both_wrong": 237,
    "net_gain_minus_loss": -5,
    "net_pct": -1.4164305949008498,
    "base_item_pct": 30.31161473087819,
    "cand_item_pct": 28.89518413597734
  },
  {
    "group": "regular_5_ops",
    "n": 94,
    "gain": 0,
    "loss": 7,
    "both_correct": 20,
    "both_wrong": 67,
    "net_gain_minus_loss": -7,
    "net_pct": -7.446808510638298,
    "base_item_pct": 28.723404255319153,
    "cand_item_pct": 21.27659574468085
  },
  {
    "group": "move_contents_5_ops",
    "n": 116,
    "gain": 5,
    "loss": 13,
    "both_correct": 36,
    "both_wrong": 62,
    "net_gain_minus_loss": -8,
    "net_pct": -6.896551724137931,
    "base_item_pct": 42.241379310344826,
    "cand_item_pct": 35.3448275862069
  },
  {
    "group": "ambiref_3_ops",
    "n": 409,
    "gain": 7,
    "loss": 20,
    "both_correct": 92,
    "both_wrong": 290,
    "net_gain_minus_loss": -13,
    "net_pct": -3.1784841075794623,
    "base_item_pct": 27.383863080684595,
    "cand_item_pct": 24.205378973105134
  },
  {
    "group": "ambiref_4_ops",
    "n": 434,
    "gain": 5,
    "loss": 20,
    "both_correct": 131,
    "both_wrong": 278,
    "net_gain_minus_loss": -15,
    "net_pct": -3.456221198156682,
    "base_item_pct": 34.7926267281106,
    "cand_item_pct": 31.336405529953915
  }
]
```

EWoK relation-group total: `{'n': 6806, 'gain': 616, 'loss': 590, 'net_gain_minus_loss': 26, 'net_pct': 0.3820158683514546}`

GlobalPIQA groups:

```json
[
  {
    "group": "GlobalPIQA_nonparallel",
    "n": 100,
    "gain": 6,
    "loss": 3,
    "both_correct": 44,
    "both_wrong": 47,
    "net_gain_minus_loss": 3,
    "net_pct": 3.0,
    "base_item_pct": 47.0,
    "cand_item_pct": 50.0
  },
  {
    "group": "GlobalPIQA_parallel",
    "n": 103,
    "gain": 8,
    "loss": 7,
    "both_correct": 22,
    "both_wrong": 66,
    "net_gain_minus_loss": 1,
    "net_pct": 0.970873786407767,
    "base_item_pct": 28.155339805825243,
    "cand_item_pct": 29.126213592233007
  }
]
```

## Interpretation for next comparator

If coherent SuperGLUE keeps the fast-path endpoint alive, ordinary scale1.75 chck_86M must be evaluated and compared at the same item level. The mechanistic question is whether coherent adds high-operation Entity / multi-step tracking beyond ordinary extra exposure without losing EWoK, GlobalPIQA, COMPS, or Supplement items. fastpath pair item family review alone shows coherent beats the destructive spanbreak arm strongly, but coherent versus protected chck82 has a negative total item balance and no EWoK repair.

JSON: `experiments/archive/representation_and_objectives/data/fastpath_entity_focus/fastpath_entity_focus.json`
