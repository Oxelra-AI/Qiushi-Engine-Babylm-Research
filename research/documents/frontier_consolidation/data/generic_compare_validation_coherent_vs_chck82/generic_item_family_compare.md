# coherence margin signal isolation generic item/family compare: coherent86 minus chck82

Status: **COMPLETE**

Base: `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` (ready)
Candidate: `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json` (ready)

Cheap7 delta: `0.1469786949768448`

| column | score delta | gains | losses | net |
|---|---:|---:|---:|---:|
| BLiMP | +0.028716 | 761 | 731 | +30 |
| Supplement | +0.712189 | 37 | 51 | -14 |
| EWoK | -0.145453 | 147 | 149 | -2 |
| Entity | +0.125958 | 73 | 66 | +7 |
| COMPS | -0.201175 | 2095 | 2232 | -137 |
| GlobalPIQA | +0.487330 | 3 | 2 | +1 |

## Fragile readouts

```json
{
  "EWoK_fragile_tagged": {
    "n": 5408,
    "gain": 113,
    "loss": 107,
    "net_gain_minus_loss": 6,
    "groups": [
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
    ]
  },
  "Entity_highop_tagged": {
    "n": 2748,
    "gain": 29,
    "loss": 28,
    "net_gain_minus_loss": 1,
    "groups": [
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
        "group": "regular_3_ops",
        "n": 425,
        "gain": 5,
        "loss": 2,
        "both_correct": 127,
        "both_wrong": 291,
        "net_gain_minus_loss": 3,
        "net_pct": 0.7058823529411765,
        "base_item_pct": 30.352941176470587,
        "cand_item_pct": 31.058823529411768
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
        "group": "ambiref_5_ops",
        "n": 123,
        "gain": 1,
        "loss": 2,
        "both_correct": 38,
        "both_wrong": 82,
        "net_gain_minus_loss": -1,
        "net_pct": -0.8130081300813008,
        "base_item_pct": 32.52032520325203,
        "cand_item_pct": 31.70731707317073
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
  },
  "EWoK_worst12": [
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
  "EWoK_best12": [
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
      "base_item_pct"
```

Use this before escalating a route: aggregate cheap7 must be decomposed into item gains/losses and fragile EWoK/Entity movement.

JSON: `experiments/archive/frontier_consolidation/data/generic_compare_validation_coherent_vs_chck82/generic_item_family_compare.json`
