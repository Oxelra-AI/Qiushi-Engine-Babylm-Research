# earlier analysis prediction schema probe

## clean_blimp
- path: `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/BLiMP/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json`
- exists: True
- top: dict len=67; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 928,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'adjunct_island_0'",
          "pred": "'Who should Derek hug Richard after shocking?'"
        }
      }
    }
  }
}
```

## clean_supplement
- path: `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/Supplement/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json`
- exists: True
- top: dict len=5; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 842,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'hypernym_0'",
          "pred": "'If she has a dog, it must be the case that she has a mammal.'"
        }
      }
    }
  }
}
```

## clean_ewok
- path: `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/EWoK/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json`
- exists: True
- top: dict len=11; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 2210,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'agent-properties_0'",
          "pred": "'Ali is in the bakery. Ali sees the candle outside. Ali believes that the candle is in the bakery.'"
        }
      }
    }
  }
}
```

## clean_comps
- path: `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/COMPS/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_COMPS/zero_shot/mlm/comps/comps/predictions.json`
- exists: True
- top: dict len=4; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 13896,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'wugs_0'",
          "pred": "'A wug is a mussel. Therefore, a wug attaches to rocks.'"
        }
      }
    }
  }
}
```

## clean_entity
- path: `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/Entity/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json`
- exists: True
- top: dict len=18; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 508,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'ambiref_0_ops_0'",
          "pred": "'the red shirt and the yellow coat and the blue jacket.'"
        }
      }
    }
  }
}
```

## adult_ewok_partial
- path: `experiments/archive/frontier_consolidation/data/gpu_priority_eval/regmax_adultprose_chck_40M/official_outputs/EWoK/chck_40M/EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json`
- exists: True
- top: dict len=11; item: dict
- numeric paths first item: `[]`
```json
{
  "type": "dict",
  "keys": [
    "predictions"
  ],
  "sample": {
    "predictions": {
      "type": "list",
      "len": 2210,
      "first": {
        "type": "dict",
        "keys": [
          "id",
          "pred"
        ],
        "sample": {
          "id": "'agent-properties_0'",
          "pred": "'Ali is in the bakery. Ali sees the candle inside. Ali believes that the candle is in the bakery.'"
        }
      }
    }
  }
}
```

