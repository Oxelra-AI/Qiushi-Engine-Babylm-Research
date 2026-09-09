# rawtoken bridge screen and route judgment final bridge synthesis

## Runs

| run | interface | held | aff | unaff | quartet_all | paraphrase | multi | wp_delta_held | wp_delta_para | wp_delta_multi |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_base20 | raw_token | 0.578125 | 0.6302083333333334 | 0.5260416666666666 | 0.17708333333333334 | 0.5911458333333334 | 0.484375 | None | None | None |
| rawmem_base20 | raw_token | 0.5546875 | 0.7135416666666666 | 0.3958333333333333 | 0.11458333333333333 | 0.4817708333333333 | 0.515625 | -0.0026041666666666297 | 0.002604166666666685 | 0.0 |
| rawmem_noevent_base20 | raw_token | 0.53125 | 0.515625 | 0.546875 | 0.052083333333333336 | 0.4947916666666667 | 0.5 | 0.0 | 0.0 | 0.0 |
| lexmem_base20 | raw_token | 0.5598958333333334 | 0.640625 | 0.4791666666666667 | 0.15625 | 0.3958333333333333 | 0.484375 | 0.0 | -0.0078125 | 0.02604166666666663 |
| lexmem_noevent_base20 | raw_token | 0.515625 | 0.5260416666666666 | 0.5052083333333334 | 0.15625 | 0.4036458333333333 | 0.5625 | 0.0 | 0.0 | 0.0 |
| vanilla_aug20 | raw_token | 0.5677083333333334 | 0.7395833333333334 | 0.3958333333333333 | 0.052083333333333336 | 0.34375 | 0.9479166666666666 | None | None | None |
| lexrec_aug20 | raw_token | 0.5 | 0.515625 | 0.484375 | 0.041666666666666664 | 0.4895833333333333 | 0.90625 | 0.0026041666666666297 | 0.0 | 0.0 |
| hardcoord_shared_base20 | hard_coordinate_ceiling | 0.8177083333333334 | 1.0 | 0.6354166666666666 | 0.6354166666666666 | 0.9192708333333334 | 0.53125 | -0.6354166666666667 | -0.9010416666666667 | -0.0625 |
| hardcoord_shared_aug20 | hard_coordinate_ceiling | 0.8489583333333334 | 1.0 | 0.6979166666666666 | 0.6979166666666666 | 0.9114583333333334 | 1.0 | -0.6979166666666667 | -0.8125 | -1.0 |

## Key comparisons

```json
{
  "hardcoord_base20_minus_vanilla_base20": {
    "held_recomb": 0.23958333333333337,
    "held_affected": 0.36979166666666663,
    "held_unaffected": 0.109375,
    "held_quartet_all": 0.45833333333333326,
    "paraphrase": 0.328125,
    "multi_event": 0.046875
  },
  "hardcoord_base20_minus_best_raw_base_held": {
    "best_raw_base_held": 0.5598958333333334,
    "hardcoord_held": 0.8177083333333334,
    "delta": 0.2578125
  },
  "hardcoord_aug20_minus_lexrec_aug20": {
    "held_recomb": 0.34895833333333337,
    "held_affected": 0.484375,
    "held_unaffected": 0.21354166666666663,
    "held_quartet_all": 0.65625,
    "paraphrase": 0.42187500000000006,
    "multi_event": 0.09375
  },
  "hardcoord_aug20_minus_vanilla_aug20": {
    "held_recomb": 0.28125,
    "held_affected": 0.26041666666666663,
    "held_unaffected": 0.3020833333333333,
    "held_quartet_all": 0.6458333333333333,
    "paraphrase": 0.5677083333333334,
    "multi_event": 0.05208333333333337
  }
}
```

## Decision

{
  "regex_assisted_phase2_interpretation": "integration_ceiling_only",
  "natural_bridge_authorized": false,
  "reason": "Hard-coordinate EntityMemory succeeds strongly under matched full-data settings and collapses under write permutation, while all tested raw-token interfaces remain near chance/vanilla and show negligible write-permutation dependence. The missing problem is latent occurrence-role assignment, not memory capacity or training duration.",
  "safe_next": "Design a raw-token latent entity/event/query assignment mechanism with paired counterfactual/equivariant objectives before frozen-DeBERTa EWoK transfer or fresh Strict-Small training."
}
