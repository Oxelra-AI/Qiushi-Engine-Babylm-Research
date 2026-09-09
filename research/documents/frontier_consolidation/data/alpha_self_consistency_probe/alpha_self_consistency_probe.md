# private scale endpoint vs mechanism synthesis alpha self-consistency probe

Status: **COMPLETE**

This is a payload-only headroom check on six discrete columns; Reading/SuperGLUE/AoA are not included.

| rule | discrete6 mean | delta vs anchor | gain | loss | net | changed |
|---|---:|---:|---:|---:|---:|---:|
| a0p75 | 50.183909 | +0.256003 | 2332 | 2418 | -86 | 4750 |
| anchor_plus_scaled_majority_tie_anchor | 50.182522 | +0.254616 | 1553 | 1575 | -22 | 3128 |
| private_unanimous_else_anchor | 50.182522 | +0.254616 | 1553 | 1575 | -22 | 3128 |
| a0p5 | 50.179076 | +0.251170 | 1560 | 1581 | -21 | 3141 |
| scaled_majority_else_anchor | 50.173869 | +0.245963 | 2330 | 2417 | -87 | 4747 |
| a0p5_unless_075_1_agree_else_that | 50.173869 | +0.245963 | 2330 | 2417 | -87 | 4747 |
| a1 | 50.096344 | +0.168438 | 3116 | 3231 | -115 | 6347 |
| a0_anchor | 49.927906 | +0.000000 | 0 | 0 | +0 | 0 |

## By-column scores

| rule | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA |
|---|---:|---:|---:|---:|---:|---:|
| a0p75 | 68.5160 | 63.6354 | 50.0196 | 28.3222 | 52.0472 | 38.5631 |
| anchor_plus_scaled_majority_tie_anchor | 68.5453 | 63.1459 | 50.0147 | 28.2302 | 52.1105 | 39.0485 |
| private_unanimous_else_anchor | 68.5453 | 63.1459 | 50.0147 | 28.2302 | 52.1105 | 39.0485 |
| a0p5 | 68.5447 | 63.1459 | 49.9961 | 28.2302 | 52.1090 | 39.0485 |
| scaled_majority_else_anchor | 68.5143 | 63.6354 | 50.0196 | 28.2631 | 52.0477 | 38.5631 |
| a0p5_unless_075_1_agree_else_that | 68.5143 | 63.6354 | 50.0196 | 28.2631 | 52.0477 | 38.5631 |
| a1 | 68.5293 | 63.6459 | 49.9060 | 28.4399 | 51.9938 | 38.0631 |
| a0_anchor | 68.4913 | 62.9378 | 50.0555 | 28.3140 | 52.1912 | 37.5777 |

## Scientific reading

- Best simple self-consistency rule on the six discrete columns is a0p75 with delta +0.256003 points vs anchor.
- Because this uses saved winners, not logits, it is only a headroom check. A future custom multi-alpha model wrapper would require a frozen label-free rule before evaluation, and should not be pursued if this winner-level headroom is weak or purely redistribution.

JSON: `experiments/archive/frontier_consolidation/data/alpha_self_consistency_probe/alpha_self_consistency_probe.json`
