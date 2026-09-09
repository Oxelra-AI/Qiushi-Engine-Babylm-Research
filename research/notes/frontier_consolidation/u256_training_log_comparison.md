# u256 endpoint mechanism — U256 training-log comparison

Status: `U256_TRAINING_LOG_COMPARISON`

- line counts: spatial repair route status 2529, U256 2530
- final counted words: spatial repair route status 100000000, U256 100000000
- final losses: spatial repair route status 2.5525617599487305, U256 2.516624725910071, delta -0.035937
- common-step LR mismatches: 2377; batch-word mismatches: 2528; masked-token mismatches: 2524

## Milestones
| target | step delta | cumulative-word delta | loss delta | masked-token delta |
|---|---:|---:|---:|---:|
| 1M | 0 | 1085 | -0.009151 | +171 |
| 5M | 1 | 29825 | -0.126043 | -383 |
| 10M | 0 | -4381 | -0.159401 | +396 |
| 20M | 0 | -8711 | -0.036991 | -147 |
| 30M | 0 | -12927 | -0.163398 | -52 |
| 40M | 0 | -17538 | +0.103720 | +155 |
| 50M | 1 | 17927 | +0.041289 | +623 |
| 60M | 1 | 13527 | -0.248326 | -62 |
| 70M | 1 | 9059 | +0.004501 | +262 |
| 80M | 1 | 5021 | -0.113048 | +266 |
| 90M | 1 | 596 | -0.046969 | -40 |
| 100M | 1 | 0 | -0.035937 | +1079 |

## Scientific reading
U256 and spatial repair route status end at the same counted 100M words, but U256 uses 2530 updates versus spatial repair route status 2529. Across common steps, LR is essentially the same at each index (mismatches 2377, max abs 3.83e-07), while batch word counts and mask counts differ at almost every step because chunking changes the training examples. U256's final MLM loss is lower by -0.035937, but previous routes showed lower loss can redistribute rather than improve official competence, so the pending full score remains decisive.

JSON: `experiments/archive/frontier_consolidation/data/u256_training_log_comparison/u256_training_log_comparison.json`
