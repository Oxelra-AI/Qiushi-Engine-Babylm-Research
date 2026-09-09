# route synthesis for independent_review old-vs-legal training dynamics mining

No new training/evaluation. Same compact-view reinvest stream, architecture, seed, and recipe; tokenizer differs. This tests whether the legal endpoint is under-optimized or instead has lower MLM loss while worse downstream competence.

## Endpoint run facts
| run | tokenizer | first loss | final loss | steps |
|---|---|---:|---:|---:|
| old_nonlegal_tok | baseline16k | 9.8098 | 2.5656 | 2529 |
| legal_step35_tok | compliant16k_reinvest10M | 9.8375 | 2.5526 | 2529 |
| bytealpha_tok | compliant16k_reinvest10M_bytealphabet | 9.8217 | 2.5713 | 2529 |

## Per-step loss delta sign
| contrast | mean delta | median | frac candidate lower loss than old | min | max | smoothed lower from exposure |
|---|---:|---:|---:|---:|---:|---:|
| legal_step35 - old | -0.00143 | -0.00097 | 0.520 | -0.20461 | 0.10400 | 395038 |
| bytealpha - old | 0.01319 | 0.00923 | 0.363 | -0.19158 | 0.13083 | 316049 |

## Window mean loss deltas
| window M words | old loss | legal loss | legal-old | bytealpha loss | bytealpha-old |
|---|---:|---:|---:|---:|---:|
| 0-1 | 9.1573 | 9.1305 | -0.0268 | 9.1252 | -0.0322 |
| 1-5 | 6.5751 | 6.5603 | -0.0148 | 6.5635 | -0.0116 |
| 5-10 | 4.8123 | 4.8070 | -0.0053 | 4.8077 | -0.0047 |
| 10-20 | 3.9435 | 3.9397 | -0.0038 | 3.9443 | 0.0008 |
| 20-40 | 3.2906 | 3.2946 | 0.0040 | 3.3438 | 0.0532 |
| 40-60 | 2.7963 | 2.7972 | 0.0009 | 2.8101 | 0.0138 |
| 60-80 | 2.5822 | 2.5802 | -0.0020 | 2.5849 | 0.0026 |
| 80-100 | 2.4790 | 2.4764 | -0.0026 | 2.4799 | 0.0010 |
| 90-100 | 2.4689 | 2.4659 | -0.0030 | 2.4689 | -0.0000 |

## Frequency-band dynamics where available
| checkpoint | legal-old high loss | mid loss | low loss | entropy | bytealpha-old high loss | mid loss | low loss | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_8M | 0.0245 | -0.1180 | -0.1209 | 0.0523 | 0.0154 | -0.1081 | -0.0844 | -0.0356 |
| chck_16M | -0.0317 | 0.1223 | -0.1241 | -0.0055 | -0.0327 | 0.1165 | -0.1454 | 0.0680 |
| chck_24M | -0.0769 | 0.0556 | 0.0040 | -0.0673 | -0.0396 | 0.0847 | 0.0881 | -0.0486 |
| chck_32M | 0.0121 | -0.0756 | -0.0747 | -0.0151 | 0.0394 | 0.0360 | 0.1239 | 0.0127 |
| chck_40M | -0.0369 | -0.0282 | 0.0682 | -0.1043 | -0.0077 | 0.0051 | 0.2050 | -0.0082 |
| chck_48M | -0.0201 | 0.0102 | 0.1229 | -0.0474 | -0.0107 | 0.0552 | -0.0092 | -0.0706 |
| chck_56M | 0.0234 | 0.1220 | -0.1313 | 0.0187 | 0.0472 | 0.1910 | -0.2333 | 0.0340 |
| chck_64M | -0.0390 | 0.0432 | -0.0610 | 0.0058 | -0.0259 | 0.0982 | -0.0371 | 0.0154 |
| chck_72M | 0.0112 | 0.0726 | -0.1139 | 0.0493 | 0.0013 | 0.0702 | -0.1622 | 0.0232 |
| chck_80M | -0.0179 | 0.1445 | 0.0575 | -0.0154 | -0.0231 | 0.0968 | -0.0085 | -0.0010 |
| chck_87M | -0.0036 | -0.0321 | 0.0537 | -0.0141 | 0.0028 | -0.0710 | 0.1365 | -0.0163 |
| chck_95M | -0.0180 | 0.0637 | -0.1169 | -0.0041 | -0.0315 | 0.0660 | -0.0476 | -0.0070 |

## Scientific reading
- Legal spatial repair route status does not have persistently lower MLM training loss; under-optimization remains plausible.
- Any bounded optimization experiment should therefore test a specific regularization/dynamics hypothesis, not a generic lr sweep. A 40M task-score ranking remains unsafe unless a dynamics quantity, not downstream score, is the target.

JSON: `experiments/archive/frontier_consolidation/data/old_legal_training_dynamics_mining/old_legal_training_dynamics_mining.json`
CSV: `experiments/archive/frontier_consolidation/data/old_legal_training_dynamics_mining/matched_step_loss_delta.csv`
