# mlm rtd gdes 20m screen plan MLM+RTD-GDES 20M screen summary

Decision: `needs_scientific_review_before_continuation`

## Training validation
- Run dir: `experiments/archive/frontier_consolidation/training/runs/mlm_rtd_lambda1_seed43022_20M`
- Status: `MLM_RTD_TRAINING_COMPLETE`
- Word exposure: 20000000
- Actual steps: 506
- Last MLM loss: 3.810459852218628
- Last RTD loss: 0.5505039095878601
- Checkpoints present: {'chck_5M': True, 'chck_10M': True, 'chck_15M': True, 'chck_20M': True}

## Cheap-column scores
| column | baseline20M | MLM+RTD20M | delta |
|---|---:|---:|---:|
| BLiMP | 59.6900 | 60.1600 | +0.4700 |
| Supplement | 55.4500 | 56.9400 | +1.4900 |
| EWoK | 50.7300 | 50.0700 | -0.6600 |
| Entity | 18.6500 | 19.3500 | +0.7000 |
| COMPS | 50.2600 | 50.2400 | -0.0200 |
| GlobalPIQA | 34.1950 | 33.2250 | -0.9700 |
| Reading | 8.6700 | 8.4350 | -0.2350 |
| cheap7 | 39.6636 | 39.7743 | +0.1107 |

## Interpretation
- 20M movement is small or mixed; continuation would need mechanism-specific reasoning, not routine maturation hope.
