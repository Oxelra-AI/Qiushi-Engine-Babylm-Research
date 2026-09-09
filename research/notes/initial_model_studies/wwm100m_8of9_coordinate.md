# 34m smoke summary — WWM100M coordinate after AoA and (Super)GLUE

Evidence JSON: `experiments/archive/initial_model_studies/data/wwm100m_8of9_coordinate.json`

Full official Overall is still not computable because EWoK full-eval data is missing.

| column/task | score | status |
|---|---:|---|
| BLiMP | 55.92 | full eval |
| BLiMP Supplement | 52.03 | full eval |
| EWoK | — | missing: local full_eval/ewok_filtered empty; upstream gated |
| Entity Tracking | 16.76 | full eval |
| COMPS | 51.66 | full eval |
| (Super)GLUE | 63.08 | official finetune predictions aggregated by validation accuracy |
| GlobalPIQA parallel | 17.48 | full eval subcomponent |
| GlobalPIQA nonparallel | 48.00 | full eval subcomponent |
| GlobalPIQA mean | 32.74 | derived |
| Reading eye | 11.68 | full eval subcomponent |
| Reading self-paced | 3.82 | full eval subcomponent |
| Reading simple mean | 7.75 | derived sensitivity |
| AoA | 0.00 | official AoA runner; warning: 0 valid words for correlation |

## (Super)GLUE subtasks

| task | accuracy | correct / total |
|---|---:|---:|
| boolq | 66.54 | 1088 / 1635 |
| multirc | 58.50 | 1418 / 2424 |
| rte | 64.03 | 89 / 139 |
| wsc | 69.23 | 36 / 52 |
| mrpc | 70.59 | 144 / 204 |
| qqp | 71.80 | 14515 / 20215 |
| mnli | 40.89 | 2007 / 4908 |

Mean (Super)GLUE: 63.08

The sensitivity mean of the eight currently available columns is not an official Overall and should not be used as a leaderboard claim.
