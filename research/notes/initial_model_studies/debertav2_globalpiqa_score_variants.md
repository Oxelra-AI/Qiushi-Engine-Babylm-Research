# debertav2 b256 coordinate — DeBERTa GlobalPIQA scoring variants

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_globalpiqa_score_variants.json`

| split | official mean acc | raw-sum acc | first-token acc | mean-vs-sum changed | sum-correct/mean-wrong | mean-correct/sum-wrong |
|---|---:|---:|---:|---:|---:|---:|
| global_piqa_parallel | 24.27 | 19.42 | 21.36 | 26 | 5 | 10 |
| global_piqa_nonparallel | 47.00 | 48.00 | 53.00 | 11 | 6 | 5 |

Official GlobalPIQA uses length-normalized completion log-probability; raw-sum and first-token are probes only, not leaderboard scores.
