# debertav2 b256 aoa result — DeBERTa-v2 b256 GlobalPIQA diagnostics

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_b256_globalpiqa_diagnostics.json`

| split | accuracy | correct/n | pred distribution | length delta |
|---|---:|---:|---|---:|
| global_piqa_parallel | 24.27 | 25/103 | {'3': 25, '0': 21, '1': 32, '2': 25} | 0.13 |
| global_piqa_nonparallel | 47.00 | 47/100 | {'1': 45, '0': 55} | 0.03 |

## Parallel categories

| category | accuracy | correct/n |
|---|---:|---:|
| object_properties_interactions | 25.00 | 15/60 |
| spatial | 23.08 | 6/26 |
| affordances | 19.05 | 4/21 |
| counting | 13.33 | 2/15 |
| time | 27.27 | 3/11 |
| object_properties | 50.00 | 1/2 |
