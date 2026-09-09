# aoa globalpiqa diagnostics — AoA and GlobalPIQA diagnostics

Evidence JSON: `experiments/archive/initial_model_studies/data/aoa_globalpiqa_diagnostics.json`

## AoA valid-word pipeline

Surprisal rows: 124640; unique target words: 328; CDI words: 504.

| drop/success category | count | examples |
|---|---:|---|
| flat_or_zero_fit_amplitude | 290 | animal, apple, arm, backyard, bad, balloon, basket, bat |
| no_child_aoa | 38 | baby, ball, basement, bench, book, camping, child, church |

## GlobalPIQA

| split | accuracy | correct/n | pred distribution | mean chosen-minus-correct answer length |
|---|---:|---:|---|---:|
| global_piqa_parallel | 17.48 | 18/103 | {'2': 35, '1': 21, '0': 22, '3': 25} | 0.25 |
| global_piqa_nonparallel | 48.00 | 48/100 | {'0': 56, '1': 44} | 0.08 |

Detailed by-label/by-category/error examples are in the JSON.
