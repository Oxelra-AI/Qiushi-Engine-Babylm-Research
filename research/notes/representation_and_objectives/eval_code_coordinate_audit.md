# eval code coordinate audit evaluation code coordinate audit

Old runner checkout: `experiments/archive/initial_model_studies/repos/babylm-eval/strict`
Pristine current checkout: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict`

Safe to use INITIAL_MODEL_STUDIES runner code for non-EWoK/non-AoA: `False`

| group | identical py hashes | old files | pristine files | diff hashes | only old | only pristine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence_zero_shot | True | 5 | 5 | 0 | 0 | 0 |
| reading | True | 3 | 3 | 0 | 0 | 0 |
| finetune | False | 6 | 6 | 1 | 0 | 0 |
| global_piqa | True | 1 | 1 | 0 | 0 | 0 |
| collator_scoring | True | 1 | 1 | 0 | 0 | 0 |
| calculate_results | True | 1 | 1 | 0 | 0 | 0 |
| utils | True | 1 | 1 | 0 | 0 | 0 |
| aoa_word | True | 6 | 6 | 0 | 0 | 0 |

Runner-relevant code differences found; inspect JSON and patch evaluation before corrected full eval.
