# eval code coordinate audit evaluation dataset coordinate audit

Old runner checkout: `experiments/archive/initial_model_studies/repos/babylm-eval/strict`
Pristine current checkout: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict`

Safe for inherited wrapper on non-EWoK/non-AoA: `False`

| Task | identical hashes | old files | pristine files | old lines | pristine lines | diff hashes | only old | only pristine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BLiMP | True | 67 | 67 | 59875 | 59875 | 0 | 0 | 0 |
| Supplement | True | 5 | 5 | 5218 | 5218 | 0 | 0 | 0 |
| EWoK | False | 11 | 11 | 6666 | 7618 | 7 | 0 | 0 |
| Entity | True | 3 | 3 | 9483 | 9483 | 0 | 0 | 0 |
| COMPS | True | 4 | 4 | 91028 | 91028 | 0 | 0 | 0 |
| GlobalPIQA_parallel | False | 1 | 0 | 103 | 0 | 0 | 1 | 0 |
| GlobalPIQA_nonparallel | False | 1 | 0 | 100 | 0 | 0 | 1 | 0 |
| Reading | True | 1 | 1 | 1727 | 1727 | 0 | 0 | 0 |
| SuperGLUE | True | 14 | 14 | 92959 | 92959 | 0 | 0 | 0 |
| AoA | True | 2 | 2 | 65536 | 65536 | 0 | 0 | 0 |

Unrepaired differences were found; inspect the JSON before evaluation.
