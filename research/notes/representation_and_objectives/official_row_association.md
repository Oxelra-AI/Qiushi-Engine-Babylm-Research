# official row association official row association

Status: **ROW_LEVEL_AGGREGATED** from existing EWoK records; no new inference.

Update-sensitive EWoK rows: 890 global rows from domains material-dynamics, physical-dynamics.

## Row-wise correlation distributions

Each official row is treated separately; within that row, the association is across models. A strong synthetic route signal would show a consistent sign over many official update rows, not only a model-average association.

| synthetic metric vs row outcome | rows with r | median r | mean r | frac positive | frac negative |
|---|---:|---:|---:|---:|---:|
| action_crossed__correct | 849 | -0.0241 | -0.0038 | 0.4923 | 0.5077 |
| action_crossed__interaction_sum | 890 | -0.0292 | -0.0339 | 0.4674 | 0.5326 |
| action_crossed__stable_failure | 818 | -0.0708 | 0.0082 | 0.4523 | 0.5477 |
| direct_negative_nonadditive_frac__correct | 849 | 0.0109 | -0.0036 | 0.5029 | 0.4971 |
| direct_negative_nonadditive_frac__interaction_sum | 890 | -0.0565 | -0.0447 | 0.4404 | 0.5596 |
| direct_negative_nonadditive_frac__stable_failure | 818 | 0.0477 | 0.0025 | 0.5183 | 0.4817 |
| direct_nonadditive_median__correct | 849 | 0.0326 | 0.0156 | 0.5253 | 0.4747 |
| direct_nonadditive_median__interaction_sum | 890 | 0.0623 | 0.0705 | 0.5730 | 0.4270 |
| direct_nonadditive_median__stable_failure | 818 | 0.0150 | -0.0309 | 0.5037 | 0.4963 |
| targetfree_action_crossed__correct | 849 | -0.0116 | -0.0149 | 0.4888 | 0.5112 |
| targetfree_action_crossed__interaction_sum | 890 | -0.1212 | -0.0798 | 0.4022 | 0.5978 |
| targetfree_action_crossed__stable_failure | 818 | 0.0259 | 0.0241 | 0.5196 | 0.4804 |
| targetfree_negative_nonadditive_frac__correct | 849 | -0.0074 | 0.0012 | 0.4676 | 0.5324 |
| targetfree_negative_nonadditive_frac__interaction_sum | 890 | -0.1193 | -0.0584 | 0.4202 | 0.5798 |
| targetfree_negative_nonadditive_frac__stable_failure | 818 | 0.0929 | 0.0087 | 0.5489 | 0.4511 |
| targetfree_nonadditive_median__correct | 849 | 0.0420 | 0.0100 | 0.5300 | 0.4700 |
| targetfree_nonadditive_median__interaction_sum | 890 | 0.1341 | 0.0815 | 0.6157 | 0.3843 |
| targetfree_nonadditive_median__stable_failure | 818 | -0.0914 | -0.0254 | 0.4242 | 0.5758 |

## Stacked model×row association

| synthetic metric vs row outcome | n | Pearson | Spearman |
|---|---:|---:|---:|
| action_crossed__correct | 7120 | -0.0054 | -0.0036 |
| action_crossed__interaction_sum | 7120 | -0.0359 | -0.0065 |
| action_crossed__stable_failure | 7120 | 0.0146 | 0.0122 |
| direct_nonadditive_median__correct | 7120 | 0.0161 | 0.0177 |
| direct_nonadditive_median__interaction_sum | 7120 | 0.0437 | 0.0310 |
| direct_nonadditive_median__stable_failure | 7120 | -0.0331 | -0.0383 |
| targetfree_action_crossed__correct | 7120 | -0.0148 | -0.0157 |
| targetfree_action_crossed__interaction_sum | 7120 | -0.0365 | -0.0304 |
| targetfree_action_crossed__stable_failure | 7120 | 0.0269 | 0.0296 |
| targetfree_nonadditive_median__correct | 7120 | 0.0092 | 0.0085 |
| targetfree_nonadditive_median__interaction_sum | 7120 | 0.0297 | 0.0372 |
| targetfree_nonadditive_median__stable_failure | 7120 | -0.0252 | -0.0257 |

JSON: `experiments/archive/representation_and_objectives/data/official_row_association/official_row_association.json`
Row records: `experiments/archive/representation_and_objectives/data/official_row_association/row_level_associations.jsonl`
