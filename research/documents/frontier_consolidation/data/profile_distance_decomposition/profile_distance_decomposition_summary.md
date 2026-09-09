# distribution proximity prediction profile-distance decomposition

This CPU-only decomposition was produced before reading any MAX-register score. It decomposes `profile_js(eval, removed_childspeech) - profile_js(eval, removed_adultprose)` into feature classes.

## Ex-Entity feature-group summary

| feature group | mean contribution | min | max | positive families | negative families | Spearman vs committed prediction |
|---|---:|---:|---:|---:|---:|---:|
| open_class_abstract_shape | 0.02449 | 0.01044 | 0.04489 | 5 | 0 | 0.20000 |
| transcript_or_markup_format | 0.01774 | 0.01774 | 0.01774 | 5 | 0 | NA |
| punctuation | 0.01299 | 0.01046 | 0.01535 | 5 | 0 | -0.30000 |
| sentence_length | -0.00856 | -0.02953 | 0.01116 | 2 | 3 | 1.00000 |
| function_category | 0.00650 | 0.00244 | 0.00870 | 5 | 0 | 0.20000 |
| function_word_identity | 0.00480 | -0.00732 | 0.01501 | 3 | 2 | 0.80000 |
| token_length | -0.00338 | -0.00346 | -0.00319 | 0 | 5 | 0.15390 |
| number_token | -0.00335 | -0.00346 | -0.00308 | 0 | 5 | 0.15390 |
| other | 0.00115 | -0.00431 | 0.00585 | 4 | 1 | -0.70000 |

## Scientific reading

The committed profile prediction is not a pure lexical-frequency account, but its scientific interpretation depends on which profile classes carry the distance. If transcript/markup features dominate, the register experiment is partly a format mismatch test. If function words, punctuation, sentence length and abstract open-class shape all contribute with the same sign, the result is closer to a structural-register proximity account.

JSON: `experiments/archive/frontier_consolidation/data/profile_distance_decomposition/profile_distance_decomposition_summary.json`
Top feature table: `experiments/archive/frontier_consolidation/data/profile_distance_decomposition/profile_top_feature_contributions.csv`
