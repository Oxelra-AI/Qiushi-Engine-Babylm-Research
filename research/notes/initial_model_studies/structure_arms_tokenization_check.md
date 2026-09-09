# structure arms tokenization check — structure arms tokenization check

Evidence JSON: `experiments/archive/initial_model_studies/data/structure_arms_tokenization_check.json`

| arm | rows | words | tokens/word | kept tokens/word | groups/word | over-256 frac | max tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_entity_state | 129995 | 9999947 | 1.5491 | 1.5336 | 0.9428 | 1.2916% | 1081 |
| matched_low | 125194 | 9999960 | 1.6253 | 1.5880 | 0.9303 | 2.4226% | 1162 |
| uniform | 125430 | 9999997 | 1.5528 | 1.5268 | 0.9364 | 2.0274% | 894 |

Interpretation: these checks should be read before interpreting training differences; large hidden differences in kept tokens, truncation, or WWM groups would weaken a structure-density result.
