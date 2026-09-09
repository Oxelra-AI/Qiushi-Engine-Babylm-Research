# earlier analysis exact candidate-choice metrics

Exact-choice metrics compare the true candidate row's score to the false candidate row's score within each query.  They prevent an always-true classifier from appearing perfect on true-row-only metrics.

## Source `surface_linear_baselines`

- input rows: 96768
- choice queries: 36864
- paired changed+unchanged records: 18432

| condition | arm | baseline/model | changed same exact | changed opposite exact | pair-both same exact | pair-both opposite exact | tie same changed |
|---|---|---|---:|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | char_tfidf | 0.000 | 1.000 | 0.000 | 0.000 | 0.977 |
| replace_k00_spread | aligned_state_bridge | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | aligned_state_bridge | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | aligned_state_bridge | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | aligned_state_bridge | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | aligned_state_bridge | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | heldheld_only | char_tfidf | 0.000 | 0.500 | 0.000 | 0.000 | 0.984 |
| replace_k00_spread | heldheld_only | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | heldheld_only | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | heldheld_only | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | heldheld_only | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | heldheld_only | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | char_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.984 |
| replace_k00_spread | inverted_state_bridge | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | char_tfidf | 0.400 | 0.429 | 0.000 | 0.000 | 0.961 |
| replace_k16_spread | aligned_state_bridge | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | heldheld_only | char_tfidf | 0.000 | 0.500 | 0.000 | 0.000 | 0.984 |
| replace_k16_spread | heldheld_only | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | heldheld_only | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | heldheld_only | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | heldheld_only | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | heldheld_only | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | char_tfidf | 0.429 | 0.750 | 0.000 | 0.000 | 0.945 |
| replace_k16_spread | inverted_state_bridge | char_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | count_word | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | majority_true | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | word_tfidf | n/a | n/a | 0.000 | 0.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | word_tfidf_anon | n/a | n/a | 0.000 | 0.000 | 1.000 |

## Scientific reading

- True-row-only accuracy is pseudolikelihood-style evidence; exact-choice is the stricter behavioral readout for two-candidate state queries.
- The earlier analysis learned run saves logits, so exact-choice margins should be the primary state/conservation metric once that run finishes.
- A high true-row score but low exact-choice score indicates over-acceptance or poor calibration, not selective state knowledge.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/exact_choice_metrics/exact_choice_metrics_report.json`
