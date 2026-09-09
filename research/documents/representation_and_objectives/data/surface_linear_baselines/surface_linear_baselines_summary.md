# earlier analysis CPU surface linear baselines

Shallow classifiers trained on the exact serialized text of the balanced k0/k16 neural probe.  These baselines are used to interpret whether a learned DeBERTa effect could be a simple surface pattern rather than event-role structure.

## Central metrics

| condition | arm | baseline | psc changed same | psc changed opposite | psc pair-both same | psc pair-both opposite | mixed true |
|---|---|---|---:|---:|---:|---:|---:|
| replace_k00_spread | aligned_state_bridge | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k00_spread | aligned_state_bridge | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k00_spread | aligned_state_bridge | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | aligned_state_bridge | char_tfidf | 0.523 | 0.547 | 0.523 | 0.547 | 0.500 |
| replace_k00_spread | aligned_state_bridge | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | aligned_state_bridge | count_word | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k00_spread | inverted_state_bridge | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k00_spread | inverted_state_bridge | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k00_spread | inverted_state_bridge | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | inverted_state_bridge | char_tfidf | 0.484 | 0.500 | 0.484 | 0.500 | 0.500 |
| replace_k00_spread | inverted_state_bridge | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | inverted_state_bridge | count_word | 0.875 | 0.875 | 0.875 | 0.875 | 0.500 |
| replace_k00_spread | heldheld_only | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k00_spread | heldheld_only | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k00_spread | heldheld_only | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | heldheld_only | char_tfidf | 0.453 | 0.469 | 0.445 | 0.453 | 0.504 |
| replace_k00_spread | heldheld_only | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k00_spread | heldheld_only | count_word | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k16_spread | aligned_state_bridge | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | aligned_state_bridge | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k16_spread | aligned_state_bridge | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | aligned_state_bridge | char_tfidf | 0.453 | 0.461 | 0.453 | 0.445 | 0.504 |
| replace_k16_spread | aligned_state_bridge | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | aligned_state_bridge | count_word | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k16_spread | inverted_state_bridge | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | inverted_state_bridge | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k16_spread | inverted_state_bridge | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | inverted_state_bridge | char_tfidf | 0.430 | 0.453 | 0.430 | 0.453 | 0.504 |
| replace_k16_spread | inverted_state_bridge | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | inverted_state_bridge | count_word | 0.875 | 0.875 | 0.875 | 0.875 | 0.500 |
| replace_k16_spread | heldheld_only | majority_true | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| replace_k16_spread | heldheld_only | word_tfidf | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| replace_k16_spread | heldheld_only | word_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | heldheld_only | char_tfidf | 0.453 | 0.469 | 0.445 | 0.453 | 0.504 |
| replace_k16_spread | heldheld_only | char_tfidf_anon | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| replace_k16_spread | heldheld_only | count_word | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |

## Reading

- A surface baseline that reaches high same-initial changed accuracy and pair-both in k16 would weaken a neural interpretation as role learning, because the text form itself would provide a shallow cue.
- A surface baseline that remains near the anti-copy/copy-initial floors while DeBERTa succeeds would support that pretrained representations or nonlinear fine-tuning are using more than ordinary lexical n-grams.
- Mixed true-statement accuracy must be interpreted with aligned versus inverted comparison; high values in both or unstable values are not signed coordinate evidence.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/surface_linear_baselines/surface_linear_baselines_report.json`
- per-row: `experiments/archive/representation_and_objectives/data/surface_linear_baselines/surface_baseline_per_row_predictions.jsonl`
