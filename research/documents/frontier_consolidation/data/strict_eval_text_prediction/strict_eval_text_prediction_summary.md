# distribution proximity prediction strict evaluation-text prediction

CPU-only strict extraction variant before MAX-register scores. EWoK uses only Context1/2 and Target1/2; COMPS uses prefix+property candidate strings; Entity uses input_prefix/options/prefix+option; BLiMP/Supplement use sentence_good/bad; Reading uses unique sentence.

| metric | beta | Pearson | Spearman | RMSE | exEntity pred | cheap6 pred |
|---|---:|---:|---:|---:|---:|---:|
| word_js | -2.0757 | -0.0085 | -0.1536 | 0.7401 | -0.0681 | -0.0823 |
| profile_js | 13.3187 | 0.3735 | 0.4571 | 0.4929 | 0.7697 | 0.8441 |

This variant checks whether the original broad key-name heuristics were driving the prediction. The profile model remains scientifically useful only insofar as the positive sign/magnitude survive this stricter extraction.

JSON: `experiments/archive/frontier_consolidation/data/strict_eval_text_prediction/strict_eval_text_prediction_summary.json`
