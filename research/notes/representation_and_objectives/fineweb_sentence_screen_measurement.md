# fineweb factual sentence rewrite source by rewrite FineWeb sentence source screen measurement

No generation, training, or evaluation was launched. The active H100 pair remains untouched.

| tier | rows | words | unique docs | word fraction | mean words | p95 words |
|---|---:|---:|---:|---:|---:|---:|
| broad_clean_row | 59,769 | 1,246,944 | 5,419 | 0.973 | 20.86 | 39 |
| complete_nonfragment | 40,770 | 861,862 | 5,357 | 0.672 | 21.14 | 38 |
| web_artifact_removed | 39,550 | 819,521 | 5,343 | 0.639 | 20.72 | 37 |
| balanced_source_by_rewrite | 32,836 | 703,975 | 5,283 | 0.549 | 21.44 | 37 |
| strict_factual_expository | 14,385 | 332,669 | 4,566 | 0.260 | 23.13 | 39 |

Interpretation: the previous broad prompt file preserves mass but includes dialogue/fragments/artifacts; the previous strict prompt file is cleaner but too narrow. The balanced source-by-rewrite tier is the intended CPU-side source asset if a future Qwen slice is needed, because it removes obvious malformed/web/list rows while preserving enough FineWeb sentence mass to test the leader-like source+rewrite structure.

JSON: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/fineweb_sentence_screen_tiers.json`

Samples: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens/fineweb_sentence_screen_samples.json`
