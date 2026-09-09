# cached fineweb quality audit cached FineWeb quality audit

Input: `experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl`. Single-document rows: 11,032, words: 1,765,120, unique docs: 5,587.

| tier | rows | words | word fraction of single-doc |
|---|---:|---:|---:|
| single_doc | 11,032 | 1,765,120 | 1.0000 |
| english_no_mojibake | 10,987 | 1,757,920 | 0.9959 |
| balanced_quality | 10,981 | 1,756,960 | 0.9954 |
| balanced_quality_startish | 1,642 | 262,720 | 0.1488 |

| flag | rows | fraction |
|---|---:|---:|
| no_flags | 10,700 | 0.9699 |
| url_or_email | 204 | 0.0185 |
| bad_substring | 86 | 0.0078 |
| nonlatin | 17 | 0.0015 |
| html | 14 | 0.0013 |
| mojibake | 14 | 0.0013 |
| symbol_or_index_like | 5 | 0.0005 |
| digit_heavy | 1 | 0.0001 |
| very_repetitive | 1 | 0.0001 |

Interpretation: cached FineWeb is not leader-quality FineWeb simplification data. A future fallback should prefer a sequence-safe and quality-filtered variant; otherwise the training signal may be diluted by web-fragment and tokenizer-noise artifacts.

JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_quality/fineweb_quality_audit.json`
