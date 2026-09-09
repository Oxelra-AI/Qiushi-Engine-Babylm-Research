# fineweb rewrite scale and route tokenizer geometry diagnostic

This CPU-only diagnostic measures token/word and approximate seq256 visibility for candidate data pools. It does not evaluate any model.

Loaded tokenizers: baseline16k, available40k

## Baseline16k summary

| pool | rows | words | tok/word mean | token p95 | trunc rows | visible word frac | entity one-piece | number one-piece |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official_pool_first5000 | 5000 | 800,000 | 1.280 | 250.0 | 237 | 0.9950 |  |  |
| clean_qwen_aligned_first5000 | 5000 | 778,916 | 1.483 | 294.0 | 1345 | 0.9764 |  |  |
| a02_high_anchor_sources_all | 3985 | 97,605 | 1.498 | 56.0 | 0 | 1.0000 | 0.221 | 0.000 |
| a02_high_precision_sources_all | 9517 | 214,771 | 1.442 | 52.0 | 0 | 1.0000 | 0.228 | 0.000 |
| a02_medium_repaired_sources_first12000 | 12000 | 270,212 | 1.433 | 58.0 | 0 | 1.0000 | 0.231 | 0.000 |
| a02_accepted_source_rewrite_rows_all | 952 | 48,655 | 1.469 | 111.0 | 0 | 1.0000 |  |  |
| a01_priority_unique_sources_first12000 | 12000 | 281,600 | 1.451 | 57.0 | 0 | 1.0000 | 0.230 | 0.000 |
| a01_seqsafe96_treatment_first12000 | 12000 | 1,624,993 | 1.365 | 231.0 | 101 | 0.9995 |  |  |

## 40k tokenizer comparison

| pool | tok/word 16k | tok/word 40k | visible frac 16k | visible frac 40k | entity one-piece 16k | entity one-piece 40k |
|---|---:|---:|---:|---:|---:|---:|
| official_pool_first5000 | 1.280 | 1.272 | 0.9950 | 0.9950 |  |  |
| clean_qwen_aligned_first5000 | 1.483 | 1.437 | 0.9764 | 0.9801 |  |  |
| a02_high_anchor_sources_all | 1.498 | 1.372 | 1.0000 | 1.0000 | 0.221 | 0.313 |
| a02_high_precision_sources_all | 1.442 | 1.325 | 1.0000 | 1.0000 | 0.228 | 0.318 |
| a02_medium_repaired_sources_first12000 | 1.433 | 1.321 | 1.0000 | 1.0000 | 0.231 | 0.318 |
| a02_accepted_source_rewrite_rows_all | 1.469 | 1.346 | 1.0000 | 1.0000 |  |  |
| a01_priority_unique_sources_first12000 | 1.451 | 1.335 | 1.0000 | 1.0000 | 0.230 | 0.320 |
| a01_seqsafe96_treatment_first12000 | 1.365 | 1.274 | 0.9995 | 0.9999 |  |  |

## Route reading

Future FineWeb training arms must match row length and token exposure, not only whitespace words. If source/rewrite pools tokenize much shorter or longer than the protected official rows, any component movement could reflect masked-token opportunity and context truncation as well as content. A 40k tokenizer may reduce fragmentation on some entities/numbers but also changes vocabulary and parameter allocation, so it should be a separate interaction experiment rather than bundled into the next data contrast.

JSON: `experiments/archive/representation_and_objectives/data/tokenizer_geometry/tokenizer_geometry_diagnostic.json`
