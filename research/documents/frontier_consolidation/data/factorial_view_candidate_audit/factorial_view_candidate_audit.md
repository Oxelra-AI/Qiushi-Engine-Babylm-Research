# lead factorial route reassessment factorial view candidate audit

CPU-only candidate-control construction; no training/evaluation/upload/submission.

## Pair-level geometry
| variant | tail coverage | source width | runs | adjacencies | function frac | Jaccard compact | compact overlap | view tok/word |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 70.26% |  |  |  | 30.60% | 1.0000 | 99.99% | 1.5439 |
| compact_scrambled | 70.26% |  |  |  | 30.60% | 1.0000 | 99.99% | 1.5439 |
| prefix_fluent | 4.85% | 60.41% | 1.00 | 100.00% | 47.12% | 0.3592 | 50.97% | 1.3337 |
| prefix_scrambled | 4.85% | 60.41% | 1.00 | 100.00% | 47.12% | 0.3592 | 50.97% | 1.3337 |
| sourcewide_onegap | 86.60% | 100.00% | 2.00 | 90.48% | 38.89% | 0.4965 | 64.79% | 1.4758 |
| sourcewide_onegap_scrambled | 86.60% | 100.00% | 2.00 | 90.48% | 38.89% | 0.4965 | 64.79% | 1.4758 |
| best_contiguous_span | 75.54% | 60.41% | 1.00 | 100.00% | 41.60% | 0.4706 | 62.57% | 1.4021 |

## Changed-block loader exposure
| variant | active tokens | trunc tokens | trunc rows | candidate groups | E[masked tokens] | active source/view tokens | pair full visible |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 607273 | 636 | 52 | 423179 | 91090.9 | 359905/247289 | 99.55% |
| compact_scrambled | 607273 | 636 | 52 | 423171 | 91090.9 | 359905/247124 | 99.55% |
| prefix_fluent | 576463 | 157 | 19 | 423424 | 86469.4 | 359964/216436 | 99.84% |
| prefix_scrambled | 576463 | 157 | 19 | 423426 | 86469.4 | 359964/216352 | 99.84% |
| sourcewide_onegap | 596804 | 487 | 34 | 423259 | 89520.6 | 359919/236822 | 99.70% |
| sourcewide_onegap_scrambled | 596804 | 487 | 34 | 423253 | 89520.6 | 359919/236699 | 99.70% |
| best_contiguous_span | 586049 | 311 | 25 | 423338 | 87907.3 | 359944/225879 | 99.79% |

## What the candidate family can and cannot identify
- Coherence/order at fixed recurrent words is cleanest within compact vs compact_scrambled, prefix_fluent vs prefix_scrambled, and sourcewide_onegap vs sourcewide_onegap_scrambled: each pair shares the same word multiset and nearly the same loader exposure by construction.
- Source-position spread is better isolated by sourcewide_onegap vs prefix_fluent than by the source wide skeleton recurrence integrated telegraphic skeletons: onegap keeps at most two contiguous source spans, preserving local sentence fragments while increasing tail/source-width exposure. But it still changes which source words recur, so it is not a perfect fixed-identity contrast.
- Natural compact style beyond source-only recurrence is read by compact vs sourcewide_onegap only after exposure and distribution differences are checked; it cannot be inferred from copy NLL alone.
- These candidates are still not trainable routes until pending DeBERTa/results are read and a final scaffold matches exact-loader exposure tightly enough.

## Derived comparison highlights
### coherence_fixed_words
- compact_minus_compact_scrambled: `{"delta_pair_view_tokens_per_word": 0.0, "delta_changed_active_tokens": 0, "delta_changed_candidate_groups": 8, "shared_word_multiset_by_construction": true}`
- prefix_fluent_minus_prefix_scrambled: `{"delta_pair_view_tokens_per_word": 0.0, "delta_changed_active_tokens": 0, "delta_changed_candidate_groups": -2, "shared_word_multiset_by_construction": true}`
- sourcewide_onegap_minus_scrambled: `{"delta_pair_view_tokens_per_word": 0.0, "delta_changed_active_tokens": 0, "delta_changed_candidate_groups": 6, "shared_word_multiset_by_construction": true}`

### position_spread_sentence_preserving
- sourcewide_onegap_minus_prefix_fluent: `{"delta_tail_content_coverage": 0.8175092517997512, "delta_source_width_frac": 0.3959397161822006, "delta_num_source_runs": 1.0, "delta_function_fraction": -0.08235951835786054, "delta_changed_active_tokens": 20341, "delta_changed_view_active_tokens": 20386}`

### natural_compact_beyond_source_only
- compact_minus_sourcewide_onegap: `{"delta_tail_content_coverage": -0.16349580873705116, "delta_jaccard_with_compact": 0.5035451723256805, "delta_function_fraction": -0.08285536380124814, "delta_changed_active_tokens": 10469, "delta_changed_view_active_tokens": 10467}`

Pair metrics CSV: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/factorial_candidate_pair_metrics.csv`
Examples: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/sourcewide_onegap_examples.jsonl`
Pair files:
- compact: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl` SHA `6d0ac85dec1718e5f5663d09123e2f62a23f7cceb0b491b83a34f7bfca59d32d`
- compact_scrambled: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_scrambled_candidate_pairs.jsonl` SHA `44862d135b0d406b66f2d7cce7197a243b2cf7438bdee3842d5739286d07057c`
- prefix_fluent: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/prefix_fluent_candidate_pairs.jsonl` SHA `b7c5199db9c4276a4ad1ce050557d5d9faf23244d9cb9845e727930441ba1c6e`
- prefix_scrambled: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/prefix_scrambled_candidate_pairs.jsonl` SHA `2436246933757dbf7247d8cb13023e4ec524f0dd5d3b1eeabc51913506c04e8d`
- sourcewide_onegap: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/sourcewide_onegap_candidate_pairs.jsonl` SHA `93f1517ded600c9cbe55321f53d7866972d9edbe9dd8238aa7f0acc2e217de92`
- sourcewide_onegap_scrambled: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/sourcewide_onegap_scrambled_candidate_pairs.jsonl` SHA `dea345668e3a55b3309c6ecb0b52087f0fbb83d0850dedaadc7835dd886e0bfe`
- best_contiguous_span: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/best_contiguous_span_candidate_pairs.jsonl` SHA `341a34b8ef9d742d14136cf5cf7fec3b591d337350c7dba344244c255a7f1079`
JSON: `experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/factorial_view_candidate_audit.json`
