# consistency route stream validation and interface pair-span stream validation

Validate route portfolio and intervention assets source/rewrite span map against the exact frozen 100M training stream and quantify auxiliary-signal placement under inherited batch order.

This is CPU-only stream validation. It does not train, evaluate, alter the corpus, or choose an intervention route.

## 100M stream
- rows: `647400`
- field_words: `100000000`
- train_sha256: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- pass_rows: `[64740, 64740, 64740, 64740, 64740, 64740, 64740, 64740, 64740, 64740]`
- pass_words: `[10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000]`
- passes_match_first_pass: `False`

## Span-map join
- span_map_examples: `3005`
- changed_source_rows_in_100m: `30050`
- span_joined_rows_in_100m: `30050`
- changed_source_unmapped_rows: `0`
- mapped_wrong_source_rows: `0`
- missing_map_examples_in_100m: `0`
- mapped_occurrences_per_example: `{'n': 3005, 'min': 10, 'max': 10, 'mean': 10.0}`
- all_mapped_examples_repeat_exactly_ten_times: `True`
- pair_visibility_stream: `{'both_visible': 121450, 'invisible': 30, 'source_only': 70}`
- row_visibility_stream: `{'all_pairs_both_visible': 29950, 'some_pairs_both_visible': 100}`
- token_truncation_occurrences: `520`

## Exposure-level auxiliary placement
- 20M: rows=129480, batches=506, aux_rows=6010, aux_both_visible_pairs=24290, batches_with_aux=506 (1.0000); aux_rows_per_aux_batch mean=11.88, min=4.0, max=23.0
- 70M: rows=453180, batches=1771, aux_rows=21035, aux_both_visible_pairs=85015, batches_with_aux=1771 (1.0000); aux_rows_per_aux_batch mean=11.88, min=2.0, max=24.0
- 80M: rows=517920, batches=2024, aux_rows=24040, aux_both_visible_pairs=97160, batches_with_aux=2023 (0.9995); aux_rows_per_aux_batch mean=11.88, min=2.0, max=24.0
- 100M: rows=647400, batches=2529, aux_rows=30050, aux_both_visible_pairs=121450, batches_with_aux=2529 (1.0000); aux_rows_per_aux_batch mean=11.88, min=2.0, max=24.0

## Interpretation
- If source-view consistency is later selected, the route portfolio and intervention assets span map can join by example_id across the repeated 100M stream only if changed_source_unmapped_rows and mapped_wrong_source_rows stay zero and every mapped example appears ten times.
- The auxiliary signal is sparse per batch but, as quantified by `pair_span_pass_distribution`, changed rows are shuffled across nearly all batches in each 10M pass rather than kept as one contiguous block; a future source-view consistency trainer should join by `example_id` and normalize the auxiliary loss inside rows/pairs that actually have both-visible source and rewrite spans.
- This validation is not score evidence and does not favor source-view consistency over static-prior masking; mature 70M/80M clean-vs-reinvest results still choose which intervention family, if any, is scientifically justified.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_span_stream_validation/pair_span_stream_validation.json`
