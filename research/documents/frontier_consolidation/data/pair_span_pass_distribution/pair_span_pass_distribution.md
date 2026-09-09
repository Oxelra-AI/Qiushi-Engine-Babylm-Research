# consistency route stream validation and interface pair-span pass distribution

Correct the stream-level picture: the 100M file contains ten 10M passes with the same examples but shuffled order, while every mapped changed example still appears exactly once per pass and ten times overall.

## Global counts
- span_ids: `3005`
- pass_changed_counts: `[3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005]`
- pass_changed_words: `[423511, 423511, 423511, 423511, 423511, 423511, 423511, 423511, 423511, 423511]`
- unique_changed_ids_by_pass: `[3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005, 3005]`
- all_pass_changed_sets_equal_span_ids: `True`
- missing_span_changed_rows: `0`
- wrong_source_span_rows: `0`
- batch_count_per_pass: `253`

## Per-pass distribution
- pass 1: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=5.0, nonempty_max=21.0
- pass 2: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=2.0, nonempty_max=23.0
- pass 3: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=4.0, nonempty_max=20.0
- pass 4: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=2.0, nonempty_max=21.0
- pass 5: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=4.0, nonempty_max=27.0
- pass 6: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=4.0, nonempty_max=21.0
- pass 7: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=4.0, nonempty_max=24.0
- pass 8: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=4.0, nonempty_max=24.0
- pass 9: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=3.0, nonempty_max=22.0
- pass 10: changed_rows=3005, changed_words=423511, batches_with_changed=253/253 (1.0000), nonempty_batch_mean=11.88, nonempty_min=3.0, nonempty_max=24.0

## Interpretation
- The 100M stream is ten shuffled 10M passes, not ten identical row orders; a future source-view consistency trainer should join by example_id rather than by row offset within pass.
- Every pass contains all 3005 changed-row examples exactly once, and no changed-source row lacks a span map, so the route portfolio and intervention assets map is usable across the repeated exposure stream.
- Changed rows are distributed across nearly all batches in every pass rather than appearing as a single contiguous block, so the auxiliary signal is sparse per batch but not temporally confined to one corpus segment.
- This remains route-neutral engineering evidence; mature 70M/80M treatment effects still decide whether source-view consistency is worth GPU testing.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_span_pass_distribution/pair_span_pass_distribution.json`
