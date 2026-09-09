# earlier analysis pair-consistency feasibility

Determine whether a future source-rewrite consistency learning signal can be implemented under the frozen compact-view-reinvest corpus without changing the data stream or reading evaluation examples.

## Counts
- rows_total: 64740
- changed_rows: 3005
- selected_pair_ids: 12155
- pair_rows_loaded: 12155
- exact_or_tiny_suffix_alignment_rows: 3005
- token_role_rows: 3005
- rows_with_true_token_truncation: 52
- pair_token_records: 12152

## Row/token stats
- pair_count: mean=4.045, median=4.000, p05=3.000, p95=5.000, min=2, max=7
- word_length: mean=140.935, median=143.000, p05=116.000, p95=159.000, min=83, max=160
- token_length: mean=202.084, median=202.000, p05=160.000, p95=244.000, min=117, max=256
- source_tokens: mean=118.912, median=119.000, p05=94.000, p95=143.000, min=68, max=174
- rewrite_tokens: mean=81.444, median=81.000, p05=63.000, p95=100.000, min=41, max=112
- source_plus_rewrite_tokens: mean=200.355, median=201.000, p05=159.200, p95=241.000, min=117, max=256
- rewrite_to_source_token_ratio: mean=0.686, median=0.686, p05=0.593, p95=0.777, min=0.47126436781609193, max=0.8782608695652174

## Pair visibility at seq256
- {'both_visible': 12145, 'source_only': 7}

## Implementation map
- The inherited trainer dataset does not return example_id, source label, pair_id, role, or token span fields; a true consistency objective requires a new dataset/collate path, not only a masking-function patch.
- The changed-block sidecar plus accepted pair metadata reconstruct source/rewrite roles for essentially all changed rows; token role spans are therefore feasible as an auxiliary supervision map derived only from the training corpus.
- A lowest-risk consistency screen would add a small auxiliary loss only on changed-block rows: encode source and rewrite token subsets from the same packed row, pool hidden states over matching pair spans, and penalize distance between source and rewrite representations after the MLM forward. This changes the learning signal but not corpus, tokenizer, data order, model architecture, or evaluation.
- Implementation should be separate from the static-prior trainer; do not mix consistency and static prior in the first test. Use one cheap short/mature screen only if the matched clean-control trajectory shows compact-view reinvestment needs a learning-signal repair rather than a representation-only repair.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_consistency_feasibility/pair_consistency_feasibility.json`
