# route portfolio and intervention assets pair-span map

Durable trainer-visible token-span map for possible source-view consistency; derived only from the frozen legal compact-view-reinvest training corpus and sidecars.

This is CPU-only training-side metadata. It does not choose or launch a route.

## Counts
- pool_rows_scanned: `64740`
- changed_rows_seen: `3005`
- rows_written: `3005`
- selected_pair_ids: `12155`
- pair_rows_loaded: `12155`
- exact_alignment_rows: `3005`
- tiny_suffix_alignment_rows: `0`
- token_truncation_rows: `52`
- pair_visibility: `{'both_visible': 12145, 'invisible': 3, 'source_only': 7}`
- row_visibility: `{'all_pairs_both_visible': 2995, 'some_pairs_both_visible': 10}`

## Row token stats
- source_tokens: mean=118.912, median=119.000, p05=94.000, p95=143.000, min=68.0, max=174.0
- rewrite_tokens: mean=81.444, median=81.000, p05=63.000, p95=100.000, min=41.0, max=112.0
- source_plus_rewrite_tokens: mean=200.355, median=201.000, p05=159.200, p95=241.000, min=117.0, max=256.0

## Pair token stats
- source_plus_rewrite_visible_tokens_per_pair_record: mean=49.533, median=46.000, p05=25.000, p95=86.000, min=0.0, max=157.0
- contiguous_ranges_per_pair_record: mean=2.421, median=2.000, p05=2.000, p95=4.000, min=0.0, max=12.0

## Use constraints
- This map is a construction asset, not evidence that source-view consistency improves BabyLM scores.
- Do not combine this objective with static-prior masking in a first GPU test; the mature 70M/80M pattern must select the intervention family first.
- A future trainer using this map should keep corpus, tokenizer, model, optimizer, seeds, exposure accounting, and evaluation fixed; only the auxiliary source-view consistency loss should change.
- Contrastive in-batch negatives are risky in tiny-data MLM pretraining; a first implementation should consider stop-gradient or low-weight symmetric representation/logit agreement over true source-rewrite pairs before adding negatives.

Map JSONL: `experiments/archive/frontier_consolidation/data/pair_span_map/pair_span_map.jsonl`
Summary JSON: `experiments/archive/frontier_consolidation/data/pair_span_map/pair_span_map_summary.json`
