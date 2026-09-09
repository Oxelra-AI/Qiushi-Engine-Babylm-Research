# earlier analysis position-matched direct register-displacement pools

CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.

## Scientific contrast
Both arms admit the same quarter_1x FineWeb text multiset (758 rows, 105,962 words, rho=0.0105962) by exact word count. The replaced clean rows are selected as child/subtitle vs Gutenberg/SimpleWiki pairs with the same word-count histogram and reduced row-index mismatch. Unselected rows remain original clean text at original positions.

## Position matching
```json
{
  "n_pairs": 758,
  "mean_abs_row_distance": 38.03298153034301,
  "median_abs_row_distance": 17.0,
  "p90_abs_row_distance": 55,
  "max_abs_row_distance": 842,
  "same_clean_row_both_arms_count": 3,
  "mean_child_purity": 0.9014164181493843,
  "mean_adult_purity": 0.6604922140089793
}
```

## Arm summaries
### childsub_posmatched
10M: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/regpos_childsub_posmatched_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/regpos_childsub_posmatched_samefw_quarter_100M.jsonl`
SHA10: `f5b4995253985ea2af40e76c0bbca1940acc67792d337475007e48ec03f3db37`
SHA100: `a99c0adeb5712f05b469a9c3980606c474dbba1189d5cb35b7ab79d02ff50b9e`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 99727 (0.941158), Gutenberg+SimpleWiki 3726 (0.035164), other 2509
Row index mean/sd/span: 3977.3 / 2473.1 / 13..7915

### adult_posmatched
10M: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/regpos_adult_posmatched_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/regpos_adult_posmatched_samefw_quarter_100M.jsonl`
SHA10: `59c441d772f463dbdd3686b3de193d829de02401c551a6a46d3b84dd045ba205`
SHA100: `2b93c7d6c1db1527fe70285576968c2f6dfb30030a7df1c8dfcde779b0ee0f55`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 13278 (0.125309), Gutenberg+SimpleWiki 84408 (0.796587), other 8276
Row index mean/sd/span: 3992.5 / 2464.7 / 0..7918

## Interpretation use
Use this position-matched direct pair as the primary register-removal H100 discriminator if the sub-dose curve makes rho≈0.011 worth resolving. The simpler direct pair remains a sensitivity check; the fixed-position relocation pair remains a different control on active FineWeb timing.

Metadata: `experiments/archive/frontier_consolidation/data/register_position_matched_direct_pools/register_position_matched_direct_pools_metadata.json`
