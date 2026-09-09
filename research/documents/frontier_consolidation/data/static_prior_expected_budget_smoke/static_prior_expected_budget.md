# earlier analysis static-prior expected-budget analysis

Exact expected-budget analysis for static-prior WWM on the trainer-visible legal-pool seq256 stream; verifies whether token-level MLM supervision is preserved after token-length normalization.

This deterministic CPU analysis uses the actual generated trainer dataset and WWM grouping. It is not a model-training result.

## Scheme `relation_only_v1`
- rows analyzed: 4096; rows at candidate max length: 314
- source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 3005, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1, 'qwen_pair_packed': 227, 'childes': 274, 'gutenberg': 209, 'open_subtitles': 205, 'simple_wiki': 110, 'bnc_spoken': 62, 'switchboard': 3}

| split | fixed token rate | static token rate | Δ token | fixed group rate | static group rate | Δ group | selected prior lift | relation lift | high1.50 lift | clip hits | max p | len-weight corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.152308 | 0.002308 | 1.041646 | 1.592308 | 1.763905 | 0 | 0.293041 | -0.106315 |
| changed | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.152519 | 0.002519 | 1.043329 | 1.587238 | 1.760374 | 0 | 0.293032 | -0.109810 |
| other | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.151769 | 0.001769 | 1.037362 | 1.607946 | 1.774614 | 0 | 0.293041 | -0.095275 |

## Scheme `relation_info_v1`
- rows analyzed: 4096; rows at candidate max length: 314
- source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 3005, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1, 'qwen_pair_packed': 227, 'childes': 274, 'gutenberg': 209, 'open_subtitles': 205, 'simple_wiki': 110, 'bnc_spoken': 62, 'switchboard': 3}

| split | fixed token rate | static token rate | Δ token | fixed group rate | static group rate | Δ group | selected prior lift | relation lift | high1.50 lift | clip hits | max p | len-weight corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.150208 | 0.000208 | 1.022704 | 1.405789 | 1.532848 | 0 | 0.280684 | -0.010559 |
| changed | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.150305 | 0.000305 | 1.023316 | 1.400186 | 1.528585 | 0 | 0.278786 | -0.015308 |
| other | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.149960 | -0.000040 | 1.021135 | 1.423069 | 1.545351 | 0 | 0.280684 | 0.005066 |

## Input files

- trainer: `experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_static_prior.py`
- pool_10m: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- prior_json: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
- limit_rows: `4096`
- seq_length: `256`
- mask_prob: `0.15`
- schemes: `['relation_only_v1', 'relation_info_v1']`

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_expected_budget_smoke/static_prior_expected_budget.json`
