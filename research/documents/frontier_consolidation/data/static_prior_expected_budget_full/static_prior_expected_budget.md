# earlier analysis static-prior expected-budget analysis

Exact expected-budget analysis for static-prior WWM on the trainer-visible legal-pool seq256 stream; verifies whether token-level MLM supervision is preserved after token-length normalization.

This deterministic CPU analysis uses the actual generated trainer dataset and WWM grouping. It is not a model-training result.

## Scheme `relation_only_v1`
- rows analyzed: 64740; rows at candidate max length: 15548
- source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 3005, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1, 'qwen_pair_packed': 12236, 'childes': 16093, 'gutenberg': 11621, 'open_subtitles': 11434, 'simple_wiki': 6619, 'bnc_spoken': 3599, 'switchboard': 132}

| split | fixed token rate | static token rate | Δ token | fixed group rate | static group rate | Δ group | selected prior lift | relation lift | high1.50 lift | clip hits | max p | len-weight corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.151793 | 0.001793 | 1.037227 | 1.609718 | 1.774779 | 0 | 0.297237 | -0.097164 |
| changed | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.152519 | 0.002519 | 1.043329 | 1.587238 | 1.760374 | 0 | 0.293032 | -0.109810 |
| other | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.151760 | 0.001760 | 1.036953 | 1.610963 | 1.775559 | 0 | 0.297237 | -0.096434 |

## Scheme `relation_info_v1`
- rows analyzed: 64740; rows at candidate max length: 15548
- source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 3005, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 1, 'qwen_pair_packed': 12236, 'childes': 16093, 'gutenberg': 11621, 'open_subtitles': 11434, 'simple_wiki': 6619, 'bnc_spoken': 3599, 'switchboard': 132}

| split | fixed token rate | static token rate | Δ token | fixed group rate | static group rate | Δ group | selected prior lift | relation lift | high1.50 lift | clip hits | max p | len-weight corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.149977 | -0.000023 | 1.021049 | 1.423945 | 1.545469 | 0 | 0.283834 | 0.003479 |
| changed | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.150305 | 0.000305 | 1.023316 | 1.400186 | 1.528585 | 0 | 0.278786 | -0.015308 |
| other | 0.150000 | 0.150000 | 0.000000 | 0.150000 | 0.149962 | -0.000038 | 1.020947 | 1.425260 | 1.546353 | 0 | 0.283834 | 0.004599 |

## Input files

- trainer: `experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_static_prior.py`
- pool_10m: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- prior_json: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
- limit_rows: `0`
- seq_length: `256`
- mask_prob: `0.15`
- schemes: `['relation_only_v1', 'relation_info_v1']`

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_expected_budget_full/static_prior_expected_budget.json`
