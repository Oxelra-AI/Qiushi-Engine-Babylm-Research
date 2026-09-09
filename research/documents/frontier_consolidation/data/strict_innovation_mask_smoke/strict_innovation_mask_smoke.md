# earlier analysis — strict content-innovation mask smoke

CPU-only; no model update, no GPU, no official evaluation.

- changed rows sampled: `768`; unchanged rows sampled: `2048`
- p_strict: `0.5`; p_copy_effective (tokens-matched): `0.10240166`; ordinary p: `0.15`
- selected-token mass ratio vs standard WWM: `1.000166`
- all checks passed: `True`

## Empirical group rates
- strict_group_rate: `0.4984991423670669`
- copy_group_rate: `0.10016390999850991`
- source_group_rate: `0.14844919465900244`
- rewrite_other_group_rate: `0.14555702917771884`
- ordinary_group_rate: `0.148289680835838`
- filler_group_rate: `0.1`

## Empirical token rates
- strict_token_rate: `0.49841226011321277`
- copy_token_rate: `0.09988016662545413`
- source_token_rate: `0.14781775207223694`
- rewrite_other_token_rate: `0.14904270986745213`
- ordinary_token_rate: `0.1478130346702495`
- filler_token_rate: `0.07692307692307693`

## Checks
- strict_rate_near_p_strict: `True`
- copy_rate_near_p_copy: `True`
- source_group_rate_near_baseline: `True`
- rewrite_other_group_rate_near_baseline: `True`
- ordinary_group_rate_near_baseline: `True`
- mass_ratio_close_to_standard: `True`

Full JSON: `experiments/archive/frontier_consolidation/data/strict_innovation_mask_smoke/strict_innovation_mask_smoke.json`
