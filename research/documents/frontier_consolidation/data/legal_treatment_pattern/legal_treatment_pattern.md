# route portfolio and intervention assets legal-tokenizer treatment pattern

Quantify the mature 70M/80M legal-tokenizer treatment pattern so the next intervention matches the observed mechanism rather than the easiest available implementation.

Comparison file: `experiments/archive/frontier_consolidation/data/legal_treatment_trajectory/legal_treatment_trajectory_comparison.json` exists=True
Comparison missing entries: `['clean_20M', 'reinvest_70M', 'clean_70M', 'reinvest_80M', 'clean_80M']`

## Mature exposure signatures

| exposure | complete | mean7 Δ | relation Δ | broad Δ | relation-broad Δ | positive cols |
|---:|:---:|---:|---:|---:|---:|---:|
| 70 | False | NA | NA | NA | NA | None |
| 80 | False | NA | NA | NA | NA | None |

## Pattern summary

- ready: `False`
- reason: `70M and 80M complete reinvest-clean pairs are both required before interpreting the mature legal-tokenizer treatment pattern.`
- missing_exposures: `[70, 80]`

Full JSON: `experiments/archive/frontier_consolidation/data/legal_treatment_pattern/legal_treatment_pattern.json`
