# earlier analysis common-window budget decomposition

Stable selected families only. Scores are read through the identity `V-C = (V-R) + (R-C)` and growth decomposition `Δ(V-R)=ΔV-ΔR`.

## Coverage
- common_10M_80M_checkpoints: ['chck_10M', 'chck_20M', 'chck_30M', 'chck_40M', 'chck_50M', 'chck_60M', 'chck_70M', 'chck_80M']
- missing_common_10M_80M: {}
- full_10M_100M_all_available_checkpoints: ['chck_10M', 'chck_20M', 'chck_30M', 'chck_40M', 'chck_50M', 'chck_60M', 'chck_70M', 'chck_80M']
- missing_full_10M_100M: {'max_repeat': ['chck_90M', 'chck_100M']}
- full_MAX_late_complete: False

## Primary common 10M-80M means
### cheap6_no_GlobalPIQA
| dose | V-C | R-C | V-R | view growth vs 1x | repeat growth vs 1x | duplicate deterioration | V-R growth vs 1x |
|---|---:|---:|---:|---:|---:|---:|---:|
| dose1 | 0.6499 | 0.7124 | -0.0625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dose1p82 | 0.6225 | 0.4811 | 0.1414 | -0.0274 | -0.2312 | 0.2312 | 0.2039 |
| dose2p64 | 0.9053 | 0.5085 | 0.3968 | 0.2554 | -0.2039 | 0.2039 | 0.4593 |

### cheap5_no_GlobalPIQA_Reading
| dose | V-C | R-C | V-R | view growth vs 1x | repeat growth vs 1x | duplicate deterioration | V-R growth vs 1x |
|---|---:|---:|---:|---:|---:|---:|---:|
| dose1 | 0.6437 | 0.8045 | -0.1608 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dose1p82 | 0.6602 | 0.5170 | 0.1432 | 0.0165 | -0.2875 | 0.2875 | 0.3040 |
| dose2p64 | 0.9837 | 0.4943 | 0.4895 | 0.3400 | -0.3102 | 0.3102 | 0.6502 |

## Reading
On the common 10M-80M window for cheap6_no_GlobalPIQA, V-R is -0.0625 at 1x, 0.1414 at 1.82x, and 0.3968 at 2.64x. But view growth relative to 1x is -0.0274/0.2554, while repeat growth is -0.2312/-0.2039; therefore the V-R increase decomposes into view component -0.0274/0.2554 and duplicate-counterfactual deterioration component 0.2312/0.2039. On the common 10M-80M window for cheap5_no_GlobalPIQA_Reading, V-R is -0.1608 at 1x, 0.1432 at 1.82x, and 0.4895 at 2.64x. But view growth relative to 1x is 0.0165/0.3400, while repeat growth is -0.2875/-0.3102; therefore the V-R increase decomposes into view component 0.0165/0.3400 and duplicate-counterfactual deterioration component 0.2875/0.3102. This changes the first-basin interpretation: the increasing V-R curve cannot be named as compact-view scaling alone. It is a budget-allocation identity in which higher-dose exact duplication loses value as it displaces independent experience, while the view arm avoids part of that loss and may add some structured re-expression value. The full 10M-100M MAX decomposition is still incomplete in the input rows because max_repeat 90M/100M are missing or partial; this is exactly the late-exposure boundary where MAX view growth turns negative, so the late CPU finisher must be read before finalizing the turnover interpretation.

Figure: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common10_80_budget_decomposition.png`

## Files
- point_csv: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common10_80_point_decomposition.csv`
- summary_csv: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common10_80_summary_by_dose_metric.csv`
- family_csv: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common10_80_family_components.csv`
- lfo_csv: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common10_80_leave_one_family_out.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/common_window_budget_decomposition/common_window_budget_decomposition_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/common_window_budget_decomposition/common_window_budget_decomposition_summary.md`
