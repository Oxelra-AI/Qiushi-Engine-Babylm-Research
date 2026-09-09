# earlier analysis balanced Entity-stratum readout

This readout recomputes official Entity prediction files by split and numops, then contrasts MAX arms under both the official 1/6 zero-op + 5/6 nonzero-op weighting and a neutral 50/50 zero-vs-nonzero weighting.

## Late checkpoint balanced contrasts

| contrast | checkpoint | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | nonzero-zero spread pp |
|---|---:|---:|---:|---:|---:|---:|
| deberta_basin1 | chck_100M | +3.891 | -9.569 | +6.583 | -1.493 | +16.152 |
| deberta_basin1 | chck_80M | +4.501 | -9.000 | +7.201 | -0.899 | +16.201 |
| deberta_basin1 | chck_90M | +3.661 | -10.414 | +6.476 | -1.969 | +16.890 |
| deberta_basin2 | chck_100M | +2.200 | -8.596 | +4.360 | -2.118 | +12.956 |
| deberta_basin2 | chck_80M | +2.695 | -10.090 | +5.252 | -2.419 | +15.342 |
| deberta_basin2 | chck_90M | +2.722 | -8.660 | +4.999 | -1.830 | +13.659 |
| deberta_breadth_minus_repeat | chck_100M | +1.090 | -13.271 | +3.963 | -4.654 | +17.234 |
| deberta_breadth_minus_repeat | chck_80M | +1.400 | -12.177 | +4.115 | -4.031 | +16.292 |
| deberta_breadth_minus_repeat | chck_90M | +0.580 | -13.663 | +3.429 | -5.117 | +17.092 |
| deberta_view_minus_breadth | chck_100M | +2.800 | +3.702 | +2.620 | +3.161 | -1.082 |
| deberta_view_minus_breadth | chck_80M | +3.101 | +3.177 | +3.086 | +3.132 | -0.091 |
| deberta_view_minus_breadth | chck_90M | +3.081 | +3.249 | +3.047 | +3.148 | -0.202 |
| roberta_max | chck_100M | +0.277 | -0.006 | +0.333 | +0.164 | +0.339 |
| roberta_max | chck_80M | +0.513 | +0.326 | +0.550 | +0.438 | +0.224 |
| roberta_max | chck_90M | -0.014 | -0.459 | +0.076 | -0.192 | +0.535 |

## Late-window means

| contrast | quantity | n | mean | min | max |
|---|---|---:|---:|---:|---:|
| deberta_basin1 | official_all18_delta_pp | 3 | +4.018 | +3.661 | +4.501 |
| deberta_basin1 | zero_ops_delta_pp | 3 | -9.661 | -10.414 | -9.000 |
| deberta_basin1 | nonzero_ops_delta_pp | 3 | +6.753 | +6.476 | +7.201 |
| deberta_basin1 | balanced_zero_nonzero_delta_pp | 3 | -1.454 | -1.969 | -0.899 |
| deberta_basin1 | nonzero_minus_zero_spread_pp | 3 | +16.414 | +16.152 | +16.890 |
| deberta_basin2 | official_all18_delta_pp | 3 | +2.539 | +2.200 | +2.722 |
| deberta_basin2 | zero_ops_delta_pp | 3 | -9.115 | -10.090 | -8.596 |
| deberta_basin2 | nonzero_ops_delta_pp | 3 | +4.870 | +4.360 | +5.252 |
| deberta_basin2 | balanced_zero_nonzero_delta_pp | 3 | -2.123 | -2.419 | -1.830 |
| deberta_basin2 | nonzero_minus_zero_spread_pp | 3 | +13.985 | +12.956 | +15.342 |
| deberta_breadth_minus_repeat | official_all18_delta_pp | 3 | +1.023 | +0.580 | +1.400 |
| deberta_breadth_minus_repeat | zero_ops_delta_pp | 3 | -13.037 | -13.663 | -12.177 |
| deberta_breadth_minus_repeat | nonzero_ops_delta_pp | 3 | +3.836 | +3.429 | +4.115 |
| deberta_breadth_minus_repeat | balanced_zero_nonzero_delta_pp | 3 | -4.601 | -5.117 | -4.031 |
| deberta_breadth_minus_repeat | nonzero_minus_zero_spread_pp | 3 | +16.873 | +16.292 | +17.234 |
| deberta_view_minus_breadth | official_all18_delta_pp | 3 | +2.994 | +2.800 | +3.101 |
| deberta_view_minus_breadth | zero_ops_delta_pp | 3 | +3.376 | +3.177 | +3.702 |
| deberta_view_minus_breadth | nonzero_ops_delta_pp | 3 | +2.918 | +2.620 | +3.086 |
| deberta_view_minus_breadth | balanced_zero_nonzero_delta_pp | 3 | +3.147 | +3.132 | +3.161 |
| deberta_view_minus_breadth | nonzero_minus_zero_spread_pp | 3 | -0.459 | -1.082 | -0.091 |
| roberta_max | official_all18_delta_pp | 3 | +0.259 | -0.014 | +0.513 |
| roberta_max | zero_ops_delta_pp | 3 | -0.046 | -0.459 | +0.326 |
| roberta_max | nonzero_ops_delta_pp | 3 | +0.320 | +0.076 | +0.550 |
| roberta_max | balanced_zero_nonzero_delta_pp | 3 | +0.137 | -0.192 | +0.438 |
| roberta_max | nonzero_minus_zero_spread_pp | 3 | +0.366 | +0.224 | +0.535 |

## Interpretation

- **deberta_macro_entity**: In both DeBERTa basins, a positive official all-18-subtask Entity V-R combined with negative 50/50 zero/nonzero V-R means the aggregate carrier is created by the evaluation mixture's 5:1 nonzero-operation weighting, not by a uniformly improved state record skill.
- **roberta_transfer**: RoBERTa rows are included if late view and repeat predictions exist. Their balanced rows should be read before treating any aggregate Entity difference as architectural transfer.
- **breadth_specificity**: Breadth rows are included if delivered; view-minus-breadth and breadth-minus-repeat are decomposed the same way so same-population sentence breadth is not judged by aggregate Entity alone.
- **margin_next**: Official prediction JSONs contain only the selected string, not option likelihoods. Threshold-free margin separation therefore requires a direct model scoring pass or the pending binding content trade predeclared predictions sampled margin task, not these prediction files alone.

## Files

- subtask_scores_csv: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_subtask_scores.csv`
- aggregate_scores_csv: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_aggregate_scores.csv`
- contrast_rows_csv: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_contrast_rows.csv`
- balanced_rows_csv: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_rows.csv`
- late_summary_csv: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_late_summary_rows.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_readout_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_readout_summary.md`
