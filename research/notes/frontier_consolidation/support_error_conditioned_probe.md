# support error conditioned probe — error-conditioned support-sharing probe

CPU-only. No model training and no new model evaluation. Existing legal40k 8x480 seed43022 official-compatible predictions are joined to gold data to ask whether low-support legal40k tokens in answer-discriminating spans are enriched in wrong examples.

## Family-level error conditioning

| family | items | score | Δ wrong-correct frac<50 | AUC wrong-high frac<50 | Δ wrong-correct frac<100 | AUC wrong-high frac<100 | Δ wrong-correct rescue<100/ge50 | AUC wrong-high opportunity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 67.77 | -0.0055 | 0.4965 | -0.0118 | 0.4929 | -0.0119 | 0.4938 |
| Supplement | 5218 | 73.32 | 0.0275 | 0.5338 | 0.0439 | 0.5538 | 0.0430 | 0.5536 |
| EWoK | 7618 | 51.13 | -0.0025 | 0.4959 | -0.0109 | 0.4915 | -0.0112 | 0.4907 |
| COMPS | 91028 | 52.30 | 0.0059 | 0.5073 | 0.0074 | 0.5080 | 0.0075 | 0.5079 |
| GlobalPIQA_parallel | 103 | 22.33 | -0.0202 | 0.4935 | -0.0146 | 0.4739 | -0.0181 | 0.4720 |
| GlobalPIQA_nonparallel | 100 | 47.00 | 0.0261 | 0.5283 | 0.0720 | 0.5709 | 0.0791 | 0.5674 |

## Scientific reading

- This probe conditions legal40k errors on the support-sharing quantity itself: low-support legal40k tokens in answer-discriminating spans whose legal16 components have much higher same-corpus support.
- GlobalPIQA_nonparallel, the column repeatedly moved by word-mean/minfreq50, shows score 47.00 with wrong-high AUC 0.5709 for disc frac<100 and 0.5674 for the continuous support-sharing opportunity.
- GlobalPIQA_parallel shows score 22.33 with wrong-high AUC 0.4739; this separates whether the support signal is only the nonparallel idiosyncrasy.
- EWoK shows score 51.13 with wrong-correct disc frac<100 delta -0.0109 and wrong-high AUC 0.4915; this is the load-bearing test for broad transfer.
- Supplement shows score 73.32 with wrong-correct disc frac<100 delta 0.0439; support-sharing must not damage the high Supplement behavior that minfreq50 lost.
- COMPS shows score 52.30 with wrong-high AUC 0.5080; its huge item count makes small effects reliable but not necessarily route-deciding.
- BLiMP shows score 67.77 with wrong-high AUC 0.4929; support-sharing should preserve BLiMP rather than repeat minfreq50's language-column trade.
- Treat positive AUC/delta as construction evidence, not as score proof. A GPU run would still need a single-variable, official-compatible trainer and a preliminary implementation check.

Full JSON: `experiments/archive/frontier_consolidation/data/support_error_conditioned_probe/support_error_conditioned_probe.json`
CSVs: `experiments/archive/frontier_consolidation/data/support_error_conditioned_probe/support_error_conditioned_by_family.csv`, `experiments/archive/frontier_consolidation/data/support_error_conditioned_probe/support_error_conditioned_items.csv`
