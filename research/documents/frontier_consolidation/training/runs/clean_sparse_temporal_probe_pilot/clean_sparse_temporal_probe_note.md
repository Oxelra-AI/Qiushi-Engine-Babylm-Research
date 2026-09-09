# clean full trajectory seed control clean-Qwen sparse temporal seed-control

Existing inherited clean-Qwen checkpoint ladders only; same selected slices as the reinvest sparse temporal run; not a full official evaluation and no training.

## Purpose
This measures whether the seed43022/seed43122 slice-time spread observed for compact_view_reinvest is already present in the inherited clean-Qwen recipe or is enlarged/changed by the compact-view reinvestment corpus.

## Seed43122 minus seed43022 by exposure
- 1M: ewok_all=0.091, supplement_all=-5.200

## Task-group summaries
- ewok_all: early_mean=0.09090909090908639, late_mean=None, final=0.09090909090908639, max_abs=0.09090909090908639
- supplement_all: early_mean=-5.200000000000003, late_mean=None, final=-5.200000000000003, max_abs=5.200000000000003

Machine-readable output: `experiments/archive/frontier_consolidation/training/runs/clean_sparse_temporal_probe_pilot/clean_sparse_temporal_probe_summary.json`
