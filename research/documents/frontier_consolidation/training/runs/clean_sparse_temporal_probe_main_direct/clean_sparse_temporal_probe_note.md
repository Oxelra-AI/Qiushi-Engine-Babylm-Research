# clean full trajectory seed control clean-Qwen sparse temporal seed-control

Existing inherited clean-Qwen checkpoint ladders only; same selected slices as the reinvest sparse temporal run; not a full official evaluation and no training.

## Purpose
This measures whether the seed43022/seed43122 slice-time spread observed for compact_view_reinvest is already present in the inherited clean-Qwen recipe or is enlarged/changed by the compact-view reinvestment corpus.

## Seed43122 minus seed43022 by exposure
- 1M: blimp_control=8.750, blimp_worst=-9.400, entity_full=0.485, ewok_all=0.091, supplement_all=-5.200
- 10M: blimp_control=-3.500, blimp_worst=-7.700, entity_full=1.172, ewok_all=3.364, supplement_all=-0.800
- 40M: blimp_control=3.188, blimp_worst=0.400, entity_full=-4.401, ewok_all=1.364, supplement_all=-4.800
- 100M: blimp_control=2.938, blimp_worst=-4.650, entity_full=-0.502, ewok_all=1.455, supplement_all=0.800

## Task-group summaries
- blimp_control: early_mean=2.625, late_mean=3.0625, final=2.9375, max_abs=8.75
- blimp_worst: early_mean=-8.55, late_mean=-2.125, final=-4.649999999999999, max_abs=9.399999999999999
- entity_full: early_mean=0.8283104602959197, late_mean=-2.451563591736866, final=-0.5022045499021388, max_abs=4.400922633571593
- ewok_all: early_mean=1.7272727272727266, late_mean=1.4090909090909065, final=1.4545454545454533, max_abs=3.363636363636367
- supplement_all: early_mean=-3.0000000000000036, late_mean=-2.0, final=0.7999999999999972, max_abs=5.200000000000003

Machine-readable output: `experiments/archive/frontier_consolidation/training/runs/clean_sparse_temporal_probe_main_direct/clean_sparse_temporal_probe_summary.json`
