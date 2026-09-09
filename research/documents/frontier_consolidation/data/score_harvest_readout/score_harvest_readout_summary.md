# earlier analysis score harvest and dose readout

This is a file-only readout from already completed official-compatible stable-family evaluations. Missing rows remain missing; no score is imputed.

## Coverage
- clean0: 8/8 rows; missing: none
- dose1_view: 10/10 rows; missing: none
- dose1_repeat: 10/10 rows; missing: none
- dose1p82_view: 10/10 rows; missing: none
- dose1p82_repeat: 10/10 rows; missing: none
- max_view: 10/10 rows; missing: none
- max_repeat: 8/10 rows; missing: chck_90M, chck_100M

## Seed-spread scale from earlier analysis
- cheap6_no_GlobalPIQA: mean |seed spread| 0.4199, median 0.3367, max 0.8600
- cheap5_no_GlobalPIQA_Reading: mean |seed spread| 0.5350, median 0.5550, max 0.9060
- EWoK_plus_Entity_sum: mean |seed spread| 2.5880, median 2.8600, max 4.4400

## Semantic leg V-R (view minus repeat)
### dose1 (10 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M, chck_90M, chck_100M)
- cheap6_no_GlobalPIQA: mean -0.0142, median 0.1358, min -0.7417, max 0.4158, @80M 0.4158333333333246, @100M 0.1283333333333374
- cheap5_no_GlobalPIQA_Reading: mean -0.1038, median 0.0420, min -0.9880, max 0.4200, @80M 0.4199999999999946, @100M 0.06800000000000495
- EWoK_plus_Entity_sum: mean -0.1090, median 0.4100, min -4.9700, max 1.9100, @80M 1.9099999999999966, @100M 0.769999999999996

### dose1p82 (10 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M, chck_90M, chck_100M)
- cheap6_no_GlobalPIQA: mean 0.2190, median 0.4537, min -0.5150, max 0.7033, @80M 0.4241666666666717, @100M 0.5758333333333354
- cheap5_no_GlobalPIQA_Reading: mean 0.2326, median 0.4890, min -0.6980, max 0.8220, @80M 0.44200000000000017, @100M 0.6440000000000055
- EWoK_plus_Entity_sum: mean -0.1580, median -0.1550, min -2.6500, max 1.6100, @80M -0.38999999999998636, @100M -0.22999999999998977

### dose2p64 (8 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M)
- cheap6_no_GlobalPIQA: mean 0.3968, median 0.4233, min -0.6142, max 1.4350, @80M 1.0225000000000009, @100M None
- cheap5_no_GlobalPIQA_Reading: mean 0.4895, median 0.5310, min -0.6020, max 1.6760, @80M 1.3019999999999996, @100M None
- EWoK_plus_Entity_sum: mean 1.4563, median 0.9250, min -1.7700, max 7.3400, @80M 3.5700000000000074, @100M None

## Clean-free view growth V_d - V_1
### dose1p82 (10 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M, chck_90M, chck_100M)
- cheap6_no_GlobalPIQA: mean -0.1003, median -0.1250, @80M -0.4308333333333252, @100M -0.26583333333333314
- cheap5_no_GlobalPIQA_Reading: mean -0.0768, median -0.0660, @80M -0.43599999999999994, @100M -0.3019999999999996
- EWoK_plus_Entity_sum: mean 0.0140, median -0.2650, @80M -0.44999999999998863, @100M -0.3199999999999932

### dose2p64 (10 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M, chck_90M, chck_100M)
- cheap6_no_GlobalPIQA: mean 0.1438, median -0.0479, @80M -0.33833333333333115, @100M -0.29666666666666686
- cheap5_no_GlobalPIQA_Reading: mean 0.2168, median 0.0240, @80M -0.2779999999999987, @100M -0.26600000000000534
- EWoK_plus_Entity_sum: mean 1.1160, median 1.2050, @80M -0.28999999999999204, @100M 0.45000000000000284

## Total leg V-C (view minus clean, 10M-80M when clean exists)
### dose1 (8 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M)
- cheap6_no_GlobalPIQA: mean 0.6499, median 0.7254, @80M 1.5749999999999957
- cheap5_no_GlobalPIQA_Reading: mean 0.6437, median 0.7080, @80M 1.7199999999999989
- EWoK_plus_Entity_sum: mean 1.5062, median 1.1400, @80M 4.359999999999999

### dose1p82 (8 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M)
- cheap6_no_GlobalPIQA: mean 0.6225, median 0.6100, @80M 1.1441666666666706
- cheap5_no_GlobalPIQA_Reading: mean 0.6602, median 0.6300, @80M 1.283999999999999
- EWoK_plus_Entity_sum: mean 1.6413, median 1.5900, @80M 3.910000000000011

### dose2p64 (8 checkpoints: chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M)
- cheap6_no_GlobalPIQA: mean 0.9053, median 1.0375, @80M 1.2366666666666646
- cheap5_no_GlobalPIQA_Reading: mean 0.9837, median 1.2090, @80M 1.4420000000000002
- EWoK_plus_Entity_sum: mean 2.7625, median 3.4300, @80M 4.070000000000007

## RoBERTa MAX view-clean available contrast
### roberta_total_transfer (2 checkpoints: chck_10M, chck_20M)
- cheap6_no_GlobalPIQA: mean -0.0454, median -0.0454, @80M None, @100M None
- cheap5_no_GlobalPIQA_Reading: mean -0.1490, median -0.1490, @80M None, @100M None
- EWoK_plus_Entity_sum: mean -1.3900, median -1.3900, @80M None, @100M None

## Scientific reading so far
V-R dose1: cheap6 mean -0.0142, cheap5 mean -0.1038 over 10 rows. V-R dose1p82: cheap6 mean 0.2190, cheap5 mean 0.2326 over 10 rows. V-R dose2p64: cheap6 mean 0.3968, cheap5 mean 0.4895 over 8 rows. V_d-V_1 dose1p82: cheap6 mean -0.1003, cheap5 mean -0.0768 over 10 rows. V_d-V_1 dose2p64: cheap6 mean 0.1438, cheap5 mean 0.2168 over 10 rows. V-C dose1: cheap6 mean 0.6499, cheap5 mean 0.6437 over 8 rows. V-C dose1p82: cheap6 mean 0.6225, cheap5 mean 0.6602 over 8 rows. V-C dose2p64: cheap6 mean 0.9053, cheap5 mean 0.9837 over 8 rows. EWoK+Entity remains secondary because earlier analysis measured much larger seed spread there than on cheap6/cheap5. Any incomplete MAX or RoBERTa rows must be reread when their evaluators finish; the second-basin MAX pair launched in earlier analysis is required before a mechanism-level synthesis.

## Files
- deberta_rows: `experiments/archive/frontier_consolidation/data/score_harvest_readout/harvested_deberta_stable_rows.csv`
- roberta_rows: `experiments/archive/frontier_consolidation/data/score_harvest_readout/harvested_roberta_stable_rows.csv`
- semantic_VR: `experiments/archive/frontier_consolidation/data/score_harvest_readout/semantic_VR.csv`
- cleanfree_growth: `experiments/archive/frontier_consolidation/data/score_harvest_readout/cleanfree_growth_Vd_minus_V1.csv`
- total_VC: `experiments/archive/frontier_consolidation/data/score_harvest_readout/total_VC.csv`
- repeat_clean_RC: `experiments/archive/frontier_consolidation/data/score_harvest_readout/repeat_clean_RC.csv`
- roberta_contrast: `experiments/archive/frontier_consolidation/data/score_harvest_readout/roberta_max_view_clean_contrast.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/score_harvest_readout/score_harvest_readout_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/score_harvest_readout/score_harvest_readout_summary.md`
