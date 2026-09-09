# bidir ranking contextual training and eval plan bidirectional trajectory ranking

Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading; AoA/CDI words, child curves, AoA predictions, AoA outputs, and SuperGLUE are not used.

Complete rows: 52 / 52
Best bidir: `bidir_seed43022` `chck_75M` equal7=43.170000
Best clean reference: `clean_qwen_seed43022` `chck_100M` equal7=43.112857

## Top rows
1. `bidir_pair_order` `bidir_seed43022` `chck_75M` equal7=43.170000 BLiMP=66.56 Supplement=61.86 EWoK=51.53 Entity=24.97 COMPS=52.1 GlobalPIQA=36.605000000000004 Reading=8.565
2. `bidir_pair_order` `bidir_seed43022` `chck_80M` equal7=43.135714 BLiMP=66.41 Supplement=62.92 EWoK=50.85 Entity=24.51 COMPS=52.31 GlobalPIQA=36.62 Reading=8.33
3. `clean_qwen` `clean_qwen_seed43022` `chck_100M` equal7=43.112857 BLiMP=66.84 Supplement=62.84 EWoK=50.19 Entity=25.76 COMPS=51.78 GlobalPIQA=36.62 Reading=7.76
4. `clean_qwen` `clean_qwen_seed43122` `chck_60M` equal7=43.099286 BLiMP=64.58 Supplement=62.85 EWoK=50.34 Entity=25.99 COMPS=51.2 GlobalPIQA=39.58 Reading=7.154999999999999
5. `bidir_pair_order` `bidir_seed43022` `chck_90M` equal7=43.097143 BLiMP=66.71 Supplement=62.9 EWoK=51.38 Entity=24.75 COMPS=52.06 GlobalPIQA=35.605000000000004 Reading=8.274999999999999
6. `clean_qwen` `clean_qwen_seed43022` `chck_95M` equal7=43.090000 BLiMP=66.79 Supplement=62.54 EWoK=50.16 Entity=25.9 COMPS=51.81 GlobalPIQA=36.62 Reading=7.8100000000000005
7. `bidir_pair_order` `bidir_seed43122` `chck_90M` equal7=43.049286 BLiMP=66.58 Supplement=61.28 EWoK=52.35 Entity=24.29 COMPS=52.17 GlobalPIQA=36.635 Reading=8.04
8. `bidir_pair_order` `bidir_seed43022` `chck_70M` equal7=43.041429 BLiMP=66.13 Supplement=62.78 EWoK=49.65 Entity=24.45 COMPS=51.67 GlobalPIQA=38.605000000000004 Reading=8.004999999999999
9. `bidir_pair_order` `bidir_seed43022` `chck_85M` equal7=43.012857 BLiMP=66.93 Supplement=62.45 EWoK=50.66 Entity=24.85 COMPS=52.24 GlobalPIQA=35.635 Reading=8.325
10. `bidir_pair_order` `bidir_seed43022` `chck_95M` equal7=42.987857 BLiMP=66.97 Supplement=61.75 EWoK=51.12 Entity=24.9 COMPS=52.25 GlobalPIQA=35.62 Reading=8.305
11. `bidir_pair_order` `bidir_seed43122` `chck_95M` equal7=42.949286 BLiMP=66.68 Supplement=61.32 EWoK=51.83 Entity=24.09 COMPS=52.21 GlobalPIQA=36.635 Reading=7.88
12. `bidir_pair_order` `bidir_seed43122` `chck_100M` equal7=42.902143 BLiMP=66.74 Supplement=60.84 EWoK=52.17 Entity=23.86 COMPS=52.22 GlobalPIQA=36.635 Reading=7.85

## Paired seed means
- `bidir_pair_order` `chck_90M` mean_equal7=43.073214 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 43.097142857142856}, {'target': 'bidir_seed43122', 'equal7_full_eval': 43.049285714285716}]
- `bidir_pair_order` `chck_95M` mean_equal7=42.968571 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 42.98785714285714}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.949285714285715}]
- `clean_qwen` `chck_60M` mean_equal7=42.927857 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.75642857142857}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 43.09928571428571}]
- `bidir_pair_order` `chck_70M` mean_equal7=42.909286 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 43.041428571428575}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.777142857142856}]
- `bidir_pair_order` `chck_85M` mean_equal7=42.901786 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 43.01285714285715}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.79071428571428}]
- `bidir_pair_order` `chck_80M` mean_equal7=42.896429 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 43.135714285714286}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.65714285714286}]
- `bidir_pair_order` `chck_100M` mean_equal7=42.890357 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 42.878571428571426}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.902142857142856}]
- `clean_qwen` `chck_95M` mean_equal7=42.860714 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 43.089999999999996}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.63142857142857}]
- `bidir_pair_order` `chck_75M` mean_equal7=42.787143 n=2 targets=[{'target': 'bidir_seed43022', 'equal7_full_eval': 43.17}, {'target': 'bidir_seed43122', 'equal7_full_eval': 42.40428571428571}]
- `clean_qwen` `chck_100M` mean_equal7=42.778214 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 43.112857142857145}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.44357142857143}]
- `clean_qwen` `chck_70M` mean_equal7=42.713929 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.55142857142857}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.87642857142857}]
- `clean_qwen` `chck_75M` mean_equal7=42.641429 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.62428571428571}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.65857142857143}]
