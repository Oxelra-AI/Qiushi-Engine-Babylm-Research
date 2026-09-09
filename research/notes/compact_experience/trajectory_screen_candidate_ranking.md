# Trajectory-screen candidate ranking

Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading trajectory summaries; no AoA/CDI words, child curves, AoA predictions, AoA outputs, or SuperGLUE are used.

Complete rows: 52 / 52

## Top no-AoA rows
1. `clean_qwen_seed43022` `chck_100M` equal7=43.112857 BLiMP=66.84 Supplement=62.84 EWoK=50.19 Entity=25.76 COMPS=51.78 GlobalPIQA=36.62 Reading=7.76
2. `clean_qwen_seed43122` `chck_60M` equal7=43.099286 BLiMP=64.58 Supplement=62.85 EWoK=50.34 Entity=25.99 COMPS=51.2 GlobalPIQA=39.58 Reading=7.154999999999999
3. `clean_qwen_seed43022` `chck_95M` equal7=43.090000 BLiMP=66.79 Supplement=62.54 EWoK=50.16 Entity=25.9 COMPS=51.81 GlobalPIQA=36.62 Reading=7.8100000000000005
4. `clean_qwen_seed43122` `chck_70M` equal7=42.876429 BLiMP=65.47 Supplement=62.3 EWoK=51.22 Entity=24.41 COMPS=51.77 GlobalPIQA=37.605000000000004 Reading=7.36
5. `clean_qwen_seed43022` `chck_60M` equal7=42.756429 BLiMP=65.53 Supplement=60.4 EWoK=50.53 Entity=27.01 COMPS=51.83 GlobalPIQA=36.105000000000004 Reading=7.890000000000001
6. `clean_qwen_seed43022` `chck_90M` equal7=42.756429 BLiMP=66.75 Supplement=62.99 EWoK=49.88 Entity=25.66 COMPS=51.65 GlobalPIQA=34.635 Reading=7.73
7. `clean_qwen_seed43022` `chck_50M` equal7=42.714286 BLiMP=64.74 Supplement=63.32 EWoK=49.39 Entity=26.42 COMPS=51.64 GlobalPIQA=35.58 Reading=7.91
8. `clean_qwen_seed43122` `chck_80M` equal7=42.693571 BLiMP=65.56 Supplement=62.04 EWoK=50.13 Entity=25.37 COMPS=51.74 GlobalPIQA=37.105000000000004 Reading=6.91
9. `clean_qwen_seed43122` `chck_75M` equal7=42.658571 BLiMP=65.27 Supplement=63.01 EWoK=51.5 Entity=24.34 COMPS=51.84 GlobalPIQA=35.635 Reading=7.015000000000001
10. `clean_qwen_seed43122` `chck_95M` equal7=42.631429 BLiMP=66.0 Supplement=61.81 EWoK=50.74 Entity=25.0 COMPS=52.02 GlobalPIQA=35.62 Reading=7.2299999999999995
11. `clean_qwen_seed43022` `chck_75M` equal7=42.624286 BLiMP=66.61 Supplement=62.34 EWoK=49.5 Entity=25.7 COMPS=51.23 GlobalPIQA=35.165 Reading=7.824999999999999
12. `clean_qwen_seed43122` `chck_85M` equal7=42.562857 BLiMP=65.86 Supplement=61.63 EWoK=50.35 Entity=24.89 COMPS=51.86 GlobalPIQA=36.09 Reading=7.26
13. `clean_qwen_seed43022` `chck_85M` equal7=42.557857 BLiMP=66.83 Supplement=61.15 EWoK=50.02 Entity=26.24 COMPS=51.65 GlobalPIQA=34.165 Reading=7.85
14. `clean_qwen_seed43022` `chck_70M` equal7=42.551429 BLiMP=65.41 Supplement=64.15 EWoK=50.34 Entity=26.12 COMPS=51.1 GlobalPIQA=33.165 Reading=7.575
15. `devcurr_seed43122` `chck_80M` equal7=42.537143 BLiMP=65.32 Supplement=59.29 EWoK=49.65 Entity=26.3 COMPS=51.78 GlobalPIQA=37.65 Reading=7.77
16. `devcurr_seed43122` `chck_85M` equal7=42.484286 BLiMP=65.91 Supplement=59.69 EWoK=49.4 Entity=24.63 COMPS=52.18 GlobalPIQA=37.62 Reading=7.960000000000001
17. `clean_qwen_seed43122` `chck_100M` equal7=42.443571 BLiMP=66.02 Supplement=61.51 EWoK=50.43 Entity=25.26 COMPS=52.06 GlobalPIQA=34.62 Reading=7.205
18. `clean_qwen_seed43122` `chck_90M` equal7=42.423571 BLiMP=65.83 Supplement=61.36 EWoK=50.5 Entity=25.65 COMPS=52.15 GlobalPIQA=34.165 Reading=7.31
19. `devcurr_seed43022` `chck_85M` equal7=42.338571 BLiMP=65.97 Supplement=62.8 EWoK=49.61 Entity=24.89 COMPS=51.68 GlobalPIQA=33.74 Reading=7.68
20. `devcurr_seed43022` `chck_95M` equal7=42.322143 BLiMP=66.02 Supplement=62.85 EWoK=50.28 Entity=23.81 COMPS=51.43 GlobalPIQA=34.225 Reading=7.64

## Best by target
- `clean_qwen_seed43022` best `chck_100M` equal7=43.112857
- `clean_qwen_seed43122` best `chck_60M` equal7=43.099286
- `devcurr_seed43022` best `chck_85M` equal7=42.338571
- `devcurr_seed43122` best `chck_80M` equal7=42.537143

## Paired seed means
- `clean_qwen` `chck_60M` mean_equal7=42.927857 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.75642857142857}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 43.09928571428571}]
- `clean_qwen` `chck_95M` mean_equal7=42.860714 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 43.089999999999996}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.63142857142857}]
- `clean_qwen` `chck_100M` mean_equal7=42.778214 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 43.112857142857145}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.44357142857143}]
- `clean_qwen` `chck_70M` mean_equal7=42.713929 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.55142857142857}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.87642857142857}]
- `clean_qwen` `chck_75M` mean_equal7=42.641429 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.62428571428571}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.65857142857143}]
- `clean_qwen` `chck_90M` mean_equal7=42.590000 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.75642857142857}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.42357142857143}]
- `clean_qwen` `chck_85M` mean_equal7=42.560357 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.55785714285714}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.56285714285714}]
- `clean_qwen` `chck_80M` mean_equal7=42.448571 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.20357142857143}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 42.69357142857143}]
- `devcurr_firstpass` `chck_85M` mean_equal7=42.411429 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.33857142857143}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.48428571428571}]
- `devcurr_firstpass` `chck_100M` mean_equal7=42.266071 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.29928571428571}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.23285714285714}]
- `devcurr_firstpass` `chck_95M` mean_equal7=42.266071 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.32214285714286}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.209999999999994}]
- `devcurr_firstpass` `chck_80M` mean_equal7=42.227500 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 41.917857142857144}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.537142857142854}]
- `devcurr_firstpass` `chck_75M` mean_equal7=42.171429 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.135}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.207857142857144}]
- `devcurr_firstpass` `chck_90M` mean_equal7=42.165000 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.090714285714284}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.239285714285714}]
- `devcurr_firstpass` `chck_60M` mean_equal7=42.103929 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.02571428571429}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 42.18214285714286}]
- `clean_qwen` `chck_50M` mean_equal7=42.031786 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.714285714285715}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 41.34928571428571}]
- `devcurr_firstpass` `chck_70M` mean_equal7=41.964643 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 42.214999999999996}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 41.714285714285715}]
- `clean_qwen` `chck_40M` mean_equal7=41.722500 n=2 targets=[{'target': 'clean_qwen_seed43022', 'equal7_full_eval': 42.21071428571429}, {'target': 'clean_qwen_seed43122', 'equal7_full_eval': 41.23428571428571}]
- `devcurr_firstpass` `chck_50M` mean_equal7=41.473214 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 41.34928571428571}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 41.597142857142856}]
- `devcurr_firstpass` `chck_40M` mean_equal7=41.376071 n=2 targets=[{'target': 'devcurr_seed43022', 'equal7_full_eval': 41.52357142857142}, {'target': 'devcurr_seed43122', 'equal7_full_eval': 41.228571428571435}]
