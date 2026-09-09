# official geometry noaoa interpretation official geometry no-AoA interpretation

BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading only; SuperGLUE and AoA not read or used.

Clean complete Overall-9 reference: 41.34429066479573
Visible leader Overall reference: 41.8

## Best profiles
- official160_b256_best: {"BLiMP": 67.92, "COMPS": 52.02, "EWoK": 51.59, "Entity": 23.95, "GlobalPIQA": 38.09, "Reading": 7.325, "Supplement": 60.96, "available": true, "checkpoint": "chck_95M", "equal7": 43.12214285714286, "recovery_R": 44.713750000000005, "required_superglue_plus_aoa_to_match_clean_overall9": 70.24361598316153, "required_superglue_plus_aoa_to_reach_41p8": 74.34499999999997, "target": "official160_b256"}
- official_cap120geom_b256_best: {"BLiMP": 69.1, "COMPS": 52.29, "EWoK": 50.95, "Entity": 24.15, "GlobalPIQA": 37.09, "Reading": 8.19, "Supplement": 59.18, "available": true, "checkpoint": "chck_80M", "equal7": 42.99285714285714, "recovery_R": 45.1325, "required_superglue_plus_aoa_to_match_clean_overall9": 71.14861598316156, "required_superglue_plus_aoa_to_reach_41p8": 75.25, "target": "official_cap120geom_b256"}
- clean_qwen_seed43022_best: {"BLiMP": 66.84, "COMPS": 51.78, "EWoK": 50.19, "Entity": 25.76, "GlobalPIQA": 36.62, "Reading": 7.76, "Supplement": 62.84, "available": true, "checkpoint": "chck_100M", "equal7": 43.112857142857145, "recovery_R": 44.1425, "required_superglue_plus_aoa_to_match_clean_overall9": 70.30861598316153, "required_superglue_plus_aoa_to_reach_41p8": 74.40999999999997, "target": "clean_qwen_seed43022"}

## Contrasts
- {"a": {"checkpoint": "chck_80M", "target": "official_cap120geom_b256"}, "available": true, "b": {"checkpoint": "chck_95M", "target": "official160_b256"}, "contrast": "cap120geom_b256_minus_official160_b256", "delta_BLiMP": 1.1799999999999926, "delta_COMPS": 0.269999999999996, "delta_EWoK": -0.6400000000000006, "delta_Entity": 0.1999999999999993, "delta_GlobalPIQA": -1.0, "delta_Reading": 0.8649999999999993, "delta_Supplement": -1.7800000000000011, "delta_equal7_full_eval": -0.12928571428572155, "delta_recovery_R": 0.41874999999999574, "required_superglue_plus_aoa_to_match_clean_overall9": 71.14861598316156, "required_superglue_plus_aoa_to_reach_41p8": 75.25}
- {"a": {"checkpoint": "chck_80M", "target": "official_cap120geom_b256"}, "available": true, "b": {"checkpoint": "chck_100M", "target": "clean_qwen_seed43022"}, "contrast": "cap120geom_b256_minus_clean_qwen_seed43022", "delta_BLiMP": 2.259999999999991, "delta_COMPS": 0.509999999999998, "delta_EWoK": 0.7600000000000051, "delta_Entity": -1.610000000000003, "delta_GlobalPIQA": 0.47000000000000597, "delta_Reading": 0.4299999999999997, "delta_Supplement": -3.6600000000000037, "delta_equal7_full_eval": -0.12000000000000455, "delta_recovery_R": 0.990000000000002, "required_superglue_plus_aoa_to_match_clean_overall9": 71.14861598316156, "required_superglue_plus_aoa_to_reach_41p8": 75.25}
- {"a": {"checkpoint": "chck_95M", "target": "official160_b256"}, "available": true, "b": {"checkpoint": "chck_100M", "target": "clean_qwen_seed43022"}, "contrast": "official160_b256_minus_clean_qwen_seed43022", "delta_BLiMP": 1.0799999999999983, "delta_COMPS": 0.240000000000002, "delta_EWoK": 1.4000000000000057, "delta_Entity": -1.8100000000000023, "delta_GlobalPIQA": 1.470000000000006, "delta_Reading": -0.4349999999999996, "delta_Supplement": -1.8800000000000026, "delta_equal7_full_eval": 0.009285714285717006, "delta_recovery_R": 0.5712500000000063, "required_superglue_plus_aoa_to_match_clean_overall9": 70.24361598316153, "required_superglue_plus_aoa_to_reach_41p8": 74.34499999999997}

## Top official-geometry rows
- `official160_b256` `chck_95M` equal7=43.12214285714286 BLiMP=67.92 Supplement=60.96 EWoK=51.59 Entity=23.95 COMPS=52.02 GlobalPIQA=38.09 Reading=7.325
- `official_cap120geom_b256` `chck_80M` equal7=42.99285714285714 BLiMP=69.1 Supplement=59.18 EWoK=50.95 Entity=24.15 COMPS=52.29 GlobalPIQA=37.09 Reading=8.19
- `official_cap120geom_b256` `chck_95M` equal7=42.887857142857136 BLiMP=69.27 Supplement=59.97 EWoK=50.45 Entity=24.42 COMPS=52.34 GlobalPIQA=35.62 Reading=8.145
- `official160_b256` `chck_80M` equal7=42.846428571428575 BLiMP=67.63 Supplement=60.48 EWoK=50.49 Entity=23.47 COMPS=51.95 GlobalPIQA=38.55 Reading=7.3549999999999995
- `official_cap120geom_b256` `chck_90M` equal7=42.83428571428572 BLiMP=69.15 Supplement=59.78 EWoK=50.27 Entity=24.48 COMPS=52.34 GlobalPIQA=35.605000000000004 Reading=8.215
- `official160_b256` `chck_100M` equal7=42.817142857142855 BLiMP=67.94 Supplement=60.92 EWoK=51.29 Entity=24.08 COMPS=52.03 GlobalPIQA=36.12 Reading=7.34
- `official_cap120geom_b256` `chck_100M` equal7=42.81214285714286 BLiMP=69.31 Supplement=59.89 EWoK=50.48 Entity=24.37 COMPS=52.34 GlobalPIQA=35.135 Reading=8.16
- `official160_b256` `chck_90M` equal7=42.779999999999994 BLiMP=67.79 Supplement=61.36 EWoK=51.24 Entity=24.07 COMPS=52.04 GlobalPIQA=35.62 Reading=7.34
- `official_cap120geom_b256` `chck_85M` equal7=42.69642857142857 BLiMP=68.99 Supplement=60.1 EWoK=50.78 Entity=23.92 COMPS=52.31 GlobalPIQA=34.635 Reading=8.14
- `official_cap120geom_b256` `chck_75M` equal7=42.67857142857143 BLiMP=68.47 Supplement=58.91 EWoK=51.32 Entity=22.6 COMPS=52.45 GlobalPIQA=36.605000000000004 Reading=8.395
