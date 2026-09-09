# bidir ranking contextual training and eval plan contextual cap-120 trajectory ranking

Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. AoA/CDI words, child curves, AoA predictions, AoA outputs, and SuperGLUE are not used.

Complete rows: 39 / 39
Best treatment equal7: `chck_80M` 42.662143 R=44.535000 P_no_sglue=41.945000
Best clean reference equal7: `chck_100M` 43.112857 R=44.142500 P_no_sglue=44.300000

## Top rows
1. `clean_qwen_reference` `clean_qwen_seed43022` `chck_100M` equal7=43.112857 R=44.142500 P_no_sglue=44.300000 BLiMP=66.84 Supplement=62.84 EWoK=50.19 Entity=25.76 COMPS=51.78 GlobalPIQA=36.62 Reading=7.76
2. `clean_qwen_reference` `clean_qwen_seed43022` `chck_95M` equal7=43.090000 R=44.142500 P_no_sglue=44.220000 BLiMP=66.79 Supplement=62.54 EWoK=50.16 Entity=25.9 COMPS=51.81 GlobalPIQA=36.62 Reading=7.8100000000000005
3. `context_cap120_control` `context_cap120_control_seed43022` `chck_80M` equal7=42.992857 R=45.132500 P_no_sglue=41.665000 BLiMP=69.1 Supplement=59.18 EWoK=50.95 Entity=24.15 COMPS=52.29 GlobalPIQA=37.09 Reading=8.19
4. `context_cap120_control` `context_cap120_control_seed43022` `chck_95M` equal7=42.886429 R=45.048750 P_no_sglue=42.195000 BLiMP=69.26 Supplement=59.97 EWoK=50.45 Entity=24.42 COMPS=52.34 GlobalPIQA=35.62 Reading=8.145
5. `context_cap120_control` `context_cap120_control_seed43022` `chck_90M` equal7=42.834286 R=44.993750 P_no_sglue=42.130000 BLiMP=69.15 Supplement=59.78 EWoK=50.27 Entity=24.48 COMPS=52.34 GlobalPIQA=35.605000000000004 Reading=8.215
6. `context_cap120_control` `context_cap120_control_seed43022` `chck_100M` equal7=42.810714 R=45.070000 P_no_sglue=42.130000 BLiMP=69.3 Supplement=59.89 EWoK=50.48 Entity=24.37 COMPS=52.34 GlobalPIQA=35.135 Reading=8.16
7. `clean_qwen_reference` `clean_qwen_seed43022` `chck_60M` equal7=42.756429 R=43.945000 P_no_sglue=43.705000 BLiMP=65.53 Supplement=60.4 EWoK=50.53 Entity=27.01 COMPS=51.83 GlobalPIQA=36.105000000000004 Reading=7.890000000000001
8. `clean_qwen_reference` `clean_qwen_seed43022` `chck_90M` equal7=42.756429 R=44.002500 P_no_sglue=44.325000 BLiMP=66.75 Supplement=62.99 EWoK=49.88 Entity=25.66 COMPS=51.65 GlobalPIQA=34.635 Reading=7.73
9. `clean_qwen_reference` `clean_qwen_seed43022` `chck_50M` equal7=42.714286 R=43.420000 P_no_sglue=44.870000 BLiMP=64.74 Supplement=63.32 EWoK=49.39 Entity=26.42 COMPS=51.64 GlobalPIQA=35.58 Reading=7.91
10. `context_cap120_control` `context_cap120_control_seed43022` `chck_85M` equal7=42.696429 R=45.055000 P_no_sglue=42.010000 BLiMP=68.99 Supplement=60.1 EWoK=50.78 Entity=23.92 COMPS=52.31 GlobalPIQA=34.635 Reading=8.14
11. `context_cap120_control` `context_cap120_control_seed43022` `chck_75M` equal7=42.680000 R=45.161250 P_no_sglue=40.755000 BLiMP=68.48 Supplement=58.91 EWoK=51.32 Entity=22.6 COMPS=52.45 GlobalPIQA=36.605000000000004 Reading=8.395
12. `context_cap120_treatment` `context_cap120_treat_seed43022` `chck_80M` equal7=42.662143 R=44.535000 P_no_sglue=41.945000 BLiMP=66.97 Supplement=60.81 EWoK=51.71 Entity=23.08 COMPS=51.6 GlobalPIQA=36.605000000000004 Reading=7.859999999999999

## Treatment minus matched control by checkpoint
- `chck_20M` delta_equal7=1.022143 treat=40.110000 control=39.087857 delta_R=0.912500 delta_P_no_sglue=2.260000 delta_BLiMP=-0.240 delta_Supplement=4.260 delta_EWoK=2.780 delta_Entity=0.260 delta_COMPS=0.000 delta_GlobalPIQA=-1.015 delta_Reading=1.110
- `chck_60M` delta_equal7=0.782143 treat=42.020714 control=41.238571 delta_R=-0.206250 delta_P_no_sglue=2.900000 delta_BLiMP=-0.940 delta_Supplement=5.020 delta_EWoK=0.700 delta_Entity=0.780 delta_COMPS=-0.040 delta_GlobalPIQA=0.500 delta_Reading=-0.545
- `chck_30M` delta_equal7=0.745000 treat=41.562143 control=40.817143 delta_R=-0.072500 delta_P_no_sglue=1.280000 delta_BLiMP=0.130 delta_Supplement=2.650 delta_EWoK=-0.630 delta_Entity=-0.090 delta_COMPS=0.100 delta_GlobalPIQA=2.945 delta_Reading=0.110
- `chck_10M` delta_equal7=0.264286 treat=37.864286 control=37.600000 delta_R=0.892500 delta_P_no_sglue=1.105000 delta_BLiMP=1.180 delta_Supplement=2.440 delta_EWoK=1.620 delta_Entity=-0.230 delta_COMPS=0.720 delta_GlobalPIQA=-3.930 delta_Reading=0.050
- `chck_40M` delta_equal7=0.027143 treat=40.722857 control=40.695714 delta_R=0.087500 delta_P_no_sglue=0.435000 delta_BLiMP=-0.310 delta_Supplement=-0.840 delta_EWoK=1.890 delta_Entity=1.710 delta_COMPS=-0.580 delta_GlobalPIQA=-1.030 delta_Reading=-0.650
- `chck_85M` delta_equal7=-0.065714 treat=42.630714 control=42.696429 delta_R=-0.625000 delta_P_no_sglue=0.785000 delta_BLiMP=-2.240 delta_Supplement=1.710 delta_EWoK=0.780 delta_Entity=-0.140 delta_COMPS=-0.670 delta_GlobalPIQA=0.470 delta_Reading=-0.370
- `chck_75M` delta_equal7=-0.187857 treat=42.492143 control=42.680000 delta_R=-0.667500 delta_P_no_sglue=1.890000 delta_BLiMP=-1.560 delta_Supplement=2.070 delta_EWoK=0.250 delta_Entity=1.710 delta_COMPS=-1.170 delta_GlobalPIQA=-2.425 delta_Reading=-0.190
- `chck_50M` delta_equal7=-0.189286 treat=41.582143 control=41.771429 delta_R=-0.511250 delta_P_no_sglue=1.080000 delta_BLiMP=-1.630 delta_Supplement=0.070 delta_EWoK=0.460 delta_Entity=2.090 delta_COMPS=-0.390 delta_GlobalPIQA=-1.440 delta_Reading=-0.485
- `chck_70M` delta_equal7=-0.320000 treat=41.587143 control=41.907143 delta_R=-0.483750 delta_P_no_sglue=-0.395000 delta_BLiMP=-1.470 delta_Supplement=0.740 delta_EWoK=0.530 delta_Entity=-1.530 delta_COMPS=-0.350 delta_GlobalPIQA=0.485 delta_Reading=-0.645
- `chck_80M` delta_equal7=-0.330714 treat=42.662143 control=42.992857 delta_R=-0.597500 delta_P_no_sglue=0.280000 delta_BLiMP=-2.130 delta_Supplement=1.630 delta_EWoK=0.760 delta_Entity=-1.070 delta_COMPS=-0.690 delta_GlobalPIQA=-0.485 delta_Reading=-0.330
- `chck_90M` delta_equal7=-0.425714 treat=42.408571 control=42.834286 delta_R=-0.555000 delta_P_no_sglue=0.355000 delta_BLiMP=-2.030 delta_Supplement=2.160 delta_EWoK=1.040 delta_Entity=-1.450 delta_COMPS=-0.750 delta_GlobalPIQA=-1.470 delta_Reading=-0.480
- `chck_100M` delta_equal7=-0.637143 treat=42.173571 control=42.810714 delta_R=-0.555000 delta_P_no_sglue=-0.120000 delta_BLiMP=-1.990 delta_Supplement=1.400 delta_EWoK=1.010 delta_Entity=-1.640 delta_COMPS=-0.810 delta_GlobalPIQA=-2.000 delta_Reading=-0.430
