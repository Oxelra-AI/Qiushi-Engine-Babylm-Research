# contextual cap120 full eval interpretation contextual cap-120 full-eval interpretation

Reads only completed corrected full-eval files for frozen no-AoA selected targets. AoA is final measurement only, not a route selector.

Visible leader reference: 41.8000
Clean-Qwen reference: 41.34429066479573

## Ranked contextual targets by corrected Overall
1. `context_cap120_treat_seed43022_80M` `chck_80M` Overall=40.922489 delta_vs_41.8=-0.877511 equal7=42.66214285714286 R=44.535 P=51.185799764542516 SuperGLUE=69.66739929362754 AoA=0.0 need_SG_plus_AoA>77.56499999999994
2. `context_cap120_control_seed43022_80M` `chck_80M` Overall=38.774508 delta_vs_41.8=-3.025492 equal7=42.99285714285714 R=45.1325 P=50.735195459910564 SuperGLUE=68.87558637973169 AoA=-20.855014589062833 need_SG_plus_AoA>75.25

## Treatment minus matched control
- {"a": "context_cap120_treat_seed43022_80M", "a_minus_b_overall": 2.147980833662082, "b": "context_cap120_control_seed43022_80M", "delta_AoA": 20.855014589062833, "delta_BLiMP": -2.1299999999999955, "delta_COMPS": -0.6899999999999977, "delta_EWoK": 0.759999999999998, "delta_Entity": -1.0700000000000003, "delta_GlobalPIQA": -0.48499999999999943, "delta_Reading": -0.33000000000000007, "delta_SuperGLUE": 0.7918129138958534, "delta_Supplement": 1.6300000000000026, "delta_equal7_from_full_eval": -0.3307142857142793, "delta_preserve_P": 0.4506043046319519, "delta_recovery_R": -0.5975000000000037}

## Treatment minus clean-Qwen reference
- {"a": "context_cap120_treat_seed43022_80M", "a_minus_b_overall": -0.42180185439266893, "b": "qwen_clean_aligned", "delta_AoA": 0.0, "delta_BLiMP": 0.12999999999999545, "delta_COMPS": -0.17999999999999972, "delta_EWoK": 1.5200000000000031, "delta_Entity": -2.6800000000000033, "delta_GlobalPIQA": -0.014999999999993463, "delta_Reading": 0.09999999999999964, "delta_SuperGLUE": -0.6412166895340334, "delta_Supplement": -2.030000000000001}
