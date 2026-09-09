# roberta selected transfer resolution RoBERTa minimal selected evaluation

Created UTC: `2026-09-02T04:09:43Z`

Checkpoints: `['chck_100M']`
Stable selected positive: `False`; stable selected negative: `True`
Mean compact-minus-repeat: `{"BLiMP": 0.490000000000002, "COMPS": 0.10999999999999943, "EWoK": 1.0499999999999972, "EWoK_plus_Entity": 0.6400000000000006, "Entity": -0.41000000000000014, "GlobalPIQA": 0.9849999999999994, "Reading": 0.28000000000000025, "Supplement": -2.210000000000001, "cheap5_no_GlobalPIQA_Reading": -0.19400000000000261, "cheap6_no_GlobalPIQA": -0.11500000000000199, "cheap7": 0.04214285714285637}`

compact is neutral/negative on the selected stable-family readout in this minimal RoBERTa panel despite positive local pseudolikelihood; this supports a local-learning/downstream split unless additional late checkpoints overturn it

No training, upload, SuperGLUE, AoA, or leaderboard submission was performed.
