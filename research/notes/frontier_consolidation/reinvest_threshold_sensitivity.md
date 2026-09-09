# full core route localization and existing ladder plan reinvest threshold sensitivity

This note corrects the reinvest projection by using `Entity_full` from the no-AoA screen as the official-like Entity column. It also applies the observed compact-core full-minus-fast split deltas as a sensitivity case. This is not a substitute for the paired full reinvest evaluation.

Reinvest official-like fast seven-column sum = 309.700, mean = 44.2429.
From that surface, required SuperGLUE+AoA: 62.399 for inherited clean-Qwen, 66.500 for 41.8, 68.300 for 42.0.
If SuperGLUE equals compact-core full (68.901), the tolerated AoA for 41.8 is -2.401.

Observed compact-core full-minus-fast official-like deltas: BLiMP +0.220, Supplement -3.750, EWoK -0.290, Entity +0.000, COMPS +0.000, GlobalPIQA +0.000, Reading +0.000
If reinvest receives the same full-split deltas, seven-column sum = 305.880, required SuperGLUE+AoA = 70.320 for 41.8, tolerated AoA at SG=68.901 is 1.419.

JSON: `experiments/archive/frontier_consolidation/data/reinvest_projection/reinvest_threshold_sensitivity.json`
