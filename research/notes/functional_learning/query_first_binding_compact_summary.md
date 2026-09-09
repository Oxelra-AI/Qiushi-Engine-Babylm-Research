# query first binding compact summary compact summary

This file records per-seed transition times and held-entity metrics for the query-first binding result.

## orig_ans_only_500
- seed 42: mid=none, strong=none, standard top4=0.271, Bswap=+0.000, Qswap=+0.000, held top4=0.266, held NLL=1.469
- seed 43: mid=none, strong=none, standard top4=0.289, Bswap=-0.003, Qswap=+0.001, held top4=0.238, held NLL=1.575
- seed 100: mid=none, strong=none, standard top4=0.240, Bswap=-0.007, Qswap=-0.000, held top4=0.242, held NLL=1.504
- mean standard: NLL=1.487, top4=0.267, Bswap=-0.003, Qswap=+0.000, selectivity=+0.001
- mean held: NLL=1.516, top4=0.249, query margin=-0.003

## qfirst_full_500
- seed 42: mid=none, strong=none, standard top4=0.246, Bswap=-0.000, Qswap=-0.003, held top4=0.180, held NLL=1.581
- seed 43: mid=400, strong=400, standard top4=1.000, Bswap=+9.721, Qswap=+9.766, held top4=0.414, held NLL=4.265
- seed 100: mid=none, strong=none, standard top4=0.307, Bswap=+0.130, Qswap=+0.121, held top4=0.254, held NLL=1.600
- mean standard: NLL=1.017, top4=0.518, Bswap=+3.284, Qswap=+3.295, selectivity=+0.344
- mean held: NLL=2.482, top4=0.283, query margin=+0.673

## qfirst_ans_only_500
- seed 42: mid=225, strong=250, standard top4=1.000, Bswap=+15.413, Qswap=+15.620, held top4=0.504, held NLL=8.202
- seed 43: mid=475, strong=500, standard top4=0.994, Bswap=+7.578, Qswap=+7.667, held top4=0.809, held NLL=0.550
- seed 100: mid=375, strong=375, standard top4=1.000, Bswap=+11.122, Qswap=+11.090, held top4=0.965, held NLL=0.094
- mean standard: NLL=0.008, top4=0.998, Bswap=+11.371, Qswap=+11.459, selectivity=+0.993
- mean held: NLL=2.949, top4=0.759, query margin=+6.480

## qfirst_w16_500
- seed 42: mid=none, strong=none, standard top4=0.254, Bswap=+0.000, Qswap=+0.002, held top4=0.227, held NLL=1.561
- seed 43: mid=none, strong=none, standard top4=0.221, Bswap=-0.004, Qswap=+0.000, held top4=0.195, held NLL=1.504
- seed 100: mid=none, strong=none, standard top4=0.289, Bswap=+0.018, Qswap=+0.014, held top4=0.254, held NLL=1.579
- mean standard: NLL=1.514, top4=0.255, Bswap=+0.005, Qswap=+0.005, selectivity=+0.000
- mean held: NLL=1.548, top4=0.225, query margin=-0.031

## qfirst_bag_ans_500
- seed 42: mid=none, strong=none, standard top4=0.256, Bswap=-0.025, Qswap=-0.005, held top4=0.262, held NLL=1.492
- seed 43: mid=none, strong=none, standard top4=0.254, Bswap=-0.000, Qswap=-0.004, held top4=0.277, held NLL=1.441
- seed 100: mid=none, strong=none, standard top4=0.252, Bswap=+0.005, Qswap=-0.005, held top4=0.293, held NLL=1.438
- mean standard: NLL=1.463, top4=0.254, Bswap=-0.007, Qswap=-0.005, selectivity=+0.001
- mean held: NLL=1.457, top4=0.277, query margin=+0.026
