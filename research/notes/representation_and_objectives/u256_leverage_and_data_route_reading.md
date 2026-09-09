# u256 leverage and data route reading — U256 leverage and data-route reading

CPU-only reading of existing artifacts. No model was trained or evaluated, and no training corpus was created.

## U256 mass

- Hidden full words recovered by U256: 128,664 per 10M pass (1.2866%).
- Active tokens: 13,706,162 -> 13,942,644 per pass (+1.725%).
- Realized masked targets over 100M: 20,568,519 -> 20,916,090 (+1.690%).

## What content U256 actually changes

- CHILDES hidden words: 101,114 (78.59% of hidden).
- CHILDES+OpenSubtitles+BNC hidden words: 107,931 (83.89% of hidden).
- FineWeb compact hidden words: 17 (0.0132% of hidden).

U256 is therefore not a compact-view or leader-like FineWeb amplifier. It mainly exposes row tails from child/dialogue/transcript material and a smaller amount of heterogeneous long-row residue.

## Score leverage

- legal40_8x480_seed43022: Overall 41.1406, needs +5.93 summed column-points to reach 41.80; largest below-leader deficits: GlobalPIQA 5.00, EWoK 4.60, COMPS 1.90, SuperGLUE 1.28, Entity 1.25. EWoK+GlobalPIQA account for 68.5% of its below-leader deficit.
- legal40_mean_two_seed: Overall 40.7804, needs +9.18 summed column-points to reach 41.80; largest below-leader deficits: GlobalPIQA 6.25, EWoK 5.33, Entity 2.47, COMPS 1.41, SuperGLUE 0.88. EWoK+GlobalPIQA account for 70.9% of its below-leader deficit.
- legal40_depth_12x384_seed43022: Overall 41.0276, needs +6.95 summed column-points to reach 41.80; largest below-leader deficits: EWoK 5.52, GlobalPIQA 4.03, SuperGLUE 1.56, Entity 1.28, COMPS 0.87. EWoK+GlobalPIQA account for 72.0% of its below-leader deficit.

A fixed-length suffix-visibility intervention would have to create a surprisingly large and broad score movement to cross from the best compliant completed endpoint. The necessary gains sit mostly in EWoK and GlobalPIQA, while the material U256 newly exposes is mostly dialogue/transcript tails rather than factual FineWeb simplification pairs.

## Data substrate contrast

- Public leader model card: 9,999,969 FineWeb simplification-pair words.
- Current compact-view corpus: 423,511 FineWeb source+compact words (4.24% of leader pair budget); 2,080,311 all generated-pair words including official-source Qwen (20.80%).
- U256 recovers only 17 FineWeb compact words per 10M pass.

## Scientific reading

- U256 is a small visibility repair: +1.73% active tokens and +1.69% masked targets over 100M, with only 128,664 fully hidden words per 10M pass.
- The recovered words are overwhelmingly not the leader-like FineWeb simplification-pair substrate: 78.59% CHILDES and only 17 FineWeb compact words per 10M pass.
- The best completed compliant legal40 8x480 seed needs +5.94 summed column-points to reach 41.80; depth needs +6.95. Most below-leader deficit is EWoK+GlobalPIQA, while U256 mainly changes dialogue/transcript and name-heavy tails.
- Therefore U256 should not be the automatic first expensive route after a weak SGCR endpoint. It remains useful only if a delivered vector specifically points to child/dialogue/entity-state tail exposure as the limiting factor, or as a compact, fixed-length comparison after a stronger data route is unavailable.
- If SGCR is weak, the bigger unresolved difference from the public leader is data substrate: our corpus has 0.4235M FineWeb compact-pair words (4.2% of the leader's FineWeb pair word budget), while the leader model card reports 9.999969M FineWeb simplification-pair words.

Files:
- JSON: `experiments/archive/representation_and_objectives/data/u256_leverage_and_data_route_reading/u256_leverage_and_data_route_reading.json`
- CSV: `experiments/archive/representation_and_objectives/data/u256_leverage_and_data_route_reading/vector_leverage_vs_leader.csv`
