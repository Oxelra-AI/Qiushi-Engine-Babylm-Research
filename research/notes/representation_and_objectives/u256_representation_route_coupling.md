# discriminating span support — U256 / representation route coupling measurement

CPU-only analysis without training or model evaluation. Existing U256 suffix measurements are compared with tokenizer-support evidence; a proposed H100 comparison depends on the depth and minfreq50 score vectors, not implementation readiness.

## U256 recovered mass as a capability intervention

- Newly visible full suffix words per 10M pass: 128,664; hidden-token total: 236,482; hidden rows: 11,407.
- Overall word gain versus row256-visible words: 0.0130; compact FineWeb suffix words: 17; CHILDES suffix words: 101114.
- Feature gain versus visible prefix mass: action 0.0154, causal/temporal 0.0075, physical/object 0.0168, spatial/state 0.0115, mental/social 0.0109, pronoun 0.0117, capitalized/name-like 0.0207.

This supports the interpretation: U256 is mostly additional visibility for dialogue/transcript tails and name/pronoun/spatial/action material, not a direct compact-view reinvestment amplifier.

## Representation support corner from existing CPU evidence

| tokenizer | family | eval-token ratio vs 40k | frac<50 | frac<100 | p10 support | p50 support |
|---|---|---:|---:|---:|---:|---:|
| legal_byte_bpe_40k | EWoK | 1.0000 | 0.0878 | 0.1495 | 56.0 | 5075.0 |
| legal_byte_bpe_40k | Entity | 1.0000 | 0.0031 | 0.1151 | 70.0 | 31217.0 |
| legal_byte_bpe_40k | GlobalPIQA | 1.0000 | 0.0975 | 0.1609 | 54.0 | 2514.0 |
| legal_byte_bpe_40k | Supplement | 1.0000 | 0.0495 | 0.0822 | 131.0 | 6052.0 |
| legal_byte_bpe_40k | SuperGLUE | 1.0000 | 0.0816 | 0.1314 | 66.0 | 5265.0 |
| legal_byte_bpe_40k_minfreq25 | EWoK | 1.0096 | 0.0745 | 0.1418 | 67.0 | 4818.0 |
| legal_byte_bpe_40k_minfreq25 | Entity | 1.0000 | 0.0015 | 0.1151 | 70.0 | 31217.0 |
| legal_byte_bpe_40k_minfreq25 | GlobalPIQA | 1.0414 | 0.0516 | 0.1132 | 86.0 | 2873.0 |
| legal_byte_bpe_40k_minfreq25 | Supplement | 1.0148 | 0.0382 | 0.0740 | 145.0 | 6015.0 |
| legal_byte_bpe_40k_minfreq25 | SuperGLUE | 1.0221 | 0.0584 | 0.1114 | 85.0 | 4818.0 |
| legal_byte_bpe_40k_minfreq50 | EWoK | 1.0598 | 0.0016 | 0.0799 | 121.0 | 6984.0 |
| legal_byte_bpe_40k_minfreq50 | Entity | 1.0015 | 0.0000 | 0.1135 | 70.0 | 31217.0 |
| legal_byte_bpe_40k_minfreq50 | GlobalPIQA | 1.0699 | 0.0225 | 0.0632 | 145.0 | 4239.0 |
| legal_byte_bpe_40k_minfreq50 | Supplement | 1.0434 | 0.0091 | 0.0469 | 188.0 | 5356.0 |
| legal_byte_bpe_40k_minfreq50 | SuperGLUE | 1.0619 | 0.0151 | 0.0724 | 134.0 | 4976.0 |
| legal_a01_16k | EWoK | 1.0972 | 0.0012 | 0.0418 | 133.0 | 5858.0 |
| legal_a01_16k | Entity | 1.0015 | 0.0000 | 0.1135 | 70.0 | 31217.0 |
| legal_a01_16k | GlobalPIQA | 1.0827 | 0.0212 | 0.0507 | 157.0 | 4621.0 |
| legal_a01_16k | Supplement | 1.0585 | 0.0073 | 0.0322 | 224.0 | 5356.0 |
| legal_a01_16k | SuperGLUE | 1.0830 | 0.0126 | 0.0501 | 166.0 | 5421.0 |

Minfreq50 strongly repairs low-support evaluation exposure on EWoK and GlobalPIQA relative to plain 40k, but it also surrenders some segmentation efficiency. Therefore the pending minfreq50 vector is essential: if its score vector preserves GlobalPIQA/Entity while depth preserves the 40k language columns, the stronger next problem is a legal representation corner rather than U256.

## Current score context for reading the pending vectors

- Legal40k 8x480 seed43022 Overall: 41.140578; visible leader: 41.80; gap: 0.659422.
- Largest positive gaps to the leader: GlobalPIQA 5.005, EWoK 4.596, COMPS 1.895, SuperGLUE 1.280, Entity 1.246

## Route consequence

When the depth and minfreq50 vectors arrive, read them together. Complementary recovery from depth and token support should move the next construction toward a support-aware legal representation coordinate. U256@0.15 remains launch-ready but should be used first only if the combined score pattern makes extra dialogue/social/state-tracking experience a credible frontier route; it is not the automatic successor to depth.

JSON: `experiments/archive/representation_and_objectives/data/u256_representation_route_coupling/u256_representation_route_coupling.json`
CSV: `experiments/archive/representation_and_objectives/data/u256_representation_route_coupling/u256_feature_enrichment_by_source.csv`, `experiments/archive/representation_and_objectives/data/u256_representation_route_coupling/support_floor_eval_family_tradeoff.csv`
