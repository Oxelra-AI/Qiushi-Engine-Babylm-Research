# route evidence synthesis AoA pass-1 sensitivity parallel

Actual coherent86 mean-surprisal curves plus beta times pass-1 candidate-vs-current log exposure deltas; after 10M exposure deltas are zero. Model AoA fits match the official threshold/sigmoid rule; child AoAs are earlier analysis precomputed values.

Current actual reproduction: clipped 0.0000, unclipped r -0.0349, p 0.6030, n 225; earlier analysis expected r -0.0349, p 0.6030, n 225.
Whole-stream frequency vs measured model AoA r -0.6214; CHILDES frequency vs child AoA r -0.2047; CHILDES-minus-whole vs child AoA r -0.3225.

## Best legal rows

| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| | after10M |
|---|---:|---:|---:|---:|---:|---:|---:|
| sort__childes_ratio | -0.75 | 0.0000 | -0.0039 | 0.9543 | 223 | 0.7718 | 0.0000 |
| sort__childes_composite | -1.5 | 0.0000 | -0.0111 | 0.8693 | 222 | 1.7884 | 0.0000 |
| sort__childes_ratio | -1.0 | 0.0000 | -0.0119 | 0.8617 | 217 | 1.0291 | 0.0000 |
| sort__childes_composite | -1.0 | 0.0000 | -0.0201 | 0.7653 | 223 | 1.1923 | 0.0000 |
| sort__childes_ratio | -0.36 | 0.0000 | -0.0210 | 0.7534 | 226 | 0.3705 | 0.0000 |
| sort__childes_composite | -0.75 | 0.0000 | -0.0210 | 0.7544 | 224 | 0.8942 | 0.0000 |
| source_childes_only_then_current | -0.36 | 0.0000 | -0.0216 | 0.7455 | 228 | 0.2352 | 0.0000 |
| sort__spoken_freq | -0.1 | 0.0000 | -0.0216 | 0.7476 | 224 | 0.1658 | 0.0000 |
| source_childes_first | -0.36 | 0.0000 | -0.0225 | 0.7345 | 229 | 0.2352 | 0.0000 |
| sort__childes_freq | -1.0 | 0.0000 | -0.0272 | 0.6871 | 222 | 1.5486 | 0.0000 |
| sort__childes_freq | -0.1 | 0.0000 | -0.0281 | 0.6769 | 223 | 0.1549 | 0.0000 |
| source_written_first_control | -0.2 | 0.0000 | -0.0342 | 0.6087 | 226 | 0.1663 | 0.0000 |
| source_childes_only_then_current | -0.2 | 0.0000 | -0.0352 | 0.5964 | 229 | 0.1307 | 0.0000 |
| sort__childes_ratio | -0.1 | 0.0000 | -0.0356 | 0.5934 | 227 | 0.1029 | 0.0000 |
| sort__whole_freq | -0.1 | 0.0000 | -0.0359 | 0.5920 | 225 | 0.2023 | 0.0000 |
| source_childes_first | -0.5 | 0.0000 | -0.0371 | 0.5797 | 225 | 0.3267 | 0.0000 |

## Oracle-only rows

| mode | beta | clipped score | unclipped r | p | n | mean |ΔS_1M| |
|---|---:|---:|---:|---:|---:|---:|
| sort__child_early | -0.36 | 0.0000 | -0.0331 | 0.6216 | 225 | 0.2461 |
| sort__child_early | -0.1 | 0.0000 | -0.0582 | 0.3841 | 226 | 0.0684 |
| sort__child_early | -0.2 | 0.0000 | -0.0611 | 0.3596 | 227 | 0.1367 |
| sort__child_early | -0.5 | 0.0000 | -0.0636 | 0.3419 | 225 | 0.3418 |
| sort__child_early | -0.75 | 0.0000 | -0.0986 | 0.1448 | 220 | 0.5128 |
| sort__child_early | -1.5 | -0.1178 | -0.1178 | 0.0909 | 207 | 1.0255 |
| sort__child_early | -1.0 | -0.1313 | -0.1313 | 0.0534 | 217 | 0.6837 |

Full JSON: `experiments/archive/relation_learning/data/aoa_pass1_sensitivity_parallel/summary.json`
