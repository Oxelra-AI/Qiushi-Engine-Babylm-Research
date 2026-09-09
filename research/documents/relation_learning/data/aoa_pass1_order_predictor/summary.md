# route evidence synthesis AoA pass-1 order predictor

Pass-1-only reordering predictor. Checkpoints after 10M retain original exposure totals, so endpoint multiset exposure is conserved by construction in this instrument.

## Validation on observed coherent86 ladder

- Fixed-effect exposure model: 9215 word-checkpoint observations, 485 words, 19 checkpoints, RMSE 0.8491, MAE 0.6189.
- Shared beta on log1p cumulative exposure: -0.361747 nats.
- Current predicted official-like curve fitness: 0.0000, p NA, n 245; unclipped r -0.0193.
- Predicted model AoA vs measured model AoA on available words: r 0.5866, Spearman 0.5871.
- Whole-stream frequency baseline vs measured model AoA: r -0.6214.
- CHILDES frequency versus child AoA bound: r -0.2047.

## Candidate pass-1 order predictions

| mode | oracle only | clipped score | p | n | unclipped r(child) | r(pred model AoA, measured model AoA) | mean |ΔS_1M| | max |ΔS| after 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sort__spoken_freq | 0 | 0.0000 | NA | 256 | 0.0445 | 0.6039 | 0.5997 | 0.0000 |
| sort__whole_freq | 0 | 0.0000 | NA | 261 | 0.0325 | 0.6085 | 0.7317 | 0.0000 |
| sort__cdi_density | 0 | 0.0000 | NA | 214 | 0.0148 | 0.5179 | 0.3188 | 0.0000 |
| sort__childes_freq | 0 | 0.0000 | NA | 255 | 0.0117 | 0.6136 | 0.5602 | 0.0000 |
| source_childes_first | 0 | 0.0000 | NA | 229 | 0.0046 | 0.6228 | 0.2364 | 0.0000 |
| sort__childes_composite | 0 | 0.0000 | NA | 253 | -0.0081 | 0.5991 | 0.4313 | 0.0000 |
| current | 0 | 0.0000 | NA | 245 | -0.0193 | 0.5866 | NA | NA |
| sort__childes_ratio | 0 | 0.0000 | NA | 233 | -0.0283 | 0.5995 | 0.3723 | 0.0000 |
| source_childes_only_then_current | 0 | 0.0000 | NA | 228 | -0.0392 | 0.6249 | 0.2364 | 0.0000 |
| source_written_first_control | 0 | 0.0000 | NA | 240 | -0.0398 | 0.5981 | 0.3008 | 0.0000 |
| sort__child_early | 1 | 0.0000 | NA | 239 | -0.1054 | 0.5635 | 0.2473 | 0.0000 |

## Word-level proxy signals

| feature | r(feature, child AoA) | p | r(feature, measured model AoA) | p |
|---|---:|---:|---:|---:|
| whole_logfreq | -0.0714 | 0.1513 | -0.6214 | 0.0000 |
| childes_logfreq | -0.2047 | 0.0000 | -0.5242 | 0.0000 |
| bnc_logfreq | 0.0421 | 0.3980 | -0.5682 | 0.0000 |
| opensub_logfreq | 0.0372 | 0.4550 | -0.6279 | 0.0000 |
| spoken_mix_logfreq | -0.1809 | 0.0002 | -0.5448 | 0.0000 |
| childes_minus_whole | -0.3225 | 0.0000 | 0.3563 | 0.0000 |

Full JSON: `experiments/archive/relation_learning/data/aoa_pass1_order_predictor/summary.json`
