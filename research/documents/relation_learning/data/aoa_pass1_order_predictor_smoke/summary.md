# route evidence synthesis AoA pass-1 order predictor

Pass-1-only reordering predictor. Checkpoints after 10M retain original exposure totals, so endpoint multiset exposure is conserved by construction in this instrument.

## Validation on observed coherent86 ladder

- Fixed-effect exposure model: 9215 word-checkpoint observations, 485 words, 19 checkpoints, RMSE 0.8483, MAE 0.6189.
- Shared beta on log1p cumulative exposure: -0.772296 nats.
- Current predicted official-like curve fitness: 0.0000, p NA, n 243; unclipped r -0.0438.
- Predicted model AoA vs measured model AoA on available words: r 0.6026, Spearman 0.5818.
- Whole-stream frequency baseline vs measured model AoA: r -0.6353.
- CHILDES frequency versus child AoA bound: r -0.1814.

## Candidate pass-1 order predictions

| mode | oracle only | clipped score | p | n | unclipped r(child) | r(pred model AoA, measured model AoA) | mean |ΔS_1M| | max |ΔS| after 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | NA | NA |
| source_childes_only_then_current | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| source_childes_first | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__childes_freq | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__childes_ratio | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__spoken_freq | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__childes_composite | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__cdi_density | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__whole_freq | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| source_written_first_control | 0 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |
| sort__child_early | 1 | 0.0000 | NA | 243 | -0.0438 | 0.6026 | 0.0000 | 0.0000 |

## Word-level proxy signals

| feature | r(feature, child AoA) | p | r(feature, measured model AoA) | p |
|---|---:|---:|---:|---:|
| whole_logfreq | -0.0699 | 0.1597 | -0.6353 | 0.0000 |
| childes_logfreq | -0.1814 | 0.0002 | -0.5124 | 0.0000 |
| bnc_logfreq | 0.0756 | 0.1282 | -0.5637 | 0.0000 |
| opensub_logfreq | 0.0475 | 0.3400 | -0.6388 | 0.0000 |
| spoken_mix_logfreq | -0.1689 | 0.0006 | -0.5184 | 0.0000 |
| childes_minus_whole | -0.2234 | 0.0000 | 0.2515 | 0.0001 |

Full JSON: `experiments/archive/relation_learning/data/aoa_pass1_order_predictor_smoke/summary.json`
