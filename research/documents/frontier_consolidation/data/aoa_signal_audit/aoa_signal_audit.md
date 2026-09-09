# semantic repair frontier and refill plan — official-row-count AoA signal audit

Official AoA scores remain the evaluator's thresholded values; this audit records raw correlations to check the signal behind AoA=0.

## reinvest_seed43022
- official thresholded AoA: 0.0 with record {'curve_fitness': 0.0, 'n_words': 241}
- rows: 152095; row_count_values: [8005]; words with model+child AoA: 241
- raw Pearson r: -0.082675, p=0.200902; raw Spearman rho: -0.071181, p=0.271039
- model AoA std: 0.383204; child AoA std: 2.697740; model AoA range: 6.120..7.987

## reinvest_seed43122
- official thresholded AoA: 0.0 with record {'curve_fitness': 0.0, 'n_words': 233}
- rows: 152095; row_count_values: [8005]; words with model+child AoA: 233
- raw Pearson r: -0.047882, p=0.467000; raw Spearman rho: -0.062074, p=0.345509
- model AoA std: 0.376626; child AoA std: 2.706359; model AoA range: 6.026..7.991

## Scientific read
- AoA=0 is the official non-significance outcome, not an evidence-free placeholder. The saved files have complete 8,005-row/checkpoint ladders and nonconstant fitted model AoAs.
- These raw values are audit evidence only and must not replace the official leaderboard AoA column.

Machine-readable output: `experiments/archive/frontier_consolidation/data/aoa_signal_audit/aoa_signal_audit.json`
