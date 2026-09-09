# earlier analysis AoA curve-fit analysis

The official AoA column uses Pearson correlation between fitted model log-step AoA and child 50pct AoA. If the p-value is above 0.1, the score object stores curve_fitness 0 even when many word curves fit; this is a measurement-side reading only and does not modify the submitted score.

## Official score object

`aoa_score.json`: `{"aoa": 0.0, "curve_fitness_record": {"curve_fitness": 0.0, "n_words": 225}}`

## Core counts

- Surprisal rows: 152095; target words: 485; rows per step values: [8005].
- Official step mean surprisal: chck_1M 9.7979, chck_100M 8.8308.
- Child statuses: {'ok': 406, 'child_never_reaches_50pct': 53, 'child_above_50pct_from_first_age': 6, 'child_sigmoid_or_bounds_failed': 20}.
- Model statuses after child AoA: {'threshold_above_upper_asymptote': 118, 'not_attempted_no_child_aoa': 79, 'flat_or_zero_amplitude': 46, 'ok': 225, 'model_fit_exception': 8, 'aoa_after_last_checkpoint': 6, 'threshold_below_lower_asymptote': 2, 'aoa_before_first_checkpoint': 1}.

## Unclipped correlation on words that fit both sides

- n = 225
- Pearson r = -0.03485418140387646, p = 0.6030225874932932
- Spearman r = -0.06471765697429414, p = 0.33385838476488117
- Official stored value after p>0.1 rule = 0.0

## Local score landscape

- AoA score files found: 79; numeric: 79; nonzero: 16; positive: 0; max: 0.0; min: -0.2085501458906283.
- Nonzero examples are written to the JSON and CSV outputs.

Full JSON: `experiments/archive/relation_learning/data/aoa_curve_fit_analysis/summary.json`
