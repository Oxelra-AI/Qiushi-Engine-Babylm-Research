# distribution proximity prediction post-run interpretation of the pre-score distribution-proximity predictor

This note was written after inspecting the CPU-only outputs of `distribution_proximity_prediction.py` and before reading any MAX-register score. It corrects the scientific emphasis of the generated summary without changing the raw computed tables.

## Raw outputs preserved

- Script: `experiments/archive/frontier_consolidation/scripts/distribution_proximity_prediction.py`
- Main output: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/distribution_proximity_prediction_summary.json`
- Generated summary: `research/documents/frontier_consolidation/data/distribution_proximity_prediction/distribution_proximity_prediction_summary.md`
- Tables: `distance_rows.csv`, `calibration_rows.csv`, `fit_summary_rows.csv`, `prediction_rows.csv`, `aggregate_prediction_rows.csv`, `metric_agreement_rows.csv`

No model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, leaderboard action, or final-facing work was used.

## What changed after inspecting the output

The word-unigram JS version is not a viable quantitative explanation of the already scored matched-geometry family movements. Under the requested common10_80 calibration over V-Cmax, B-Cmax, R-Cmax and ex-Entity families, its coefficient is negative and its calibration association is essentially absent:

- word-JS beta = -2.2447 score points / JS bit;
- n = 15 family-arm points;
- Pearson(x, score) = 0.0030, Spearman = -0.1214;
- RMSE = 0.7389 points and centered R2 is strongly negative.

This is scientifically informative: the broad V/B/R-minus-clean movement is not explained by simple lowercased word-unigram closeness to the evaluation families. In fact, for several families the proportional removed clean block is lexically closer than the admitted FineWeb block while the FineWeb-admitting arms score higher. The word-JS register prediction is therefore a low-power lexical-frequency control, not the main committed mechanism prediction. Its aggregate prediction is tiny and negative: exEntity5 -0.069 points and cheap6 -0.085 points, both below the earlier analysis spread scale.

## Viable committed prediction: structural/function profile distance

The non-open-lexical profile metric removes open-class lexical identities and keeps function/closed-class words, punctuation, transcript markers, sentence-length bins, token-shape bins, and suffix categories. On the same common10_80 calibration rows it gives a positive coefficient and a materially better, though still imperfect, calibration:

- profile-JS beta = +14.2848 score points / JS bit;
- n = 15 family-arm points;
- Pearson(x, score) = 0.3856, Spearman = 0.4714;
- RMSE = 0.4884 points;
- centered R2 = 0.1327.

The metric is not a proof of mechanism, but it makes a large, falsifiable pre-score prediction. Positive means `childspeech_removed - adultprose_removed`, i.e. the arm that sacrifices developmental/speech rows should score higher than the arm that sacrifices Gutenberg/SimpleWiki, because the adult-prose block is closer to the evaluation profiles and costs more to remove.

| family | profile JS(child removed)-JS(adult removed) | committed prediction in score points |
|---|---:|---:|
| BLiMP | 0.05754 | +0.822 |
| Supplement | 0.07699 | +1.100 |
| EWoK | 0.04911 | +0.702 |
| COMPS | 0.00904 | +0.129 |
| Reading | 0.06923 | +0.989 |
| Entity | 0.08146 | +1.164, read separately because prior Entity effects were operation-mixture-sensitive |

Aggregate predictions:

- exEntity5: +0.748 points, about 4.24x the earlier analysis exEntity mature spread reference 0.1765;
- cheap6 including Entity: +0.817 points, about 1.95x the earlier analysis cheap6 pointwise spread reference 0.4199.

The distance ordering itself is not purely a lexical artifact: word/profile ex-Entity child-adult distance ordering has Pearson 0.796 and Spearman 0.900 with 5/5 sign agreement. The crucial difference is that the word-unigram calibration has the wrong sign and no fit, while the function/profile calibration has a positive coefficient and a usable fit.

Secondary profile calibrations define a plausible magnitude range rather than replacing the main commitment:

- common10_80 distinct-content V/B only: exEntity5 +0.883, cheap6 +0.965;
- late80_100 distinct-content V/B only: exEntity5 +0.434, cheap6 +0.474;
- late80_100 all three V/B/R, diluted by duplicate decay: exEntity5 +0.206, cheap6 +0.225.

Because the queued MAX-register scorer targets chck_10M..80M first, the primary pre-score expected magnitude is the common10_80 profile prediction: a broad ex-Entity positive contrast around +0.7 to +0.9 points, with Supplement/Reading/BLiMP/EWoK much larger than COMPS. A future late-window readout should also be checked, but its expected magnitude may be smaller if persistence behaves like the distinct-content late calibration.

## Required persistence condition

The existing matched-geometry vectors impose a constraint on any principle. R-Cmax exEntity5 is +0.4361 over common10_80 but only +0.0343 over late80_100, whereas V-Cmax and B-Cmax remain +0.3853 and +0.3350 late. Thus simple proximity is incomplete unless it includes a distinct-content/persistence term: duplicated text can initially mimic the admitted-FineWeb movement but does not preserve broad ex-Entity value late, while distinct admitted content does.

## Future score interpretation fixed before scores land

The profile-distance version of the fixed-budget proximity account gains support if the MAX-register contrast is positive and roughly follows the committed family ordering on ex-Entity families, especially Supplement, Reading, BLiMP, and EWoK being clearly positive while COMPS is small. It is weakened if:

1. exEntity5 is near zero or negative when compared with the +0.748 common-window prediction;
2. the sign reverses on most adult/editable-English-close families;
3. only Entity moves while ex-Entity families stay flat;
4. observed family ordering is unrelated to both word/profile distance-difference ordering; or
5. the result requires treating duplicate recurrence as persistently equivalent to distinct admitted content despite the already measured R-Cmax decay.

This record should be used with `prediction_rows.csv` and `aggregate_prediction_rows.csv` when `register_max_readout.py` is rerun after register scores arrive.
