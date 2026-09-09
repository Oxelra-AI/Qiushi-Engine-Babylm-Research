# distribution proximity prediction distribution-proximity prediction before MAX-register scores

This CPU-only result was written before reading any MAX-register score. It commits a quantitative prediction for `childspeech_removed - adultprose_removed`, where positive means the arm that sacrifices CHILDES/OpenSubtitles/BNC/Switchboard scores higher than the arm that sacrifices Gutenberg/SimpleWiki. Both arms admit the identical matched max dose execution note MAX FineWeb compact-view block, so the admitted side cancels in the contrast.

## Exact text blocks used

- admitted view block: first 7923 rows of `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl`, 1118587 row words.
- admitted breadth block: first 7923 rows of `experiments/archive/frontier_consolidation/data/dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_10M.jsonl`, 1118587 row words.
- admitted repeat block: first 7923 rows of `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl`, 1118587 row words.
- proportional removed clean block: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl`, 1118720 row words.
- childspeech removed block: 1118720 row words, sources {'childes': 578080, 'open_subtitles': 411040, 'bnc_spoken': 124160, 'switchboard': 5440}.
- adultprose removed block: 1118720 row words, sources {'gutenberg': 710560, 'simple_wiki': 408160}.
- The 133-word arm-specific topup is reported in earlier analysis metadata and is not used as the removed block; it is 0.0119% of the MAX substitution words.

## Calibration equation

For each evaluation family and V/B/R arm, the fitted scalar is

`score_delta_points = beta * (JS(eval_family, removed_proportional_clean) - JS(eval_family, admitted_arm))`.

The primary calibration uses the three matched-geometry vectors V-Cmax, B-Cmax, and R-Cmax on BLiMP, Supplement, EWoK, COMPS, and Reading over common10_80, because the queued register scorer targets chck_10M..chck_80M.

Primary word-JS beta: **-2.245 score points / JS bit** over n=15; Pearson x-y=0.003, Spearman=-0.121, RMSE=0.739 points.
Profile-JS companion beta on the same rows: **14.285 score points / JS bit**; Pearson x-y=0.386, Spearman=0.471, RMSE=0.488 points.

## Committed primary word-JS prediction

| family | JS(child removed)-JS(adult removed) | predicted score points | sign |
|---|---:|---:|---|
| BLiMP | 0.01867 | -0.042 | negative_adultprose_arm_higher |
| Supplement | 0.05951 | -0.134 | negative_adultprose_arm_higher |
| EWoK | 0.02882 | -0.065 | negative_adultprose_arm_higher |
| COMPS | 0.01173 | -0.026 | negative_adultprose_arm_higher |
| Reading | 0.03547 | -0.080 | negative_adultprose_arm_higher |
| Entity | 0.07401 | -0.166 | negative_adultprose_arm_higher |

Primary word-JS aggregate exEntity5 prediction: **-0.069 points**, -0.39x the earlier analysis exEntity mature seed-spread reference 0.1765. Primary cheap6 prediction including Entity: **-0.085 points**, -0.20x the earlier analysis cheap6 pointwise spread reference 0.4199.

## Non-open-lexical profile companion

The profile metric removes open-class identities and retains closed-class/function words, punctuation, transcript markers, sentence length, token shape, and suffix categories. It tests whether the expected family ordering survives beyond lexical frequency.

| family | profile distance child-adult | predicted score points | sign |
|---|---:|---:|---|
| BLiMP | 0.05754 | 0.822 | positive_childspeech_arm_higher |
| Supplement | 0.07699 | 1.100 | positive_childspeech_arm_higher |
| EWoK | 0.04911 | 0.702 | positive_childspeech_arm_higher |
| COMPS | 0.00904 | 0.129 | positive_childspeech_arm_higher |
| Reading | 0.06923 | 0.989 | positive_childspeech_arm_higher |
| Entity | 0.08146 | 1.164 | positive_childspeech_arm_higher |

Profile aggregate exEntity5 prediction: **0.748 points**; cheap6 prediction: **0.817 points**. Word/profile exEntity distance-order Pearson=0.796, Spearman=0.900, sign agreement 5/5.

## Required persistence clause from the existing matched-geometry vectors

Under matched geometry, R-Cmax exEntity5 falls from 0.4361 common10_80 to 0.0343 late80_100, while V-Cmax and B-Cmax remain 0.3853 and 0.3350 late. R/mean(V,B) is 0.746 common and 0.095 late. Therefore a viable principle is not just distance-to-target; repeated exposure to the same admitted text loses broad ex-Entity value late, while distinct admitted content persists. The register pair should be read as a test of the sacrificed-register term under distinct admitted content, not as support for unlimited duplication.

## How future MAX-register scores bear on this account

If the observed childspeech_removed - adultprose_removed vector is positive and roughly follows the committed family ordering on ex-Entity families, the fixed-budget proximity account gains quantitative support: scarce words buy downstream competence when they are spent near the target distribution and when the admitted content remains distinct. If the contrast is near zero or reversed on the adult/editable-English-close families, or if only Entity moves while ex-Entity remains flat, the account loses force and the interpretation should shift toward admitted-content idiosyncrasy, DeBERTa coordinate effects, or structure beyond these two distance views.

## Files

- summary JSON: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/distribution_proximity_prediction_summary.json`
- distance rows: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/distance_rows.csv`
- calibration rows: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/calibration_rows.csv`
- family predictions: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/prediction_rows.csv`
- aggregate predictions: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/aggregate_prediction_rows.csv`
- metric agreement: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/metric_agreement_rows.csv`
