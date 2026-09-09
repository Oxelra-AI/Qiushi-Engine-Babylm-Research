# distribution proximity prediction distribution-proximity prediction before MAX-register scores

This note records the CPU-only prediction made before reading any MAX-register score. It supersedes the first auto-generated prose emphasis while preserving the raw outputs under `data/distribution_proximity_prediction/`.

## Raw outputs

- Script: `experiments/archive/frontier_consolidation/scripts/distribution_proximity_prediction.py`
- Main output: `experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/distribution_proximity_prediction_summary.json`
- Generated summary: `research/documents/frontier_consolidation/data/distribution_proximity_prediction/distribution_proximity_prediction_summary.md`
- Corrected interpretation: `research/documents/frontier_consolidation/data/distribution_proximity_prediction/prediction_interpretation_after_inspection.md`
- Tables: `distance_rows.csv`, `calibration_rows.csv`, `fit_summary_rows.csv`, `prediction_rows.csv`, `aggregate_prediction_rows.csv`, `metric_agreement_rows.csv`

No model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, leaderboard action, or final-facing work was used.

## Exact contrast

The future contrast is `childspeech_removed - adultprose_removed`. Positive means the arm that sacrifices CHILDES/OpenSubtitles/BNC/Switchboard scores higher than the arm that sacrifices Gutenberg/SimpleWiki. Both arms admit the identical matched max dose execution note MAX FineWeb compact-view block; the admitted block therefore cancels in the direct register contrast.

Text blocks reconstructed by the script:

- admitted view block: first 7,923 rows of `dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl`, 1,118,587 words;
- admitted breadth block: first 7,923 rows of `dose_2p64x_breadth_rowholdout_pools/compact_breadth_dose2p64x_10M.jsonl`, 1,118,587 words;
- admitted repeat block: first 7,923 rows of `dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl`, 1,118,587 words;
- proportional removed clean block: `dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl`, 1,118,720 words;
- childspeech removed block: 1,118,720 words = CHILDES 578,080 + OpenSubtitles 411,040 + BNC Spoken 124,160 + Switchboard 5,440;
- adultprose removed block: 1,118,720 words = Gutenberg 710,560 + SimpleWiki 408,160.

The 133-word topup is intentionally excluded from the removed block; it is only 0.0119% of the MAX substitution words and is separately documented in earlier analysis metadata.

## Calibration equation

For each evaluation family and V/B/R arm:

`score_delta_points = beta * (JS(eval_family, removed_proportional_clean) - JS(eval_family, admitted_arm))`.

This uses the matched-geometry reference and margin state vectors V-Cmax, B-Cmax, R-Cmax. The primary common-window target is BLiMP, Supplement, EWoK, COMPS, and Reading over chck_10M..80M, because the queued register scorer targets the same window.

## What the lexical control showed

The word-unigram JS version did **not** explain the already scored family movements:

- beta = -2.2447 score points / JS bit;
- Pearson(x, score) = 0.0030, Spearman = -0.1214;
- RMSE = 0.7389 points;
- exEntity5 register prediction = -0.069 points; cheap6 = -0.085 points.

This weakens a simple lexical-frequency proximity account. It is kept as a pre-score negative/low-power lexical control, not as the main committed mechanism prediction.

## Committed viable prediction: non-open-lexical profile distance

The profile metric removes open-class lexical identities and keeps function/closed-class words, punctuation, transcript markers, sentence-length bins, token-shape bins, and suffix categories. It fit the existing matched-geometry movements materially better:

- beta = +14.2848 score points / JS bit;
- Pearson(x, score) = 0.3856, Spearman = 0.4714;
- RMSE = 0.4884 points;
- centered R2 = 0.1327.

Committed family predictions for `childspeech_removed - adultprose_removed`:

| family | profile JS(child removed)-JS(adult removed) | predicted score points |
|---|---:|---:|
| BLiMP | 0.05754 | +0.822 |
| Supplement | 0.07699 | +1.100 |
| EWoK | 0.04911 | +0.702 |
| COMPS | 0.00904 | +0.129 |
| Reading | 0.06923 | +0.989 |
| Entity | 0.08146 | +1.164, read separately because prior Entity effects were operation-mixture-sensitive |

Aggregate predictions:

- exEntity5 = +0.748 points, about 4.24x the earlier analysis exEntity mature spread reference 0.1765;
- cheap6 = +0.817 points, about 1.95x the earlier analysis cheap6 pointwise spread reference 0.4199.

Secondary profile calibrations provide a magnitude range: common-window V/B only gives exEntity5 +0.883; late V/B only gives +0.434; late all-three, diluted by duplicate recurrence decay, gives +0.206. The main commitment for the queued common-window register scorer remains a positive exEntity5 contrast around +0.7 to +0.9, with Supplement/Reading/BLiMP/EWoK positive and COMPS small.

## Persistence condition already required by existing results

R-Cmax exEntity5 falls from +0.4361 over common10_80 to +0.0343 late80_100, while V-Cmax and B-Cmax remain +0.3853 and +0.3350 late. Therefore the principle cannot be simple distance-to-target alone. It needs a distinct-content/persistence term: duplicated text can initially resemble the admitted-FineWeb movement but does not preserve broad ex-Entity value late; distinct admitted content does.

## Future register-score reading fixed now

The profile-distance fixed-budget account gains support if the observed MAX-register contrast is positive and roughly follows the committed ex-Entity family ordering, especially Supplement, Reading, BLiMP, and EWoK clearly positive while COMPS is small. It is weakened if exEntity5 is near zero or negative, if most adult/editable-English-close families reverse sign, if only Entity moves, if the family ordering is unrelated to both distance metrics, or if the result forces duplicated recurrence to be treated as persistently equivalent to distinct admitted content despite the measured R-Cmax decay.

## Robustness of the profile prediction to removing obvious register markers

A follow-up CPU-only pre-score decomposition is saved at `experiments/archive/frontier_consolidation/data/profile_distance_decomposition` and `experiments/archive/frontier_consolidation/data/profile_ablation_prediction`. It shows the full profile child-minus-adult distance is not carried only by explicit transcript markers. Ex-Entity signed JS contributions average: open-class abstract shape +0.02449, transcript/markup format +0.01774, punctuation +0.01299, function category +0.00650, function-word identity +0.00480, with sentence length (-0.00856), token length (-0.00338), and number tokens (-0.00335) partly opposing it. Thus transcript notation is visible but not the sole component.

Ablated prediction fits remain mostly positive before any register score is read:

| profile ablation | exEntity predicted points | cheap6 predicted points | interpretation |
|---|---:|---:|---|
| full profile | +0.748 | +0.817 | main committed common-window prediction |
| no transcript/markup format | +0.658 | +0.749 | positive signal survives without speaker/tier/bracket/url-like format counters |
| no transcript/markup and no punctuation | +0.380 | +0.470 | reduced but still positive broad prediction |
| function-only | +0.240 | +0.298 | weak calibration but positive sign |
| abstract-shape-only | +0.659 | +0.667 | positive signal also survives without function identities/punctuation |

This strengthens the pre-score prediction as a structural/profile-register proximity test rather than a pure CHILDES transcript-marker artifact. It also narrows what future scores can show: a near-zero observed exEntity contrast would falsify not only the full-profile prediction but also these ablated positive versions; a moderate positive contrast around +0.3 to +0.6 would support a weaker structural-profile account even if it undershoots the full +0.748 prediction.

## Strict evaluation-text extraction robustness

independent_review noted that the first predictor used heuristic extraction for EWoK/COMPS/Entity. I therefore ran `experiments/archive/frontier_consolidation/scripts/strict_eval_text_prediction.py` before any MAX-register scores were read. The stricter variant uses only task stimulus fields: BLiMP/Supplement sentence_good/bad, EWoK Context1/2 and Target1/2 (excluding Concept/Type/Diff metadata), COMPS composed prefix+property candidate strings, Entity input_prefix/options/prefix+option, and Reading unique sentences.

The result reproduces the same scientific split:

| metric | beta | Pearson | Spearman | RMSE | exEntity predicted points | cheap6 predicted points |
|---|---:|---:|---:|---:|---:|---:|
| word_js | -2.0757 | -0.0085 | -0.1536 | 0.7401 | -0.0681 | -0.0823 |
| profile_js | +13.3187 | 0.3735 | 0.4571 | 0.4929 | +0.7697 | +0.8441 |

Strict profile family predictions are BLiMP +0.766, Supplement +1.025, EWoK +0.931, COMPS +0.203, Reading +0.922, Entity +1.216. Thus the committed positive profile-register prediction is not a byproduct of including EWoK/COMPS metadata-like fields in the first extraction.
