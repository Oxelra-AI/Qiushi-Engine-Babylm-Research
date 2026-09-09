# earlier analysis — EWoK semantic-hazard alignment

CPU-only analysis over selected compact_view_reinvest packets. It tests whether the EWoK domains with negative current-official treatment interaction are enriched for the force/modal/attribution/negation/causal/coreference hazards used by the candidate repair.

Selected block: 12155 rows / 423511 pair words; force-any row fraction 0.248; pronoun-or-force row fraction 0.276.

## EWoK-domain alignment (most negative official DiD first)
| EWoK domain | DiD | TE43022 | TE43122 | rows mapped | force-any frac | pronoun+force frac | modal loss | new causal | lost neg | content recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| material-dynamics | -17.532 | 9.870 | -7.662 | 2908 | 0.263 | 0.288 | 0.134 | 0.054 | 0.041 | 0.644 |
| physical-dynamics | -10.833 | 7.500 | -3.333 | 2908 | 0.263 | 0.288 | 0.134 | 0.054 | 0.041 | 0.644 |
| spatial-relations | -8.367 | 2.653 | -5.714 | 2279 | 0.250 | 0.275 | 0.116 | 0.063 | 0.033 | 0.653 |
| physical-interactions | -5.935 | 4.317 | -1.619 | 2908 | 0.263 | 0.288 | 0.134 | 0.054 | 0.041 | 0.644 |
| social-relations | -2.132 | 0.065 | -2.067 | 1187 | 0.223 | 0.246 | 0.113 | 0.046 | 0.029 | 0.668 |
| quantitative-properties | -0.955 | 0.637 | -0.318 | 974 | 0.285 | 0.302 | 0.177 | 0.042 | 0.037 | 0.668 |
| physical-relations | -0.489 | 2.323 | 1.834 | 2908 | 0.263 | 0.288 | 0.134 | 0.054 | 0.041 | 0.644 |
| agent-properties | 1.719 | -0.995 | 0.724 | 8162 | 0.245 | 0.274 | 0.125 | 0.055 | 0.032 | 0.667 |
| social-properties | 2.744 | 3.963 | 6.707 | 1187 | 0.223 | 0.246 | 0.113 | 0.046 | 0.029 | 0.668 |
| social-interactions | 6.463 | -5.782 | 0.680 | 2579 | 0.234 | 0.259 | 0.124 | 0.041 | 0.038 | 0.652 |
| material-properties | 8.235 | 7.059 | 15.294 | 1589 | 0.279 | 0.304 | 0.133 | 0.069 | 0.037 | 0.646 |

## Correlations across EWoK domains
- force_any_row_fraction_vs_DiD: pearson=-0.1712278601785298, spearman=-0.15856029077679654, n=11
- pronoun_or_force_row_fraction_vs_DiD: pearson=-0.1812573644641374, spearman=-0.11192491113656226, n=11
- mean_content_recall_vs_DiD: pearson=0.3556744144885093, spearman=0.37308303712187424, n=11
- mean_length_ratio_vs_DiD: pearson=0.2631245035475041, spearman=0.30779350562554625, n=11
- question_force_flip_row_fraction_vs_DiD: pearson=0.20969945989969496, spearman=0.37308303712187424, n=11
- lost_modal_or_hedge_row_fraction_vs_DiD: pearson=-0.062162270392953016, spearman=-0.32644765748163995, n=11
- lost_attribution_row_fraction_vs_DiD: pearson=-0.02170411418744554, spearman=-0.21452274634507768, n=11
- lost_negation_row_fraction_vs_DiD: pearson=-0.3887162153694742, spearman=-0.41971841676210847, n=11
- gained_negation_row_fraction_vs_DiD: pearson=-0.3293352257251661, spearman=-0.38241011304992106, n=11
- new_causal_marker_without_source_row_fraction_vs_DiD: pearson=-0.10369404956694343, spearman=-0.10259783520851541, n=11
- rewrite_starts_unresolved_pronoun_row_fraction_vs_DiD: pearson=-0.23766274927204112, spearman=0.027981227784140566, n=11

## Negative domain group
Domains ['material-dynamics', 'physical-dynamics', 'spatial-relations', 'physical-interactions']: unique mapped rows 3528 / pair words 134525; force-any frac 0.249; pronoun+force frac 0.274.
Top enrichment deltas vs complement:
- lost_negation: subset 0.038, complement 0.032, delta +0.006, log_odds +0.172
- entity_recall_lt_1: subset 0.021, complement 0.017, delta +0.004, log_odds +0.236
- gained_negation: subset 0.012, complement 0.010, delta +0.002, log_odds +0.150
- lost_attribution: subset 0.030, complement 0.029, delta +0.001, log_odds +0.052
- force_any: subset 0.249, complement 0.248, delta +0.001, log_odds +0.005
- new_causal_marker_without_source: subset 0.053, complement 0.053, delta +0.001, log_odds +0.017
- number_recall_lt_1: subset 0.000, complement 0.000, delta +0.000, log_odds +0.894
- rewrite_starts_unresolved_pronoun: subset 0.036, complement 0.038, delta -0.002, log_odds -0.043

## Scientific read
- This is heuristic: compact corpus domain tags are broad and do not prove that a particular EWoK item was learned from a particular row. It only tests whether the proposed repair's flags are enriched in the domains where official EWoK DiD is negative.
- A strong positive alignment between hazard burden and negative DiD would support materializing a force-repair corpus; weak or opposite alignment would favor checkpoint/optimization/consolidation work or a different relation-domain data idea.

Machine-readable output: `experiments/archive/frontier_consolidation/data/ewok_hazard_alignment/ewok_hazard_alignment.json`
