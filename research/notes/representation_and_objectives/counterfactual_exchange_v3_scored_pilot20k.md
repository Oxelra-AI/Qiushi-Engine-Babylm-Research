# counterfactual exchange clean subset — rule-certified counterfactual exchange probe
Created: 2026-08-31T05:11:32Z
## Build
Cases scored: **900** / input 900
Generated pair word count if trained once: **16516** words; by family: `{'temporal_order': 286, 'spatial_vertical': 474, 'causal_direction': 91, 'comparative_scalar': 57, 'comparative_more_less': 10}`.

## Frozen checkpoint arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.05221081309848362 | -0.020676617092556422 | -0.374469846089681 | compact > rowblock > interleaved | 0.4266806591881646 |
| target_perm_m | -0.02614082548353407 | -0.0021600394778781467 | 0.0027610630459255644 | interleaved > rowblock > compact | 0.028901888529459633 |
| ctx0_margin | -0.7138868003421359 | -0.35575646188524035 | -0.4599707169002957 | rowblock > interleaved > compact | 0.35813033845689557 |
| ctx1_margin | 0.7660976134406196 | 0.33507984479268393 | 0.0855008708106147 | compact > rowblock > interleaved | 0.6805967426300048 |
| target_main_bias | -1.4799844137827556 | -0.6908363066779243 | -0.5454715877109104 | interleaved > rowblock > compact | 0.9345128260718452 |

## Family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| causal_direction | 90 | 0.6739725748697917 | -0.12275417115953234 | 0.10733036465115017 | compact > interleaved > rowblock | 0.796726746029324 |
| comparative_more_less | 10 | 0.08887939453125 | 0.03137197494506836 | 0.0514068603515625 | compact > interleaved > rowblock | 0.05750741958618164 |
| comparative_scalar | 56 | 0.017848423549107144 | 0.13462611607142858 | 0.20722522054399764 | interleaved > rowblock > compact | 0.1893767969948905 |
| spatial_vertical | 469 | -0.2262541947842661 | 0.6172256286718698 | -0.7529742763494887 | rowblock > compact > interleaved | 1.3701999050213585 |
| temporal_order | 275 | 0.32929858814586294 | -1.108700662092729 | -0.020568507801402697 | compact > interleaved > rowblock | 1.437999250238592 |

## Interpretation hook
The object is a no-update probe. It is useful as a possible training object only if true exchange margins are not merely saturated local copying, if arm separation exceeds the target-permutation control, and if the direction follows the known relation-hard-row tradeoff rather than the compact broad-capability ordering.

Files:
- summary: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k_scored/counterfactual_exchange_score_summary.json`
- scores: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k_scored/counterfactual_exchange_scores.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k_scored/counterfactual_exchange_scores.csv`
