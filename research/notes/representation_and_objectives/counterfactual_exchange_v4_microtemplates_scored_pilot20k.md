# counterfactual exchange clean subset — rule-certified counterfactual exchange probe
Created: 2026-08-31T05:19:45Z
## Build
Cases scored: **46** / input 46

## Frozen checkpoint arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.12020745484725288 | -0.12417731077774712 | 0.06913558296535326 | compact > interleaved > rowblock | 0.244384765625 |
| target_perm_m | 0.04224171845809273 | 0.06525719684103261 | -0.07198157517806343 | rowblock > compact > interleaved | 0.13723877201909604 |
| ctx0_margin | -0.86376953125 | 0.4293053668478261 | -0.11269080120584239 | rowblock > interleaved > compact | 1.2930748980978262 |
| ctx1_margin | 0.9839769860972529 | -0.5534826776255732 | 0.18182638417119565 | compact > interleaved > rowblock | 1.5374596637228262 |
| target_main_bias | -1.847746517347253 | 0.9827880444733993 | -0.294517185377038 | rowblock > interleaved > compact | 2.8305345618206523 |

## Family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| causal_exact | 33 | 0.16587511698404947 | -0.08415450471820253 | 0.21340295040246213 | interleaved > compact > rowblock | 0.29755745512066467 |
| comparative_exact | 13 | 0.004281850961538462 | -0.22577366462120643 | -0.29708158052884615 | compact > rowblock > interleaved | 0.3013634314903846 |

## Interpretation hook
The object is a no-update probe. It is useful as a possible training object only if true exchange margins are not merely saturated local copying, if arm separation exceeds the target-permutation control, and if the direction follows the known relation-hard-row tradeoff rather than the compact broad-capability ordering.

Files:
- summary: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_v4_microtemplates_pilot20k_scored/counterfactual_exchange_score_summary.json`
- scores: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_v4_microtemplates_pilot20k_scored/counterfactual_exchange_scores.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_v4_microtemplates_pilot20k_scored/counterfactual_exchange_scores.csv`
