# counterfactual exchange clean subset — rule-certified counterfactual exchange probe
Created: 2026-08-31T05:21:33Z
## Build
Cases scored: **2372** / input 2372
Generated pair word count if trained once: **42700** words; by family: `{'temporal_order': 866, 'spatial_vertical': 1234, 'causal_direction': 149, 'comparative_scalar': 109, 'comparative_more_less': 14}`.

## Frozen checkpoint arm means
| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.08474952495118217 | -0.0384411441938069 | -0.3858450399884486 | compact > rowblock > interleaved | 0.47059456493963076 |
| target_perm_m | -0.01035335011667184 | -0.0029020325530439766 | 0.0016682172103032696 | interleaved > rowblock > compact | 0.012021567326975109 |
| ctx0_margin | -0.7267406018950284 | -0.33200065442321675 | -0.44925657693692445 | rowblock > interleaved > compact | 0.3947399474718117 |
| ctx1_margin | 0.8114901268462106 | 0.29355951022940985 | 0.06341153694847583 | compact > rowblock > interleaved | 0.7480785898977348 |
| target_main_bias | -1.538230728741239 | -0.6255601646526265 | -0.5126681138854002 | interleaved > rowblock > compact | 1.0255626148558386 |

## Family true_m means
| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| causal_direction | 149 | 0.7922301836461829 | -0.1379852870966764 | 0.15624838707431052 | compact > interleaved > rowblock | 0.9302154707428594 |
| comparative_more_less | 14 | 0.04304722377232143 | 0.06685522624424525 | 0.11902945382254464 | interleaved > rowblock > compact | 0.0759822300502232 |
| comparative_scalar | 109 | -0.037712482137417576 | 0.03545760233467872 | 0.109364903301274 | interleaved > rowblock > compact | 0.1470773854386916 |
| spatial_vertical | 1234 | -0.17030884342409996 | 0.7038665938416022 | -0.7558259755324699 | rowblock > compact > interleaved | 1.459692569374072 |
| temporal_order | 866 | 0.3425550670051134 | -1.090063296619794 | -0.02240574772859005 | compact > interleaved > rowblock | 1.4326183636249075 |

## Interpretation hook
The object is a no-update probe. It is useful as a possible training object only if true exchange margins are not merely saturated local copying, if arm separation exceeds the target-permutation control, and if the direction follows the known relation-hard-row tradeoff rather than the compact broad-capability ordering.

Files:
- summary: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_full_scored/counterfactual_exchange_score_summary.json`
- scores: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_full_scored/counterfactual_exchange_scores.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_full_scored/counterfactual_exchange_scores.csv`
