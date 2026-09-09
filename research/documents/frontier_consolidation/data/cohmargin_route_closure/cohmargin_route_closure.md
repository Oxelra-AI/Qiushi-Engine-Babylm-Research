# coherence margin signal isolation coherence-margin route closure

Status: **COHERENCE_MARGIN_ROUTE_CLOSED_AFTER_4M_PILOT**

Pilot cheap7: `36.692142857142855`; protected chck82 cheap7: `43.95944987645173`; delta `-7.267307`.

| column | pilot | chck82 | delta |
|---|---:|---:|---:|
| BLiMP | 53.970000 | 68.491284 | -14.521284 |
| Supplement | 46.570000 | 62.937811 | -16.367811 |
| EWoK | 48.940000 | 50.055453 | -1.115453 |
| Entity | 16.500000 | 28.314042 | -11.814042 |
| COMPS | 50.300000 | 52.191175 | -1.891175 |
| GlobalPIQA | 34.195000 | 37.577670 | -3.382670 |
| Reading | 6.370000 | 8.148714 | -1.778714 |

## Mechanism readings

Training trace: margin_loss mean `0.7998403205591089`, last10 `0.7989563912153244`, logged NLL gap mean `-0.0006274628855559664`, last10 `-0.0003145403374219313`.
NLL probe gaps: pilot train64 `0.0009750703694095194`, pilot holdout64 `0.0003258801299841141`, ref4m train64 `0.025009877468731484`, ref4m holdout64 `0.017921989642364473`.

The pilot is broadly destructive on official-compatible cheap7 and fails the intended mechanism probe: the margin term stayed near softplus(0.20) and coherent-vs-disrupted NLL separation remained near zero. The route should not continue to 20M, nor should lambda-zero official evaluation be spent after this pilot result.

The lambda-zero isolate task was cancelled once the pilot cheap7 result made the route noncompetitive and scientifically unpromising; its partial logs are not an endpoint result.

JSON: `experiments/archive/frontier_consolidation/data/cohmargin_route_closure/cohmargin_route_closure.json`
