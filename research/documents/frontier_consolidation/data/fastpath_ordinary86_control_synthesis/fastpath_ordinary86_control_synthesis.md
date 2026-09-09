# earlier analysis fast-path ordinary86 control synthesis

Status: **COMPLETE**
Source: `experiments/archive/representation_and_objectives/data/fastpath_vs_ordinary86_item_review/fastpath_vs_ordinary86_item_review.json`

## Cheap7 table

| arm | cheap7 |
|---|---:|
| chck82 | 43.95944987645173 |
| ordinary86 | 43.770714285714284 |
| shuffled_private86 | 44.01285714285714 |
| coherent4M | 44.10642857142857 |
| spanbreak4M | 43.121428571428574 |

## Decisive comparisons

### coherent_minus_ordinary86

Aggregate: `{'total_gain_items': 4796, 'total_loss_items': 5029, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.3799999999999996, 'discrete_reconstructed_mean_delta': 0.3796021811646971, 'total_gain_minus_loss': -233, 'loss_to_gain_ratio': 1.048582151793161}`

| column | payload Δ | item net |
|---|---:|---:|
| BLiMP | 0.04999999999999716 | 42 |
| Supplement | 0.9699999999999989 | -21 |
| EWoK | -0.23000000000000398 | -27 |
| Entity | -0.13999999999999702 | -14 |
| COMPS | -0.29999999999999716 | -217 |
| GlobalPIQA | 1.9299999999999997 | 4 |

### ordinary86_minus_chck82

Aggregate: `{'total_gain_items': 5366, 'total_loss_items': 5248, 'total_common_items': 170722, 'discrete_payload_mean_delta': -0.21207259098336273, 'discrete_reconstructed_mean_delta': -0.21116388710188913, 'total_gain_minus_loss': 118, 'loss_to_gain_ratio': 0.9780096906448006}`

| column | payload Δ | item net |
|---|---:|---:|
| BLiMP | -0.021284036519858773 | -12 |
| Supplement | -0.25781125620019907 | 7 |
| EWoK | 0.08454667446724073 | 25 |
| Entity | 0.2659580697012238 | 21 |
| COMPS | 0.09882490556403667 | 80 |
| GlobalPIQA | -1.4426699029126198 | -3 |

### coherent_minus_chck82

Aggregate: `{'total_gain_items': 3116, 'total_loss_items': 3231, 'total_common_items': 170722, 'discrete_payload_mean_delta': 0.16792740901663686, 'discrete_reconstructed_mean_delta': 0.16843829406280797, 'total_gain_minus_loss': -115, 'loss_to_gain_ratio': 1.0369062901155328}`

| column | payload Δ | item net |
|---|---:|---:|
| BLiMP | 0.028715963480138385 | 30 |
| Supplement | 0.7121887437997998 | -14 |
| EWoK | -0.14545332553276324 | -2 |
| Entity | 0.1259580697012268 | 7 |
| COMPS | -0.2011750944359605 | -137 |
| GlobalPIQA | 0.48733009708737995 | 1 |

## Scientific reading

ordinary86 is the exposure-matched ordinary-continuation control for coherent86. Coherent86 has a robust aggregate endpoint edge, but common official-item transitions show negative item balance versus ordinary86 and versus the anchor. The current evidence therefore supports coherent86 as a stronger endpoint candidate, not as a demonstrated general slow-fast learning principle.

Before training a retention-modified fast path, test whether the learned private residual has a useful inference-time amplitude: alpha 0 is the protected anchor, alpha 1 is coherent86, and intermediate alpha can reveal whether score gains and item erosion are separable without new training.

JSON: `experiments/archive/frontier_consolidation/data/fastpath_ordinary86_control_synthesis/fastpath_ordinary86_control_synthesis.json`
