# reciprocal multiview mechanism and scaffold reciprocal view-lift comparison

First: `chck82` `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_probe.json`
Second: `chck100` `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck100_128/reciprocal_view_lift_probe.json`
Records: 2048 -> 2048

## Copy/text strata
| pair_type | target | copy_by_text | n first | n second | mean lift first | mean lift second | Δ second-first | pos first | pos second |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | other | False | 73 | 73 | 0.061620 | 0.040022 | -0.021598 | 0.397260 | 0.424658 |
| compact | other | True | 439 | 439 | 5.371522 | 5.316419 | -0.055104 | 0.936219 | 0.933941 |
| compact | source | False | 188 | 188 | -0.106480 | -0.112485 | -0.006005 | 0.462766 | 0.473404 |
| compact | source | True | 324 | 324 | 5.131248 | 5.106071 | -0.025177 | 0.935185 | 0.932099 |
| repeat | other | True | 512 | 512 | 5.120093 | 5.143040 | 0.022947 | 0.982422 | 0.986328 |
| repeat | source | False | 162 | 162 | -0.312812 | -0.310210 | 0.002602 | 0.419753 | 0.364198 |
| repeat | source | True | 350 | 350 | 4.339225 | 4.285412 | -0.053813 | 0.965714 | 0.965714 |

## Mechanistic reading
Large copy-stratum lift is expected and mostly reflects visible lexical repetition. The load-bearing signal for semantic-view invariance is compact non-copy lift in both target directions. If this non-copy lift is small or falls from 82M to 100M while official relation/state scores also fall, it is a plausible diagnostic of reciprocal semantic support but not yet a trainable objective. Any training proposal must avoid merely increasing exact-copy targets.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_comparison_82_100/reciprocal_view_lift_comparison.json`
