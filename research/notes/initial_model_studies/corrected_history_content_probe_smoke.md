# corrected history content probe corrected history/content probe

Evidence JSON: `experiments/archive/initial_model_studies/data/corrected_history_content_probe_smoke.json`

Cases: 50; mean entity gap 33.1; mean unrelated gap 34.0; unique targets 35

## Content prediction effect

| model | n | loss orig | entity-cf delta | unrelated-cf delta | extra entity effect | SEM | frac(entity>unrel) |
|---|---:|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 50 | 6.6574 | +0.0030 | +0.0242 | -0.0212 | 0.0166 | 0.280 |

## Representation extra divergence (unrelated cosine - entity cosine; positive means same-entity history matters more than unrelated perturbation)

| model | L2 | L4 | L6 | L8 |
|---|---:|---:|---:|---:|
| wwm43_40M | +0.00262 | +0.00187 | +0.00151 | +0.00266 |

Interpretation: MCEC-style training is supported only if the same-entity history replacement has a larger effect than the matched unrelated-history replacement on masked content prediction, not merely on hidden-state geometry.
