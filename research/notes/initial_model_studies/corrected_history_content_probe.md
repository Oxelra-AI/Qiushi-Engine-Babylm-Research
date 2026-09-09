# corrected history content probe corrected history/content probe

Evidence JSON: `experiments/archive/initial_model_studies/data/corrected_history_content_probe.json`

Cases: 300; mean entity gap 37.8; mean unrelated gap 38.7; unique targets 193

## Content prediction effect

| model | n | loss orig | entity-cf delta | unrelated-cf delta | extra entity effect | SEM | frac(entity>unrel) |
|---|---:|---:|---:|---:|---:|---:|---:|
| wwm42_40M | 297 | 7.0607 | +0.0088 | +0.0112 | -0.0024 | 0.0063 | 0.202 |
| wwm42_100M | 297 | 6.5020 | +0.0054 | +0.0132 | -0.0078 | 0.0104 | 0.246 |
| wwm43_40M | 297 | 7.0047 | -0.0012 | +0.0090 | -0.0102 | 0.0071 | 0.276 |
| wwm43_100M | 297 | 6.4811 | -0.0003 | +0.0125 | -0.0127 | 0.0134 | 0.249 |
| amlm42_40M | 297 | 6.7201 | +0.0042 | +0.0068 | -0.0026 | 0.0053 | 0.249 |
| amlm42_100M | 297 | 6.2779 | -0.0120 | +0.0073 | -0.0193 | 0.0181 | 0.242 |
| amlm43_40M | 297 | 6.7314 | +0.0019 | +0.0056 | -0.0038 | 0.0100 | 0.266 |
| amlm43_100M | 297 | 6.2548 | +0.0079 | +0.0158 | -0.0079 | 0.0146 | 0.279 |

## Representation extra divergence (unrelated cosine - entity cosine; positive means same-entity history matters more than unrelated perturbation)

| model | L2 | L4 | L6 | L8 |
|---|---:|---:|---:|---:|
| wwm42_40M | -0.00060 | +0.00024 | +0.00031 | +0.00113 |
| wwm42_100M | -0.00024 | +0.01774 | +0.01440 | +0.01684 |
| wwm43_40M | -0.00072 | -0.00029 | -0.00011 | +0.00109 |
| wwm43_100M | -0.00037 | +0.01475 | +0.01135 | +0.01372 |
| amlm42_40M | +0.00012 | +0.00039 | +0.00140 | +0.00291 |
| amlm42_100M | +0.01415 | +0.01016 | +0.01647 | +0.01747 |
| amlm43_40M | +0.00176 | +0.00221 | +0.00206 | +0.00349 |
| amlm43_100M | +0.01879 | +0.01693 | +0.01401 | +0.01650 |

Interpretation: MCEC-style training is supported only if the same-entity history replacement has a larger effect than the matched unrelated-history replacement on masked content prediction, not merely on hidden-state geometry.
