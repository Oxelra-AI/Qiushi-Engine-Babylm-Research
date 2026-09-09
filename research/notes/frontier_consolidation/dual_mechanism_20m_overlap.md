# dual mechanism 20m overlap — 20M overlap of residual-side-capacity and faithful-visibility mechanisms

CPU-only analysis of saved predictions. It compares spatial repair route status legal 20M, adapter128 scale1.75 20M, and U256 faithful-visibility 20M on common discrete items. Reading is not included because the current saved-prediction parser handles only the six discrete zero-shot columns.

## Aggregate

- Six-discrete reconstructed mean Δ vs spatial repair route status: scale1.75 +0.7458; U256 +1.2609.
- Aggregate repair shared fraction of union: 0.2582.
- Aggregate damage shared fraction of union: 0.2478.
- Aggregate counts: `{'both_miss_base_wrong': 46240, 'u256_only_repair': 16587, 'both_preserve_base_correct': 57155, 'both_damage': 8256, 'u256_only_damage': 15946, 'both_repair': 8780, 'scale_only_damage': 9121, 'scale_only_repair': 8637}`.

## Column overlap

| column | Δ scale | Δ U256 | repair Jaccard | damage Jaccard | shared repairs / union | shared damages / union | group-net r | oracle extra repairs over best | strongest opposite groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BLiMP | -0.074 | +0.898 | 0.282 | 0.244 | 0.282 | 0.244 | 0.142 | 1716 | existential_there_quantifiers_2 s=+32 u=-250; principle_A_reconstruction s=+110 u=-35; superlative_quantifiers_2 s=-187 u=+127; wh_island s=-103 u=+149 |
| Supplement | +0.815 | +2.440 | 0.373 | 0.294 | 0.373 | 0.294 | 0.058 | 136 | subject_aux_inversion s=-13 u=+110 |
| EWoK | +1.492 | -0.479 | 0.284 | 0.256 | 0.284 | 0.256 | 0.540 | 565 | physical-dynamics s=+15 u=-1; quantitative-properties s=+13 u=-3; agent-properties s=-9 u=+15 |
| Entity | -0.428 | -0.280 | 0.199 | 0.374 | 0.199 | 0.374 | 0.703 | 208 | move_contents_2_ops s=+1 u=-10; regular_3_ops s=+2 u=-7; move_contents_4_ops s=-2 u=+10; move_contents_5_ops s=-1 u=+1 |
| COMPS | -0.287 | +0.528 | 0.248 | 0.243 | 0.248 | 0.243 | 0.268 | 6006 | wugs s=-53 u=+152; wugs_dist_before s=-67 u=+66 |
| GlobalPIQA | +2.956 | +4.456 | 0.222 | 0.125 | 0.222 | 0.125 | 1.000 | 6 |  |

## Scientific reading

- Low repair/damage overlap means the two early gains are mostly different item decisions rather than one common easy-score movement. This can support later combined-mechanism reasoning only after the 100M endpoints show that the individual mature effects are real, because several previous early gains reversed at maturity.
- Strong opposite groups are especially important: a combined route is scientifically plausible only if one mechanism repairs the other's mature losing families without destroying the winning families. These 20M overlaps therefore generate hypotheses for interpreting the arriving 100M endpoints; they are not evidence that a combined training run would succeed.
- The scale1.75 100M endpoint has now finished training and requires full evaluation. If full evaluation clears 41.8, independent legality/reproducibility/submission checks are needed before treating it as the study's accepted endpoint.

JSON: `experiments/archive/frontier_consolidation/data/dual_mechanism_20m_overlap/dual_mechanism_20m_overlap.json`
