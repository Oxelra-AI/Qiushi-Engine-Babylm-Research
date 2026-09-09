# structure density pretrain validation — Structure-density selection: pre-training validation

## Design constraints

All arms must match unique corpus amount, repetition distribution, and training geometry, and that the separation is genuinely structure, not source or vocabulary topic.

Evidence: `data/structure_density_revision_157/manifest.json`

### Source proportions

All arms stay within 3.7% max deviation from the uniform reference:

| arm | max source deviation from uniform |
|---|---:|
| high_entity_state | 1.89% |
| high_physical | 0.29% |
| high_social_causal | 3.69% |
| matched_low | 0.71% |

### Unique budget and repetition

| arm | unique words | repetition | achieved exposure |
|---|---:|---:|---:|
| high_entity_state | 2,410,655 | 4.15× | 9,999,947 |
| high_physical | 2,496,526 | 4.01× | 9,999,960 |
| high_social_causal | 2,412,315 | 4.15× | 9,999,992 |
| matched_low | 2,502,046 | 4.00× | 9,999,960 |
| uniform | 2,497,380 | 4.00× | 9,999,997 |

Spread: 3.7% unique budget, ~4% repetition difference between highest and lowest.

### Structure density separation (score means per window)

| arm | entity_state | physical | social_causal | composite |
|---|---:|---:|---:|---:|
| high_entity_state | **0.266** | 0.072 | 0.066 | 0.405 |
| high_physical | 0.170 | **0.128** | 0.053 | 0.350 |
| high_social_causal | 0.192 | 0.071 | **0.106** | 0.369 |
| matched_low | 0.112 | 0.061 | 0.037 | **0.210** |
| uniform | 0.178 | 0.077 | 0.059 | 0.315 |

Key separations:
- entity_state: high 0.266 vs low 0.112 (2.4×)
- physical: high 0.128 vs low 0.061 (2.1×)
- social_causal: high 0.106 vs low 0.037 (2.9×)

### Residual confounds (noted, not blocking)

1. **Repetition**: high_entity_state and high_social_causal repeat ~4.15× vs ~4.0× for others. This is a 4% difference in unique coverage that could slightly favor or penalize repeated arms. The effect direction is ambiguous (more repetition can help memorization but hurt generalization).

2. **Cross-axis correlation**: high_entity_state also has somewhat elevated composite (0.405 vs uniform 0.315) because entity-rich windows tend to have other cues. The matched_low arm has low composite (0.210), so it is genuinely low on all axes.

3. **Source proportion**: high_social_causal has 3.69% max deviation (slightly more simple_wiki, slightly less gutenberg). This is small but detectable.

### Verdict: acceptable for first screen

The 2-3× density separation on target axes is much larger than the residual ~4% confounds. If any arm shows a 2+ point gain on Entity, EWoK, or GlobalPIQA over matched_low, the confounds cannot explain it. If effects are <1 point, a tighter design or multi-seed replication would be needed.

## Training plan

First decisive comparison: `high_entity_state` vs `matched_low` vs `uniform`

- Fixed base: S1 DeBERTa-v2 12×384/intermediate1280, baseline16k, AdamW, flat WWM, seq 256, batch 256/micro 128
- Exposure: each arm's achieved exposure (~10M)
- lr_total_steps: 0 (auto, should be ~830 steps for all arms since unique budgets are similar)
- Seeds: 42/456/789
- Checkpoint at full exposure

If entity_state shows Entity/EWoK gain over matched_low without large BLiMP/Reading damage, it supports the density principle. If not, the bottleneck may be in the objective (plain WWM doesn't force the model to use relation structure), requiring the second-stage correspondence-aware masking test.
