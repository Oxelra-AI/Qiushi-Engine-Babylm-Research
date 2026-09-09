# dense focus common target analysis Dense Focus Common-Target Analysis

## The key qualitative shift: dense focus IMPROVES source-responsive behavior

### Common-target deltas vs coherent86 parent:

| Metric | Sparse Focus u80 | Dense Focus u80 | Difference |
|--------|-----------------|-----------------|------------|
| Overall mean | -0.0102 | **+0.2079** | +0.218 |
| no_source | +0.018 | +0.020 | +0.002 |
| source_altered | **-0.058** | **+0.340** | **+0.398** |
| source_original | +0.009 | +0.264 | +0.255 |
| held_source | -0.018 | **+0.366** | +0.384 |
| trained_content | +0.017 | +0.133 | +0.116 |

### Interpretation

**Sparse focus (earlier analysis):** Improved practiced-surface reconstruction but
DEGRADED source_altered behavior (-0.058) and had near-zero held_source
change. The model learned to predict familiar Qwen view tokens better
but didn't improve its use of source information.

**Dense focus (this result):** STRONG positive movement on ALL conditions,
including +0.340 on source_altered and +0.366 on held_source. The model
didn't just learn familiar surfaces — it improved its general ability
to predict correct targets in paired content contexts.

The source_altered improvement (+0.340) being larger than source_original
(+0.264) suggests the model became more robust to source perturbation,
not that it learned source-independent shortcuts. The no_source delta
is small (+0.020), confirming the improvement requires some contextual
evidence rather than just memorized priors.

### Context from compact baselines

| Baseline | Overall | source_altered | held_source |
|----------|---------|---------------|-------------|
| compact_equal_epoch | +0.160 | +0.133 | +0.118 |
| compact_wordmatched | +0.200 | +0.166 | +0.152 |
| dense_focus | **+0.208** | **+0.340** | **+0.366** |

Dense focus exceeds compact baselines by ~2× on source_altered and ~2.5×
on held_source. This is the strongest common-target result.

### Why dense focus works

Dense focus gives ALL 132,283 Qwen pair groups focused credit across
80 updates (176,607 total focus targets, 23% of all targets), versus
sparse focus's random 15% sampling (28,590 targets, 4.6%).

The 6× more focus credit means the model sees comprehensive, consistent
training on every paired content word. This may enable it to learn the
full structure of source-rewrite correspondence rather than just local
surface patterns.

### Combined evidence (Cheap7 + common-target)

Dense focus at update 80:
- Cheap7 mean: 44.809 (+0.245 vs coherent86)
- Entity: 28.44 (+0.66 vs coherent86)
- Common-target: +0.208 overall, +0.340 source_altered
- Both BROAD COMPETENCE and MECHANISTIC READOUTS improved

This is qualitatively different from all previous functional_learning late experiments,
which showed either broad improvement without mechanism (ordinary WWM)
or mechanism without broad improvement (sparse focus).
