# consolidated margin and preservation evidence — EWoK Per-Item Margin Analysis: Scientific Interpretation

## Key Findings

### 1. Late-checkpoint dynamics are negligible for EWoK

All 90M→100M mean margin deltas are < 0.02 in absolute value. The frac_100_better
hovers near 0.50 for every domain. Net item flips range from -10 to +14, out of
hundreds of items per domain. **The EWoK relation instability was already present
by 90M; the last 10M words of training do not create or fix it.** This rules out
late-training destabilization as the mechanism and closes the late-stopping route
for EWoK improvement.

### 2. Dynamics vs Properties split in the treatment interaction

The per-item 2×2 DiD reveals a clean separation between two domain families:

**Positive DiD (seed43022 benefits more from treatment)**:
| Domain               | TE43022 | TE43122 | DiD    |
|----------------------|---------|---------|--------|
| material-dynamics    | +0.341  | -0.224  | +0.564 |
| physical-dynamics    | +0.375  | -0.048  | +0.423 |
| spatial-relations    | +0.072  | -0.183  | +0.255 |
| physical-relations   | +0.152  | -0.069  | +0.222 |
| physical-interactions| +0.034  | -0.069  | +0.102 |

These are all **dynamic/relational** domains — they test reasoning about processes,
changes, spatial positions, and physical causation.

**Negative DiD (seed43122 benefits more or both benefit)**:
| Domain               | TE43022 | TE43122 | DiD    |
|----------------------|---------|---------|--------|
| material-properties  | +0.142  | +0.505  | -0.363 |
| social-properties    | +0.125  | +0.396  | -0.270 |
| social-interactions  | -0.435  | -0.037  | -0.398 |

These are about **static properties/knowledge** — what things are made of, social
roles, beliefs.

**Near-zero DiD (stable across seeds)**:
- agent-properties (-0.009), quantitative-properties (-0.016), social-relations (-0.047)

### 3. Confidently wrong, not diffuse wandering

In the negatively-affected domains, seed43122 reinvest doesn't just show near-zero
margins. It shows large fractions of **confidently wrong** predictions:

| Domain               | reinvest43022 conf_wrong | reinvest43122 conf_wrong |
|----------------------|--------------------------|--------------------------|
| material-dynamics    | 0.210                    | 0.340                    |
| spatial-relations    | 0.345                    | 0.449                    |
| physical-interactions| 0.304                    | 0.340                    |

In spatial-relations, nearly half (0.449) of reinvest43122 items are confidently
wrong (|margin| > 1.0 in the wrong direction). This is not uncertainty — the model
has learned directionally wrong associations.

### 4. Treatment qualitatively reverses seed advantage

Material-dynamics is the clearest example:
- Clean baseline: seed43022 = 48.3%, seed43122 = 54.3% (seed43122 is BETTER)
- Reinvest treatment: seed43022 = 58.2%, seed43122 = 46.6% (seed43022 is MUCH BETTER)

The compact-view reinvestment **flips which seed performs better** at material
dynamics. This can't be explained by data quality alone — the same corpus produces
opposite treatment directions at different initializations.

### 5. Flip structure

Seed43022 treatment flip analysis (clean→reinvest):
- material-dynamics: 127 lost, 203 gained, **+76 net** (largest net gain)
- social-interactions: 55 lost, 38 gained, **-17 net** (largest net loss)

Seed43122 treatment flip analysis (clean→reinvest):
- material-dynamics: 192 lost, 133 gained, **-59 net** (largest net loss)
- material-properties: 8 lost, 34 gained, **+26 net** (large gain)

## Scientific Interpretation

The margin microscope reveals that compact-view reinvestment creates a
**dynamics-vs-properties learning asymmetry** that interacts with initialization:

1. **Property knowledge** (material-properties, social-properties) improves
   stably across seeds. The reinvested source diversity teaches the model
   more facts about what things are and what they do. This is robust because
   static associations can be learned even from compressed views.

2. **Dynamic/relational knowledge** (material-dynamics, physical-dynamics,
   spatial-relations) improves strongly at one seed but degrades at another.
   The compact views may damage temporal sequences, causal chains, and
   spatial transformation descriptions, creating seed-dependent relational
   representations. The items aren't "barely wrong" — they're confidently
   wrong, meaning the model learned wrong relational associations.

3. **Social-interactions** is a special case: seed43022 loses TE=-0.435 while
   seed43122 is near zero. This may reflect loss of dialogue and social
   context that happened to be well-represented in the non-Qwen rows
   removed to make room for FineWeb.

## Implications for repair

1. **Late stopping is ruled out** for EWoK improvement. The relational
   patterns are formed early and don't change meaningfully in the last 10M.

2. **Broad semantic-force filtering would not target the right mechanism.**
   The problem is not that hazardous rows cluster in physical domains
   (earlier analysis hazard alignment showed weak correlation). The problem is
   that the compact transformation systematically degrades dynamic
   relational markers (verbs of change, spatial prepositions, causal
   connectives) while preserving static property markers, and this
   degradation interacts with optimization dynamics.

3. **A benchmark-independent relational preservation criterion** should
   target rows whose compact views lose dynamic relational structure:
   - Dynamic verb preservation (break→broke, melt→melts, flow→flowing)
   - Spatial preposition chain preservation (above, below, inside, between)
   - Causal connective preservation (because, so, therefore)
   - Temporal sequence preservation (then, after, before)
   These are properties of the training views, not of the evaluation.

4. **Mixing verbatim repetition for dynamic-rich rows** with compaction
   for static-rich rows could stabilize the treatment effect across seeds
   while preserving the reinvested source diversity that drives property gains.

## Preserved artifacts

- Per-item margins (all 5 models): `ewok_margins/{model}/ewok_margins.json`
- Comprehensive analysis: `ewok_margin_analysis/ewok_margin_analysis.json`
- This note: `ewok_margin_analysis_interpretation.md`
