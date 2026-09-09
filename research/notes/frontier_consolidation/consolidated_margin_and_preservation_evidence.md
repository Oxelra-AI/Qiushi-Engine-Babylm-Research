# consolidated margin and preservation evidence — EWoK margin + relational preservation: consolidated route evidence

## EWoK per-item margin analysis

Five models scored on all 7,618 EWoK items with per-item PLL margins:
clean43022, clean43122, reinvest43022, reinvest43122, reinvest43022_90M.
Domain accuracies match official scores exactly, validating the scorer.

### Finding 1: late-checkpoint dynamics are negligible
All 90M→100M mean margin deltas < 0.02; frac_100_better ≈ 0.50; net flips tiny.
**Late-stopping and checkpoint-averaging are ruled out as EWoK repair routes.**
The relational knowledge pattern is formed by 90M and doesn't change in the last 10M.

### Finding 2: dynamics-vs-properties split in treatment interaction
Per-item 2×2 DiD shows a clean separation:
- **Positive DiD** (seed43022 benefits more): material-dynamics +0.564, physical-dynamics +0.423, spatial-relations +0.255, physical-relations +0.222, physical-interactions +0.102 — all DYNAMIC/RELATIONAL domains
- **Negative DiD** (seed43122 benefits more): social-interactions -0.398, material-properties -0.363, social-properties -0.270 — STATIC/PROPERTY domains
- **Near-zero**: agent-properties -0.009, quantitative-properties -0.016

### Finding 3: confidently wrong, not diffuse wandering
In negatively-affected domains, reinvest43122 shows large confident-wrong fractions:
- spatial-relations: 44.9% confidently wrong (|margin|>1.0 in wrong direction)
- material-dynamics: 34.0% confidently wrong (vs 21.0% for reinvest43022)
The model has learned directionally wrong associations, not uncertain predictions.

### Finding 4: treatment flips seed advantage in material-dynamics
- Clean: seed43022=48.3%, seed43122=54.3% (seed43122 BETTER)
- Reinvest: seed43022=58.2%, seed43122=46.6% (seed43022 MUCH BETTER)
The compact-view reinvestment qualitatively reverses which seed is better at
material dynamics. This cannot be explained by data quality alone.

## Relational content preservation in compact views

Analyzed all 18,682 accepted compact rewrites. General word retention: 61.0%.

Corrected per-category retention (separating copulas from property adjectives):
| Category          | Retention | Excess vs general |
|-------------------|-----------|-------------------|
| Dynamic verbs     | 91.1%     | +30.1%            |
| Causal markers    | 89.4%     | +28.4%            |
| Property adjectives| 81.3%    | +20.3%            |
| Temporal markers  | 69.2%     | +8.2%             |
| **Spatial preps** | **54.8%** | **-6.2%**         |
| Copulas           | 33.5%     | -27.5%            |

**Spatial prepositions are the only content-word category preserved below the
general retention rate.** Dynamic verbs, property adjectives, and causal markers
are all well-preserved (81-91%).

### Mechanism interpretation

The compact transformation is a good content preserver — it retains essential
meaning-bearing words. But it disproportionately drops spatial relational markers
(above, below, inside, between, through, etc.). Since physical/dynamic EWoK domains
test understanding of physical processes in spatial contexts, this selective spatial
loss creates an unstable foundation:

- The model sees the ACTION (break, fall, melt) → preserved at 91%
- But loses the spatial/physical CONTEXT (above, inside, between) → preserved at only 55%
- For some seeds, the remaining context is enough to learn correct associations
- For other seeds, the model learns confidently wrong associations

This is fundamentally a learning-dynamics issue, not a data-quality issue. The compact
views provide a degraded but not destroyed signal for physical/spatial relations, and
optimization dynamics amplify or suppress this degradation depending on initialization.

## Artifacts

- Per-item margins: `ewok_margins/{model}/ewok_margins.{json,csv}`
- Margin analysis: `ewok_margin_analysis/ewok_margin_analysis.{json,md}`
- Relational preservation: `relational_preservation/relational_preservation.{json,md}`
- Interpretation: `ewok_margin_analysis_interpretation.md`

## Route implications

1. **Late-stopping route is closed** for EWoK
2. **Broad force-filtering is not well-targeted** (earlier analysis hazard alignment + this analysis)
3. **Spatial preposition preservation** is the most specific benchmark-independent repair
   criterion: compact rows that drop spatial preps should be preferentially replaced
   with less-compressed or verbatim views
4. However, the repair is small in magnitude (6% below general for one marker type)
   and the problem is fundamentally about learning dynamics × data interaction
5. The current frozen endpoint at 42.0331 is above-leader and should be preserved
6. The most impactful next work may be improving OTHER weak columns (GlobalPIQA,
   AoA, Reading) rather than chasing EWoK stability
