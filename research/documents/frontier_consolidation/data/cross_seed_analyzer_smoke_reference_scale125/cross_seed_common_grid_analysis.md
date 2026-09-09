# lead cross seed decision framework cross-seed/common-grid structural analyzer

This readout was defined before reading the seed43122 grid. It treats per-column peak-vector structure as the primary evidence for whether the late phase is structural or stochastic.

## Trajectories

| label | status | best | best cheap7 | 82M | 84M | 100M | 82→100 Δ | connected near-best span M | local peak excess | cheap column peak spread M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | ok | chck_84M | 44.123571 | 43.958571 | 44.123571 | 43.542143 | -0.416429 | 2.000000 | 0.258929 | 20.000000 |
| scale1p25_seed43022_dense | ok | chck_86M | 43.537857 | 43.331429 | 43.500714 | 43.349286 | 0.017857 | 2.000000 | 0.153929 | 18.000000 |

## Column peak vectors

| label | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | 94 | 76 | 74 | 88 | 78 | 84 | 78 |
| scale1p25_seed43022_dense | 88 | 96 | 78 | 86 | 88 | 84 | 94 |

## Comparisons to reference

| label | aggregate peak shift M | mean abs column peak shift M | peak Pearson | peak Spearman | other spread M | mean Δcheap7 | mean Δcheap6(no GP) | mean Δcheap5(no GP/Reading) | mean ΔEWoK/Entity | positive cheap7 rows | broad positive rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p25_seed43022_dense | 2.000000 | 8.285714 | -0.047785 | -0.027273 | 18.000000 | -0.410982 | -0.815208 | -0.949125 | -0.425313 | 1 | 0 |

## Predeclared reading

- Structural aligned: column peak vectors correlate and aggregate peak also aligns; study residual-capacity allocation dynamics.
- Structural but misaligned: column dispersion resembles reference but aggregate peak shifts; study allocation dynamics plus stochastic stabilization.
- Unstructured: column peak vectors do not correlate and no similar late broad phase appears; treat seed43022 chck84 as fragile endpoint and study stochastic competence stabilization before new training.
- Any stabilization or averaging probe must improve cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, and nonnegative EWoK/Entity; volatile-column-only gains are not enough.

JSON: `experiments/archive/frontier_consolidation/data/cross_seed_analyzer_smoke_reference_scale125/cross_seed_common_grid_analysis.json`
