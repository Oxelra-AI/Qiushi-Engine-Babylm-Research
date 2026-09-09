# deberta grid tooling and topology packing confound DeBERTa common-grid trajectory interpreter

This file is a fixed score-readout tool for the 70M--100M selected DeBERTa grid. It records measurements only; it does not promote any route by itself.

## Trajectory summaries

| label | status | best | best cheap7 | 82M cheap7 | 100M cheap7 | 82→100 Δ | final-best Δ | high-band span M | local peak excess | column peak spread M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | ok | chck_82M | 44.414286 | 44.414286 | 43.814286 | -0.600000 | -0.600000 | 6.0 | 0.175000 | 0.0 |
| scale1p25_seed43022_dense | ok | chck_82M | 44.464286 | 44.464286 | 43.864286 | -0.600000 | -0.600000 | 6.0 | 0.175000 | 0.0 |
| scale1p75_seed43122_dense | ok | chck_82M | 44.412857 | 44.412857 | 43.812857 | -0.600000 | -0.600000 | 6.0 | 0.175000 | 0.0 |

## Adapter energy context

| label | scale | mean effective up/stock | ratio to reference | n |
|---|---:|---:|---:|---:|
| scale1p75_seed43022_reference | 1.75 | 0.104429 | 1.000 | 16 |
| scale1p25_seed43022_dense | 1.25 | 0.081414 | 0.780 | 16 |
| scale1p75_seed43122_dense | 1.75 | 0.105149 | 1.007 | 16 |

## Comparisons against reference

| label | common | mean Δcheap7 | mean Δcheap6(no GlobalPIQA) | mean Δcheap5(no GP/Reading) | mean ΔEWoK/Entity | mean Δvolatile | positive Δcheap7 rows | wide-family positive rows | best Δ endpoint | best Δcheap7 | volatile gain share at best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| scale1p25_seed43022_dense | 16 | 0.050000 | 0.050000 | 0.050000 | 0.050000 | 0.050000 | 16 | 16 | chck_74M | 0.050000 | 0.286 |
| scale1p75_seed43122_dense | 16 | -0.001429 | -0.030000 | -0.030000 | -0.030000 | 0.070000 | 0 | 0 | chck_70M | -0.001429 | 1.000 |

## Reading the measurements

- scale1p25_seed43022_dense: best shift vs reference = +0.0M
- scale1p25_seed43022_dense: high-band span change vs reference = +0.0M
- scale1p25_seed43022_dense: mean deltas cheap7=+0.050000, cheap6(no GlobalPIQA)=+0.050000, EWoK/Entity=+0.050000.
- scale1p25_seed43022_dense: mean effective adapter-up/stock ratio is 0.780 of the reference mean.
- Read this as the seed/mask-matched lower-residual-energy contrast, not as a new data intervention.
- scale1p75_seed43122_dense: best shift vs reference = +0.0M
- scale1p75_seed43122_dense: high-band span change vs reference = +0.0M
- scale1p75_seed43122_dense: mean deltas cheap7=-0.001429, cheap6(no GlobalPIQA)=-0.030000, EWoK/Entity=-0.030000.
- Read this as joint initialization plus mask-stream robustness, not as an isolated seed-only contrast.
- Reference 82M-to-100M cheap7 movement in the common grid: -0.600000.
- A useful amplitude/timing result should move the high-score band or reduce late falloff while preserving broad family support; volatile-column-only movement remains weak evidence.

Important: run `selected_mlm_integrity_check.py` on each selected-evaluation output before treating these measurements as evidence. No leaderboard submission is part of this workflow.

JSON: `experiments/archive/frontier_consolidation/data/deberta_interpreter_smoke/out/deberta_common_grid_interpretation.json`
