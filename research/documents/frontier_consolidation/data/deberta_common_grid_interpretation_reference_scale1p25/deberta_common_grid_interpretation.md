# deberta grid tooling and topology packing confound DeBERTa common-grid trajectory interpreter

This file is a fixed score-readout tool for the 70M--100M selected DeBERTa grid. It records measurements only; it does not promote any route by itself.

## Trajectory summaries

| label | status | best | best cheap7 | 82M cheap7 | 100M cheap7 | 82→100 Δ | final-best Δ | high-band span M | connected span M | local peak excess | column peak spread M |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | ok | chck_84M | 44.123571 | 43.958571 | 43.542143 | -0.416429 | -0.581429 | 2.0 | 2.0 | 0.258929 | 20.0 |
| scale1p25_seed43022_dense | ok | chck_86M | 43.537857 | 43.331429 | 43.349286 | 0.017857 | -0.188571 | 16.0 | 2.0 | 0.271429 | 18.0 |

## Adapter energy context

| label | scale | mean effective up/stock | ratio to reference | 82M energy | 100M energy | 82→100 energy Δ | slope per 2M | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | 1.75 | 0.104429 | 1.000 | 0.104443 | 0.104487 | 0.000044 | 0.000019 | 16 |
| scale1p25_seed43022_dense | 1.25 | 0.081414 | 0.780 | 0.081440 | 0.081476 | 0.000035 | 0.000020 | 16 |

## Comparisons against reference

| label | common | mean Δcheap7 | mean Δcheap6(no GlobalPIQA) | mean Δcheap5(no GP/Reading) | mean ΔEWoK/Entity | mean Δvolatile | positive Δcheap7 rows | wide-family positive rows | best Δ endpoint | best Δcheap7 | volatile gain share at best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| scale1p25_seed43022_dense | 16 | -0.410982 | -0.815208 | -0.949125 | -0.425313 | 0.934375 | 1 | 0 | chck_90M | 0.045714 | 0.980 |

## Reading the measurements

- scale1p25_seed43022_dense: best shift vs reference = +2.0M
- scale1p25_seed43022_dense: high-band span change vs reference = +14.0M
- scale1p25_seed43022_dense: connected high-band span change vs reference = +0.0M.
- scale1p25_seed43022_dense: mean deltas cheap7=-0.410982, cheap6(no GlobalPIQA)=-0.815208, EWoK/Entity=-0.425313.
- scale1p25_seed43022_dense: mean effective adapter-up/stock ratio is 0.780 of the reference mean.
- Read this as the seed/mask-matched lower-residual-energy contrast, not as a new data intervention.
- Reference 82M-to-100M cheap7 movement in the common grid: -0.416429.
- A useful amplitude/timing result should move the high-score band or reduce late falloff while preserving broad family support; volatile-column-only movement remains weak evidence.

Important: run `selected_mlm_integrity_check.py` on each selected-evaluation output before treating these measurements as evidence. No leaderboard submission is part of this workflow.

JSON: `experiments/archive/frontier_consolidation/data/deberta_common_grid_interpretation_reference_scale1p25/deberta_common_grid_interpretation.json`
