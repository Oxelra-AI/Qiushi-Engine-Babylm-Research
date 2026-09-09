# dense6x384 10m curve profile fixed — dense6x384 10M exposure curve official-compatible profile

Run: `experiments/archive/initial_model_studies/training/runs/babylm_compare_dense6x384_10M_curve`

Evidence JSON: `experiments/archive/initial_model_studies/data/dense6x384_10m_curve_profile_fixed.json`

Purpose: determine whether the capacity controlled 1m profile and next scale dense6x384 baseline was mainly exposure-limited, or whether Entity/Reading remain mechanism-limited even after the full 10M-word corpus pass.

| revision | BLiMP | Supplement | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| chck_1M | 56.31 | 55.20 | 51.73 | 17.88 | 49.73 | 8.98 | 2.24 |
| chck_5M | 56.31 | 55.20 | 51.73 | 17.88 | 49.73 | 8.98 | 2.24 |
| chck_10M | 56.31 | 55.20 | 51.73 | 17.88 | 49.73 | 8.98 | 2.24 |

## Deltas: final minus first profiled checkpoint

- BLiMP: +0.00 (56.31 → 56.31)
- Supplement: +0.00 (55.20 → 55.20)
- EWoK: +0.00 (51.73 → 51.73)
- Entity: +0.00 (17.88 → 17.88)
- COMPS: +0.00 (49.73 → 49.73)
- Reading eye: +0.00 (8.98 → 8.98)
- Reading SPR: +0.00 (2.24 → 2.24)

## Scientific reading

The 10M dense curve produces a broad improvement in the local NLP probes relative to its own 1M checkpoint, especially BLiMP/Supplement/EWoK/COMPS if their deltas are positive, and must be used as the baseline before judging new mechanisms. However, this does not by itself solve the official BabyLM target: the full nine-column Overall is still unmeasured, and AoA, GlobalPIQA, and full (Super)GLUE remain absent.

Entity Tracking remains a central bottleneck: the improvement from 1M to 10M is small relative to the weakness observed in all earlier 1M candidates. This supports reopening state-memory and/or entity-consistency mechanisms rather than merely increasing dense exposure.

Reading does not show a clear monotone gain with exposure in this fast profile, so developmental/human-like behavior remains unresolved and should shape the next innovation route.

## Next research implication

Use dense6x384-10M as the current exposure baseline, not as the final research idea. The next equal-exposure innovation should target the remaining Entity/Reading/AoA bottleneck, most plausibly a compact state-memory or developmental data-order/objective route from `research/notes/initial_model_studies/entity_reading_aoa_route_reopen_plan.md`, compared against this dense baseline.
