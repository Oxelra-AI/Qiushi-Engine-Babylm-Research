# routeB closure and scale1p75 eval — Scale-1.75 GlobalPIQA Hard-Surface Anatomy

## Key finding: First real hard-margin improvement

The adapter128 scale-1.75 80M model is the **first compliant model** to show
substantial movement on GlobalPIQA hard ranks and margins. This is qualitatively
different from all previous attempts.

## Comparison: scale-1.75 80M vs anchor legal40k 80M

### Parallel (103 rows, 4-choice)
| Metric | Scale-1.75 | Anchor | Delta |
|--------|-----------|--------|-------|
| Accuracy | **26.21%** | 22.33% | **+3.88** |
| Hard52 accuracy | 1.92% (1/52) | 0.00% (0/52) | +1 row |
| Hard52 mean margin | **1.717 nats** | 1.957 nats | **-0.240 (improved)** |
| Hard52 median margin | **1.371 nats** | 1.805 nats | **-0.434 (improved)** |
| Hard52 rank2 | 10 | 8 | +2 |
| Hard52 rank3 | 21 | 23 | -2 (improved) |
| Hard52 rank4 | 20 | 21 | -1 (improved) |
| All rows rank1 | **27** | 23 | +4 |
| All rows mean margin | **1.043** | 1.259 | **-0.216 (improved)** |

### Nonparallel (100 rows, 2-choice)
| Metric | Scale-1.75 | Anchor | Delta |
|--------|-----------|--------|-------|
| Accuracy | **50.0%** | 47.0% | **+3.0** |

### Category breakdown (parallel)
| Category (n) | Scale-1.75 acc | Anchor acc | Delta |
|--------------|---------------|-----------|-------|
| tool_affordance (8) | **37.5%** | 0.0% | **+37.5** |
| direction_spatial (12) | **16.7%** | 8.3% | **+8.3** |
| physical_object (73) | **31.5%** | 26.0% | **+5.5** |
| spatial (22) | **13.6%** | 9.1% | **+4.5** |
| affordances (21) | **28.6%** | 23.8% | +4.8 |
| object_properties (36) | **36.1%** | 33.3% | +2.8 |
| time, counting (7) | 0.0% | 0.0% | 0.0 |

## Interpretation

1. **Hard52 margin reduction (-0.240 nats)** is the largest observed in any compliant model.
   Previous attempts: reheat +0.02, mature avg +0.00, role-switch +0.10 (worsened).

2. **Tool affordance category flipped** from 0% to 37.5%. This category requires
   understanding which tool is appropriate for which task under different contexts.

3. **Direction_spatial doubled** from 8.3% to 16.7%. These rows test spatial relationship
   alternatives under competing contexts.

4. **The movement is broad**: 4 more rank-1 placements overall, 2 more rank-2 placements
   in hard52, consistent margin reduction, and category-wide improvement.

5. **The adapter mechanism**: amplified residual adapters (scale 1.75) create a
   separate learning channel that appears to acquire some context-conditioned
   alternative discrimination that the base MLM backbone alone cannot.

## Remaining weakness: EWoK decline
- Scale-1.75 80M: EWoK 49.24 vs aoa mincontext discrepancy audit 80M: 51.01 (Δ = -1.77)
- The adapter improves 4-choice context binding but worsens binary EWoK
- This may indicate partial/asymmetric context sensitivity that helps
  multi-option discrimination more than binary plausibility judgment

## Implications for SOTA
If the 100M model maintains this hard-margin trend while SuperGLUE is adequate:
- Overall could cross 41.80
- The scale-1.75 adapter would be the first architecture to simultaneously
  improve broad cheap7 AND reduce the context-conditioned binding deficit
