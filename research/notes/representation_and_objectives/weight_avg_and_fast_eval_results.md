# decision framework and clean collation plan — Weight averaging and fast evaluation results

## Cross-seed weight averaging: CATASTROPHIC FAILURE

Cross-seed average of reinv43022 + reinv43122 at chck_100M produced a model
with equal7 = **37.825** through the official babylm-eval pipeline.

Individual seed fast equal7: seed43022=44.29, seed43122=42.93, mean=43.61.
The averaged model scores **5.8 points below the individual mean** and destroys
performance on every column: BLiMP 55.64 (was ~66), Supplement 51.6 (was ~63),
Entity 16.89 (was ~27), GlobalPIQA_parallel 18.45 (was ~25).

**Interpretation**: The two initialization seeds found incompatible weight-space
local optima. Linear interpolation in parameter space does not interpolate
between their functions — it produces an incoherent model. This eliminates
weight averaging as a path to a robust multi-seed endpoint.

## Tail averaging: NEUTRAL

- tail_avg_022 (seed43022, chck_90M-100M): equal7 = **44.27** (individual: 44.29)
- tail_avg_122 (seed43122, chck_90M-100M): equal7 = **42.90** (individual: 42.93)

Tail averaging neither helps nor hurts. Both seeds have already converged to
stable local optima by 90M exposure. No checkpoint consolidation benefit.

### Tail avg 022 column detail (vs seed43022 individual)
| Column | tail_avg_022 | seed43022 | delta |
|--------|-------------|-----------|-------|
| BLiMP | 66.57 | 66.87 | -0.30 |
| Supplement | 66.40 | 63.28 | +3.12* |
| EWoK | 53.82 | 53.54 | +0.28 |
| Entity_full | 27.76 | 27.75 | +0.01 |
| COMPS | 52.01 | 51.97 | +0.04 |
| GlobalPIQA | 35.12 | 35.62 | -0.50 |
| Reading | 8.21 | 8.24 | -0.03 |

*Supplement +3.12 is notable — fast eval vs official may differ in subset.
Overall neutral but tail_avg_022 might be worth full evaluation if Supplement
movement is real at full scale.

## Standalone PLL margin scorer: systematic accuracy deficit

The standalone EWoK PLL scorer (weight_avg_and_ewok_margins.py)
produced accuracies ~3 points below official scorer for all models:
- reinv430: 50.79% (official: 53.54%)
- reinv431: 49.50% (official: 51.89%)
- clean430: 49.97% (official: 50.66%)
- clean431: 49.95% (official: 51.48%)

Likely cause: difference between standalone PLL (mask-all-tokens) and the
official scorer's data preprocessing in dataset.py, possibly including which
tokens are masked and temperature optimization. The margin data's absolute
values are unreliable; relative patterns may still be informative but
must be validated against official scores.

## Decision consequences

1. **Cross-seed averaging eliminated** as a robustness path
2. **Tail averaging neutral** — no checkpoint consolidation benefit
3. **Expected two-seed Overall** remains (42.03+41.25)/2 = 41.64, below leader 41.8
4. **seed43022 remains the single above-leader endpoint** at 42.033
5. **Tail_avg_022 Supplement gain** worth investigating via full evaluation

## Proposed Follow-Up Comparisons

Since averaging and consolidation are ruled out, the remaining paths are:
- **A**: Submit seed43022 as-is (42.033 > 41.8) — valid but single-seed
- **B**: Third pretraining seed (~4h GPU) for better expected-score estimate
- **C**: Investigate tail_avg_022's Supplement gain via full official evaluation

Path A is defensible now. Paths B and C improve confidence without changing
the underlying method.

## Artifacts
- `experiments/archive/representation_and_objectives/data/fast_eval_averaged/fast_eval_averaged_summary.json`
- `experiments/archive/representation_and_objectives/data/weight_avg_and_margins` (models + margins)
- Averaged models: `models/cross_seed_avg_100M/`, `models/tail_avg_seed43022_90_100M/`,
  `models/tail_avg_seed43122_90_100M/`
