# evaluation runbook and interpretation selected MLM trajectory comparison

Reference: `ref`. Near-best band uses cheap7 within 0.200 of each trajectory peak.

## Trajectory summaries

| label | n | best | best cheap7 | final-best | near-best endpoints | cheap6(no GPIQA) at best | EWoK/Entity at best |
|---|---:|---|---:|---:|---|---:|---:|
| ref | 2 | chck_72M | 43.000000 | +0.000000 | chck_72M | 44.000000 | 38.500000 |
| alt | 2 | chck_72M | 43.300000 | +0.000000 | chck_72M | 44.016667 | 38.650000 |

## Contrasts against reference

| label | common | mean Δcheap7 | mean Δcheap6(no GPIQA) | mean ΔEWoK/Entity | positive Δcheap7 rows | broad rows | best Δ endpoint | best Δcheap7 | GPIQA gain share at best |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| alt | 2 | +0.235714 | +0.108333 | +0.100000 | 2 | 2 | chck_72M | +0.300000 | 0.800 |

Interpretation note: the file records measurements only. The scale1.25 run should be read against the seed/mask-matched reference; the scale1.75 seed43122 run cannot separate model initialization from mask-stream randomness.

JSON: `experiments/archive/frontier_consolidation/data/comparator_smoke/mlm_compare/selected_mlm_trajectory_comparison.json`
