# earlier analysis causal GPT compact-vs-repeat selected contrast

Common endpoints: 2; compact active-token exposure asymmetry: +0.2216% relative to repeat.

Mean Δcheap7: +0.000000; positive cheap7 endpoints: 1/2; broad positive endpoints (Δcheap7>0 and ≥4 positive columns): 1/2.

Mean Δcheap6(no GlobalPIQA): +0.166667; mean Δcheap5(no GlobalPIQA/Reading): +0.150000; mean Δrelation_state(EWoK+Entity): +0.375000.

Best endpoint: chck_20M Δcheap7=+0.500000, +cols=4/7; worst endpoint: chck_50M Δcheap7=-0.500000.

| endpoint | compact | repeat | Δcheap7 | Δcheap6 noG | Δcheap5 noG/R | Δrel/state | Δsyntax | Δvolatile | +cols | max +col share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_20M | 40.000000 | 39.500000 | +0.500000 | 0.416667 | 0.300000 | -0.250000 | 0.666667 | 1.000000 | 4 | 0.250 |
| chck_50M | 41.000000 | 41.500000 | -0.500000 | -0.083333 | 0.000000 | 1.000000 | -0.666667 | -0.250000 | 2 | 0.500 |

Interpretation: this comparison can support the compact-view principle only if compact wins across multiple checkpoints and families. A spike in GlobalPIQA/Reading or one grammaticality family remains a redistribution endpoint.

JSON: `experiments/archive/frontier_consolidation/data/causal_comparator_smoke/out/causal_compact_repeat_selected_contrast.json`
