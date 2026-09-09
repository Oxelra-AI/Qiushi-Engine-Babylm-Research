# earlier analysis causal GPT compact-vs-repeat selected contrast

Common endpoints: 6; compact active-token exposure asymmetry: +0.2216% relative to repeat.

Mean Δcheap7: +0.077738; positive cheap7 endpoints: 3/6; broad positive endpoints (Δcheap7>0 and ≥4 positive columns): 2/6.

Mean Δcheap6(no GlobalPIQA): -0.144583; mean Δcheap5(no GlobalPIQA/Reading): -0.151333; mean Δrelation_state(EWoK+Entity): -0.319167.

Best endpoint: chck_82M Δcheap7=+0.550000, +cols=4/7; worst endpoint: chck_70M Δcheap7=-0.692857.

| endpoint | compact | repeat | Δcheap7 | Δcheap6 noG | Δcheap5 noG/R | Δrel/state | Δsyntax | Δvolatile | +cols | max +col share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_20M | 38.081429 | 38.194286 | -0.112857 | 0.282500 | 0.236000 | 0.345000 | 0.163333 | -0.985000 | 5 | 0.543 |
| chck_50M | 40.149286 | 40.212143 | -0.062857 | -0.237500 | -0.318000 | 0.650000 | -0.963333 | 0.575000 | 5 | 0.323 |
| chck_70M | 39.903571 | 40.596429 | -0.692857 | -0.563333 | -0.594000 | -1.225000 | -0.173333 | -0.940000 | 2 | 0.931 |
| chck_82M | 40.821429 | 40.271429 | +0.550000 | -0.103333 | -0.058000 | -0.230000 | 0.056667 | 2.070000 | 4 | 0.875 |
| chck_90M | 40.675714 | 40.347143 | +0.328571 | -0.114167 | -0.076000 | -0.725000 | 0.356667 | 1.340000 | 3 | 0.727 |
| chck_100M | 40.711429 | 40.255000 | +0.456429 | -0.131667 | -0.098000 | -0.730000 | 0.323333 | 1.842500 | 4 | 0.804 |

Interpretation: this comparison can support the compact-view principle only if compact wins across multiple checkpoints and families. A spike in GlobalPIQA/Reading or one grammaticality family remains a redistribution endpoint.

JSON: `experiments/archive/frontier_consolidation/data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.json`
