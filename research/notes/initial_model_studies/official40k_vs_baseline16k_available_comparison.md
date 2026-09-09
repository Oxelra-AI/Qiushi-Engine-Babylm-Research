# official40k accum training validation — official40k vs baseline16k DeBERTa available-column comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/official40k_vs_baseline16k_available_comparison.json`

official40k preserves the 8x480 non-embedding backbone but increases total parameters through embeddings and was trained with microbatch128 accumulation2.

| column | baseline16k | official40k | delta | leader | 40k-leader |
|---|---:|---:|---:|---:|---:|
| BLiMP | 66.76 | 66.65 | -0.11 | 67.20 | -0.55 |
| Supplement | 59.88 | 59.27 | -0.61 | 56.01 | +3.26 |
| Entity | 22.62 | 22.16 | -0.46 | 28.45 | -6.29 |
| COMPS | 52.19 | 52.32 | +0.13 | 53.57 | -1.25 |
| GlobalPIQA parallel | 24.27 | 26.21 | +1.94 | — | — |
| GlobalPIQA nonparallel | 47.00 | 47.00 | +0.00 | — | — |
| GlobalPIQA mean | 35.63 | 36.61 | +0.97 | 39.67 | -3.06 |
| Reading eye | 9.95 | 1.74 | -8.21 | — | — |
| Reading self-paced | 5.29 | 0.04 | -5.25 | — | — |
| Reading mean | 7.62 | 0.89 | -6.73 | 5.42 | -4.53 |
