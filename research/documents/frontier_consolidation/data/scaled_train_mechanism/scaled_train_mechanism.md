# adapter scale sweep plan scaled-train mechanism readout

| run | cfg scale | adapter RMS mean/max | up norm sum | down norm sum | stock relL2 vs spatial repair route status | stock cosine vs spatial repair route status |
|---|---:|---:|---:|---:|---:|---:|
| scale1_train_eval1 | 1.00 | 0.056568/0.069869 | 28.231 | 66.339 | 0.4681 | 0.89024 |
| scale1p75_train | 1.75 | 0.071835/0.079807 | 26.201 | 65.561 | 0.4913 | 0.87907 |
| scale2p00_train | 2.00 | 0.076060/0.085352 | 25.184 | 65.187 | 0.5003 | 0.87449 |

## Pairwise stock displacement
- scale1p75_train vs adapter matched horizon plan scale1 stock: relL2 0.4696, cosine 0.88971.
- scale2p00_train vs adapter matched horizon plan scale1 stock: relL2 0.4794, cosine 0.88498.
- scale2.00 vs scale1.75 stock: relL2 0.4598, cosine 0.89424.
