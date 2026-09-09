# corrected multiseed gauge affine and next raw binding frozen-affine scalar probe from saved no-anchor outputs

This is a CPU-only representational check: no weights are updated; one-dimensional affine rules are fit from direct h0/h2 eval anchor coordinates and tested on h1/h3 graph-transfer rows.

## tied_bs+1_noanchor_seed29000
train_cmp=0.875 heldheld_closure=0.625 raw_direct=0.0 raw_graph=0.25 d_direct=[-8.166,8.764] d_graph=[-7.939,15.642]

| fit | bs target | calib acc | graph same canon | graph same arm | graph all canon | graph all arm | xt graph canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_threshold | 1 | 1.000 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| best_threshold | -1 | 1.000 | 0.250 | 0.750 | 0.250 | 0.750 | 0.250 |
| least_squares | 1 | 1.000 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| least_squares | -1 | 1.000 | 0.250 | 0.750 | 0.250 | 0.750 | 0.250 |

