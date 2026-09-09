# corrected multiseed gauge affine and next raw binding frozen-affine scalar tests from saved outputs

Post-training one-dimensional affine calibration from direct h0/h2 state rows; tested on unanchored h1/h3 graph state rows. This is representational sufficiency, not proof that training learned the affine map.

## tied_heldheld_only_seed28801
condition=tied arm=heldheld_only seed=28801 train_cmp=1.000 hh_closure=1.000 raw_direct=0.000 raw_graph=0.000 d_direct=[-12.893,10.806] d_graph=[-12.847,13.696]

| affine fit | anchor sign | calib | direct canon | graph same canon | graph same arm | graph all canon | graph all arm | xt graph canon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_threshold | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| least_squares | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| best_threshold | -1 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| least_squares | -1 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |

## tied_bs+1_noanchor_seed29000
condition=tied arm=aligned_state_bridge seed=29000 train_cmp=0.875 hh_closure=0.625 raw_direct=0.000 raw_graph=0.250 d_direct=[-8.166,8.764] d_graph=[-7.939,15.642]

| affine fit | anchor sign | calib | direct canon | graph same canon | graph same arm | graph all canon | graph all arm | xt graph canon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_threshold | +1 | 1.000 | 1.000 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| least_squares | +1 | 1.000 | 1.000 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| best_threshold | -1 | 1.000 | 0.000 | 0.250 | 0.750 | 0.250 | 0.750 | 0.250 |
| least_squares | -1 | 1.000 | 0.000 | 0.250 | 0.750 | 0.250 | 0.750 | 0.250 |

