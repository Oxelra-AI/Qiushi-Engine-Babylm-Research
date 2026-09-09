# scale1p75 pre80m state scale1.75 50M maturation decision

## Prefix replication
- Embedded 20M prefix exact: **True**
- safetensors hash equal: True
- log prefix equal ignoring elapsed seconds: True
- tensor exact equal: True

## 20M reference rows

| row | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status legal 20M | 39.6636 | 59.690 | 55.450 | 50.730 | 18.650 | 50.260 | 34.195 | 8.670 |
| adapter matched horizon plan trained scale1 evaluated scale1 | 39.3893 | 60.380 | 56.230 | 49.490 | 18.650 | 50.440 | 32.225 | 8.310 |
| adapter matched horizon plan scale1 checkpoint evaluated at scale1.75 | 39.7550 | 60.560 | 57.120 | 50.740 | 18.470 | 50.460 | 32.710 | 8.225 |
| adapter matched horizon plan scale1 checkpoint evaluated at scale2.00 | 39.8193 | 60.610 | 57.350 | 51.080 | 18.460 | 50.330 | 32.710 | 8.195 |
| train-time adapter scale 1.75 | 40.3057 | 59.610 | 56.270 | 52.220 | 18.220 | 49.970 | 37.150 | 8.700 |
| train-time adapter scale 2.00 | 39.7229 | 59.920 | 56.210 | 50.540 | 17.450 | 50.460 | 35.120 | 8.360 |
| train-time adapter scale1.625 20M | 39.5864 | 59.310 | 56.620 | 49.550 | 18.890 | 50.920 | 33.210 | 8.605 |

## Matched 50M rows

| row | cheap7 | Δ cheap7 vs spatial repair route status 50M | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status legal chck_50M | 41.9729 | +0.0000 | 63.800 | 59.110 | 48.460 | 26.260 | 51.610 | 36.580 | 7.990 |
| adapter128 scale1.75 chck_50M | 42.3843 | +0.4114 | 65.640 | 59.070 | 47.730 | 29.050 | 52.340 | 35.105 | 7.755 |

### 50M per-column deltas scale1.75 minus spatial repair route status

| column | delta |
|---|---:|
| BLiMP | +1.840 |
| Supplement | -0.040 |
| EWoK | -0.730 |
| Entity | +2.790 |
| COMPS | +0.730 |
| GlobalPIQA | -1.475 |
| Reading | -0.235 |

## Interpretation

- The 50M run embeds the exact 20M scale1.75 trajectory: same normalized training prefix and bitwise-identical chck_20M model tensors.
- Scale1.75 retains cheap7 advantage +0.4114, but damaging columns remain: {'EWoK': -0.730000000000004, 'GlobalPIQA': -1.4749999999999943}.

Route signal: `analyze_tradeoff_before_80M`
