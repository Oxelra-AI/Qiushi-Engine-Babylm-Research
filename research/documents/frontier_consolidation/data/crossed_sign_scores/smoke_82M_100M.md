# Crossed-sign scores: smoke_82M_100M

Probe SHA: `376e0e1f11c5f71f...`  Items: 250
Best composite ckpt: **chck_82M** | Best CS ckpt: **chck_82M** | Official: chck_82M

## Correlations with cheap7

- composite_vs_cheap7_pearson: N/A
- composite_vs_cheap7_spearman: N/A
- crossed_sign_vs_cheap7_pearson: N/A
- crossed_sign_vs_cheap7_spearman: N/A
- n_shared: **2.0000**

## Per-checkpoint

### chck_82M (cheap7=43.9594498765)
  composite=-0.3498  crossed_sign=0.2800
  - agent_action: Δ=1.3028 cs=0.311 bias=-1.543
  - entity_state: Δ=-10.8171 cs=0.000 bias=-1.001
  - property_bind: Δ=0.7705 cs=0.200 bias=0.365
  - spatial_put: Δ=16.0914 cs=1.000 bias=-0.110
  - temporal_order: Δ=-9.0964 cs=0.044 bias=0.350
  - entity_state depth_2: Δ=-12.2095 cs=0.000 n=35
  - entity_state depth_3: Δ=-9.8500 cs=0.000 n=25
  - entity_state depth_4: Δ=-8.3617 cs=0.000 n=10

### chck_100M (cheap7=43.5431599193)
  composite=-0.5195  crossed_sign=0.2640
  - agent_action: Δ=0.6098 cs=0.267 bias=-1.297
  - entity_state: Δ=-11.0559 cs=0.000 bias=-0.957
  - property_bind: Δ=0.8499 cs=0.156 bias=0.321
  - spatial_put: Δ=16.2447 cs=1.000 bias=-0.158
  - temporal_order: Δ=-9.2462 cs=0.044 bias=0.432
  - entity_state depth_2: Δ=-12.4162 cs=0.000 n=35
  - entity_state depth_3: Δ=-10.1230 cs=0.000 n=25
  - entity_state depth_4: Δ=-8.6270 cs=0.000 n=10

## Per-family correlations

- agent_action: N/A
- entity_state: N/A
- property_bind: N/A
- spatial_put: N/A
- temporal_order: N/A
