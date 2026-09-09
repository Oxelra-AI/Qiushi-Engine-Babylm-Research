# answer credit alignment answer-credit alignment test

Continuation starts from the same query-first bound answer-only preparation checkpoints used in accumulated mechanism synthesis.
The two arms keep the context rows and answer-position marginal RWT vocabulary, but train the answer target as bag-independent rather than the queried entity's attribute.
Held entity tokens remain absent from all continuation rows.

## Final means

| arm | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bag_static_1over17 | 0.225 | -0.040 | -0.007 | 0.261 | 0.247 | 1.256 | 1.471 |
| bag_interleaved_ans_full | 0.219 | -0.071 | -0.016 | 0.240 | 0.230 | 1.337 | 1.610 |

## Per-seed trajectories

### Seed 42
**bag_static_1over17**
  be=  0 h4=0.484 hB=+3.833 hSel=+0.365 tr4=0.986 ctx=0.000 cw=0.0
  be=  1 h4=0.477 hB=+3.310 hSel=+0.361 tr4=0.811 ctx=32.456 cw=0.0588
  be=  5 h4=0.184 hB=-0.134 hSel=-0.019 tr4=0.227 ctx=9.680 cw=0.0588
  be= 10 h4=0.199 hB=-0.120 hSel=-0.019 tr4=0.232 ctx=4.176 cw=0.0588
  be= 25 h4=0.199 hB=-0.129 hSel=-0.021 tr4=0.242 ctx=1.695 cw=0.0588
  be= 50 h4=0.266 hB=+0.002 hSel=+0.011 tr4=0.271 ctx=1.391 cw=0.0588
  be= 75 h4=0.266 hB=-0.044 hSel=-0.001 tr4=0.301 ctx=1.370 cw=0.0588
  be=100 h4=0.234 hB=-0.065 hSel=-0.001 tr4=0.258 ctx=1.361 cw=0.0588
  be=125 h4=0.219 hB=-0.072 hSel=-0.008 tr4=0.299 ctx=1.357 cw=0.0588
  be=150 h4=0.199 hB=-0.060 hSel=-0.004 tr4=0.350 ctx=1.350 cw=0.0588
  be=200 h4=0.199 hB=-0.118 hSel=-0.018 tr4=0.279 ctx=1.343 cw=0.0588
  be=250 h4=0.234 hB=-0.120 hSel=-0.011 tr4=0.230 ctx=1.334 cw=0.0588
  be=300 h4=0.203 hB=-0.124 hSel=-0.017 tr4=0.275 ctx=1.325 cw=0.0588
  be=350 h4=0.195 hB=-0.125 hSel=-0.021 tr4=0.291 ctx=1.313 cw=0.0588
  be=400 h4=0.141 hB=-0.128 hSel=-0.020 tr4=0.232 ctx=1.302 cw=0.0588
  be=450 h4=0.121 hB=-0.136 hSel=-0.019 tr4=0.268 ctx=1.283 cw=0.0588
  be=500 h4=0.188 hB=-0.130 hSel=-0.022 tr4=0.297 ctx=1.270 cw=0.0588

**bag_interleaved_ans_full**
  be=  0 h4=0.484 hB=+3.833 hSel=+0.365 tr4=0.986 ctx=0.000 cw=0.0
  be=  1 h4=0.469 hB=+2.517 hSel=+0.344 tr4=0.801 ctx=36.068 cw=0.0
  be=  5 h4=0.449 hB=+0.715 hSel=+0.166 tr4=0.512 ctx=16.454 cw=0.0
  be= 10 h4=0.266 hB=+0.071 hSel=+0.013 tr4=0.285 ctx=6.782 cw=1.0
  be= 25 h4=0.207 hB=+0.026 hSel=+0.007 tr4=0.305 ctx=2.189 cw=0.0
  be= 50 h4=0.285 hB=+0.044 hSel=+0.016 tr4=0.311 ctx=1.563 cw=1.0
  be= 75 h4=0.262 hB=+0.017 hSel=+0.011 tr4=0.375 ctx=1.478 cw=0.0
  be=100 h4=0.250 hB=-0.021 hSel=-0.002 tr4=0.283 ctx=1.506 cw=1.0
  be=125 h4=0.289 hB=-0.083 hSel=-0.005 tr4=0.252 ctx=1.403 cw=0.0
  be=150 h4=0.254 hB=-0.056 hSel=+0.003 tr4=0.281 ctx=1.438 cw=1.0
  be=200 h4=0.270 hB=-0.104 hSel=-0.010 tr4=0.250 ctx=1.411 cw=1.0
  be=250 h4=0.230 hB=-0.110 hSel=-0.023 tr4=0.232 ctx=1.388 cw=1.0
  be=300 h4=0.199 hB=-0.158 hSel=-0.025 tr4=0.209 ctx=1.391 cw=1.0
  be=350 h4=0.219 hB=-0.169 hSel=-0.021 tr4=0.262 ctx=1.401 cw=1.0
  be=400 h4=0.199 hB=-0.182 hSel=-0.018 tr4=0.262 ctx=1.382 cw=1.0
  be=450 h4=0.207 hB=-0.192 hSel=-0.030 tr4=0.244 ctx=1.360 cw=1.0
  be=500 h4=0.188 hB=-0.227 hSel=-0.047 tr4=0.219 ctx=1.351 cw=1.0

### Seed 43
**bag_static_1over17**
  be=  0 h4=0.820 hB=+5.504 hSel=+0.726 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.402 hB=+1.056 hSel=+0.183 tr4=0.449 ctx=29.884 cw=0.0588
  be=  5 h4=0.227 hB=-0.000 hSel=-0.005 tr4=0.266 ctx=8.076 cw=0.0588
  be= 10 h4=0.270 hB=-0.031 hSel=-0.005 tr4=0.207 ctx=2.531 cw=0.0588
  be= 25 h4=0.246 hB=-0.044 hSel=-0.003 tr4=0.238 ctx=1.455 cw=0.0588
  be= 50 h4=0.285 hB=-0.024 hSel=-0.010 tr4=0.230 ctx=1.383 cw=0.0588
  be= 75 h4=0.238 hB=-0.031 hSel=-0.006 tr4=0.246 ctx=1.367 cw=0.0588
  be=100 h4=0.230 hB=+0.006 hSel=-0.004 tr4=0.219 ctx=1.363 cw=0.0588
  be=125 h4=0.238 hB=-0.019 hSel=-0.003 tr4=0.232 ctx=1.355 cw=0.0588
  be=150 h4=0.289 hB=-0.014 hSel=+0.001 tr4=0.248 ctx=1.350 cw=0.0588
  be=200 h4=0.238 hB=-0.011 hSel=+0.003 tr4=0.256 ctx=1.341 cw=0.0588
  be=250 h4=0.289 hB=-0.006 hSel=+0.001 tr4=0.246 ctx=1.325 cw=0.0588
  be=300 h4=0.238 hB=+0.014 hSel=+0.004 tr4=0.242 ctx=1.307 cw=0.0588
  be=350 h4=0.273 hB=+0.015 hSel=+0.006 tr4=0.268 ctx=1.292 cw=0.0588
  be=400 h4=0.207 hB=-0.000 hSel=+0.003 tr4=0.279 ctx=1.277 cw=0.0588
  be=450 h4=0.211 hB=+0.001 hSel=+0.003 tr4=0.260 ctx=1.257 cw=0.0588
  be=500 h4=0.238 hB=+0.016 hSel=+0.004 tr4=0.252 ctx=1.249 cw=0.0588

**bag_interleaved_ans_full**
  be=  0 h4=0.820 hB=+5.504 hSel=+0.726 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.559 hB=+2.153 hSel=+0.327 tr4=0.609 ctx=34.356 cw=0.0
  be=  5 h4=0.195 hB=-0.063 hSel=-0.017 tr4=0.209 ctx=13.799 cw=0.0
  be= 10 h4=0.238 hB=-0.013 hSel=-0.003 tr4=0.236 ctx=4.486 cw=1.0
  be= 25 h4=0.230 hB=-0.031 hSel=-0.004 tr4=0.234 ctx=1.653 cw=0.0
  be= 50 h4=0.266 hB=-0.007 hSel=-0.012 tr4=0.238 ctx=1.548 cw=1.0
  be= 75 h4=0.250 hB=-0.021 hSel=-0.003 tr4=0.256 ctx=1.447 cw=0.0
  be=100 h4=0.262 hB=-0.009 hSel=-0.008 tr4=0.232 ctx=1.439 cw=1.0
  be=125 h4=0.254 hB=-0.015 hSel=-0.006 tr4=0.260 ctx=1.411 cw=0.0
  be=150 h4=0.262 hB=-0.025 hSel=+0.000 tr4=0.225 ctx=1.460 cw=1.0
  be=200 h4=0.258 hB=-0.022 hSel=-0.003 tr4=0.248 ctx=1.408 cw=1.0
  be=250 h4=0.250 hB=-0.007 hSel=+0.008 tr4=0.266 ctx=1.377 cw=1.0
  be=300 h4=0.242 hB=-0.006 hSel=-0.001 tr4=0.225 ctx=1.381 cw=1.0
  be=350 h4=0.250 hB=+0.006 hSel=-0.002 tr4=0.246 ctx=1.389 cw=1.0
  be=400 h4=0.270 hB=-0.013 hSel=-0.002 tr4=0.223 ctx=1.354 cw=1.0
  be=450 h4=0.266 hB=-0.005 hSel=-0.006 tr4=0.260 ctx=1.386 cw=1.0
  be=500 h4=0.223 hB=+0.007 hSel=-0.001 tr4=0.232 ctx=1.343 cw=1.0

### Seed 100
**bag_static_1over17**
  be=  0 h4=0.953 hB=+7.774 hSel=+0.914 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.551 hB=+1.994 hSel=+0.293 tr4=0.551 ctx=26.724 cw=0.0588
  be=  5 h4=0.203 hB=-0.082 hSel=-0.008 tr4=0.250 ctx=8.579 cw=0.0588
  be= 10 h4=0.246 hB=-0.046 hSel=-0.005 tr4=0.305 ctx=4.078 cw=0.0588
  be= 25 h4=0.234 hB=-0.024 hSel=-0.007 tr4=0.301 ctx=1.636 cw=0.0588
  be= 50 h4=0.234 hB=+0.006 hSel=+0.002 tr4=0.268 ctx=1.390 cw=0.0588
  be= 75 h4=0.281 hB=-0.022 hSel=+0.006 tr4=0.283 ctx=1.365 cw=0.0588
  be=100 h4=0.242 hB=-0.016 hSel=-0.011 tr4=0.236 ctx=1.357 cw=0.0588
  be=125 h4=0.285 hB=-0.020 hSel=-0.005 tr4=0.287 ctx=1.347 cw=0.0588
  be=150 h4=0.246 hB=-0.033 hSel=-0.008 tr4=0.225 ctx=1.338 cw=0.0588
  be=200 h4=0.293 hB=-0.036 hSel=-0.005 tr4=0.277 ctx=1.323 cw=0.0588
  be=250 h4=0.281 hB=-0.017 hSel=-0.001 tr4=0.275 ctx=1.311 cw=0.0588
  be=300 h4=0.254 hB=-0.012 hSel=-0.001 tr4=0.268 ctx=1.294 cw=0.0588
  be=350 h4=0.262 hB=-0.005 hSel=-0.002 tr4=0.297 ctx=1.272 cw=0.0588
  be=400 h4=0.234 hB=+0.013 hSel=+0.002 tr4=0.268 ctx=1.265 cw=0.0588
  be=450 h4=0.238 hB=+0.008 hSel=-0.000 tr4=0.270 ctx=1.251 cw=0.0588
  be=500 h4=0.250 hB=-0.006 hSel=-0.002 tr4=0.234 ctx=1.248 cw=0.0588

**bag_interleaved_ans_full**
  be=  0 h4=0.953 hB=+7.774 hSel=+0.914 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.637 hB=+2.749 hSel=+0.416 tr4=0.648 ctx=30.268 cw=0.0
  be=  5 h4=0.496 hB=+0.301 hSel=+0.111 tr4=0.480 ctx=13.654 cw=0.0
  be= 10 h4=0.266 hB=+0.031 hSel=+0.008 tr4=0.303 ctx=6.298 cw=1.0
  be= 25 h4=0.273 hB=+0.049 hSel=+0.014 tr4=0.338 ctx=2.238 cw=0.0
  be= 50 h4=0.262 hB=+0.031 hSel=+0.010 tr4=0.293 ctx=1.493 cw=1.0
  be= 75 h4=0.281 hB=+0.027 hSel=+0.011 tr4=0.281 ctx=1.467 cw=0.0
  be=100 h4=0.227 hB=+0.028 hSel=-0.005 tr4=0.250 ctx=1.434 cw=1.0
  be=125 h4=0.254 hB=-0.010 hSel=+0.002 tr4=0.275 ctx=1.388 cw=0.0
  be=150 h4=0.270 hB=-0.021 hSel=-0.002 tr4=0.230 ctx=1.412 cw=1.0
  be=200 h4=0.262 hB=-0.016 hSel=+0.009 tr4=0.271 ctx=1.408 cw=1.0
  be=250 h4=0.273 hB=-0.033 hSel=-0.003 tr4=0.232 ctx=1.455 cw=1.0
  be=300 h4=0.262 hB=-0.028 hSel=+0.001 tr4=0.221 ctx=1.414 cw=1.0
  be=350 h4=0.211 hB=-0.022 hSel=+0.001 tr4=0.229 ctx=1.372 cw=1.0
  be=400 h4=0.281 hB=+0.007 hSel=+0.005 tr4=0.242 ctx=1.350 cw=1.0
  be=450 h4=0.262 hB=-0.015 hSel=+0.003 tr4=0.271 ctx=1.343 cw=1.0
  be=500 h4=0.246 hB=+0.008 hSel=+0.001 tr4=0.268 ctx=1.317 cw=1.0

## Interpretation

If these bag-independent answer-credit arms lose binding while Step021b's bound static w=1/17 and bound interleaving preserve it, the preservation cannot be explained by answer-token exposure, RWT-family rehearsal, or temporal alternation alone. It depends on answer gradients that remain aligned with the query-conditioned relation. Conversely, if they preserve, the previous result would reduce to generic answer/RWT rehearsal or loss allocation without relation-specific credit.
