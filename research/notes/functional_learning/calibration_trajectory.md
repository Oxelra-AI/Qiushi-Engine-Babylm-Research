# accumulated mechanism synthesis calibration and compatible-trajectory test

Tests whether held-symbol binding loss during full-objective continuation
is inherent (body context prediction conflicts with the binding marker)
or transitional (gradient shock from uncalibrated context positions).

## Arm means at final epoch

| arm | n | start h4 | final h4 | final hB | final hSel | final train4 | final blk4 | final ctx_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_full | 3 | 0.753 | 0.418 | 3.111 | 0.256 | 0.921 | 0.258 | 1.233 |
| calibrate_out50_then_full | 3 | 0.753 | 0.423 | 3.307 | 0.296 | 0.915 | 0.257 | 1.233 |
| gradual_ctx100_then_full | 3 | 0.753 | 0.587 | 5.702 | 0.442 | 1.000 | 0.264 | 1.229 |
| slow_lr25_then_full | 3 | 0.753 | 0.486 | 4.125 | 0.314 | 1.000 | 0.258 | 1.231 |
| interleaved_ans_full | 3 | 0.753 | 0.746 | 7.765 | 0.665 | 1.000 | 0.238 | 1.316 |

## Per-seed trajectories

### Seed 42

**direct_full**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.873 h4=0.484 hB=+4.376 hSel=+0.378 ctx_ce=31.349 cw=1.0
  be=  5 [all_fixed     ] tr4=0.332 h4=0.250 hB=-0.149 hSel=-0.006 ctx_ce=7.628 cw=1.0
  be= 10 [all_fixed     ] tr4=0.299 h4=0.277 hB=+0.090 hSel=+0.022 ctx_ce=3.129 cw=1.0
  be= 25 [all_fixed     ] tr4=0.402 h4=0.371 hB=+0.492 hSel=+0.128 ctx_ce=1.457 cw=1.0
  be= 50 [all_fixed     ] tr4=0.896 h4=0.473 hB=+2.327 hSel=+0.350 ctx_ce=1.366 cw=1.0
  be= 75 [all_fixed     ] tr4=0.971 h4=0.480 hB=+2.993 hSel=+0.350 ctx_ce=1.353 cw=1.0
  be=100 [all_fixed     ] tr4=0.986 h4=0.480 hB=+3.350 hSel=+0.349 ctx_ce=1.340 cw=1.0
  be=125 [all_fixed     ] tr4=0.998 h4=0.473 hB=+3.726 hSel=+0.346 ctx_ce=1.332 cw=1.0
  be=150 [all_fixed     ] tr4=0.998 h4=0.480 hB=+3.906 hSel=+0.348 ctx_ce=1.319 cw=1.0
  be=200 [all_fixed     ] tr4=0.994 h4=0.480 hB=+4.276 hSel=+0.342 ctx_ce=1.296 cw=1.0
  be=250 [all_fixed     ] tr4=0.996 h4=0.465 hB=+4.367 hSel=+0.300 ctx_ce=1.272 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.457 hB=+4.286 hSel=+0.294 ctx_ce=1.259 cw=1.0
  be=350 [all_fixed     ] tr4=0.986 h4=0.441 hB=+4.386 hSel=+0.258 ctx_ce=1.246 cw=1.0
  be=400 [all_fixed     ] tr4=0.996 h4=0.449 hB=+4.438 hSel=+0.258 ctx_ce=1.245 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.449 hB=+4.651 hSel=+0.271 ctx_ce=1.243 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.449 hB=+4.332 hSel=+0.270 ctx_ce=1.232 cw=1.0

**calibrate_out50_then_full**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [out_only_fixed] tr4=0.986 h4=0.484 hB=+3.834 hSel=+0.365 ctx_ce=36.195 cw=1.0
  be=  5 [out_only_fixed] tr4=0.988 h4=0.484 hB=+3.836 hSel=+0.366 ctx_ce=35.856 cw=1.0
  be= 10 [out_only_fixed] tr4=0.988 h4=0.484 hB=+3.837 hSel=+0.367 ctx_ce=35.502 cw=1.0
  be= 25 [out_only_fixed] tr4=0.992 h4=0.484 hB=+3.839 hSel=+0.368 ctx_ce=33.570 cw=1.0
  be= 50 [out_only_fixed] tr4=0.990 h4=0.484 hB=+3.839 hSel=+0.368 ctx_ce=30.988 cw=1.0
  be= 75 [all_fixed     ] tr4=0.803 h4=0.457 hB=+1.817 hSel=+0.306 ctx_ce=1.432 cw=1.0
  be=100 [all_fixed     ] tr4=0.959 h4=0.480 hB=+2.904 hSel=+0.356 ctx_ce=1.360 cw=1.0
  be=125 [all_fixed     ] tr4=0.984 h4=0.480 hB=+3.383 hSel=+0.354 ctx_ce=1.352 cw=1.0
  be=150 [all_fixed     ] tr4=0.994 h4=0.480 hB=+3.760 hSel=+0.356 ctx_ce=1.337 cw=1.0
  be=200 [all_fixed     ] tr4=0.994 h4=0.480 hB=+4.331 hSel=+0.352 ctx_ce=1.311 cw=1.0
  be=250 [all_fixed     ] tr4=0.998 h4=0.484 hB=+4.585 hSel=+0.338 ctx_ce=1.283 cw=1.0
  be=300 [all_fixed     ] tr4=0.996 h4=0.477 hB=+4.380 hSel=+0.336 ctx_ce=1.266 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.457 hB=+4.388 hSel=+0.301 ctx_ce=1.248 cw=1.0
  be=400 [all_fixed     ] tr4=0.998 h4=0.473 hB=+4.823 hSel=+0.307 ctx_ce=1.245 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.453 hB=+4.764 hSel=+0.290 ctx_ce=1.244 cw=1.0
  be=500 [all_fixed     ] tr4=0.984 h4=0.473 hB=+4.728 hSel=+0.278 ctx_ce=1.233 cw=1.0

**gradual_ctx100_then_full**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_ramp      ] tr4=0.980 h4=0.484 hB=+4.394 hSel=+0.375 ctx_ce=31.667 cw=0.0199
  be=  5 [all_ramp      ] tr4=0.936 h4=0.484 hB=+4.611 hSel=+0.371 ctx_ce=8.007 cw=0.0595
  be= 10 [all_ramp      ] tr4=0.967 h4=0.480 hB=+4.372 hSel=+0.365 ctx_ce=2.699 cw=0.109
  be= 25 [all_ramp      ] tr4=0.988 h4=0.477 hB=+4.157 hSel=+0.356 ctx_ce=1.401 cw=0.2575
  be= 50 [all_ramp      ] tr4=1.000 h4=0.477 hB=+4.410 hSel=+0.357 ctx_ce=1.352 cw=0.505
  be= 75 [all_ramp      ] tr4=1.000 h4=0.480 hB=+4.297 hSel=+0.343 ctx_ce=1.337 cw=0.7525
  be=100 [all_ramp      ] tr4=1.000 h4=0.477 hB=+4.192 hSel=+0.341 ctx_ce=1.319 cw=1.0
  be=125 [all_fixed     ] tr4=0.992 h4=0.457 hB=+3.850 hSel=+0.334 ctx_ce=1.301 cw=1.0
  be=150 [all_fixed     ] tr4=0.990 h4=0.473 hB=+3.565 hSel=+0.331 ctx_ce=1.280 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.461 hB=+3.533 hSel=+0.296 ctx_ce=1.253 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.438 hB=+3.019 hSel=+0.246 ctx_ce=1.245 cw=1.0
  be=300 [all_fixed     ] tr4=0.994 h4=0.457 hB=+3.573 hSel=+0.273 ctx_ce=1.241 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.445 hB=+3.347 hSel=+0.255 ctx_ce=1.233 cw=1.0
  be=400 [all_fixed     ] tr4=0.998 h4=0.449 hB=+3.299 hSel=+0.249 ctx_ce=1.236 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.430 hB=+3.001 hSel=+0.214 ctx_ce=1.236 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.441 hB=+3.813 hSel=+0.243 ctx_ce=1.229 cw=1.0

**slow_lr25_then_full**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.986 h4=0.484 hB=+3.916 hSel=+0.367 ctx_ce=35.745 cw=1.0
  be=  5 [all_fixed     ] tr4=0.980 h4=0.484 hB=+4.182 hSel=+0.377 ctx_ce=31.264 cw=1.0
  be= 10 [all_fixed     ] tr4=0.875 h4=0.484 hB=+4.405 hSel=+0.377 ctx_ce=25.859 cw=1.0
  be= 25 [all_fixed     ] tr4=0.494 h4=0.461 hB=+2.435 hSel=+0.281 ctx_ce=13.473 cw=1.0
  be= 50 [all_fixed     ] tr4=0.895 h4=0.469 hB=+2.252 hSel=+0.351 ctx_ce=1.373 cw=1.0
  be= 75 [all_fixed     ] tr4=0.969 h4=0.480 hB=+3.133 hSel=+0.348 ctx_ce=1.352 cw=1.0
  be=100 [all_fixed     ] tr4=0.996 h4=0.480 hB=+3.594 hSel=+0.350 ctx_ce=1.337 cw=1.0
  be=125 [all_fixed     ] tr4=0.998 h4=0.477 hB=+4.040 hSel=+0.349 ctx_ce=1.323 cw=1.0
  be=150 [all_fixed     ] tr4=0.998 h4=0.480 hB=+4.195 hSel=+0.325 ctx_ce=1.308 cw=1.0
  be=200 [all_fixed     ] tr4=0.986 h4=0.469 hB=+4.355 hSel=+0.297 ctx_ce=1.273 cw=1.0
  be=250 [all_fixed     ] tr4=0.998 h4=0.461 hB=+4.110 hSel=+0.289 ctx_ce=1.256 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.473 hB=+4.293 hSel=+0.326 ctx_ce=1.249 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.445 hB=+4.051 hSel=+0.285 ctx_ce=1.238 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.473 hB=+4.097 hSel=+0.286 ctx_ce=1.240 cw=1.0
  be=450 [all_fixed     ] tr4=0.982 h4=0.480 hB=+4.709 hSel=+0.326 ctx_ce=1.239 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.480 hB=+4.776 hSel=+0.308 ctx_ce=1.231 cw=1.0

**interleaved_ans_full**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_interleaved] tr4=0.961 h4=0.477 hB=+3.826 hSel=+0.352 ctx_ce=36.229 cw=0.0
  be=  5 [all_interleaved] tr4=0.939 h4=0.488 hB=+4.927 hSel=+0.353 ctx_ce=14.678 cw=0.0
  be= 10 [all_interleaved] tr4=0.852 h4=0.496 hB=+4.252 hSel=+0.367 ctx_ce=6.107 cw=1.0
  be= 25 [all_interleaved] tr4=0.930 h4=0.477 hB=+3.823 hSel=+0.338 ctx_ce=1.929 cw=0.0
  be= 50 [all_interleaved] tr4=0.906 h4=0.492 hB=+3.407 hSel=+0.364 ctx_ce=1.562 cw=1.0
  be= 75 [all_interleaved] tr4=0.998 h4=0.496 hB=+3.532 hSel=+0.377 ctx_ce=1.447 cw=0.0
  be=100 [all_interleaved] tr4=0.998 h4=0.492 hB=+3.952 hSel=+0.377 ctx_ce=1.429 cw=1.0
  be=125 [all_interleaved] tr4=1.000 h4=0.492 hB=+4.178 hSel=+0.376 ctx_ce=1.426 cw=0.0
  be=150 [all_interleaved] tr4=1.000 h4=0.484 hB=+4.331 hSel=+0.370 ctx_ce=1.436 cw=1.0
  be=200 [all_interleaved] tr4=1.000 h4=0.488 hB=+4.594 hSel=+0.368 ctx_ce=1.401 cw=1.0
  be=250 [all_interleaved] tr4=0.998 h4=0.508 hB=+4.451 hSel=+0.381 ctx_ce=1.384 cw=1.0
  be=300 [all_interleaved] tr4=0.906 h4=0.516 hB=+4.781 hSel=+0.363 ctx_ce=1.402 cw=1.0
  be=350 [all_interleaved] tr4=1.000 h4=0.574 hB=+5.155 hSel=+0.471 ctx_ce=1.391 cw=1.0
  be=400 [all_interleaved] tr4=1.000 h4=0.566 hB=+5.424 hSel=+0.468 ctx_ce=1.361 cw=1.0
  be=450 [all_interleaved] tr4=1.000 h4=0.574 hB=+5.549 hSel=+0.477 ctx_ce=1.374 cw=1.0
  be=500 [all_interleaved] tr4=1.000 h4=0.570 hB=+5.739 hSel=+0.455 ctx_ce=1.331 cw=1.0

### Seed 43

**direct_full**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.740 h4=0.672 hB=+3.517 hSel=+0.484 ctx_ce=28.999 cw=1.0
  be=  5 [all_fixed     ] tr4=0.297 h4=0.289 hB=-0.015 hSel=-0.007 ctx_ce=6.647 cw=1.0
  be= 10 [all_fixed     ] tr4=0.242 h4=0.230 hB=-0.027 hSel=-0.001 ctx_ce=2.015 cw=1.0
  be= 25 [all_fixed     ] tr4=0.281 h4=0.273 hB=-0.031 hSel=-0.005 ctx_ce=1.407 cw=1.0
  be= 50 [all_fixed     ] tr4=0.238 h4=0.230 hB=-0.041 hSel=-0.010 ctx_ce=1.365 cw=1.0
  be= 75 [all_fixed     ] tr4=0.264 h4=0.258 hB=-0.028 hSel=-0.011 ctx_ce=1.353 cw=1.0
  be=100 [all_fixed     ] tr4=0.234 h4=0.234 hB=-0.032 hSel=-0.007 ctx_ce=1.347 cw=1.0
  be=125 [all_fixed     ] tr4=0.232 h4=0.262 hB=-0.032 hSel=-0.007 ctx_ce=1.332 cw=1.0
  be=150 [all_fixed     ] tr4=0.264 h4=0.281 hB=+0.056 hSel=+0.019 ctx_ce=1.310 cw=1.0
  be=200 [all_fixed     ] tr4=0.545 h4=0.270 hB=+0.852 hSel=+0.135 ctx_ce=1.270 cw=1.0
  be=250 [all_fixed     ] tr4=0.588 h4=0.352 hB=+1.063 hSel=+0.178 ctx_ce=1.251 cw=1.0
  be=300 [all_fixed     ] tr4=0.773 h4=0.418 hB=+1.869 hSel=+0.258 ctx_ce=1.241 cw=1.0
  be=350 [all_fixed     ] tr4=0.762 h4=0.367 hB=+1.825 hSel=+0.247 ctx_ce=1.238 cw=1.0
  be=400 [all_fixed     ] tr4=0.762 h4=0.371 hB=+2.032 hSel=+0.271 ctx_ce=1.237 cw=1.0
  be=450 [all_fixed     ] tr4=0.773 h4=0.391 hB=+2.005 hSel=+0.255 ctx_ce=1.233 cw=1.0
  be=500 [all_fixed     ] tr4=0.764 h4=0.406 hB=+2.291 hSel=+0.301 ctx_ce=1.233 cw=1.0

**calibrate_out50_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [out_only_fixed] tr4=1.000 h4=0.824 hB=+5.503 hSel=+0.726 ctx_ce=34.220 cw=1.0
  be=  5 [out_only_fixed] tr4=1.000 h4=0.816 hB=+5.499 hSel=+0.726 ctx_ce=33.399 cw=1.0
  be= 10 [out_only_fixed] tr4=1.000 h4=0.816 hB=+5.495 hSel=+0.727 ctx_ce=32.960 cw=1.0
  be= 25 [out_only_fixed] tr4=1.000 h4=0.812 hB=+5.480 hSel=+0.724 ctx_ce=31.192 cw=1.0
  be= 50 [out_only_fixed] tr4=0.998 h4=0.812 hB=+5.455 hSel=+0.721 ctx_ce=28.881 cw=1.0
  be= 75 [all_fixed     ] tr4=0.262 h4=0.270 hB=-0.033 hSel=-0.008 ctx_ce=1.397 cw=1.0
  be=100 [all_fixed     ] tr4=0.254 h4=0.242 hB=-0.040 hSel=-0.008 ctx_ce=1.368 cw=1.0
  be=125 [all_fixed     ] tr4=0.246 h4=0.273 hB=-0.040 hSel=-0.008 ctx_ce=1.353 cw=1.0
  be=150 [all_fixed     ] tr4=0.266 h4=0.246 hB=-0.024 hSel=-0.003 ctx_ce=1.338 cw=1.0
  be=200 [all_fixed     ] tr4=0.516 h4=0.344 hB=+0.937 hSel=+0.155 ctx_ce=1.297 cw=1.0
  be=250 [all_fixed     ] tr4=0.525 h4=0.352 hB=+1.116 hSel=+0.182 ctx_ce=1.260 cw=1.0
  be=300 [all_fixed     ] tr4=0.789 h4=0.410 hB=+1.751 hSel=+0.244 ctx_ce=1.247 cw=1.0
  be=350 [all_fixed     ] tr4=0.742 h4=0.379 hB=+2.074 hSel=+0.259 ctx_ce=1.240 cw=1.0
  be=400 [all_fixed     ] tr4=0.768 h4=0.395 hB=+2.340 hSel=+0.290 ctx_ce=1.239 cw=1.0
  be=450 [all_fixed     ] tr4=0.771 h4=0.395 hB=+2.229 hSel=+0.260 ctx_ce=1.234 cw=1.0
  be=500 [all_fixed     ] tr4=0.762 h4=0.387 hB=+2.370 hSel=+0.286 ctx_ce=1.234 cw=1.0

**gradual_ctx100_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_ramp      ] tr4=0.959 h4=0.836 hB=+5.436 hSel=+0.714 ctx_ce=29.298 cw=0.0199
  be=  5 [all_ramp      ] tr4=0.975 h4=0.922 hB=+4.675 hSel=+0.762 ctx_ce=6.801 cw=0.0595
  be= 10 [all_ramp      ] tr4=1.000 h4=0.887 hB=+4.868 hSel=+0.714 ctx_ce=1.768 cw=0.109
  be= 25 [all_ramp      ] tr4=1.000 h4=0.836 hB=+5.552 hSel=+0.698 ctx_ce=1.381 cw=0.2575
  be= 50 [all_ramp      ] tr4=1.000 h4=0.820 hB=+6.158 hSel=+0.710 ctx_ce=1.357 cw=0.505
  be= 75 [all_ramp      ] tr4=0.998 h4=0.816 hB=+6.061 hSel=+0.661 ctx_ce=1.331 cw=0.7525
  be=100 [all_ramp      ] tr4=1.000 h4=0.750 hB=+5.563 hSel=+0.554 ctx_ce=1.305 cw=1.0
  be=125 [all_fixed     ] tr4=1.000 h4=0.660 hB=+4.962 hSel=+0.463 ctx_ce=1.276 cw=1.0
  be=150 [all_fixed     ] tr4=1.000 h4=0.637 hB=+5.130 hSel=+0.439 ctx_ce=1.257 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.648 hB=+5.611 hSel=+0.481 ctx_ce=1.239 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.648 hB=+5.718 hSel=+0.484 ctx_ce=1.243 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.699 hB=+6.205 hSel=+0.502 ctx_ce=1.233 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.723 hB=+6.452 hSel=+0.531 ctx_ce=1.233 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.691 hB=+6.658 hSel=+0.529 ctx_ce=1.234 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.766 hB=+7.555 hSel=+0.600 ctx_ce=1.227 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.742 hB=+6.920 hSel=+0.552 ctx_ce=1.227 cw=1.0

**slow_lr25_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=1.000 h4=0.824 hB=+5.628 hSel=+0.740 ctx_ce=33.728 cw=1.0
  be=  5 [all_fixed     ] tr4=0.971 h4=0.809 hB=+5.505 hSel=+0.717 ctx_ce=28.269 cw=1.0
  be= 10 [all_fixed     ] tr4=0.721 h4=0.730 hB=+3.433 hSel=+0.499 ctx_ce=22.457 cw=1.0
  be= 25 [all_fixed     ] tr4=0.324 h4=0.289 hB=+0.210 hSel=+0.033 ctx_ce=11.304 cw=1.0
  be= 50 [all_fixed     ] tr4=0.246 h4=0.246 hB=-0.039 hSel=-0.008 ctx_ce=1.373 cw=1.0
  be= 75 [all_fixed     ] tr4=0.244 h4=0.250 hB=-0.026 hSel=-0.014 ctx_ce=1.353 cw=1.0
  be=100 [all_fixed     ] tr4=0.230 h4=0.238 hB=-0.021 hSel=-0.006 ctx_ce=1.340 cw=1.0
  be=125 [all_fixed     ] tr4=0.236 h4=0.277 hB=+0.053 hSel=+0.012 ctx_ce=1.312 cw=1.0
  be=150 [all_fixed     ] tr4=0.533 h4=0.352 hB=+0.892 hSel=+0.151 ctx_ce=1.285 cw=1.0
  be=200 [all_fixed     ] tr4=0.641 h4=0.363 hB=+1.238 hSel=+0.215 ctx_ce=1.253 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.543 hB=+3.454 hSel=+0.379 ctx_ce=1.246 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.547 hB=+4.163 hSel=+0.388 ctx_ce=1.238 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.477 hB=+4.066 hSel=+0.328 ctx_ce=1.236 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.512 hB=+4.247 hSel=+0.336 ctx_ce=1.236 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.492 hB=+4.409 hSel=+0.314 ctx_ce=1.230 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.543 hB=+4.948 hSel=+0.388 ctx_ce=1.231 cw=1.0

**interleaved_ans_full**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_interleaved] tr4=0.998 h4=0.809 hB=+5.690 hSel=+0.687 ctx_ce=34.310 cw=0.0
  be=  5 [all_interleaved] tr4=0.910 h4=0.906 hB=+4.634 hSel=+0.742 ctx_ce=13.074 cw=0.0
  be= 10 [all_interleaved] tr4=0.967 h4=0.918 hB=+5.000 hSel=+0.776 ctx_ce=4.874 cw=1.0
  be= 25 [all_interleaved] tr4=1.000 h4=0.922 hB=+6.088 hSel=+0.808 ctx_ce=1.594 cw=0.0
  be= 50 [all_interleaved] tr4=1.000 h4=0.887 hB=+6.575 hSel=+0.784 ctx_ce=1.480 cw=1.0
  be= 75 [all_interleaved] tr4=1.000 h4=0.863 hB=+6.988 hSel=+0.780 ctx_ce=1.468 cw=0.0
  be=100 [all_interleaved] tr4=1.000 h4=0.910 hB=+7.614 hSel=+0.831 ctx_ce=1.504 cw=1.0
  be=125 [all_interleaved] tr4=1.000 h4=0.902 hB=+7.850 hSel=+0.815 ctx_ce=1.424 cw=0.0
  be=150 [all_interleaved] tr4=1.000 h4=0.895 hB=+8.118 hSel=+0.818 ctx_ce=1.418 cw=1.0
  be=200 [all_interleaved] tr4=1.000 h4=0.961 hB=+8.103 hSel=+0.917 ctx_ce=1.433 cw=1.0
  be=250 [all_interleaved] tr4=1.000 h4=0.957 hB=+8.321 hSel=+0.892 ctx_ce=1.372 cw=1.0
  be=300 [all_interleaved] tr4=1.000 h4=0.957 hB=+8.629 hSel=+0.868 ctx_ce=1.362 cw=1.0
  be=350 [all_interleaved] tr4=1.000 h4=0.938 hB=+8.581 hSel=+0.820 ctx_ce=1.392 cw=1.0
  be=400 [all_interleaved] tr4=1.000 h4=0.836 hB=+7.921 hSel=+0.696 ctx_ce=1.310 cw=1.0
  be=450 [all_interleaved] tr4=1.000 h4=0.797 hB=+7.464 hSel=+0.634 ctx_ce=1.370 cw=1.0
  be=500 [all_interleaved] tr4=1.000 h4=0.855 hB=+9.252 hSel=+0.789 ctx_ce=1.310 cw=1.0

### Seed 100

**direct_full**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.822 h4=0.648 hB=+3.353 hSel=+0.447 ctx_ce=25.865 cw=1.0
  be=  5 [all_fixed     ] tr4=0.293 h4=0.266 hB=-0.108 hSel=-0.008 ctx_ce=6.550 cw=1.0
  be= 10 [all_fixed     ] tr4=0.281 h4=0.223 hB=-0.091 hSel=-0.015 ctx_ce=2.917 cw=1.0
  be= 25 [all_fixed     ] tr4=0.355 h4=0.305 hB=+0.118 hSel=+0.042 ctx_ce=1.442 cw=1.0
  be= 50 [all_fixed     ] tr4=0.510 h4=0.477 hB=+0.760 hSel=+0.175 ctx_ce=1.356 cw=1.0
  be= 75 [all_fixed     ] tr4=0.539 h4=0.496 hB=+1.338 hSel=+0.227 ctx_ce=1.340 cw=1.0
  be=100 [all_fixed     ] tr4=0.494 h4=0.430 hB=+1.342 hSel=+0.191 ctx_ce=1.322 cw=1.0
  be=125 [all_fixed     ] tr4=0.520 h4=0.359 hB=+1.142 hSel=+0.151 ctx_ce=1.304 cw=1.0
  be=150 [all_fixed     ] tr4=0.580 h4=0.344 hB=+1.133 hSel=+0.139 ctx_ce=1.275 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.578 hB=+2.965 hSel=+0.353 ctx_ce=1.256 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.426 hB=+2.510 hSel=+0.219 ctx_ce=1.244 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.461 hB=+2.990 hSel=+0.260 ctx_ce=1.239 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.422 hB=+2.937 hSel=+0.217 ctx_ce=1.232 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.426 hB=+3.260 hSel=+0.248 ctx_ce=1.237 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.281 hB=+2.154 hSel=+0.064 ctx_ce=1.232 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.398 hB=+2.710 hSel=+0.196 ctx_ce=1.233 cw=1.0

**calibrate_out50_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [out_only_fixed] tr4=1.000 h4=0.953 hB=+7.770 hSel=+0.914 ctx_ce=30.362 cw=1.0
  be=  5 [out_only_fixed] tr4=1.000 h4=0.953 hB=+7.756 hSel=+0.914 ctx_ce=29.747 cw=1.0
  be= 10 [out_only_fixed] tr4=1.000 h4=0.949 hB=+7.739 hSel=+0.914 ctx_ce=29.389 cw=1.0
  be= 25 [out_only_fixed] tr4=1.000 h4=0.949 hB=+7.691 hSel=+0.914 ctx_ce=27.801 cw=1.0
  be= 50 [out_only_fixed] tr4=1.000 h4=0.949 hB=+7.617 hSel=+0.913 ctx_ce=25.147 cw=1.0
  be= 75 [all_fixed     ] tr4=0.402 h4=0.316 hB=+0.310 hSel=+0.095 ctx_ce=1.405 cw=1.0
  be=100 [all_fixed     ] tr4=0.539 h4=0.359 hB=+1.028 hSel=+0.184 ctx_ce=1.352 cw=1.0
  be=125 [all_fixed     ] tr4=0.514 h4=0.379 hB=+1.276 hSel=+0.180 ctx_ce=1.333 cw=1.0
  be=150 [all_fixed     ] tr4=0.996 h4=0.539 hB=+2.375 hSel=+0.361 ctx_ce=1.304 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.629 hB=+4.131 hSel=+0.539 ctx_ce=1.265 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.562 hB=+3.959 hSel=+0.495 ctx_ce=1.246 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.516 hB=+3.578 hSel=+0.437 ctx_ce=1.241 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.484 hB=+3.297 hSel=+0.392 ctx_ce=1.233 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.477 hB=+3.603 hSel=+0.430 ctx_ce=1.237 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.391 hB=+2.795 hSel=+0.332 ctx_ce=1.231 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.410 hB=+2.823 hSel=+0.322 ctx_ce=1.233 cw=1.0

**gradual_ctx100_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_ramp      ] tr4=0.988 h4=0.848 hB=+5.346 hSel=+0.654 ctx_ce=25.904 cw=0.0199
  be=  5 [all_ramp      ] tr4=0.996 h4=0.918 hB=+3.833 hSel=+0.711 ctx_ce=6.569 cw=0.0595
  be= 10 [all_ramp      ] tr4=1.000 h4=0.793 hB=+3.662 hSel=+0.609 ctx_ce=2.402 cw=0.109
  be= 25 [all_ramp      ] tr4=1.000 h4=0.656 hB=+3.812 hSel=+0.510 ctx_ce=1.377 cw=0.2575
  be= 50 [all_ramp      ] tr4=1.000 h4=0.605 hB=+3.752 hSel=+0.491 ctx_ce=1.341 cw=0.505
  be= 75 [all_ramp      ] tr4=1.000 h4=0.535 hB=+3.208 hSel=+0.404 ctx_ce=1.317 cw=0.7525
  be=100 [all_ramp      ] tr4=1.000 h4=0.496 hB=+2.779 hSel=+0.331 ctx_ce=1.294 cw=1.0
  be=125 [all_fixed     ] tr4=1.000 h4=0.477 hB=+2.830 hSel=+0.332 ctx_ce=1.272 cw=1.0
  be=150 [all_fixed     ] tr4=1.000 h4=0.492 hB=+3.104 hSel=+0.352 ctx_ce=1.251 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.574 hB=+5.084 hSel=+0.491 ctx_ce=1.243 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.594 hB=+5.752 hSel=+0.541 ctx_ce=1.233 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.570 hB=+5.012 hSel=+0.508 ctx_ce=1.233 cw=1.0
  be=350 [all_fixed     ] tr4=1.000 h4=0.613 hB=+6.424 hSel=+0.555 ctx_ce=1.228 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.602 hB=+6.415 hSel=+0.560 ctx_ce=1.233 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.598 hB=+6.020 hSel=+0.548 ctx_ce=1.229 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.578 hB=+6.371 hSel=+0.531 ctx_ce=1.231 cw=1.0

**slow_lr25_then_full**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=1.000 h4=0.934 hB=+7.657 hSel=+0.899 ctx_ce=29.930 cw=1.0
  be=  5 [all_fixed     ] tr4=0.996 h4=0.848 hB=+6.059 hSel=+0.713 ctx_ce=25.381 cw=1.0
  be= 10 [all_fixed     ] tr4=0.791 h4=0.629 hB=+2.984 hSel=+0.419 ctx_ce=20.537 cw=1.0
  be= 25 [all_fixed     ] tr4=0.299 h4=0.227 hB=-0.295 hSel=-0.023 ctx_ce=11.136 cw=1.0
  be= 50 [all_fixed     ] tr4=0.439 h4=0.391 hB=+0.438 hSel=+0.108 ctx_ce=1.368 cw=1.0
  be= 75 [all_fixed     ] tr4=0.535 h4=0.434 hB=+1.326 hSel=+0.209 ctx_ce=1.333 cw=1.0
  be=100 [all_fixed     ] tr4=0.496 h4=0.410 hB=+1.315 hSel=+0.157 ctx_ce=1.308 cw=1.0
  be=125 [all_fixed     ] tr4=0.998 h4=0.629 hB=+3.317 hSel=+0.465 ctx_ce=1.288 cw=1.0
  be=150 [all_fixed     ] tr4=1.000 h4=0.617 hB=+3.435 hSel=+0.469 ctx_ce=1.258 cw=1.0
  be=200 [all_fixed     ] tr4=1.000 h4=0.523 hB=+3.266 hSel=+0.425 ctx_ce=1.248 cw=1.0
  be=250 [all_fixed     ] tr4=1.000 h4=0.496 hB=+3.125 hSel=+0.361 ctx_ce=1.239 cw=1.0
  be=300 [all_fixed     ] tr4=1.000 h4=0.457 hB=+2.908 hSel=+0.323 ctx_ce=1.236 cw=1.0
  be=350 [all_fixed     ] tr4=0.998 h4=0.473 hB=+3.275 hSel=+0.372 ctx_ce=1.230 cw=1.0
  be=400 [all_fixed     ] tr4=1.000 h4=0.398 hB=+2.697 hSel=+0.250 ctx_ce=1.235 cw=1.0
  be=450 [all_fixed     ] tr4=1.000 h4=0.391 hB=+2.672 hSel=+0.237 ctx_ce=1.231 cw=1.0
  be=500 [all_fixed     ] tr4=1.000 h4=0.434 hB=+2.651 hSel=+0.247 ctx_ce=1.232 cw=1.0

**interleaved_ans_full**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_interleaved] tr4=1.000 h4=0.941 hB=+8.487 hSel=+0.931 ctx_ce=30.402 cw=0.0
  be=  5 [all_interleaved] tr4=0.990 h4=0.961 hB=+5.107 hSel=+0.842 ctx_ce=11.703 cw=0.0
  be= 10 [all_interleaved] tr4=1.000 h4=0.945 hB=+4.199 hSel=+0.767 ctx_ce=5.218 cw=1.0
  be= 25 [all_interleaved] tr4=1.000 h4=0.789 hB=+5.003 hSel=+0.682 ctx_ce=1.732 cw=0.0
  be= 50 [all_interleaved] tr4=1.000 h4=0.738 hB=+5.272 hSel=+0.663 ctx_ce=1.532 cw=1.0
  be= 75 [all_interleaved] tr4=1.000 h4=0.703 hB=+5.176 hSel=+0.639 ctx_ce=1.451 cw=0.0
  be=100 [all_interleaved] tr4=1.000 h4=0.660 hB=+5.078 hSel=+0.589 ctx_ce=1.479 cw=1.0
  be=125 [all_interleaved] tr4=1.000 h4=0.652 hB=+5.045 hSel=+0.570 ctx_ce=1.456 cw=0.0
  be=150 [all_interleaved] tr4=1.000 h4=0.605 hB=+4.592 hSel=+0.495 ctx_ce=1.428 cw=1.0
  be=200 [all_interleaved] tr4=1.000 h4=0.555 hB=+4.337 hSel=+0.439 ctx_ce=1.385 cw=1.0
  be=250 [all_interleaved] tr4=1.000 h4=0.535 hB=+4.317 hSel=+0.387 ctx_ce=1.393 cw=1.0
  be=300 [all_interleaved] tr4=1.000 h4=0.836 hB=+6.770 hSel=+0.757 ctx_ce=1.372 cw=1.0
  be=350 [all_interleaved] tr4=1.000 h4=0.809 hB=+7.246 hSel=+0.736 ctx_ce=1.336 cw=1.0
  be=400 [all_interleaved] tr4=1.000 h4=0.836 hB=+7.654 hSel=+0.754 ctx_ce=1.342 cw=1.0
  be=450 [all_interleaved] tr4=1.000 h4=0.828 hB=+8.152 hSel=+0.768 ctx_ce=1.295 cw=1.0
  be=500 [all_interleaved] tr4=1.000 h4=0.812 hB=+8.306 hSel=+0.752 ctx_ce=1.306 cw=1.0

## Interpretation

If any compatible trajectory (gradual ramp, slow lr, interleaved reinforcement) preserves held binding while context CE drops to normal levels (~1.5 nats), the conflict is transitional and a compatible learning path exists.  If all fail, context prediction through the body inherently conflicts with the binding computation.

Readout calibration may fail to reduce context CE enough (body objective ablation showed out_only barely reduced it from ~34 to ~28 in 25 epochs), so its failure alone does not establish inherent conflict.  The gradual ramp is the strongest test because it limits gradient magnitude at every epoch.

