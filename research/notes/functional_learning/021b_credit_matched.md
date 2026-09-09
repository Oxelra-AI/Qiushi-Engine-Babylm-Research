# accumulated mechanism synthesis calibration and compatible-trajectory test

Tests whether held-symbol binding loss during full-objective continuation
is inherent (body context prediction conflicts with the binding marker)
or transitional (gradient shock from uncalibrated context positions).

## Arm means at final epoch

| arm | n | start h4 | final h4 | final hB | final hSel | final train4 | final blk4 | final ctx_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_ctx_half | 3 | 0.753 | 0.445 | 4.011 | 0.307 | 0.992 | 0.255 | 1.234 |
| constant_ctx_tenth | 3 | 0.753 | 0.652 | 6.754 | 0.515 | 1.000 | 0.270 | 1.234 |
| constant_ctx_1over17 | 3 | 0.753 | 0.694 | 7.548 | 0.574 | 1.000 | 0.264 | 1.238 |

## Per-seed trajectories

### Seed 42

**constant_ctx_half**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.879 h4=0.484 hB=+4.372 hSel=+0.377 ctx_ce=31.351 cw=0.5
  be=  5 [all_fixed     ] tr4=0.508 h4=0.441 hB=+1.204 hSel=+0.212 ctx_ce=7.704 cw=0.5
  be= 10 [all_fixed     ] tr4=0.682 h4=0.484 hB=+1.826 hSel=+0.318 ctx_ce=3.125 cw=0.5
  be= 25 [all_fixed     ] tr4=0.920 h4=0.477 hB=+2.627 hSel=+0.357 ctx_ce=1.448 cw=0.5
  be= 50 [all_fixed     ] tr4=0.979 h4=0.480 hB=+3.351 hSel=+0.355 ctx_ce=1.366 cw=0.5
  be= 75 [all_fixed     ] tr4=0.990 h4=0.480 hB=+3.829 hSel=+0.346 ctx_ce=1.352 cw=0.5
  be=100 [all_fixed     ] tr4=0.996 h4=0.480 hB=+4.217 hSel=+0.353 ctx_ce=1.340 cw=0.5
  be=125 [all_fixed     ] tr4=0.998 h4=0.480 hB=+4.580 hSel=+0.357 ctx_ce=1.333 cw=0.5
  be=150 [all_fixed     ] tr4=0.998 h4=0.480 hB=+4.744 hSel=+0.351 ctx_ce=1.320 cw=0.5
  be=200 [all_fixed     ] tr4=0.959 h4=0.473 hB=+5.270 hSel=+0.299 ctx_ce=1.296 cw=0.5
  be=250 [all_fixed     ] tr4=1.000 h4=0.480 hB=+4.930 hSel=+0.335 ctx_ce=1.273 cw=0.5
  be=300 [all_fixed     ] tr4=1.000 h4=0.496 hB=+5.035 hSel=+0.353 ctx_ce=1.262 cw=0.5
  be=350 [all_fixed     ] tr4=1.000 h4=0.488 hB=+4.850 hSel=+0.352 ctx_ce=1.247 cw=0.5
  be=400 [all_fixed     ] tr4=0.998 h4=0.484 hB=+4.769 hSel=+0.336 ctx_ce=1.245 cw=0.5
  be=450 [all_fixed     ] tr4=1.000 h4=0.473 hB=+4.929 hSel=+0.336 ctx_ce=1.243 cw=0.5
  be=500 [all_fixed     ] tr4=0.975 h4=0.480 hB=+5.197 hSel=+0.328 ctx_ce=1.233 cw=0.5

**constant_ctx_tenth**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.906 h4=0.484 hB=+4.393 hSel=+0.377 ctx_ce=31.384 cw=0.1
  be=  5 [all_fixed     ] tr4=0.881 h4=0.492 hB=+4.251 hSel=+0.377 ctx_ce=8.118 cw=0.1
  be= 10 [all_fixed     ] tr4=0.947 h4=0.484 hB=+4.127 hSel=+0.368 ctx_ce=3.223 cw=0.1
  be= 25 [all_fixed     ] tr4=0.984 h4=0.477 hB=+4.403 hSel=+0.353 ctx_ce=1.464 cw=0.1
  be= 50 [all_fixed     ] tr4=0.998 h4=0.480 hB=+4.815 hSel=+0.361 ctx_ce=1.366 cw=0.1
  be= 75 [all_fixed     ] tr4=1.000 h4=0.484 hB=+5.207 hSel=+0.362 ctx_ce=1.351 cw=0.1
  be=100 [all_fixed     ] tr4=0.988 h4=0.480 hB=+4.884 hSel=+0.343 ctx_ce=1.341 cw=0.1
  be=125 [all_fixed     ] tr4=1.000 h4=0.488 hB=+5.036 hSel=+0.362 ctx_ce=1.336 cw=0.1
  be=150 [all_fixed     ] tr4=1.000 h4=0.484 hB=+5.348 hSel=+0.362 ctx_ce=1.325 cw=0.1
  be=200 [all_fixed     ] tr4=1.000 h4=0.488 hB=+5.113 hSel=+0.351 ctx_ce=1.304 cw=0.1
  be=250 [all_fixed     ] tr4=1.000 h4=0.500 hB=+5.006 hSel=+0.353 ctx_ce=1.287 cw=0.1
  be=300 [all_fixed     ] tr4=1.000 h4=0.492 hB=+4.637 hSel=+0.351 ctx_ce=1.272 cw=0.1
  be=350 [all_fixed     ] tr4=1.000 h4=0.520 hB=+5.912 hSel=+0.390 ctx_ce=1.259 cw=0.1
  be=400 [all_fixed     ] tr4=1.000 h4=0.500 hB=+5.337 hSel=+0.349 ctx_ce=1.250 cw=0.1
  be=450 [all_fixed     ] tr4=1.000 h4=0.492 hB=+5.284 hSel=+0.335 ctx_ce=1.247 cw=0.1
  be=500 [all_fixed     ] tr4=1.000 h4=0.500 hB=+5.203 hSel=+0.332 ctx_ce=1.235 cw=0.1

**constant_ctx_1over17**
  be=  0 [prep          ] tr4=0.986 h4=0.484 hB=+3.833 hSel=+0.365 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.934 h4=0.484 hB=+4.417 hSel=+0.378 ctx_ce=31.425 cw=0.0588
  be=  5 [all_fixed     ] tr4=0.945 h4=0.488 hB=+4.568 hSel=+0.374 ctx_ce=8.331 cw=0.0588
  be= 10 [all_fixed     ] tr4=0.975 h4=0.480 hB=+4.427 hSel=+0.367 ctx_ce=3.338 cw=0.0588
  be= 25 [all_fixed     ] tr4=0.986 h4=0.484 hB=+4.433 hSel=+0.352 ctx_ce=1.485 cw=0.0588
  be= 50 [all_fixed     ] tr4=0.996 h4=0.480 hB=+4.739 hSel=+0.367 ctx_ce=1.372 cw=0.0588
  be= 75 [all_fixed     ] tr4=0.998 h4=0.480 hB=+5.084 hSel=+0.360 ctx_ce=1.356 cw=0.0588
  be=100 [all_fixed     ] tr4=0.998 h4=0.488 hB=+5.209 hSel=+0.363 ctx_ce=1.346 cw=0.0588
  be=125 [all_fixed     ] tr4=1.000 h4=0.484 hB=+5.242 hSel=+0.361 ctx_ce=1.342 cw=0.0588
  be=150 [all_fixed     ] tr4=1.000 h4=0.484 hB=+5.242 hSel=+0.352 ctx_ce=1.331 cw=0.0588
  be=200 [all_fixed     ] tr4=1.000 h4=0.484 hB=+5.482 hSel=+0.363 ctx_ce=1.321 cw=0.0588
  be=250 [all_fixed     ] tr4=0.988 h4=0.504 hB=+5.397 hSel=+0.373 ctx_ce=1.303 cw=0.0588
  be=300 [all_fixed     ] tr4=1.000 h4=0.488 hB=+4.568 hSel=+0.326 ctx_ce=1.294 cw=0.0588
  be=350 [all_fixed     ] tr4=1.000 h4=0.488 hB=+4.344 hSel=+0.324 ctx_ce=1.270 cw=0.0588
  be=400 [all_fixed     ] tr4=1.000 h4=0.484 hB=+4.277 hSel=+0.327 ctx_ce=1.261 cw=0.0588
  be=450 [all_fixed     ] tr4=0.994 h4=0.445 hB=+4.440 hSel=+0.277 ctx_ce=1.253 cw=0.0588
  be=500 [all_fixed     ] tr4=1.000 h4=0.547 hB=+5.485 hSel=+0.404 ctx_ce=1.244 cw=0.0588

### Seed 43

**constant_ctx_half**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.752 h4=0.703 hB=+3.599 hSel=+0.498 ctx_ce=29.000 cw=0.5
  be=  5 [all_fixed     ] tr4=0.289 h4=0.254 hB=+0.003 hSel=+0.003 ctx_ce=6.648 cw=0.5
  be= 10 [all_fixed     ] tr4=0.279 h4=0.258 hB=-0.018 hSel=-0.002 ctx_ce=2.018 cw=0.5
  be= 25 [all_fixed     ] tr4=0.279 h4=0.285 hB=-0.028 hSel=-0.003 ctx_ce=1.408 cw=0.5
  be= 50 [all_fixed     ] tr4=0.236 h4=0.230 hB=-0.032 hSel=-0.007 ctx_ce=1.366 cw=0.5
  be= 75 [all_fixed     ] tr4=0.234 h4=0.254 hB=-0.027 hSel=-0.016 ctx_ce=1.355 cw=0.5
  be=100 [all_fixed     ] tr4=0.230 h4=0.254 hB=-0.001 hSel=+0.004 ctx_ce=1.351 cw=0.5
  be=125 [all_fixed     ] tr4=0.473 h4=0.320 hB=+0.677 hSel=+0.102 ctx_ce=1.337 cw=0.5
  be=150 [all_fixed     ] tr4=0.504 h4=0.336 hB=+1.041 hSel=+0.154 ctx_ce=1.317 cw=0.5
  be=200 [all_fixed     ] tr4=0.549 h4=0.281 hB=+1.083 hSel=+0.159 ctx_ce=1.279 cw=0.5
  be=250 [all_fixed     ] tr4=1.000 h4=0.520 hB=+3.332 hSel=+0.368 ctx_ce=1.254 cw=0.5
  be=300 [all_fixed     ] tr4=1.000 h4=0.496 hB=+4.168 hSel=+0.376 ctx_ce=1.245 cw=0.5
  be=350 [all_fixed     ] tr4=1.000 h4=0.523 hB=+4.476 hSel=+0.387 ctx_ce=1.240 cw=0.5
  be=400 [all_fixed     ] tr4=1.000 h4=0.516 hB=+4.600 hSel=+0.369 ctx_ce=1.239 cw=0.5
  be=450 [all_fixed     ] tr4=1.000 h4=0.547 hB=+5.119 hSel=+0.400 ctx_ce=1.234 cw=0.5
  be=500 [all_fixed     ] tr4=1.000 h4=0.547 hB=+4.950 hSel=+0.388 ctx_ce=1.235 cw=0.5

**constant_ctx_tenth**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.830 h4=0.754 hB=+4.121 hSel=+0.578 ctx_ce=29.022 cw=0.1
  be=  5 [all_fixed     ] tr4=0.816 h4=0.840 hB=+2.936 hSel=+0.571 ctx_ce=6.995 cw=0.1
  be= 10 [all_fixed     ] tr4=0.992 h4=0.898 hB=+4.399 hSel=+0.705 ctx_ce=2.219 cw=0.1
  be= 25 [all_fixed     ] tr4=1.000 h4=0.836 hB=+5.479 hSel=+0.717 ctx_ce=1.407 cw=0.1
  be= 50 [all_fixed     ] tr4=1.000 h4=0.820 hB=+6.404 hSel=+0.713 ctx_ce=1.365 cw=0.1
  be= 75 [all_fixed     ] tr4=1.000 h4=0.812 hB=+6.811 hSel=+0.701 ctx_ce=1.350 cw=0.1
  be=100 [all_fixed     ] tr4=1.000 h4=0.801 hB=+7.138 hSel=+0.686 ctx_ce=1.340 cw=0.1
  be=125 [all_fixed     ] tr4=1.000 h4=0.859 hB=+7.775 hSel=+0.749 ctx_ce=1.323 cw=0.1
  be=150 [all_fixed     ] tr4=1.000 h4=0.832 hB=+7.585 hSel=+0.724 ctx_ce=1.303 cw=0.1
  be=200 [all_fixed     ] tr4=1.000 h4=0.762 hB=+7.554 hSel=+0.621 ctx_ce=1.262 cw=0.1
  be=250 [all_fixed     ] tr4=1.000 h4=0.730 hB=+7.385 hSel=+0.569 ctx_ce=1.253 cw=0.1
  be=300 [all_fixed     ] tr4=1.000 h4=0.750 hB=+8.088 hSel=+0.602 ctx_ce=1.244 cw=0.1
  be=350 [all_fixed     ] tr4=1.000 h4=0.719 hB=+7.724 hSel=+0.569 ctx_ce=1.237 cw=0.1
  be=400 [all_fixed     ] tr4=1.000 h4=0.715 hB=+7.751 hSel=+0.542 ctx_ce=1.239 cw=0.1
  be=450 [all_fixed     ] tr4=1.000 h4=0.766 hB=+7.171 hSel=+0.633 ctx_ce=1.236 cw=0.1
  be=500 [all_fixed     ] tr4=1.000 h4=0.793 hB=+7.476 hSel=+0.614 ctx_ce=1.234 cw=0.1

**constant_ctx_1over17**
  be=  0 [prep          ] tr4=1.000 h4=0.820 hB=+5.504 hSel=+0.726 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.877 h4=0.777 hB=+4.453 hSel=+0.613 ctx_ce=29.056 cw=0.0588
  be=  5 [all_fixed     ] tr4=0.967 h4=0.938 hB=+4.310 hSel=+0.745 ctx_ce=7.251 cw=0.0588
  be= 10 [all_fixed     ] tr4=1.000 h4=0.918 hB=+4.921 hSel=+0.739 ctx_ce=2.257 cw=0.0588
  be= 25 [all_fixed     ] tr4=1.000 h4=0.844 hB=+5.982 hSel=+0.730 ctx_ce=1.411 cw=0.0588
  be= 50 [all_fixed     ] tr4=1.000 h4=0.816 hB=+6.839 hSel=+0.715 ctx_ce=1.366 cw=0.0588
  be= 75 [all_fixed     ] tr4=1.000 h4=0.801 hB=+7.268 hSel=+0.672 ctx_ce=1.351 cw=0.0588
  be=100 [all_fixed     ] tr4=1.000 h4=0.793 hB=+7.218 hSel=+0.672 ctx_ce=1.342 cw=0.0588
  be=125 [all_fixed     ] tr4=1.000 h4=0.793 hB=+7.213 hSel=+0.632 ctx_ce=1.327 cw=0.0588
  be=150 [all_fixed     ] tr4=1.000 h4=0.762 hB=+6.950 hSel=+0.601 ctx_ce=1.308 cw=0.0588
  be=200 [all_fixed     ] tr4=1.000 h4=0.879 hB=+9.281 hSel=+0.777 ctx_ce=1.278 cw=0.0588
  be=250 [all_fixed     ] tr4=1.000 h4=0.879 hB=+8.880 hSel=+0.717 ctx_ce=1.258 cw=0.0588
  be=300 [all_fixed     ] tr4=1.000 h4=0.828 hB=+8.275 hSel=+0.671 ctx_ce=1.246 cw=0.0588
  be=350 [all_fixed     ] tr4=1.000 h4=0.824 hB=+8.101 hSel=+0.644 ctx_ce=1.238 cw=0.0588
  be=400 [all_fixed     ] tr4=1.000 h4=0.801 hB=+8.147 hSel=+0.635 ctx_ce=1.240 cw=0.0588
  be=450 [all_fixed     ] tr4=1.000 h4=0.785 hB=+8.262 hSel=+0.621 ctx_ce=1.234 cw=0.0588
  be=500 [all_fixed     ] tr4=1.000 h4=0.758 hB=+8.239 hSel=+0.589 ctx_ce=1.234 cw=0.0588

### Seed 100

**constant_ctx_half**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.844 h4=0.652 hB=+3.468 hSel=+0.458 ctx_ce=25.865 cw=0.5
  be=  5 [all_fixed     ] tr4=0.316 h4=0.254 hB=-0.098 hSel=-0.011 ctx_ce=6.609 cw=0.5
  be= 10 [all_fixed     ] tr4=0.309 h4=0.227 hB=-0.074 hSel=-0.010 ctx_ce=2.936 cw=0.5
  be= 25 [all_fixed     ] tr4=0.430 h4=0.395 hB=+0.307 hSel=+0.084 ctx_ce=1.445 cw=0.5
  be= 50 [all_fixed     ] tr4=0.531 h4=0.484 hB=+1.364 hSel=+0.238 ctx_ce=1.356 cw=0.5
  be= 75 [all_fixed     ] tr4=0.545 h4=0.465 hB=+1.539 hSel=+0.222 ctx_ce=1.340 cw=0.5
  be=100 [all_fixed     ] tr4=0.502 h4=0.395 hB=+1.399 hSel=+0.166 ctx_ce=1.325 cw=0.5
  be=125 [all_fixed     ] tr4=1.000 h4=0.453 hB=+2.373 hSel=+0.295 ctx_ce=1.309 cw=0.5
  be=150 [all_fixed     ] tr4=1.000 h4=0.488 hB=+3.084 hSel=+0.364 ctx_ce=1.278 cw=0.5
  be=200 [all_fixed     ] tr4=1.000 h4=0.418 hB=+2.393 hSel=+0.271 ctx_ce=1.257 cw=0.5
  be=250 [all_fixed     ] tr4=1.000 h4=0.367 hB=+2.185 hSel=+0.240 ctx_ce=1.244 cw=0.5
  be=300 [all_fixed     ] tr4=1.000 h4=0.309 hB=+2.038 hSel=+0.216 ctx_ce=1.240 cw=0.5
  be=350 [all_fixed     ] tr4=1.000 h4=0.328 hB=+2.104 hSel=+0.226 ctx_ce=1.231 cw=0.5
  be=400 [all_fixed     ] tr4=1.000 h4=0.297 hB=+2.018 hSel=+0.194 ctx_ce=1.238 cw=0.5
  be=450 [all_fixed     ] tr4=1.000 h4=0.273 hB=+1.792 hSel=+0.160 ctx_ce=1.232 cw=0.5
  be=500 [all_fixed     ] tr4=1.000 h4=0.309 hB=+1.886 hSel=+0.206 ctx_ce=1.233 cw=0.5

**constant_ctx_tenth**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.967 h4=0.711 hB=+4.319 hSel=+0.534 ctx_ce=25.872 cw=0.1
  be=  5 [all_fixed     ] tr4=0.977 h4=0.898 hB=+3.318 hSel=+0.655 ctx_ce=7.057 cw=0.1
  be= 10 [all_fixed     ] tr4=1.000 h4=0.895 hB=+3.357 hSel=+0.636 ctx_ce=3.009 cw=0.1
  be= 25 [all_fixed     ] tr4=1.000 h4=0.699 hB=+4.197 hSel=+0.576 ctx_ce=1.431 cw=0.1
  be= 50 [all_fixed     ] tr4=1.000 h4=0.625 hB=+4.221 hSel=+0.520 ctx_ce=1.353 cw=0.1
  be= 75 [all_fixed     ] tr4=1.000 h4=0.578 hB=+4.024 hSel=+0.456 ctx_ce=1.341 cw=0.1
  be=100 [all_fixed     ] tr4=1.000 h4=0.531 hB=+3.799 hSel=+0.417 ctx_ce=1.328 cw=0.1
  be=125 [all_fixed     ] tr4=1.000 h4=0.691 hB=+7.159 hSel=+0.655 ctx_ce=1.315 cw=0.1
  be=150 [all_fixed     ] tr4=1.000 h4=0.719 hB=+7.445 hSel=+0.604 ctx_ce=1.291 cw=0.1
  be=200 [all_fixed     ] tr4=1.000 h4=0.770 hB=+8.031 hSel=+0.687 ctx_ce=1.265 cw=0.1
  be=250 [all_fixed     ] tr4=1.000 h4=0.758 hB=+9.137 hSel=+0.685 ctx_ce=1.244 cw=0.1
  be=300 [all_fixed     ] tr4=1.000 h4=0.754 hB=+8.968 hSel=+0.678 ctx_ce=1.239 cw=0.1
  be=350 [all_fixed     ] tr4=1.000 h4=0.727 hB=+8.223 hSel=+0.667 ctx_ce=1.233 cw=0.1
  be=400 [all_fixed     ] tr4=1.000 h4=0.707 hB=+7.926 hSel=+0.618 ctx_ce=1.237 cw=0.1
  be=450 [all_fixed     ] tr4=1.000 h4=0.680 hB=+7.764 hSel=+0.604 ctx_ce=1.231 cw=0.1
  be=500 [all_fixed     ] tr4=1.000 h4=0.664 hB=+7.582 hSel=+0.600 ctx_ce=1.234 cw=0.1

**constant_ctx_1over17**
  be=  0 [prep          ] tr4=1.000 h4=0.953 hB=+7.774 hSel=+0.914 ctx_ce=0.000 cw=0.0
  be=  1 [all_fixed     ] tr4=0.973 h4=0.750 hB=+4.716 hSel=+0.575 ctx_ce=25.878 cw=0.0588
  be=  5 [all_fixed     ] tr4=0.996 h4=0.930 hB=+3.870 hSel=+0.724 ctx_ce=7.173 cw=0.0588
  be= 10 [all_fixed     ] tr4=1.000 h4=0.895 hB=+3.943 hSel=+0.692 ctx_ce=3.045 cw=0.0588
  be= 25 [all_fixed     ] tr4=1.000 h4=0.734 hB=+5.016 hSel=+0.627 ctx_ce=1.435 cw=0.0588
  be= 50 [all_fixed     ] tr4=1.000 h4=0.664 hB=+4.979 hSel=+0.570 ctx_ce=1.354 cw=0.0588
  be= 75 [all_fixed     ] tr4=1.000 h4=0.590 hB=+4.705 hSel=+0.495 ctx_ce=1.343 cw=0.0588
  be=100 [all_fixed     ] tr4=1.000 h4=0.574 hB=+4.558 hSel=+0.475 ctx_ce=1.330 cw=0.0588
  be=125 [all_fixed     ] tr4=1.000 h4=0.512 hB=+4.096 hSel=+0.386 ctx_ce=1.315 cw=0.0588
  be=150 [all_fixed     ] tr4=1.000 h4=0.758 hB=+8.635 hSel=+0.728 ctx_ce=1.303 cw=0.0588
  be=200 [all_fixed     ] tr4=1.000 h4=0.762 hB=+8.950 hSel=+0.728 ctx_ce=1.275 cw=0.0588
  be=250 [all_fixed     ] tr4=1.000 h4=0.848 hB=+9.841 hSel=+0.824 ctx_ce=1.253 cw=0.0588
  be=300 [all_fixed     ] tr4=1.000 h4=0.848 hB=+10.409 hSel=+0.805 ctx_ce=1.246 cw=0.0588
  be=350 [all_fixed     ] tr4=1.000 h4=0.855 hB=+10.380 hSel=+0.806 ctx_ce=1.235 cw=0.0588
  be=400 [all_fixed     ] tr4=1.000 h4=0.828 hB=+9.687 hSel=+0.789 ctx_ce=1.239 cw=0.0588
  be=450 [all_fixed     ] tr4=1.000 h4=0.793 hB=+9.200 hSel=+0.745 ctx_ce=1.233 cw=0.0588
  be=500 [all_fixed     ] tr4=1.000 h4=0.777 hB=+8.919 hSel=+0.731 ctx_ce=1.236 cw=0.0588

## Interpretation

If any compatible trajectory (gradual ramp, slow lr, interleaved reinforcement) preserves held binding while context CE drops to normal levels (~1.5 nats), the conflict is transitional and a compatible learning path exists.  If all fail, context prediction through the body inherently conflicts with the binding computation.

Readout calibration may fail to reduce context CE enough (body objective ablation showed out_only barely reduced it from ~34 to ~28 in 25 epochs), so its failure alone does not establish inherent conflict.  The gradual ramp is the strongest test because it limits gradient magnitude at every epoch.

