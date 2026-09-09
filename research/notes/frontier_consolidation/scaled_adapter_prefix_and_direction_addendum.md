# Scaled-Adapter Prefix Identity and Update Direction

Status: completed 20M-prefix comparison, matched 50M evaluation and mechanism measurements; later exposure remained conditional.

The standalone 20M scale-1.75 run and the 20M checkpoint inside the 50M run matched over 506 normalized training rows. Their final common row had 20,008,711 words, loss 3.765514850616455 and learning rate 0.0009460119426666896. Checkpoint hashes and all 218 tensors matched exactly, with global L2 difference 0.0. The early cheap7 +0.642 signal was therefore not explained by a mismatched prefix.

| 50M coordinate | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legal reference | 41.9729 | 63.800 | 59.110 | 48.460 | 26.260 | 51.610 | 36.580 | 7.990 |
| Adapter-128, scale 1.75 | 42.3843 | 65.640 | 59.070 | 47.730 | 29.050 | 52.340 | 35.105 | 7.755 |

The cheap7 difference was +0.4114, with gains in BLiMP, Entity and COMPS but losses in EWoK and GlobalPIQA. This justified further trajectory inspection, not a uniform-improvement claim.

Stock-weight cosine/relative-L2 against the reference were 0.8791/0.4913 at 20M and 0.7559/0.6982 at 50M. The 20M-to-50M update-norm ratio was 0.9991, but update-vector cosine was only 0.3050. Adapter RMS changed from 0.066006 to 0.066315, up-projection RMS from 0.012710 to 0.018637. Core stable-rank means were 32.754 versus 32.963. The interpretation was a redirected residual/backbone co-trajectory at similar update magnitude, without a broad spectral-rank signature.

An attempted 80M invocation failed before training because its argument names were incompatible. It supplied no negative scientific result. The corrected proposed continuation retained bottleneck 128, scale 1.75 and the 2,529-update horizon, and required 70M/80M cheap-column evidence before 100M/full evaluation.
