# lamb update allocation closure LAMB training dynamics

Existing logs only; no new training.

## Run endpoints

| run | rows | last step | words | loss | tail20 loss | complete 20M |
|---|---:|---:|---:|---:|---:|---:|
| spatial repair route status AdamW legal reference | 2529 | 2529 | 100000000 | 2.5526 | 2.4787 | True |
| Muon hidden wd-matched | 506 | 506 | 20000000 | 3.5528 | 3.4967 | True |
| LAMB lr=0.005 partial timeout | 342 | 342 | 13525431 | 4.7883 | 4.7768 | False |
| LAMB lr=0.007 complete | 506 | 506 | 20000000 | 6.8976 | 6.8812 | True |

## Matched/nearest losses

| target words | spatial repair route status loss | Muon loss Δ | LAMB005 loss Δ | LAMB007 loss Δ |
|---:|---:|---:|---:|---:|
| 1000000 | 8.2528 | +0.4163 | +0.1419 | -0.1542 |
| 2000000 | 6.7716 | +0.0897 | +0.2165 | +0.0897 |
| 5000000 | 5.8707 | +0.1419 | +0.2520 | +1.2799 |
| 10000000 | 4.3420 | -0.1108 | +1.0887 | +2.5911 |
| 14000000 | 3.8706 | -0.1405 | +0.9177 | +2.9504 |
| 15000000 | 3.9411 | -0.1376 | +0.8471 | +2.9636 |
| 20000000 | 3.7556 | -0.2028 | +1.0327 | +3.1420 |

## Last available comparison

| run | words | loss | spatial repair route status loss at nearest words | Δloss | last LR |
|---|---:|---:|---:|---:|---:|
| lamb005_partial | 13525431 | 4.7883 | 4.0233 | +0.7649 | 0.00132697 |
| lamb007_20M | 20000000 | 6.8976 | 3.7556 | +3.1420 | 0 |
| muon_wdmatched_20M | 20000000 | 3.5528 | 3.7556 | -0.2028 | 0.0075681 |

Interpretation: lr0.007 completing 20M with a high tail loss would mean this arm is not a mature-candidate optimizer even if some cheap columns move. lr0.005 timing out after 14M is a resource/runtime fact; its scientific status should be judged by the 14M cheap evaluation and by whether its update-allocation readout preserves the intended Adam-direction mechanism.
