# paired factorial result and bridge paired factorial and update probe

## Why this run exists

corruption vs loss design used different arm-dependent corruption streams. This run pairs the corrupted arms by deterministic epoch/batch seeds and resets dropout seeds before each forward pass. It also removes duplicated groups and cycles query side independently from source order.

## Endpoint results

| Arm | Held flip | Held U | Held R | Held N | Train flip | bg corrupted positions | answer labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired_corrupted_answer_only | 2/12 | +1.937 | -2.007 | +1.844 | 33/36 | 187731 | 36000 |
| paired_corrupted_answer_plus_bg | 1/12 | +1.284 | -0.979 | +1.306 | 20/36 | 187731 | 36000 |

## Update probe

- Answer/bg gradient cosine at answer-only snapshot: +0.0413
- Norms: answer 2.39095, bg 8.39645, combined 8.82466
- Clip coefficients if max norm 1.0: answer 0.4182, bg 0.1191, combined 0.1133

One-step fixed-batch consequences:

| Update | Δ answer loss | Δ bg loss | Δ held flip | Δ held U | Δ held R |
|---|---:|---:|---:|---:|---:|
| answer | -0.023793 | -0.023685 | 0 | -0.072462 | +0.077705 |
| bg | -0.003784 | -0.107630 | 0 | -0.090957 | +0.094410 |
| combined | -0.009868 | -0.104371 | 0 | -0.095032 | +0.098392 |

## Interpretation

Paired corrupted answer-only: held flip=2/12 U=+1.937 R=-2.007 N=+1.844
Paired corrupted answer+bg: held flip=1/12 U=+1.284 R=-0.979 N=+1.306
The two corrupted arms have matched cumulative background-corruption counts under the deterministic stream.
The paired endpoint tests whether adding background loss under identical corruption/dropout streams changes recipient-sensitive behavior in this controlled template optimizer; it does not by itself establish ordinary ALN dynamics or a universal gradient-direction law.
Update probe at answer-only snapshot: ||g_answer||=2.391, ||g_bg||=8.396, cos(answer,bg)=+0.0413, combined clip coef=0.1133.
