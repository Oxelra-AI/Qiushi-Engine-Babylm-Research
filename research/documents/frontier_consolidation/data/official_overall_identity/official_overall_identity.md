# earlier analysis official Strict-Small Overall(AoA0) identity

This records the live Space formula from the earlier analysis clone and applies it to current endpoints.

## Source-grounded formula

- `read_evals.py` SHA `f7e99e7c8e8ed07353ce5c7277fd226f9b88b0201aeed12fdabf4e0fd821c76c` lines 363-377 define `overall_average = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GLUE, GlobalPIQA, Reading, AoA)`.
- `eval_submission.py` SHA `15e6bb9317cb225919d45812b115d8d5c3c0a943b31a560afaa795b6d8d3d05c` lines 80-87 maps missing/malformed scalar AoA to 0.0; `read_evals.py` lines 204-208 zero AoA when `aoa_surprisals` is absent.
- `check_validity.py` SHA `ce762ff75c0e2118c3ae57ae68e9f09a4075733c575c31dc7877736007f0b4ff` lines 209-220 and 292-295 allow missing AoA and missing fast results, although `submit.py` later warns that fast history is required for challenge submissions.

Therefore for these MLM carriers with scalar AoA 0 and known cheap7:

`Overall(AoA0) = (7 * cheap7 + SuperGLUE) / 9`

## Rows

| arm | cheap7 | SuperGLUE used | Overall formula | Δ vs chck82 | SG needed to match chck82 |
|---|---:|---:|---:|---:|---:|
| chck82_anchor | 43.959449876452 | 69.766181371312 | 41.942481167386 | +0.000000000000 |  |
| coherent86_private_alpha1 | 44.106428571429 | 69.777968264287 | 42.058107584921 | +0.115626417535 | 68.737330506474 |
| coherent86_private_alpha0p5 | 44.177857142857 | 69.777968264287 | 42.113663140476 | +0.171181973090 | 68.237330506474 |
| coherent86_private_alpha0p75 | 44.181428571429 | 69.777968264287 | 42.116440918254 | +0.173959750868 | 68.212330506474 |

chck82 identity residual: `+0.000000000000e+00`.

Caveat: this establishes the arithmetic identity for the evaluated version of the leaderboard application. A public challenge-counted row still depends on an accepted submission and the application recomputing the submitted results. This analysis is not evidence of leaderboard acceptance.

JSON: `experiments/archive/frontier_consolidation/data/official_overall_identity/official_overall_identity.json`
