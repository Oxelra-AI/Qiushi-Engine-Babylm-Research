# Per-Relation Transport Analysis: h1 (dist=1) vs h3 (dist=2)

Separates h1 and h3 in existing saved outputs to test whether
transport quality varies with graph distance from anchored nodes.

Graph: h0--h1, h0--h2, h1--h2, h2--h3
Anchors: h0, h2 (state training rows with bridge_sign flip)
h1: distance 1 from h0 (direct edge), distance 1 from h2 (direct edge)
h3: distance 1 from h2, distance 2 from h0 (via h2)

## earlier analysis learned-equality full-alpha, shared_trunk, seed29930

| Relation | GraphDist | Matched | OppFrac | AntiCorr | MeanMarg+ | MeanMarg- | AbsMarg+ | AbsMarg- |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | anchor | 96 | 1.000 | 1.0000 | 1.275 | 1.391 | 18.451 | 13.613 |
| h2_norp | anchor | 96 | 1.000 | 1.0000 | -2.597 | 1.432 | 17.760 | 15.608 |
| h1_mep | dist=1 | 96 | 1.000 | 1.0000 | 0.653 | 0.728 | 17.779 | 13.782 |
| h3_ziv | dist=2 | 96 | 1.000 | 1.0000 | -2.846 | 1.400 | 16.863 | 14.878 |

## earlier analysis frozen posalign, shared_trunk, seed29920

| Relation | GraphDist | Matched | OppFrac | AntiCorr | MeanMarg+ | MeanMarg- | AbsMarg+ | AbsMarg- |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | anchor | 96 | 1.000 | 1.0000 | 0.516 | 3.075 | 11.212 | 14.487 |
| h2_norp | anchor | 96 | 1.000 | 1.0000 | 2.530 | 5.461 | 12.665 | 16.360 |
| h1_mep | dist=1 | 96 | 1.000 | 1.0000 | -0.501 | 4.545 | 10.938 | 16.111 |
| h3_ziv | dist=2 | 96 | 1.000 | 1.0000 | 2.454 | 5.030 | 11.766 | 16.410 |

