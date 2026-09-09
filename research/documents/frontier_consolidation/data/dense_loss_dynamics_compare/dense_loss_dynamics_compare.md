# dense and causal evaluation refined plan dense loss/dynamics comparison

CPU-only analysis; not a replacement for official cheap7 scoring.

## Reference loss does not explain the known 82M cheap7 peak

- Pearson raw loss vs known cheap7 over available reference points: -0.5397372123990473 (n=4)
- Pearson ±1M mean loss vs known cheap7: 0.41369162647396723
- The reference rolling loss keeps improving after 82M while cheap7 falls by 100M, so loss-derived peak prediction remains unsupported.

## Loss snapshots

| M | ref loss | s1.75/43122 loss | s1.25/43022 loss | ref cheap7 |
|---:|---:|---:|---:|---:|
| 20 | 3.765514850616455 | 3.839345693588257 | 3.7635979652404785 |  |
| 50 | 2.841961145401001 | 2.8187849521636963 | 2.8609042167663574 |  |
| 70 | 2.539531946182251 | 2.5937697887420654 | 2.535033941268921 |  |
| 76 | 2.533445358276367 | 2.57184100151062 | 2.537837028503418 |  |
| 78 | 2.5940093994140625 | 2.5663001537323 | 2.5968456268310547 | 43.70214285714286 |
| 80 | 2.5337066650390625 | 2.4070653915405273 | 2.5303800106048584 | 43.81214285714286 |
| 82 | 2.5113165378570557 | 2.553424119949341 | 2.5210344791412354 | 43.95944987645173 |
| 84 | 2.455108165740967 | 2.5246009826660156 | 2.466435670852661 |  |
| 86 | 2.5804686546325684 | 2.5552079677581787 | 2.5774495601654053 |  |
| 90 | 2.5149521827697754 | 2.533233642578125 | 2.518465757369995 |  |
| 94 | 2.4787936210632324 | 2.491201400756836 | 2.4867584705352783 |  |
| 100 | 2.5441808700561523 | 2.6495566368103027 | 2.5401194095611572 | 43.543159919261925 |

## Dynamics interpretation

The sparse frequency-band traces begin producing nonzero band metrics at 20M and are too coarse to choose endpoints, but they can reveal gross collapse; no obvious collapse signal appears in the training artifacts alone. Official cheap7 family trajectories are still the decisive evidence.

JSON: `experiments/archive/frontier_consolidation/data/dense_loss_dynamics_compare/dense_loss_dynamics_compare.json`
