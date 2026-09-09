# mature muon reversal and switch repair mature matched-decay Muon mechanism summary

This readout uses training logs and hidden-matrix checkpoints only; it is not a BabyLM evaluation result.

## Training-loss milestones

| words | spatial repair route status loss | Muon 80-run loss | Δ loss | Muon logged group0 lr | spatial repair route status lr |
|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 8.3555 | 8.7782 | +0.4227 | 0.0013245 | 0.000165563 |
| 5,000,000 | 5.9490 | 6.0520 | +0.1030 | 0.0066755 | 0.000834437 |
| 10,000,000 | 4.3420 | 4.2312 | -0.1108 | 0.00796374 | 0.000995467 |
| 20,000,000 | 3.7556 | 3.4842 | -0.2714 | 0.0075681 | 0.000946012 |
| 30,000,000 | 3.4103 | 3.1409 | -0.2694 | 0.00677754 | 0.000847192 |
| 40,000,000 | 3.0069 | 2.8429 | -0.1640 | 0.00567956 | 0.000709945 |
| 50,000,000 | 2.8504 | 2.7312 | -0.1192 | 0.00440094 | 0.000550118 |
| 60,000,000 | 2.8293 | 2.7492 | -0.0801 | 0.00307316 | 0.000384145 |
| 70,000,000 | 2.5260 | 2.4406 | -0.0854 | 0.00184795 | 0.000230994 |
| 80,000,000 | 2.5405 | 2.3978 | -0.1427 | 0.000857669 | 0.000107618 |

## Hidden attention/FFN matrix spectra

| model | Frobenius/init | stable rank | entropy rank | top8 energy | stable-rank Δ vs init | top8 Δ vs init |
|---|---:|---:|---:|---:|---:|---:|
| reference_20M | 1.2179 | 43.96 | 274.82 | 0.1332 | -109.01 | +0.0798 |
| muon_wdmatch_20M | 1.6019 | 121.29 | 325.65 | 0.0607 | -31.68 | +0.0072 |
| reference_70M | 1.4692 | 42.84 | 247.69 | 0.1370 | -110.13 | +0.0835 |
| muon_wdmatch_70M | 2.1085 | 108.48 | 322.97 | 0.0622 | -44.49 | +0.0087 |
| reference_80M | 1.4731 | 42.80 | 247.59 | 0.1368 | -110.17 | +0.0833 |
| muon_wdmatch_80M | 2.1112 | 107.99 | 322.94 | 0.0623 | -44.98 | +0.0088 |

## Muon minus spatial repair route status at matched exposure

| ckpt | stable-rank Δ | entropy-rank Δ | top8-energy Δ | Frobenius-norm Δ |
|---|---:|---:|---:|---:|
| 20M | +77.33 | +50.83 | -0.0726 | +4.61 |
| 70M | +65.64 | +75.28 | -0.0748 | +7.53 |
| 80M | +65.19 | +75.34 | -0.0745 | +7.51 |
