# source free transfer synthesis — Source-free transfer test

Items: 3960 total (2726 train-doc, 1234 test-doc)
Documents: 3171 train, 1358 test
Source-absent targets: 1864
Probe: bottleneck 32, dropout 0.15, L2 0.001, 30 epochs
Probe seeds: 3

## Raw frozen NLL (diagnostic)
  free: train 6.9887, test 6.8451
  true: train 3.9269, test 4.0403
  shuffled: train 6.8965, test 6.7513
  true-vs-shuffled test: -2.7110

## spatial repair route status baseline test NLL (no probe): 6.8451

## Transfer results (train-conditioned → eval source-free)
| Condition | Mean eval NLL | Std | vs spatial repair route status baseline |
|-----------|--------------|-----|-------------------|
| true_conditioned | 8.2811 | 0.1049 | +1.4360 |
| shuffled_conditioned | 8.4501 | 0.0908 | +1.6050 |
| source_free | 8.5312 | 0.0321 | +1.6861 |
| shared_true | 7.3997 | 0.0377 | +0.5547 |
| shared_shuffled | 8.4068 | 0.0738 | +1.5617 |

## Decisive comparison: true vs shuffled (source-free eval)
  True mean: 8.2811
  Shuffled mean: 8.4501
  Delta (true - shuffled): -0.1690
  Paired bootstrap CI: [-0.4112, -0.0977]

  Interpretation: TRANSFER EXISTS

JSON: `experiments/archive/frontier_consolidation/data/transfer_all_edit/source_free_transfer_test.json`
