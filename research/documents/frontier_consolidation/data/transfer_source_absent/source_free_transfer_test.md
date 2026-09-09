# source free transfer synthesis — Source-free transfer test

Items: 1864 total (1287 train-doc, 577 test-doc)
Documents: 3171 train, 1358 test
Source-absent targets: 1864
Probe: bottleneck 32, dropout 0.15, L2 0.001, 30 epochs
Probe seeds: 3

## Raw frozen NLL (diagnostic)
  free: train 6.6627, test 6.4930
  true: train 7.1369, test 7.1819
  shuffled: train 6.5250, test 6.4278
  true-vs-shuffled test: 0.7541

## spatial repair route status baseline test NLL (no probe): 6.4930

## Transfer results (train-conditioned → eval source-free)
| Condition | Mean eval NLL | Std | vs spatial repair route status baseline |
|-----------|--------------|-----|-------------------|
| true_conditioned | 6.1378 | 0.0444 | -0.3552 |
| shuffled_conditioned | 6.0367 | 0.0288 | -0.4563 |
| source_free | 5.8455 | 0.0368 | -0.6474 |
| shared_true | 5.5977 | 0.0375 | -0.8953 |
| shared_shuffled | 5.9210 | 0.0254 | -0.5720 |

## Decisive comparison: true vs shuffled (source-free eval)
  True mean: 6.1378
  Shuffled mean: 6.0367
  Delta (true - shuffled): 0.1012
  Paired bootstrap CI: [-0.0571, 0.1544]

  Interpretation: SHUFFLED BETTER (anti-transfer)

JSON: `experiments/archive/frontier_consolidation/data/transfer_source_absent/source_free_transfer_test.json`
