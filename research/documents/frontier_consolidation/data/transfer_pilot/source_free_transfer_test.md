# source free transfer synthesis — Source-free transfer test

Items: 128 total (86 train-doc, 42 test-doc)
Documents: 3171 train, 1358 test
Source-absent targets: 63
Probe: bottleneck 16, dropout 0.15, L2 0.001, 5 epochs
Probe seeds: 1

## Raw frozen NLL (diagnostic)
  free: train 7.0828, test 7.1002
  true: train 4.2238, test 4.1249
  shuffled: train 6.9709, test 6.9944
  true-vs-shuffled test: -2.8696

## spatial repair route status baseline test NLL (no probe): 7.1002

## Transfer results (train-conditioned → eval source-free)
| Condition | Mean eval NLL | Std | vs spatial repair route status baseline |
|-----------|--------------|-----|-------------------|
| true_conditioned | 7.0878 | 0.0000 | -0.0125 |
| shuffled_conditioned | 7.0853 | 0.0000 | -0.0149 |
| source_free | 7.0743 | 0.0000 | -0.0259 |
| shared_true | 7.0833 | 0.0000 | -0.0170 |
| shared_shuffled | 7.0753 | 0.0000 | -0.0249 |

## Decisive comparison: true vs shuffled (source-free eval)
  True mean: 7.0878
  Shuffled mean: 7.0853
  Delta (true - shuffled): 0.0024
  Paired bootstrap CI: [-0.0048, 0.0098]

  Interpretation: NO CLEAR TRANSFER

JSON: `experiments/archive/frontier_consolidation/data/transfer_pilot/source_free_transfer_test.json`
