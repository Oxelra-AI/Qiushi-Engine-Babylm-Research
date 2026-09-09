# aoa overall update — loss-curve bin comparison

CPU-only comparison of training logs for the two compact_view_reinvest seeds.

## Ten-million-word bin differences (43122 minus 43022)
- 00-010M: mean loss -0.0104554, last loss +0.0209327, mask-rate mean -0.000304365, masked-token mean -16.9524
- 10-020M: mean loss -0.00357445, last loss +0.0824435, mask-rate mean +9.09091e-05, masked-token mean +5.17391
- 20-030M: mean loss -0.00126084, last loss +0.0637751, mask-rate mean -0.000188142, masked-token mean -10.8617
- 30-040M: mean loss -0.00833072, last loss -0.0702455, mask-rate mean -5.88933e-05, masked-token mean -3.15415
- 40-050M: mean loss +0.00131927, last loss +0.0241406, mask-rate mean +7.98419e-05, masked-token mean +4.62451
- 50-060M: mean loss +0.00252643, last loss -0.00599694, mask-rate mean +0.000340316, masked-token mean +19.4427
- 60-070M: mean loss -0.00435944, last loss +0.0853753, mask-rate mean +2.7668e-06, masked-token mean +0.217391
- 70-080M: mean loss -0.00345147, last loss -0.122045, mask-rate mean -2.13439e-05, masked-token mean -1.38735
- 80-090M: mean loss -0.00845415, last loss -0.0405858, mask-rate mean -0.000157708, masked-token mean -8.93676
- 90-100M: mean loss +0.000160846, last loss +0.0759079, mask-rate mean +0.000152174, masked-token mean +8.74704

## Reading
The two runs have identical step count and exposure schedule. Mean loss differences oscillate around zero across most of training, while the very last minibatch loss is higher for seed43122. This makes the sparse task-slice trajectory more informative than the scalar final loss for explaining the weaker seed.

Figure: `experiments/archive/frontier_consolidation/figures/compact_reinvest_seed_loss_curves.png`

Machine-readable output: `experiments/archive/frontier_consolidation/data/loss_curve_bin_compare/loss_curve_bin_compare.json`
