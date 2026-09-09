# curriculum result adapter amplitude and paired tail relaunch — adapter residual amplitude key-column readout

Inference-only scaling of already-trained live adapter128 20M checkpoint. This is a route readout, not a submission tuning rule.

aoa mincontext discrepancy audit key4 anchor (Supplement/EWoK/GlobalPIQA/Reading): 37.2612; live scale=1 key4: 36.5637.

| scale | key4 | Δkey4 vs aoa mincontext discrepancy audit | Supp | EWoK | GPpar | GPnon | GP | Reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 36.911 | -0.350 | 56.120 | 48.980 |  |  | 34.165 | 8.380 |
| 0.25 | 36.909 | -0.352 | 55.890 | 49.210 | 22.330 | 46.000 | 34.165 | 8.370 |
| 0.50 | 36.710 | -0.551 | 55.900 | 49.390 | 20.390 | 46.000 | 33.195 | 8.355 |
| 0.75 | 36.659 | -0.602 | 56.270 | 49.320 | 19.420 | 46.000 | 32.710 | 8.335 |
| 1.00 | 36.564 | -0.697 | 56.230 | 49.490 |  |  | 32.225 | 8.310 |
| 1.25 | 36.889 | -0.372 | 56.890 | 49.660 | 18.450 | 47.000 | 32.725 | 8.280 |
| 1.50 | 37.005 | -0.256 | 56.890 | 50.170 | 19.420 | 46.000 | 32.710 | 8.250 |

## Interpretation
- Some scale is better than live scale=1 on the harmed key columns but still below the aoa mincontext discrepancy audit key4 anchor; amplitude control can reduce damage but does not rescue ordinary adapters by itself at 20M.
- GlobalPIQA is maximized near zero adapter output, consistent with paired tail smoke and adapter ablation's direct-residual damage decomposition.
