# changed state bias threat and route decision Entity numops changed-state-bias readout

File-only recomputation from official Entity prediction JSONs. The key split is 0-op versus nonzero-op rows. Bias toward changed state predicts nonzero-op gains with 0-op loss or no help.

## MAX view-minus-repeat by basin/checkpoint

| basin | checkpoint | all Δ pp | zero-op Δ pp | nonzero-op Δ pp | nonzero minus zero Δ pp |
|---|---:|---:|---:|---:|---:|
| seed43022_first_basin | chck_10M | -0.969 | -0.716 | -1.020 | -0.304 |
| seed43022_first_basin | chck_20M | -1.809 | -0.901 | -1.991 | -1.089 |
| seed43022_first_basin | chck_30M | +1.802 | -3.051 | +2.773 | +5.824 |
| seed43022_first_basin | chck_40M | +1.941 | -11.036 | +4.536 | +15.572 |
| seed43022_first_basin | chck_50M | +2.433 | -8.052 | +4.530 | +12.582 |
| seed43022_first_basin | chck_60M | +6.491 | -12.442 | +10.278 | +22.720 |
| seed43022_first_basin | chck_70M | +2.094 | -12.445 | +5.002 | +17.446 |
| seed43022_first_basin | chck_80M | +4.501 | -9.000 | +7.201 | +16.201 |
| seed43022_first_basin | chck_90M | +3.661 | -10.414 | +6.476 | +16.890 |

## Interpretation

- **missing_prediction_files**: 0 missing prediction files; partial readouts are explicitly marked in the plan.
- **late_zero_nonzero_test**: Compare late_summary zero_ops and nonzero_ops. A larger nonzero-minus-zero spread supports the changed-state-bias alternative; positive zero-op together with nonzero gains weakens pure bias.
- **permuted_card_decision**: If second-basin official Entity reproduces but numops movement is mostly nonzero-op gain with zero-op loss, delay permuted training and reinterpret Entity as a bias-sensitive benchmark carrier. If zero-op is not harmed and binding affected/unaffected does not show the signed bias, the permuted correspondence arm remains decisive.

CSV files in this directory contain all split and per-subtask rows.
