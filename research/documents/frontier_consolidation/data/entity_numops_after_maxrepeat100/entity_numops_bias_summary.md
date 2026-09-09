# changed state bias threat and route decision Entity numops changed-state-bias readout

File-only recomputation from official Entity prediction JSONs. The key split is 0-op versus nonzero-op rows. Bias toward changed state predicts nonzero-op gains with 0-op loss or no help.

## MAX view-minus-repeat by basin/checkpoint

| basin | checkpoint | all Δ pp | zero-op Δ pp | nonzero-op Δ pp | nonzero minus zero Δ pp |
|---|---:|---:|---:|---:|---:|
| seed43022_first_basin | chck_80M | +4.501 | -9.000 | +7.201 | +16.201 |
| seed43022_first_basin | chck_90M | +3.661 | -10.414 | +6.476 | +16.890 |
| seed43022_first_basin | chck_100M | +3.891 | -9.569 | +6.583 | +16.152 |

## Interpretation

- **missing_prediction_files**: 0 missing prediction files; partial readouts are explicitly marked in the plan.
- **late_zero_nonzero_test**: Compare late_summary zero_ops and nonzero_ops. A larger nonzero-minus-zero spread supports the changed-state-bias alternative; positive zero-op together with nonzero gains weakens pure bias.
- **permuted_card_decision**: If second-basin official Entity reproduces but numops movement is mostly nonzero-op gain with zero-op loss, delay permuted training and reinterpret Entity as a bias-sensitive benchmark carrier. If zero-op is not harmed and binding affected/unaffected does not show the signed bias, the permuted correspondence arm remains decisive.

CSV files in this directory contain all split and per-subtask rows.
