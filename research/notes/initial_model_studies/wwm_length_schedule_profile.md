# wwm cross seed stability interpretation — WWM fixed base: length schedule 64→128→256 vs fixed 256 at 1M

Evidence JSON: `experiments/archive/initial_model_studies/data/wwm_length_schedule_profile.json`

Only the sequence-length schedule changes. The masking mode remains WWM; tokenizer, model size, optimizer schedule, exposure, seeds, and selected examples/order/source mix are matched to each seed's fixed-256 WWM baseline.

| seed | mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | scheduled | 56.93 | 47.60 | 50.27 | 18.25 | 49.70 | 8.92 | 4.30 | 6.71 |
| 43 | scheduled | 57.41 | 48.00 | 50.82 | 17.26 | 49.71 | 9.88 | 3.49 | 6.82 |

## Schedule minus fixed-WWM baseline

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.76 | -3.20 | 1.27 | 0.19 | -0.50 | -0.90 | 0.54 |
| 43 | 0.78 | -2.80 | -0.54 | -0.37 | -0.72 | 0.06 | -0.23 |

## Mean delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.77 | -3.00 | 0.36 | -0.09 | -0.61 | -0.42 | 0.15 |
