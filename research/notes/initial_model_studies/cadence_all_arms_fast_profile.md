# cadence all arms fast profile matched cadence all-arms fast profile

JSON: `experiments/archive/initial_model_studies/data/cadence_all_arms_fast_profile.json`
Log: `research/notes/initial_model_studies/cadence_all_arms_fast_profile.log`

All configs equal to fixed reference: True

## Training target totals

| arm | active tokens total | masked targets total | loss last | seq lens | mask minmax |
|---|---:|---:|---:|---|---|
| fixed256_ref | 1335322 | 199522 | 6.0478 | [256] | [0.15, 0.15] |
| length_only | 1122911 | 167557 | 6.2482 | [128, 256, 512] | [0.15, 0.15] |
| mask_decay_only | 1335322 | 302811 | 6.2586 | [256] | [0.1531, 0.3] |
| combined | 1122911 | 244267 | 6.2432 | [128, 256, 512] | [0.1531, 0.3] |

## Raw scores

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| fixed256_ref | 56.54 | 47.60 | 52.27 | 16.85 | 50.03 | 5.995 |
| length_only | 57.08 | 50.40 | 49.27 | 17.49 | 50.70 | 5.875 |
| mask_decay_only | 56.13 | 48.00 | 50.64 | 17.96 | 50.01 | 6.140 |
| combined | 56.72 | 51.60 | 48.73 | 16.41 | 49.94 | 5.875 |

## Deltas vs fixed256_ref

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| length_only | +0.54 | +2.80 | -3.00 | +0.64 | +0.67 | -0.120 |
| mask_decay_only | -0.41 | +0.40 | -1.63 | +1.11 | -0.02 | +0.145 |
| combined | +0.18 | +4.00 | -3.54 | -0.44 | -0.09 | -0.120 |
