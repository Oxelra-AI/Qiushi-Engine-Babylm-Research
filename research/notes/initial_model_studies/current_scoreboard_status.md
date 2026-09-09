# current scoreboard status — Current BabyLM score-coordinate state

Evidence JSON: `experiments/archive/initial_model_studies/data/current_scoreboard_status.json`

## Current complete internal coordinate

Protected DeBERTa-v2 8×480 WWM remains the only true complete internal 9/9 coordinate: Overall 40.5269.

## S1/S2 100M coordinate state

| model | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | Reading | SuperGLUE | AoA | true 9/9? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | 66.84 | 60.31 | 52.02 | 20.24 | 52.26 | 37.61 | 7.25 | 65.12 | — | no |
| S2 | 64.24 | 59.09 | 51.64 | 18.47 | 50.61 | 38.63 | 7.52 | 65.60 | — | no |

S1/S2 AoA cannot be computed in the same way as the trusted protected coordinate from current files because the early checkpoints `chck_1M`–`chck_9M` were not saved. SuperGLUE has completed and is included above; S1/S2 remain non-9/9 coordinates because AoA is unavailable from the saved checkpoint sequence.

## Route B 10M readout

Route B surface adapter is not scale-worthy: surface minus token-ID gives Entity +0.02, COMPS +0.36, EWoK -2.67, GlobalPIQA +2.47, Reading 0.00.
