# seed43022 dose practical predictions registered seed43022 practical dose pattern before seed43122 scoring

Seed43022 cheap7 is not best understood as a scalar non-monotone result.  The full vector in `data/seed43022_cheap7/summary.json` shows that the cheap7 scalar is strongly affected by GlobalPIQA, the noisiest two-subcolumn average:

| dose | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base0 | 43.5407 | 68.62 | 62.90 | 49.08 | 27.46 | 52.30 | 36.105 | 8.32 |
| dose21 | 43.9221 | 67.61 | 62.37 | 50.14 | 27.94 | 51.92 | 39.605 | 7.87 |
| dose25 | 42.9793 | 66.00 | 63.31 | 52.52 | 26.76 | 52.33 | 32.635 | 7.30 |

Contrasts:
- dose21-base: cheap7 +0.3814, but GlobalPIQA alone is +3.50.  Averaging the other six columns gives -0.1400.
- dose25-base: cheap7 -0.5614, with GlobalPIQA -3.47.  Averaging the other six columns gives -0.0800.

The stable-looking structure inside the flat six-column mean is a relation-typed substitution trade:
- BLiMP declines monotonically: 68.62 -> 67.61 -> 66.00.
- EWoK rises monotonically: 49.08 -> 50.14 -> 52.52.
- Reading also drifts downward: 8.32 -> 7.87 -> 7.30.
- Supplement, Entity, and COMPS remain small relative to known seed floors.

Scientific prediction for seed43122, registered before running its cheap7 scorer: if the marginal aligned-restatement dose is a real relation-typed transfer rather than seed noise, seed43122 should reproduce BLiMP monotone down and EWoK monotone up across base0 -> dose21 -> dose25.  The cheap7 scalar itself is not the main scientific readout because GlobalPIQA can flip the sign.

Connection to the data-efficient learning principle: the added pairs practice a context-conditioned changed-form relation.  EWoK has a context-consistency choice format closer to that relation, while BLiMP uses single-sentence minimal pairs and plausibly reflects competence supported by the ordinary text displaced under the fixed 100M-word budget.  Replication would make the practical dose result a benchmark-level instance of relation-conditioned transfer plus substitution cost, not a new Overall route.
