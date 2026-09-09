# effective input repaired contrast result repaired cue contrast

Fast effective-input pilot: names are canonicalized as `<C>/<O>` before the GRU, so name invariance is supplied. Train-only cue tokens test whether variation helps only when a visible nuisance can compete with the relation solution.

Seed=30000, epochs=40, device=cpu

## Final held-out performance

| condition | sign | train_cmp_current | no-cue direct | no-cue graph | matched-cue direct | matched-cue graph |
|---|---|---:|---:|---:|---:|---:|
| clean | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_locked | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_locked_noise | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_varied_dropout | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Budget curve: no-cue graph transport

| epoch | clean bs+1 | cue_locked bs+1 | cue_locked_noise bs+1 | cue_varied_dropout bs+1 |
|---:|---:|---:|---:|---:|
| 1 | 0.396 | 0.396 | 0.417 | 0.396 |
| 5 | 0.458 | 0.458 | 0.458 | 0.458 |
| 10 | 0.438 | 0.438 | 0.375 | 0.438 |
| 15 | 0.688 | 0.750 | 0.708 | 0.750 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| 35 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 1.000 | 1.000 | 1.000 | 1.000 |

## Per-relation final no-cue accuracies

| condition | sign | h0 | h1 | h2 | h3 |
|---|---|---:|---:|---:|---:|
| clean | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_locked | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_locked_noise | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| cue_varied_dropout | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
