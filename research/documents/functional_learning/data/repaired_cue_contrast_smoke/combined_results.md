# effective input repaired contrast result repaired cue contrast

Fast effective-input pilot: names are canonicalized as `<C>/<O>` before the GRU, so name invariance is supplied. Train-only cue tokens test whether variation helps only when a visible nuisance can compete with the relation solution.

Seed=30000, epochs=3, device=cpu

## Final held-out performance

| condition | sign | train_cmp_current | no-cue direct | no-cue graph | matched-cue direct | matched-cue graph |
|---|---|---:|---:|---:|---:|---:|
| clean | +1 | 0.625 | 0.500 | 0.250 | 0.500 | 0.250 |
| cue_locked | +1 | 0.625 | 0.500 | 0.250 | 0.500 | 0.250 |

## Budget curve: no-cue graph transport

| epoch | clean bs+1 | cue_locked bs+1 |
|---:|---:|---:|
| 1 | 0.396 | 0.396 |
| 2 | 0.375 | 0.417 |
| 3 | 0.250 | 0.250 |

## Per-relation final no-cue accuracies

| condition | sign | h0 | h1 | h2 | h3 |
|---|---|---:|---:|---:|---:|
| clean | +1 | 0.500 | 0.500 | 0.500 | 0.000 |
| cue_locked | +1 | 0.500 | 0.500 | 0.500 | 0.000 |
