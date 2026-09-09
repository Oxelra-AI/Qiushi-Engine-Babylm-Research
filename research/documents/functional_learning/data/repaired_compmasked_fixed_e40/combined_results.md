# effective input repaired contrast result repaired cue contrast

Fast effective-input pilot: names are canonicalized as `<C>/<O>` before the GRU, so name invariance is supplied. Train-only cue tokens test whether variation helps only when a visible nuisance can compete with the relation solution. In `comp_*_masked` conditions, direct state anchors remain clean while comparison-event relation verbs are masked with probability 0.75. In `packet_*` conditions, a row-level packet token is shared by both events in a training comparison row or broken independently across the two events; packet tokens are absent at eval.

Seed=30000, epochs=40, device=cpu

## Final held-out performance

| condition | sign | train_cmp_current | no-cue direct | no-cue graph | matched-cue direct | matched-cue graph | wrong-cue direct | wrong-cue graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| comp_clean_masked | +1 | 0.495 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| comp_cue_locked_masked | +1 | 1.000 | 1.000 | 0.792 | 1.000 | 0.979 | 1.000 | 0.750 |
| comp_cue_locked_noise_masked | +1 | 1.000 | 1.000 | 0.938 | 1.000 | 1.000 | 1.000 | 0.500 |
| comp_cue_varied_dropout_masked | +1 | 0.526 | 1.000 | 0.750 | 1.000 | 0.750 | 1.000 | 0.750 |

## Budget curve: no-cue graph transport

| epoch | comp_clean_masked bs+1 | comp_cue_locked_masked bs+1 | comp_cue_locked_noise_masked bs+1 | comp_cue_varied_dropout_masked bs+1 |
|---:|---:|---:|---:|---:|
| 1 | 0.500 | 0.500 | 0.500 | 0.500 |
| 5 | 0.500 | 0.500 | 0.500 | 0.500 |
| 10 | 0.375 | 0.375 | 0.375 | 0.396 |
| 15 | 0.500 | 0.500 | 0.500 | 0.500 |
| 20 | 0.500 | 0.500 | 0.500 | 0.500 |
| 25 | 0.750 | 0.646 | 0.771 | 0.708 |
| 30 | 0.750 | 0.562 | 0.896 | 0.750 |
| 35 | 1.000 | 0.896 | 0.792 | 0.750 |
| 40 | 1.000 | 0.792 | 0.938 | 0.750 |

## Per-relation final no-cue accuracies

| condition | sign | h0 | h1 | h2 | h3 |
|---|---|---:|---:|---:|---:|
| comp_clean_masked | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| comp_cue_locked_masked | +1 | 1.000 | 0.583 | 1.000 | 1.000 |
| comp_cue_locked_noise_masked | +1 | 1.000 | 0.875 | 1.000 | 1.000 |
| comp_cue_varied_dropout_masked | +1 | 1.000 | 0.500 | 1.000 | 1.000 |
