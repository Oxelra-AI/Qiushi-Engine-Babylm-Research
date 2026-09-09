# effective input repaired contrast result repaired cue contrast

Fast effective-input pilot: names are canonicalized as `<C>/<O>` before the GRU, so name invariance is supplied. Train-only cue tokens test whether variation helps only when a visible nuisance can compete with the relation solution. In `comp_*_masked` conditions, direct state anchors remain clean while comparison-event relation verbs are masked with probability 0.75. In `packet_*` conditions, a row-level packet token is shared by both events in a training comparison row or broken independently across the two events; packet tokens are absent at eval.

Seed=30000, epochs=40, device=cpu

## Final held-out performance

| condition | sign | train_cmp_current | no-cue direct | no-cue graph | matched-cue direct | matched-cue graph | wrong-cue direct | wrong-cue graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_locked | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_locked_noise | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_broken | +1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Budget curve: no-cue graph transport

| epoch | clean bs+1 | packet_locked bs+1 | packet_locked_noise bs+1 | packet_broken bs+1 |
|---:|---:|---:|---:|---:|
| 1 | 0.500 | 0.500 | 0.500 | 0.500 |
| 5 | 0.500 | 0.500 | 0.500 | 0.500 |
| 10 | 0.375 | 0.375 | 0.375 | 0.375 |
| 15 | 0.750 | 0.771 | 0.771 | 0.792 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 1.000 | 1.000 | 1.000 | 1.000 |
| 35 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 1.000 | 1.000 | 1.000 | 1.000 |

## Per-relation final no-cue accuracies

| condition | sign | h0 | h1 | h2 | h3 |
|---|---|---:|---:|---:|---:|
| clean | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_locked | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_locked_noise | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
| packet_broken | +1 | 1.000 | 1.000 | 1.000 | 1.000 |
