# state coherent vs corrupted 1m profile — procedural state stories: coherent vs corrupted

Evidence JSON: `experiments/archive/initial_model_studies/data/state_coherent_vs_corrupted_1m_profile.json`

900k official words + 100k hand-coded procedural state stories. Coherent/corrupted arms share official examples, story templates, target answer multiset, target mask positions, word counts, kept-token totals, and update geometry; corrupted swaps decisive bindings so events no longer support the recorded target state.

| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | state_coherent | 53.09 | 51.20 | 49.18 | 18.31 | 50.43 | 10.34 | 3.77 |
| 42 | state_corrupted | 53.10 | 50.00 | 50.82 | 18.31 | 50.07 | 10.34 | 3.77 |
| 43 | state_coherent | 54.72 | 52.00 | 51.55 | 18.28 | 49.96 | 10.04 | 3.52 |
| 43 | state_corrupted | 54.69 | 51.60 | 49.45 | 18.28 | 50.38 | 10.04 | 3.52 |

## coherent minus corrupted

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.01 | 1.20 | -1.64 | 0.00 | 0.36 | 0.00 | 0.00 |
| 43 | 0.03 | 0.40 | 2.10 | 0.00 | -0.42 | 0.00 | 0.00 |

## mean delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.80 | 0.23 | 0.00 | -0.03 | 0.00 | 0.00 |
