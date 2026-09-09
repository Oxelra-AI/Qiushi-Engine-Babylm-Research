# chck84 late item dynamics synthesis synthesis: `chck_84M` as a narrow late competence-allocation peak

This note records CPU/file-only analysis while SuperGLUE evaluation for `chck_84M` and the DeBERTa common-grid resume remained incomplete. No partial output from those evaluations was interpreted.

Script and outputs:

- Script: `experiments/archive/frontier_consolidation/scripts/reference_late_item_dynamics.py`
- Output directory: `experiments/archive/frontier_consolidation/data/reference_late_item_dynamics`
- Main report: `reference_late_item_dynamics_summary.md`
- Tables: `score_curve.csv`, `late_item_dynamics.csv`, `column_dynamics_summary.csv`, `column_subtask_dynamics_summary.csv`, `subtask_score_curve.csv`

The script initially duplicated a directory prefix during path resolution; this was repaired with explicit root discovery. A second repair corrected the bootstrap lens so Reading is included as an exact scalar contribution in cheap6/cheap7 while classification subtasks are resampled.

## What was measured

The analysis uses already completed selected cheap-task prediction payloads for the scale1.75 seed43022 reference trajectory:

`chck_70M, 72M, 74M, 76M, 78M, 80M, 82M, 84M, 86M, 88M, 90M, 92M, 100M`.

It reconstructs official-like item correctness for 170,722 classification items using the chck84 item movement synthesis task parsers, aligns item correctness across checkpoints, and summarizes endpoint scores, local peak shape, gain persistence, reversals, and subtask-resampling stability.

## Main empirical result

`chck_84M` is a real narrow local peak in the protected seed/mask trajectory:

- `chck_84M - chck_82M` cheap7: **+0.163992**
- `chck_86M - chck_84M` cheap7: **-0.352133**
- `chck_84M` local excess over mean(`chck_82M`, `chck_86M`): **+0.258062 cheap7**
- local excess remains without GlobalPIQA/Reading: **+0.102375 cheap5**
- relation/state local excess is smaller but positive: **+0.051195**

This supports treating `chck_84M` as an endpoint branch, not a pure GlobalPIQA or Reading accident.

## The gain is not monotone accumulation

Correctness movement between 82M, 84M, and 86M shows large churn relative to the net endpoint movement:

| column | net 82→84 | gains 82→84 | losses 82→84 | fraction of 82→84 gains lost by 86 | fraction of 82→84 gains persisting through 92 |
|---|---:|---:|---:|---:|---:|
| BLiMP | -139 | 1015 | 1154 | 0.354 | 0.448 |
| Supplement | -10 | 51 | 61 | 0.392 | 0.353 |
| EWoK | -3 | 237 | 240 | 0.354 | 0.422 |
| Entity | +11 | 134 | 123 | 0.425 | 0.276 |
| COMPS | +82 | 3001 | 2919 | 0.352 | 0.441 |
| GlobalPIQA | +1 | 5 | 4 | 0.200 | 0.400 |

Even where official-style column scores improve, many individual decisions flip both directions. The late phase is better described as competence allocation/reallocation than as smooth additional knowledge acquisition.

## Stability lens

Subtask-unit bootstrap over classification subtasks, with Reading included exactly in cheap6/cheap7, gives:

- `84M - 82M`: cheap7 median **+0.1633**, 90% interval **[-0.3956, +0.7247]**, P(delta≤0)=0.261; cheap6-no-GP median **+0.1015**, P(delta≤0)=0.138; syntax/surface median **-0.1162**, P(delta≤0)=0.822.
- `86M - 84M`: cheap7 median **-0.3555**, 90% interval **[-0.5865, -0.1193]**, P(delta≤0)=0.996.
- `100M - 84M`: cheap7 median **-0.5757**, P(delta≤0)=0.965; cheap6-no-GP median **-0.3375**, P(delta≤0)=0.978; relation/state median **-1.0429**, P(delta≤0)=0.999.

The 84M advantage over 82M is positive but not uniformly stable under subtask resampling; the later decline from 84M to 100M is much more stable and is concentrated in relation/state. This strengthens the interpretation that the late reference trajectory enters a narrow high-capability allocation phase and then leaves it.

## Consequences for the endpoint branch

- Pending `chck_84M` SuperGLUE remains decisive for endpoint arithmetic. earlier analysis thresholds remain the exact arithmetic: SuperGLUE ≥68.6173 beats protected `chck_82M`; ≥69.135 reaches Overall 42.0; ≥70.2242 matches coherent86 alpha0.75 projected Overall(AoA0).
- `chck_84M` is scientifically cleaner than coherent86 alpha0.75 because it is an ordinary legal training checkpoint on the same trajectory, not an inference-time private-scale endpoint. chck84 vs alpha075 endpoint overlap synthesis showed alpha0.75 only exceeds `chck_84M` by +0.0576 cheap7 and loses without GlobalPIQA, especially on EWoK+Entity.
- But `chck_84M` is still one seed/mask trajectory. It should not be elevated into a general residual-capacity learning law until the already-trained scale1.75 seed43122 common grid is scored and compared.

## Consequences for mechanism search

The late-peak object is now sharper:

1. The 84M peak is not explained by immediately preceding compact-pair exposure (earlier analysis showed 82M→84M has no compact-pair-block words).
2. The 84M peak is not a pure volatile-column event (chck84 item movement synthesis/189 show cheap5 and relation/state local excess remain positive, though smaller).
3. The peak is not monotone competence accumulation; it is a high-churn weighted allocation phase.
4. The 84M→100M decline is more stable than the 82M→84M rise and is relation/state-heavy.

Thus the next decisive evidence is not another local endpoint tweak. It is the seed43122 common-grid comparison: does a different init+mask stream also show a narrow late peak with a comparable family profile, or is this a seed-specific alignment of the official benchmark vector? Only after that should new H100 training for compact-order or reciprocal/topology variants be reconsidered, and only with directional-fork scores in hand.
