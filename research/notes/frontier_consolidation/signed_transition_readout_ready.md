# signed transition readout ready: signed transition readout for seed43122 robustness

CPU/file-only analysis without model inference. The seed43122 selected grid remains pending.

## Why this was needed

The lead cross seed decision framework/196 readout emphasized per-column peak vectors and best-endpoint item overlap. The important correction is that static best-endpoint correct-set overlap is dominated by easy items and task base rates, so it can be high even when the late learning dynamics are unrelated. The seed43122 result should instead be read by **signed item transitions** over matched local windows:

- within each trajectory, find its own selected-score `cheap7` peak over the completed 70M--100M common grid;
- define the rise window = previous checkpoint -> peak;
- define the fall window = peak -> next checkpoint;
- ask whether the same tasks/subtasks/families strengthen during the rise and erode during the fall across trajectories.

Endpoint overlap remains supporting context only.

## Tool built

`experiments/archive/frontier_consolidation/scripts/signed_transition_signature_analyzer.py`

It consumes existing selected-grid directories and per-target payloads. It does not run model inference. It:

1. chooses each trajectory's own peak by selected-row metric (default `cheap7`);
2. loads only the pre-peak/peak/post-peak payloads;
3. recomputes official-like item correctness using the chck84 item movement synthesis parsers;
4. records signed transitions per item (`-1`, `0`, `+1`) for rise and fall;
5. summarizes raw item-weighted churn and official-like score-coordinate column movement;
6. computes pairwise signed-transition similarity to a reference at item, official-like column, and subtask levels;
7. saves top reference moving subtasks so seed43122 can be read by concrete capability families.

The tool now separates two quantities that can disagree:

- `column_transitions`: official-like column movement, averaged over subtasks and aligned with the selected score coordinate;
- `column_transitions_item_weighted`: raw item-count movement, useful for churn anatomy but not the leaderboard score coordinate.

This repair was necessary because the first smoke showed raw item-weighted BLiMP/Supplement/EWoK movement can contradict selected-score deltas under the official subtask/group averaging.

## Calibration on completed reference + scale1.25 grids

Output: `experiments/archive/frontier_consolidation/data/signed_transition_signature_smoke_reference_scale125_v2`.

Trajectory windows:

- reference scale1.75 seed43022: `chck_82M -> chck_84M -> chck_86M`, selected cheap7 `43.958571 -> 44.123571 -> 43.770714`.
- scale1.25 seed43022: `chck_84M -> chck_86M -> chck_88M`, selected cheap7 `43.500714 -> 43.537857 -> 43.267143`.

Official-like column signed movement (rise net %, fall net %):

| trajectory | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA |
|---|---:|---:|---:|---:|---:|---:|
| reference rise | -0.240112 | +0.545878 | +0.018014 | +0.261068 | +0.014408 | +0.543689 |
| reference fall | +0.222349 | -0.808011 | +0.066870 | +0.007431 | +0.086869 | -1.985437 |
| scale1.25 rise | +0.391830 | -0.060258 | -0.846942 | +0.472413 | +0.274451 | 0.000000 |
| scale1.25 fall | +0.133812 | -0.232664 | -0.368463 | -0.115231 | +0.131980 | -1.456311 |

Transition similarity of scale1.25 to reference:

- Official-like column level: rise Pearson `-0.077226`, weighted Pearson `-0.248842`, cosine `-0.016266`, sign agreement on reference moving columns `1/3`; fall Pearson `0.915932`, weighted Pearson `0.773866`, cosine `0.933815`, sign agreement on reference moving columns `2/2`; rise+fall pattern agreement on reference movers `1/3`.
- Subtask level (107 groups): rise Pearson `-0.367627`, weighted Pearson `-0.376681`, cosine `-0.369552`, sign agreement on reference moving subtasks `0.448276`; fall Pearson `0.190754`, weighted Pearson `0.147471`, cosine `0.189981`, sign agreement `0.505747`; rise+fall pattern agreement `0.359223`.
- Raw signed item transitions over all 170,722 classification items: reference-changed items almost never have the same signed transition in scale1.25 (rise sign agreement `0.035331`, fall `0.026227`; signed-transition cosine `-0.005920` / `-0.004251`; gain Jaccard `0.017865` / `0.014653`; loss Jaccard `0.017375` / `0.015328`). In most reference-changed items the scale1.25 trajectory is unchanged (`0.923301` rise and `0.943800` fall).

Calibration interpretation: scale1.25 is a useful negative control. It partially shares the coarse falling-column direction (especially Supplement and GlobalPIQA decline after its own peak) but fails to reproduce the rise into the reference peak and does not share the actual signed item transitions or most top-moving subtasks. This is stronger evidence than best-endpoint overlap that scale1.25 is not the same late-competence event shifted by two million words.

## Command to run after seed43122 delivers

After seed43122 selected-grid completion and a passing `scripts/selected_mlm_integrity_check.py` on `experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/signed_transition_signature_analyzer.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/signed_transition_signature_all3
```

Then run the already prepared score-level and item-overlap tools:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/cross_seed_common_grid_analyzer.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/cross_seed_common_grid_analysis_all3 \
  --strict-complete

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/multitrajectory_item_dynamics.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/multitrajectory_item_dynamics_all3 \
  --bootstrap 2000
```

## Scientific reading after seed43122

Use the signed-transition analyzer first, then the lead cross seed decision framework/196 summaries as supporting evidence.

- **Robust residual-capacity late phase:** seed43122 has a comparable late peak and the same official-like columns and major subtasks strengthen and erode with the same signs; subtask-level rise/fall correlations are materially positive; reference moving subtasks mostly recur. Then the phenomenon is a real residual-capacity allocation dynamic, and the next work should localize/stabilize the underlying family competences across existing checkpoints.
- **Stochastic late phase:** seed43122 has a late peak but the rise/fall family signatures differ or only coarse fall directions recur, with low subtask-pattern agreement and low signed item-transition overlap. Then the phase exists but its family alignment is stochastic; the next scientific target is stochastic competence stabilization, not tuning the seed43022 `chck_84M` endpoint.
- **Unstructured seed result:** no comparable late broad-family peak or unrelated transition signature. Then seed43022 `chck_84M` is an endpoint asset, while the general learning-principle route must be rebuilt around stabilization or a different inductive mechanism.

Do not use static best-endpoint correct-set Jaccard as the primary robustness decision.
