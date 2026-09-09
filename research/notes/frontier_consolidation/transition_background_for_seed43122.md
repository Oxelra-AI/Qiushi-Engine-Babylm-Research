# transition background for seed43122: fixed-window and all-window background for seed43122 transition readout

## Why this was needed

The signed transition readout ready signed-transition analyzer is useful, but its default window is each trajectory's own cheap7-selected local peak. That creates an automatic pre-peak rise and post-peak fall. A trajectory can therefore look superficially similar to the reference simply because both windows were selected by the same aggregate score geometry, not because the same capability families strengthen and erode.

The transition background for seed43122 companion tool fixes that by comparing every eligible late three-checkpoint window in a target trajectory against the fixed reference signature:

`scale1p75_seed43022_reference: chck_82M -> chck_84M -> chck_86M`

It separately marks:

- the same-coordinate fixed window `82M->84M->86M`
- the target's own cheap7 peak window
- the remaining late-window background

The intended use after seed43122 selected-grid evaluation completes is to ask whether seed43122's own-peak and/or fixed-coordinate windows stand out against its own all-window background at official-like column, subtask, and signed-item levels. This prevents static best-endpoint overlap or selection-induced rise/fall from deciding robustness.

## Script and validated calibration

Script:

`scripts/transition_window_background.py`

Validated calibration output:

`data/transition_window_background_reference_scale125_calibration_v3/transition_window_background_summary.md`

The first run exposed a units bug: `multitrajectory item dynamics ready.endpoint_int()` returns `84` for `chck_84M`, not raw words. The script was patched to use 2M spacing as integer `2`. A second edge-case patch treats `own_peak_and_fixed_82_84_86` as both roles if a target's own peak is exactly at 84M. The v3 calibration is the usable record.

## Calibration lesson from the failed scale1.25 trajectory

The calibration on completed reference + scale1.25 grids shows why the transition background for seed43122 background is scientifically necessary.

Scale1.25 own peak is `chck_84M -> chck_86M -> chck_88M`, not the fixed `82M->84M->86M`. Its own-peak coarse column pattern can look deceptively similar by peak selection:

- own-peak `column_official_subtask_mean` rise+fall cosine = **0.706816**, rank 1/14 against its all-window background.

But finer structure rejects it:

- own-peak `column_subtask` rise+fall weighted Pearson = **-0.241053**, rank 13/14.
- own-peak `column_subtask` rise+fall cosine = **-0.202040**, rank 12/14.
- own-peak signed item-pattern cosine = **-0.005165**, rank 13/14.
- signed transition readout ready already showed near-zero signed item-transition overlap with the reference-changed items.

The fixed `82M->84M->86M` scale1.25 window has better subtask and item ranks, but absolute magnitudes remain modest:

- fixed `column_subtask` rise+fall weighted Pearson = **0.335592**, rank 1/14.
- fixed `column_subtask` rise+fall cosine = **0.296070**, rank 1/14.
- fixed item-pattern cosine = **0.016419**, rank 1/14.
- fixed same-signed fraction on reference-changed transitions = **0.049773**, rank 5/14.

This means rank alone is not enough: a failed trajectory can have its best-matching late window at the same coordinate simply because the internal background is weak. Robustness requires both (i) standing out against background and (ii) meaningful absolute similarity in the same columns/subtasks/items, together with the score-level facts from lead cross seed decision framework and the signed-transition facts from signed transition readout ready.

## Exact command after seed43122 delivers

After seed43122 selected-grid completion and a passing integrity check, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/transition_window_background.py \
  --reference scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --target scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --target scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --target scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --out-dir experiments/archive/frontier_consolidation/data/transition_window_background_all3
```

Then run the already-prepared signed transition readout ready, lead cross seed decision framework, and multitrajectory item dynamics ready readouts. The intended order is:

1. `selected_mlm_integrity_check.py` on the seed43122 selected grid.
2. `transition_window_background.py` to remove the peak-selection artifact.
3. `signed_transition_signature_analyzer.py` for own-peak signed transition comparison.
4. `cross_seed_common_grid_analyzer.py` for score/peak-vector structure.
5. `multitrajectory_item_dynamics.py` for static endpoint overlap as supporting context only.

After these, make the route decision rather than building another layer of tooling.

## How to decide after seed43122

Use the fixed-window/background result as follows.

- If seed43122 has an own peak near the same late region and both its own-peak and fixed `82M->84M->86M` windows stand out against its all-window background with meaningful absolute similarity at column, subtask, and signed-item levels, then the scale1.75 residual-capacity late phase is a more robust learning-dynamics object. The next scientific target should be stabilizing this broad competence allocation, not simply tuning a single checkpoint.

- If seed43122 has a late peak but its fixed/own windows match only at coarse column level, or different subtasks/items move, then residual capacity creates stochastic competence allocation. The next route should target stabilization across stochastic trajectories or return to mechanism design; do not treat seed43022 `chck_84M` as a general law.

- If seed43122 lacks a comparable late broad-family peak, then ordinary `chck_84M` remains an endpoint asset only. The research should not spend more work tuning the original peak or reopening closed alpha/anchor/retention/edit/IG/graph/coherence-margin routes.

The public `chck_82M` endpoint remains protected; `chck_84M` has a validated truthful carrier and HF revision; coherent86 alpha0.75 remains the strongest local projected endpoint but is redistributive.
