# dense focus official and mechanism state dense-focus candidate readout

## True coherent86 reference

- Zero-shot/Reading payload: `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json`
- SuperGLUE payload: `experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json`
- cheap7: 44.181428571429
- projected Overall with AoA=0: 42.121024709967

## Fast-screen dense movement

- BLiMP: computed delta -0.552239; payload delta -0.5499999999999972
- Supplement: computed delta -0.400000; payload delta -0.4000000000000057
- EWoK: computed delta +0.363636; payload delta 0.35999999999999943
- Entity: computed delta +0.658300; payload delta 0.6600000000000001
- COMPS: computed delta +0.106543; payload delta 0.10000000000000142
- GlobalPIQA_parallel: computed delta +0.970874; payload delta 0.9700000000000024
- GlobalPIQA_nonparallel: computed delta +2.000000; payload delta 2.0
- GlobalPIQA: computed delta +1.485437; payload delta 1.4849999999999994
- Reading: computed delta +0.055000; payload delta 0.054999999999999716
- cheap7_mean: computed delta +0.245240; payload delta 0.24428571428572354

## Dense/sparse objective distinction

- Dense focus targets: 176607 of 769465 total targets; selected groups 132283; lambda_focus 0.15.
- Sparse focus targets: 28590 of 621448 total targets; selected groups 21479; lambda_focus 0.15.
- Interpretation: the intervention broadens second-view content masking and removes local content clues while keeping the paired source visible; it does not multiply the aggregate focus coefficient.

## Common-target movement for dense u0080

- mean delta: +0.207875
- no_source: mean +0.020241, median +0.019950, n=25
- source_altered: mean +0.339728, median +0.267437, n=25
- source_original: mean +0.263657, median +0.262742, n=25
- held_source: mean +0.366223, median +0.165000, n=24
- trained_content: mean +0.133359, median +0.136511, n=51

## Official arithmetic sensitivity

With AoA fixed at 0, each +1.0 summed point across the eight non-AoA official columns changes Overall by +1/9 = +0.111111. Dense beats coherent86 when (dense seven non-SG columns - coherent seven non-SG columns) + (dense SuperGLUE - coherent SuperGLUE) > 0.
If the official non-SuperGLUE seven-column mean moved like the fast screen and SuperGLUE were unchanged, projected Overall would be approximately 42.311766662061. This is only a sensitivity calculation, not evidence about full official columns.

## Dense official payload state at script time

{
  "path": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
  "exists": true,
  "size": 6804,
  "target": "dense_focus_seed62064_u0080",
  "present_tasks": [
    "BLiMP",
    "EWoK",
    "Entity",
    "Supplement"
  ],
  "finished_tasks": [
    "BLiMP",
    "EWoK",
    "Entity",
    "Supplement"
  ],
  "unfinished_or_incomplete_tasks": [],
  "scores_seen": {
    "BLiMP": 68.06,
    "Supplement": 63.04,
    "EWoK": 49.92,
    "Entity": 29.37,
    "COMPS": null,
    "SuperGLUE": null,
    "GlobalPIQA": null,
    "Reading": null,
    "AoA": null
  },
  "cheap7_seen": null,
  "projected_overall_aoa0_seen": null,
  "official_overall_field": null
}
