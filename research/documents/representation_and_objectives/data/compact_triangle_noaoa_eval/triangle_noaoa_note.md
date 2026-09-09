# triangle oom repair and gc preflight compact-view triangle no-AoA readout

Summary JSON: `experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/triangle_noaoa_summary.json`

This is a mechanism readout, not a final endpoint package. It compares compact-view reinvestment against exact-repeat reinvestment and source-own adjacency breaking under the same DeBERTa-v2 8x480/baseline16k/WWM/AdamW coordinate.

| target | BLiMP | Supplement | EWoK | Entity | Entity_full | COMPS | GlobalPIQA_mean | Reading | equal7_mean | equal7_full_entity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_view_reinvest | 66.630 | 66.400 | 53.090 | 28.070 | 27.750 | 51.970 | 35.620 | 8.240 | 44.289 | 44.243 |
| compact_repeat_reinvest | 66.680 | 62.000 | 50.910 | 20.170 | 20.700 | 51.770 | 33.635 | 7.940 | 41.872 | 41.948 |
| adjbreak_reinvest | 68.250 | 62.800 | 48.450 | 27.050 | 25.720 | 51.840 | 35.105 | 7.895 | 43.056 | 42.866 |

## Training states

- **compact_view_reinvest**: metrics=True, endpoint=True, word_exposure=100000000, steps=2529, loss_first=9.809807777404785, loss_last=2.565617084503174, checkpoints=100
- **compact_repeat_reinvest**: metrics=True, endpoint=True, word_exposure=100000000, steps=2529, loss_first=9.8112154006958, loss_last=2.577287197113037, checkpoints=100
- **adjbreak_reinvest**: metrics=True, endpoint=True, word_exposure=100000000, steps=2529, loss_first=9.809857368469238, loss_last=2.6621243953704834, checkpoints=100

## Mechanism contrasts

- **view_minus_repeat_reinvest**: `{"BLiMP": -0.05, "COMPS": 0.2, "EWoK": 2.18, "Entity": 7.9, "Entity_full": 7.05, "GlobalPIQA_mean": 1.985, "GlobalPIQA_nonparallel": 3.0, "GlobalPIQA_parallel": 0.97, "Reading": 0.3, "Reading_eye": 0.28, "Reading_self_paced": 0.32, "Supplement": 4.4, "equal7_full_entity": 2.295, "equal7_mean": 2.416429}`
- **view_minus_adjbreak_reinvest**: `{"BLiMP": -1.62, "COMPS": 0.13, "EWoK": 4.64, "Entity": 1.02, "Entity_full": 2.03, "GlobalPIQA_mean": 0.515, "GlobalPIQA_nonparallel": 2.0, "GlobalPIQA_parallel": -0.97, "Reading": 0.345, "Reading_eye": -0.19, "Reading_self_paced": 0.88, "Supplement": 3.6, "equal7_full_entity": 1.377143, "equal7_mean": 1.232857}`
- **adjbreak_profile**: `{"broad_delta_mean_view_minus_adjbreak": 0.891, "targeted_delta_mean_view_minus_adjbreak": 1.68, "reading_rule": "If broad columns all drop strongly, adjbreak may be a generic incoherence control rather than a clean source-own consolidation ablation."}`
- **repeat_minus_adjbreak_reinvest**: `{"BLiMP": -1.57, "COMPS": -0.07, "EWoK": 2.46, "Entity": -6.88, "Entity_full": -5.02, "GlobalPIQA_mean": -1.47, "GlobalPIQA_nonparallel": -1.0, "GlobalPIQA_parallel": -1.94, "Reading": 0.045, "Reading_eye": -0.47, "Reading_self_paced": 0.56, "Supplement": -0.8, "equal7_full_entity": -0.917857, "equal7_mean": -1.183571}`

## Interpretation reminders

- Fast equal7 gaps below 1.356 are not a single-seed mechanism verdict without full scoring or a tie-breaking seed.
- If `adjbreak_reinvest` drops across nearly every broad column, it may be a destructive incoherence control rather than evidence specifically for source-own rewrite consolidation.

## Implementation-equivalence guard

- Summary path: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json`
- Interpretation: `trajectory_equivalent_to_checked_horizon`
- Allows historical compact-view reference for causal triangle interpretation: `True`
- If this guard is false, the table may be used only as raw score evidence; train/evaluate `compact_view_reinvest` under the same GC implementation before mechanism interpretation.
