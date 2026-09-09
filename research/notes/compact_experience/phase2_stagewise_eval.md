# phase2 stagewise eval — Corrected Phase 2 stagewise fast evaluation

Summary JSON: `experiments/archive/compact_experience/data/phase2_stagewise_eval/phase2_stagewise_eval_summary.json`

This evaluates the corrected 12×384/LAMB/40k stagewise Phase-2 runs from fixedinit and phase2 dynamics. It uses the same sentence-zero-shot and Reading invocation pattern as the repaired mixture eval repaired/fixedinit and phase2 dynamics screens. It is not a complete official nine-column Overall because SuperGLUE and AoA are not yet evaluated.

## Training verification

- `phase2s_official`: all_passed=True, words=100000000, steps=5984, params=34677184, vocab=40000, loss 10.636→2.304, modes={'wwm': 4519, 'token': 1465}, first_token={'step': 4520, 'stage': 3, 'cumulative_word_exposure': 70035200, 'mask_mode': 'token', 'loss': 2.5721404552459717}, failed=[]
- `phase2s_mix25`: all_passed=True, words=100000000, steps=5984, params=34677184, vocab=40000, loss 10.640→2.285, modes={'wwm': 4519, 'token': 1465}, first_token={'step': 4520, 'stage': 3, 'cumulative_word_exposure': 70035200, 'mask_mode': 'token', 'loss': 2.7093849182128906}, failed=[]

## Fast-screen table

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt | wproxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase2s_official | 68.610 | 64.000 | 45.910 | 22.830 | 22.650 | 52.160 | 34.225 | 5.625 | 41.909 | 41.883 | 47.956 |
| phase2s_mix25 | 66.510 | 60.800 | 51.730 | 25.060 | 24.780 | 52.710 | 34.635 | 6.075 | 42.503 | 42.463 | 48.574 |
| initial_model_baseline | 67.350 | 65.200 | 49.640 | 21.240 | 21.780 | 53.110 | 36.120 | 7.330 | 42.856 | 42.933 | 32.273 |

## Contrasts

- **phase2s_mix25_minus_phase2s_official**: BLiMP -2.100, Supplement -3.200, EWoK +5.820, Entity +2.230, Entity_full +2.130, COMPS +0.550, GlobalPIQA_mean +0.410, Reading +0.450, equal7_mean +0.594, equal7_full_entity +0.580, weighted_fast_proxy +0.618
- **phase2s_official_minus_initial_model_studies_baseline**: BLiMP +1.260, Supplement -1.200, EWoK -3.730, Entity +1.590, Entity_full +0.870, COMPS -0.950, GlobalPIQA_mean -1.895, Reading -1.705, equal7_mean -0.947, equal7_full_entity -1.050, weighted_fast_proxy +15.683
- **phase2s_mix25_minus_initial_model_studies_baseline**: BLiMP -0.840, Supplement -4.400, EWoK +2.090, Entity +3.820, Entity_full +3.000, COMPS -0.400, GlobalPIQA_mean -1.485, Reading -1.255, equal7_mean -0.353, equal7_full_entity -0.470, weighted_fast_proxy +16.302
- **phase2s_official_minus_leader_local_columns**: BLiMP +1.410, Supplement +7.990, EWoK -10.160, Entity -5.620, COMPS -1.410, GlobalPIQA_mean -5.445, Reading +0.205
- **phase2s_mix25_minus_leader_local_columns**: BLiMP -0.690, Supplement +4.790, EWoK -4.340, Entity -3.390, COMPS -0.860, GlobalPIQA_mean -5.035, Reading +0.655

## Scientific reading

The summary JSON should be used for the authoritative numbers. The comparison tests whether the low-dose aligned-data gain survives when tokenizer, architecture, optimizer, sequence curriculum, initialization and training RNG are held fixed in the stronger 12×384/LAMB/40k recipe.
