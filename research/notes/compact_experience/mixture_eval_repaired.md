# mixture eval repaired — Repaired mixture dose-response evaluation

Summary JSON: `experiments/archive/compact_experience/data/mixture_eval/mixture_eval_summary.json`

The original mixture experiment design evaluator was cancelled after it used non-existent task modules. This repaired run uses the paired alignment accounting audit/syncse relation ladder eval `sentence_zero_shot` and `reading` invocations.

Important interpretation constraint: the three new mixture arms record `extra_init_seed=-1`; b256 fixedseq pair eval raw showed this means seed43 does not fix model initialization in the inherited fullcycle trainer. This curve is exploratory evidence for a dose-response, not a basis for locking Phase 2 without shared-initialization replication and an official-data Phase 2 control.

## Dose-response table

| target | frac | BLiMP | Supp | EWoK | Entity_fast | Entity_full | COMPS | GPIQA_mean | Reading | equal7_fast | equal7_fullEnt | wproxy | extra_init_seed | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_model_baseline | 0% | 67.35 | 65.20 | 49.64 | 21.24 | 21.78 | 53.11 | 36.12 | 7.330 | 42.856 | 42.933 | 32.273 | -1 | 2.61 |
| mix_25pct | 25% | 68.12 | 65.60 | 51.09 | 20.84 | 22.70 | 51.98 | 38.12 | 8.405 | 43.451 | 43.716 | 32.738 | -1 | 2.53 |
| mix_50pct | 50% | 65.85 | 65.20 | 50.55 | 22.76 | 23.76 | 52.31 | 30.68 | 7.910 | 42.180 | 42.323 | 31.776 | -1 | 2.67 |
| mix_75pct | 75% | 66.91 | 65.60 | 50.45 | 25.60 | 25.64 | 51.53 | 33.12 | 7.420 | 42.947 | 42.953 | 32.343 | -1 | 2.29 |
| aligned_100pct | 100% | 64.13 | 60.80 | 51.18 | 27.74 | 29.16 | 51.42 | 37.52 | 7.160 | 42.850 | 43.053 | 32.265 | -1 | 1.95 |

## Contrasts vs INITIAL_MODEL_STUDIES baseline

- **mix_25pct_minus_initial_model_studies_baseline**: equal7_fast +0.595, equal7_fullEnt +0.784, Entity_fast -0.40, Entity_full +0.92, BLiMP +0.77, Supplement +0.40, wproxy +0.465
- **mix_50pct_minus_initial_model_studies_baseline**: equal7_fast -0.676, equal7_fullEnt -0.610, Entity_fast +1.52, Entity_full +1.98, BLiMP -1.50, Supplement +0.00, wproxy -0.496
- **mix_75pct_minus_initial_model_studies_baseline**: equal7_fast +0.091, equal7_fullEnt +0.020, Entity_fast +4.36, Entity_full +3.86, BLiMP -0.44, Supplement +0.40, wproxy +0.070
- **aligned_100pct_minus_initial_model_studies_baseline**: equal7_fast -0.006, equal7_fullEnt +0.120, Entity_fast +6.50, Entity_full +7.38, BLiMP -3.22, Supplement -4.40, wproxy -0.007

Best by fast-Entity equal7: `mix_25pct` = 43.451.
Best by full-Entity equal7: `mix_25pct` = 43.716.
