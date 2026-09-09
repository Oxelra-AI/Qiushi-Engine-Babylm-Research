# syncse relation ladder eval — SynCSE semantic-relation ladder evaluation

Summary JSON: `experiments/archive/compact_experience/data/syncse_relation_ladder_eval/relation_ladder_eval_summary.json`

This evaluation is a relation-type probe. SynCSE `hard_neg` is not a pure same-topic non-synonymous coherent control; examples often preserve entities/lexical fields while introducing semantic incompatibility, negation, or role reversal. Therefore the hard-neg contrast is informative about hard-negative adjacency, but the cross-relation interaction must not be used as direct attribution to rewrite correspondence beyond generic coherence.

## Scores

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | equal-7 | weighted fast proxy | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hardneg_coherent | 63.490 | 57.600 | 46.910 | 29.160 | 51.160 | 35.620 | 6.835 | 41.539 | 31.277 | 2.465 |
| hardneg_mismatched | 63.430 | 53.200 | 49.820 | 18.920 | 50.290 | 37.090 | 7.310 | 40.009 | 30.137 | 3.024 |
| paraphrase_aligned | 63.640 | 59.200 | 48.270 | 31.120 | 51.460 | 31.665 | 7.465 | 41.831 | 31.507 | 1.933 |
| paraphrase_mismatched | 65.190 | 54.800 | 48.360 | 17.530 | 51.530 | 39.120 | 7.950 | 40.640 | 30.622 | 3.059 |
| initial_model_baseline | 67.340 | 65.200 | 49.640 | 21.240 | 53.110 | 36.120 | 7.330 | 42.854 | 32.272 | 2.609 |

## Contrasts

| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal-7 | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hardneg_coherent_minus_hardneg_mismatched | 0.060 | 4.400 | -2.910 | 10.240 | 0.870 | -1.470 | -0.475 | 1.531 | 1.140 |
| paraphrase_aligned_minus_paraphrase_mismatched | -1.550 | 4.400 | -0.090 | 13.590 | -0.070 | -7.455 | -0.485 | 1.191 | 0.885 |
| paraphrase_aligned_minus_hardneg_coherent | 0.150 | 1.600 | 1.360 | 1.960 | 0.300 | -3.955 | 0.630 | 0.292 | 0.230 |
| paraphrase_mismatched_minus_hardneg_mismatched | 1.760 | 1.600 | -1.460 | -1.390 | 1.240 | 2.030 | 0.640 | 0.631 | 0.485 |
| paraphrase_aligned_minus_initial_model_studies_baseline | -3.700 | -6.000 | -1.370 | 9.880 | -1.650 | -4.455 | 0.135 | -1.023 | -0.765 |
| hardneg_coherent_minus_initial_model_studies_baseline | -3.850 | -7.600 | -2.730 | 7.920 | -1.950 | -0.500 | -0.495 | -1.315 | -0.995 |
| relation_type_interaction_paraphrase_minus_hardneg | -1.610 | 0.000 | 2.820 | 3.350 | -0.940 | -5.985 | -0.010 | -0.339 | -0.255 |
