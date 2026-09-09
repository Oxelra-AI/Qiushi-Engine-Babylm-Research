# zero reading and superglue item evidence zero-shot/Reading item stability

This parses actual official prediction files for coherent86, dense seeds, and clean preservation seeds. It does not use pending SuperGLUE values.

## Payload score validation

| model | column | n items | subtasks | payload | computed official-like | item-weighted | payload-computed |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent86 | BLiMP | 59875 | 67 | 68.5100 | 68.5160 | 68.4042 | -0.005970 |
| coherent86 | Supplement | 5218 | 5 | 63.6400 | 63.6354 | 73.9747 | 0.004631 |
| coherent86 | EWoK | 7618 | 11 | 50.0200 | 50.0196 | 49.9869 | 0.000384 |
| coherent86 | Entity | 6780 | 18 | 28.3200 | 28.3222 | 27.8319 | -0.002219 |
| coherent86 | COMPS | 91028 | 4 | 52.0500 | 52.0472 | 52.5135 | 0.002825 |
| coherent86 | GlobalPIQA_parallel | 103 | 1 | 29.1300 | 29.1262 | 29.1262 | 0.003786 |
| coherent86 | GlobalPIQA_nonparallel | 100 | 1 | 48.0000 | 48.0000 | 48.0000 | 0.000000 |
| coherent86 | Reading | 1726 | - | {'Reading_eye': 10.74, 'Reading_self_paced': 5.59, 'Reading': 8.165} | - | - | - |
| dense_seed62064 | BLiMP | 59875 | 67 | 68.0600 | 68.0655 | 67.9499 | -0.005520 |
| dense_seed62064 | Supplement | 5218 | 5 | 63.0400 | 63.0375 | 73.8789 | 0.002507 |
| dense_seed62064 | EWoK | 7618 | 11 | 49.9200 | 49.9225 | 50.1575 | -0.002478 |
| dense_seed62064 | Entity | 6780 | 18 | 29.3700 | 29.3670 | 28.4513 | 0.003048 |
| dense_seed62064 | COMPS | 91028 | 4 | 52.1500 | 52.1537 | 52.6464 | -0.003718 |
| dense_seed62064 | GlobalPIQA_parallel | 103 | 1 | 30.1000 | 30.0971 | 30.0971 | 0.002913 |
| dense_seed62064 | GlobalPIQA_nonparallel | 100 | 1 | 50.0000 | 50.0000 | 50.0000 | 0.000000 |
| dense_seed62064 | Reading | 1726 | - | {'Reading_eye': 10.7, 'Reading_self_paced': 5.74, 'Reading': 8.219999999999999} | - | - | - |
| dense_seed62065 | BLiMP | 59875 | 67 | 68.0200 | 68.0283 | 67.9115 | -0.008300 |
| dense_seed62065 | Supplement | 5218 | 5 | 63.0900 | 63.0934 | 73.8406 | -0.003405 |
| dense_seed62065 | EWoK | 7618 | 11 | 49.7700 | 49.7699 | 50.0394 | 0.000069 |
| dense_seed62065 | Entity | 6780 | 18 | 29.4200 | 29.4157 | 28.4661 | 0.004308 |
| dense_seed62065 | COMPS | 91028 | 4 | 52.1500 | 52.1455 | 52.6453 | 0.004543 |
| dense_seed62065 | GlobalPIQA_parallel | 103 | 1 | 30.1000 | 30.0971 | 30.0971 | 0.002913 |
| dense_seed62065 | GlobalPIQA_nonparallel | 100 | 1 | 50.0000 | 50.0000 | 50.0000 | 0.000000 |
| dense_seed62065 | Reading | 1726 | - | {'Reading_eye': 10.68, 'Reading_self_paced': 5.74, 'Reading': 8.21} | - | - | - |
| clean_seed62064 | BLiMP | 59875 | 67 | 68.2600 | 68.2662 | 68.1553 | -0.006160 |
| clean_seed62064 | Supplement | 5218 | 5 | 63.2800 | 63.2775 | 73.9939 | 0.002530 |
| clean_seed62064 | EWoK | 7618 | 11 | 49.8200 | 49.8240 | 49.9737 | -0.003978 |
| clean_seed62064 | Entity | 6780 | 18 | 29.4000 | 29.3999 | 28.4661 | 0.000138 |
| clean_seed62064 | COMPS | 91028 | 4 | 52.1600 | 52.1581 | 52.6728 | 0.001876 |
| clean_seed62064 | GlobalPIQA_parallel | 103 | 1 | 30.1000 | 30.0971 | 30.0971 | 0.002913 |
| clean_seed62064 | GlobalPIQA_nonparallel | 100 | 1 | 50.0000 | 50.0000 | 50.0000 | 0.000000 |
| clean_seed62064 | Reading | 1726 | - | {'Reading_eye': 10.69, 'Reading_self_paced': 5.71, 'Reading': 8.2} | - | - | - |
| clean_seed62065 | BLiMP | 59875 | 67 | 68.2200 | 68.2274 | 68.1136 | -0.007416 |
| clean_seed62065 | Supplement | 5218 | 5 | 63.2900 | 63.2909 | 73.9747 | -0.000879 |
| clean_seed62065 | EWoK | 7618 | 11 | 49.7200 | 49.7234 | 49.9344 | -0.003442 |
| clean_seed62065 | Entity | 6780 | 18 | 29.4500 | 29.4488 | 28.5251 | 0.001233 |
| clean_seed62065 | COMPS | 91028 | 4 | 52.1400 | 52.1353 | 52.6541 | 0.004706 |
| clean_seed62065 | GlobalPIQA_parallel | 103 | 1 | 30.1000 | 30.0971 | 30.0971 | 0.002913 |
| clean_seed62065 | GlobalPIQA_nonparallel | 100 | 1 | 50.0000 | 50.0000 | 50.0000 | 0.000000 |
| clean_seed62065 | Reading | 1726 | - | {'Reading_eye': 10.69, 'Reading_self_paced': 5.7, 'Reading': 8.195} | - | - | - |

## Clean seed62064/seed62065 item replication

| column | n | correct agree | pred-string agree | shared gains | shared losses | gain Jaccard | loss Jaccard | opposite changed | net item pp seed64 | net item pp seed65 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 0.9981 | 0.9981 | 551 | 694 | 0.9371 | 0.9036 | 0 | -0.2489 | -0.2906 |
| Supplement | 5218 | 0.9994 | 0.9994 | 37 | 36 | 0.9737 | 0.9474 | 0 | 0.0192 | 0.0000 |
| EWoK | 7618 | 0.9967 | 0.9967 | 158 | 160 | 0.9294 | 0.9249 | 0 | -0.0131 | -0.0525 |
| Entity | 6780 | 0.9973 | 0.9937 | 181 | 136 | 0.9526 | 0.9379 | 0 | 0.6342 | 0.6932 |
| COMPS | 91028 | 0.9973 | 0.9973 | 2529 | 2389 | 0.9554 | 0.9503 | 0 | 0.1593 | 0.1406 |
| GlobalPIQA_parallel | 103 | 1.0000 | 1.0000 | 2 | 1 | 1.0000 | 1.0000 | 0 | 0.9709 | 0.9709 |
| GlobalPIQA_nonparallel | 100 | 1.0000 | 1.0000 | 2 | 0 | 1.0000 | None | 0 | 2.0000 | 2.0000 |

## Dense-to-clean seed-matched decomposition

Counts are item-weighted: clean recovery of dense parent losses, clean loss of items dense kept, kept dense gains, dropped dense gains, and new clean gains beyond dense.

| column | seed pair | n | recover dense loss | lose dense-kept parent item | keep dense gain | drop dense gain | new clean gain | shared parent loss | net clean-dense item pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 62064 | 59875 | 338 | 17 | 562 | 214 | 16 | 710 | 0.2054 |
| BLiMP | 62065 | 59875 | 367 | 22 | 546 | 239 | 15 | 713 | 0.2021 |
| Supplement | 62064 | 5218 | 22 | 0 | 37 | 17 | 1 | 37 | 0.1150 |
| Supplement | 62065 | 5218 | 22 | 0 | 36 | 16 | 1 | 37 | 0.1342 |
| EWoK | 62064 | 7618 | 68 | 9 | 151 | 85 | 12 | 155 | -0.1838 |
| EWoK | 62065 | 7618 | 72 | 10 | 155 | 80 | 10 | 159 | -0.1050 |
| Entity | 62064 | 6780 | 41 | 7 | 173 | 46 | 13 | 136 | 0.0147 |
| Entity | 62065 | 6780 | 45 | 4 | 172 | 50 | 13 | 134 | 0.0590 |
| COMPS | 62064 | 91028 | 1019 | 73 | 2545 | 985 | 63 | 2390 | 0.0264 |
| COMPS | 62065 | 91028 | 1094 | 66 | 2510 | 1078 | 58 | 2374 | 0.0088 |
| GlobalPIQA_parallel | 62064 | 103 | 0 | 0 | 2 | 0 | 0 | 1 | 0.0000 |
| GlobalPIQA_parallel | 62065 | 103 | 0 | 0 | 2 | 0 | 0 | 1 | 0.0000 |
| GlobalPIQA_nonparallel | 62064 | 100 | 0 | 0 | 2 | 0 | 0 | 0 | 0.0000 |
| GlobalPIQA_nonparallel | 62065 | 100 | 0 | 0 | 2 | 0 | 0 | 0 | 0.0000 |

## Reading vector stability

| pair | n | pred Pearson | prev_pred Pearson | mean signed diff a-b | mean abs diff |
|---|---:|---:|---:|---:|---:|
| clean_seed62064__coherent86 | 1726 | 0.99830 | 0.99725 | 0.01379 | 0.20810 |
| clean_seed62064__dense_seed62064 | 1726 | 0.99972 | 0.99955 | -0.04754 | 0.09410 |
| clean_seed62064__dense_seed62065 | 1726 | 0.99971 | 0.99952 | -0.04711 | 0.09755 |
| clean_seed62064__clean_seed62065 | 1726 | 0.99999 | 0.99999 | 0.00078 | 0.01188 |
| clean_seed62065__coherent86 | 1726 | 0.99830 | 0.99726 | 0.01301 | 0.20749 |
| clean_seed62065__dense_seed62064 | 1726 | 0.99972 | 0.99954 | -0.04832 | 0.09455 |
| clean_seed62065__dense_seed62065 | 1726 | 0.99971 | 0.99952 | -0.04789 | 0.09704 |

## Family movements most relevant to interpretation

### BLiMP

Clean64 largest positive subtask deltas vs coherent86:

- matrix_question_npi_licensor_present: coherent 41.66, dense64 45.53, clean64 44.56, clean65 44.35, clean64-parent 2.91
- existential_there_quantifiers_2: coherent 40.94, dense64 44.02, clean64 43.36, clean65 43.25, clean64-parent 2.41
- wh_questions_object_gap: coherent 54.60, dense64 56.34, clean64 56.69, clean65 56.69, clean64-parent 2.10
- ellipsis_n_bar_2: coherent 96.38, dense64 97.95, clean64 97.34, clean65 97.34, clean64-parent 0.97
- sentential_subject_island: coherent 32.78, dense64 34.34, clean64 33.61, clean65 33.51, clean64-parent 0.83
- distractor_agreement_relative_clause: coherent 28.59, dense64 29.39, clean64 29.28, clean65 29.05, clean64-parent 0.69
Clean64 largest negative subtask deltas vs coherent86:

- only_npi_licensor_present: coherent 80.39, dense64 77.55, clean64 78.00, clean65 78.12, clean64-parent -2.38
- distractor_agreement_relational_noun: coherent 49.87, dense64 46.83, clean64 47.84, clean65 47.97, clean64-parent -2.03
- wh_vs_that_no_gap: coherent 88.73, dense64 86.64, clean64 86.76, clean65 86.88, clean64-parent -1.97
- superlative_quantifiers_2: coherent 82.25, dense64 79.41, clean64 80.32, clean65 80.73, clean64-parent -1.93
- principle_A_domain_1: coherent 77.02, dense64 73.52, clean64 75.60, clean65 75.71, clean64-parent -1.42
- determiner_noun_agreement_1: coherent 88.91, dense64 86.65, clean64 87.51, clean65 87.41, clean64-parent -1.40

### Supplement

Clean64 largest positive subtask deltas vs coherent86:

- subject_aux_inversion: coherent 80.79, dense64 80.92, clean64 80.92, clean65 80.86, clean64-parent 0.13
- qa_congruence_tricky: coherent 49.70, dense64 49.09, clean64 49.70, clean65 49.70, clean64-parent 0.00
- turn_taking: coherent 67.50, dense64 67.50, clean64 67.50, clean65 67.50, clean64-parent 0.00
- hypernym: coherent 49.88, dense64 48.93, clean64 49.52, clean65 49.64, clean64-parent -0.36
- qa_congruence_easy: coherent 70.31, dense64 68.75, clean64 68.75, clean65 68.75, clean64-parent -1.56
Clean64 largest negative subtask deltas vs coherent86:

- qa_congruence_easy: coherent 70.31, dense64 68.75, clean64 68.75, clean65 68.75, clean64-parent -1.56
- hypernym: coherent 49.88, dense64 48.93, clean64 49.52, clean65 49.64, clean64-parent -0.36
- qa_congruence_tricky: coherent 49.70, dense64 49.09, clean64 49.70, clean65 49.70, clean64-parent 0.00
- turn_taking: coherent 67.50, dense64 67.50, clean64 67.50, clean65 67.50, clean64-parent 0.00
- subject_aux_inversion: coherent 80.79, dense64 80.92, clean64 80.92, clean65 80.86, clean64-parent 0.13

### EWoK

Clean64 largest positive subtask deltas vs coherent86:

- material-dynamics: coherent 50.52, dense64 54.68, clean64 52.08, clean65 52.73, clean64-parent 1.56
- quantitative-properties: coherent 48.73, dense64 48.41, clean64 49.68, clean65 49.04, clean64-parent 0.96
- spatial-relations: coherent 46.53, dense64 46.73, clean64 47.14, clean65 47.14, clean64-parent 0.61
- social-relations: coherent 50.65, dense64 50.97, clean64 51.23, clean65 50.90, clean64-parent 0.58
- social-interactions: coherent 54.08, dense64 54.76, clean64 53.74, clean65 52.72, clean64-parent -0.34
- agent-properties: coherent 50.36, dense64 50.14, clean64 49.95, clean65 50.00, clean64-parent -0.41
Clean64 largest negative subtask deltas vs coherent86:

- social-properties: coherent 46.65, dense64 45.12, clean64 44.82, clean65 44.82, clean64-parent -1.83
- material-properties: coherent 51.76, dense64 51.18, clean64 50.59, clean65 50.59, clean64-parent -1.18
- physical-dynamics: coherent 52.50, dense64 50.83, clean64 51.67, clean65 51.67, clean64-parent -0.83
- physical-relations: coherent 49.88, dense64 49.39, clean64 49.14, clean65 49.14, clean64-parent -0.73
- physical-interactions: coherent 48.56, dense64 46.94, clean64 48.02, clean65 48.20, clean64-parent -0.54
- agent-properties: coherent 50.36, dense64 50.14, clean64 49.95, clean65 50.00, clean64-parent -0.41

### Entity

Clean64 largest positive subtask deltas vs coherent86:

- ambiref_5_ops: coherent 31.71, dense64 36.59, clean64 35.77, clean65 35.77, clean64-parent 4.07
- move_contents_2_ops: coherent 21.55, dense64 25.31, clean64 24.81, clean65 25.06, clean64-parent 3.26
- regular_2_ops: coherent 21.98, dense64 24.44, clean64 24.69, clean65 24.20, clean64-parent 2.72
- move_contents_5_ops: coherent 43.10, dense64 44.83, clean64 45.69, clean65 45.69, clean64-parent 2.59
- move_contents_3_ops: coherent 26.60, dense64 28.57, clean64 29.06, clean65 29.06, clean64-parent 2.46
- regular_4_ops: coherent 27.58, dense64 30.41, clean64 29.90, clean65 29.90, clean64-parent 2.32
Clean64 largest negative subtask deltas vs coherent86:

- regular_0_ops: coherent 40.62, dense64 35.78, clean64 36.94, clean65 36.94, clean64-parent -3.68
- move_contents_0_ops: coherent 45.54, dense64 42.44, clean64 42.83, clean65 42.83, clean64-parent -2.71
- ambiref_0_ops: coherent 25.20, dense64 23.82, clean64 23.43, clean65 24.21, clean64-parent -1.77
- move_contents_1_ops: coherent 17.16, dense64 16.70, clean64 16.70, clean65 16.25, clean64-parent -0.46
- ambiref_2_ops: coherent 24.94, dense64 24.46, clean64 24.70, clean65 24.70, clean64-parent -0.24
- ambiref_1_ops: coherent 14.95, dense64 14.02, clean64 15.19, clean65 14.95, clean64-parent 0.23

### COMPS

Clean64 largest positive subtask deltas vs coherent86:

- wugs_dist_before: coherent 37.48, dense64 38.54, clean64 38.38, clean65 38.39, clean64-parent 0.90
- base: coherent 53.24, dense64 53.42, clean64 53.48, clean65 53.47, clean64-parent 0.24
- wugs: coherent 52.02, dense64 51.69, clean64 51.88, clean65 51.89, clean64-parent -0.14
- wugs_dist_in_between: coherent 65.44, dense64 64.97, clean64 64.90, clean65 64.80, clean64-parent -0.55
Clean64 largest negative subtask deltas vs coherent86:

- wugs_dist_in_between: coherent 65.44, dense64 64.97, clean64 64.90, clean65 64.80, clean64-parent -0.55
- wugs: coherent 52.02, dense64 51.69, clean64 51.88, clean65 51.89, clean64-parent -0.14
- base: coherent 53.24, dense64 53.42, clean64 53.48, clean65 53.47, clean64-parent 0.24
- wugs_dist_before: coherent 37.48, dense64 38.54, clean64 38.38, clean65 38.39, clean64-parent 0.90

### GlobalPIQA_parallel

Clean64 largest positive subtask deltas vs coherent86:

- GlobalPIQA_parallel: coherent 29.13, dense64 30.10, clean64 30.10, clean65 30.10, clean64-parent 0.97
Clean64 largest negative subtask deltas vs coherent86:

- GlobalPIQA_parallel: coherent 29.13, dense64 30.10, clean64 30.10, clean65 30.10, clean64-parent 0.97

### GlobalPIQA_nonparallel

Clean64 largest positive subtask deltas vs coherent86:

- GlobalPIQA_nonparallel: coherent 48.00, dense64 50.00, clean64 50.00, clean65 50.00, clean64-parent 2.00
Clean64 largest negative subtask deltas vs coherent86:

- GlobalPIQA_nonparallel: coherent 48.00, dense64 50.00, clean64 50.00, clean65 50.00, clean64-parent 2.00

## Scientific reading

The clean seeds should be read as a repeated displacement only where item agreement and shared parent-relative gain/loss sets are high. GlobalPIQA is a small fixed item set and its equality across clean seeds should not be mistaken for broad robustness by itself. Entity gains are meaningful only if they concentrate coherently by operation/split and repeat across seeds. BLiMP/Supplement/EWoK losses remain real where both clean seeds share losses against coherent86. The dense-to-clean decomposition indicates whether preservation truly recovers dense-induced parent losses on zero-shot items or whether the score difference is mostly unrelated item churn; this complements, but does not replace, the pending SuperGLUE and exact `(M,S)` official comparison.
