# earlier analysis official-macro Entity transition check

This is a file-only recomputation from `entity_transition_rows.csv`. It averages gain/loss and option-response quantities over the same split×numops subtasks used in the official Entity score, rather than micro-weighting examples.

## Late 80/90/100M macro transition means

| contrast | group | Δ acc pp | wrong→correct pp | correct→wrong pp | both-correct pp | both-wrong pp | Δ option0 pp | Δ option1 pp | Δ option2 pp | Δ option3 pp | Δ option4 pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VminusB | all_18_subtasks | +2.994 | +9.648 | +6.653 | +18.441 | +65.258 | +2.994 | -1.530 | -0.963 | -0.274 | -0.227 |
| VminusB | zero_ops | +3.376 | +11.468 | +8.092 | +27.032 | +53.409 | +3.376 | +0.312 | -1.145 | -0.437 | -2.107 |
| VminusB | nonzero_ops | +2.918 | +9.283 | +6.366 | +16.723 | +67.628 | +2.918 | -1.898 | -0.927 | -0.241 | +0.149 |
| VminusB | numops_0 | +3.376 | +11.468 | +8.092 | +27.032 | +53.409 | +3.376 | +0.312 | -1.145 | -0.437 | -2.107 |
| VminusB | numops_1 | +0.724 | +6.469 | +5.745 | +8.793 | +78.993 | +0.724 | +1.349 | -3.177 | +1.546 | -0.442 |
| VminusB | numops_2 | +1.390 | +7.700 | +6.309 | +14.765 | +71.226 | +1.390 | -0.468 | +0.768 | -1.466 | -0.224 |
| VminusB | numops_3 | +1.994 | +8.756 | +6.762 | +17.693 | +66.789 | +1.994 | -1.415 | +1.317 | -1.535 | -0.362 |
| VminusB | numops_4 | +5.159 | +10.843 | +5.684 | +21.447 | +62.026 | +5.159 | -3.872 | -0.627 | -0.714 | +0.054 |
| VminusB | numops_5 | +5.321 | +12.650 | +7.329 | +20.918 | +59.103 | +5.321 | -5.086 | -2.915 | +0.962 | +1.719 |
| VminusB | split_ambiref | +2.948 | +9.876 | +6.927 | +17.367 | +65.830 | +2.948 | -1.533 | -1.596 | +0.413 | -0.232 |
| VminusB | split_regular | +3.346 | +9.986 | +6.639 | +18.080 | +65.295 | +3.346 | -1.953 | -0.539 | -1.242 | +0.387 |
| VminusB | split_move_contents | +2.688 | +9.081 | +6.394 | +19.877 | +64.649 | +2.688 | -1.104 | -0.755 | +0.007 | -0.836 |
| BminusR | all_18_subtasks | +1.023 | +10.245 | +9.222 | +14.850 | +65.684 | +1.023 | +0.770 | -6.489 | +2.628 | +2.068 |
| BminusR | zero_ops | -13.037 | +6.241 | +19.278 | +28.883 | +45.599 | -13.037 | +4.813 | +2.324 | +2.371 | +3.529 |
| BminusR | nonzero_ops | +3.836 | +11.046 | +7.210 | +12.043 | +69.701 | +3.836 | -0.039 | -8.251 | +2.679 | +1.775 |
| BminusR | numops_0 | -13.037 | +6.241 | +19.278 | +28.883 | +45.599 | -13.037 | +4.813 | +2.324 | +2.371 | +3.529 |
| BminusR | numops_1 | +1.243 | +7.305 | +6.063 | +7.233 | +79.399 | +1.243 | -5.054 | -3.376 | +3.020 | +4.168 |
| BminusR | numops_2 | +2.464 | +10.253 | +7.789 | +10.821 | +71.137 | +2.464 | -1.843 | -7.954 | +3.925 | +3.408 |
| BminusR | numops_3 | +6.038 | +12.564 | +6.526 | +11.891 | +69.019 | +6.038 | +0.021 | -8.567 | +2.177 | +0.331 |
| BminusR | numops_4 | +4.184 | +12.139 | +7.955 | +14.992 | +64.914 | +4.184 | +1.253 | -10.049 | +2.510 | +2.102 |
| BminusR | numops_5 | +5.250 | +12.968 | +7.719 | +15.278 | +64.034 | +5.250 | +5.428 | -11.312 | +1.766 | -1.131 |
| BminusR | split_ambiref | +0.625 | +9.413 | +8.788 | +14.881 | +66.917 | +0.625 | +0.248 | -2.517 | +1.186 | +0.458 |
| BminusR | split_regular | +1.014 | +10.763 | +9.749 | +13.957 | +65.531 | +1.014 | +2.347 | -8.976 | +3.728 | +1.888 |
| BminusR | split_move_contents | +1.431 | +10.559 | +9.127 | +15.711 | +64.602 | +1.431 | -0.285 | -7.973 | +2.970 | +3.858 |
| VminusR | all_18_subtasks | +4.018 | +11.715 | +7.698 | +16.374 | +64.213 | +4.018 | -0.760 | -7.452 | +2.354 | +1.841 |
| VminusR | zero_ops | -9.661 | +7.318 | +16.979 | +31.182 | +44.522 | -9.661 | +5.125 | +1.179 | +1.934 | +1.422 |
| VminusR | nonzero_ops | +6.753 | +12.595 | +5.842 | +13.412 | +68.152 | +6.753 | -1.937 | -9.178 | +2.438 | +1.924 |
| VminusR | numops_0 | -9.661 | +7.318 | +16.979 | +31.182 | +44.522 | -9.661 | +5.125 | +1.179 | +1.934 | +1.422 |
| VminusR | numops_1 | +1.967 | +7.856 | +5.889 | +7.406 | +78.848 | +1.967 | -3.704 | -6.554 | +4.566 | +3.726 |
| VminusR | numops_2 | +3.854 | +10.188 | +6.335 | +12.276 | +71.201 | +3.854 | -2.311 | -7.186 | +2.459 | +3.184 |
| VminusR | numops_3 | +8.032 | +13.002 | +4.970 | +13.447 | +68.581 | +8.032 | -1.394 | -7.249 | +0.642 | -0.031 |
| VminusR | numops_4 | +9.343 | +15.587 | +6.243 | +16.704 | +61.466 | +9.343 | -2.619 | -10.676 | +1.796 | +2.156 |
| VminusR | numops_5 | +10.571 | +16.341 | +5.770 | +17.227 | +60.662 | +10.571 | +0.342 | -14.227 | +2.727 | +0.587 |
| VminusR | split_ambiref | +3.574 | +10.504 | +6.931 | +16.739 | +65.827 | +3.574 | -1.286 | -4.114 | +1.600 | +0.226 |
| VminusR | split_regular | +4.360 | +12.446 | +8.086 | +15.620 | +63.848 | +4.360 | +0.394 | -9.514 | +2.486 | +2.275 |
| VminusR | split_move_contents | +4.119 | +12.195 | +8.076 | +16.763 | +62.966 | +4.119 | -1.389 | -8.728 | +2.976 | +3.021 |

## Scientific reading

- Macro V-B is positive in the official arithmetic: all18 +2.994, zero-op +3.376, nonzero +2.918 pp.
- That macro V-B still has substantial item churn: all18 wrong→correct +9.648 pp and correct→wrong +6.653 pp. The net is positive because gains exceed losses, not because items are stable records.
- Macro B-R retains the operation prior shape: zero-op -13.037 versus nonzero +3.836 pp.
- These official-macro transition rates do not overturn the paired binding conservation result, where V-B decreases both-correct state. They mainly confirm that official Entity V-B is a real score surface but do not identify source correspondence.

## Files

- macro_rows_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_official_macro_transition_rows.csv`
- late_summary_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_official_macro_transition_late_summary.csv`
- summary_md: `research/documents/representation_and_objectives/data/conservation_transition_readout/entity_official_macro_transition_summary.md`
