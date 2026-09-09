# ewok contrast preservation anatomy — EWoK interaction-margin probe

This diagnostic recomputes EWoK four-score pseudo-log-likelihood matrices on a stratified subset of existing checkpoint(s). It does not train or run full official evaluation.

## legal40_depth_12x384_43022

Rows `132`, device `cpu`, saved-official sign agreement `1.000`.

| subset | n | official_t1 positive | official_t2 positive | interaction positive | interaction median | both within-context positive | near-zero interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_selected | 132 | 0.3333333333333333 | 0.5757575757575758 | 0.38636363636363635 | -0.19032256468199193 | 0.03787878787878788 | 0.25 |
| legal40_depth_both_wrong_subset | 66 | 0.0 | 0.7272727272727273 | 0.30303030303030304 | -0.5804807008244097 | 0.045454545454545456 | 0.18181818181818182 |
| all_legal_wrong_subset | 33 | 0.0 | 0.7272727272727273 | 0.15151515151515152 | -0.6735787154175341 | 0.0 | 0.12121212121212122 |
| depth_saved_correct_subset | 44 | 1.0 | 0.29545454545454547 | 0.5454545454545454 | 0.09133931808173656 | 0.045454545454545456 | 0.29545454545454547 |
| depth_saved_wrong_subset | 88 | 0.0 | 0.7159090909090909 | 0.3068181818181818 | -0.44478777050971985 | 0.03409090909090909 | 0.22727272727272727 |

## legal40_8x480_43022

Rows `132`, device `cpu`, saved-official sign agreement `1.000`.

| subset | n | official_t1 positive | official_t2 positive | interaction positive | interaction median | both within-context positive | near-zero interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_selected | 132 | 0.3333333333333333 | 0.49242424242424243 | 0.5454545454545454 | 0.07534271478652954 | 0.045454545454545456 | 0.1893939393939394 |
| legal40_depth_both_wrong_subset | 66 | 0.0 | 0.6666666666666666 | 0.48484848484848486 | -0.12740413658320904 | 0.030303030303030304 | 0.19696969696969696 |
| all_legal_wrong_subset | 33 | 0.0 | 0.6666666666666666 | 0.48484848484848486 | -0.17595478892326355 | 0.0 | 0.15151515151515152 |
| depth_saved_correct_subset | 44 | 0.5 | 0.3181818181818182 | 0.5454545454545454 | 0.10179969668388367 | 0.022727272727272728 | 0.18181818181818182 |
| depth_saved_wrong_subset | 88 | 0.25 | 0.5795454545454546 | 0.5454545454545454 | 0.07171012629987672 | 0.056818181818181816 | 0.19318181818181818 |

Interpretation rule: persistent official errors with negative or near-zero interaction support a relational compatibility learning deficit; positive interaction but official failure points toward scorer/calibration/context-prior artifacts; large sensitivity to mean or PMI margins weakens the case for a training objective.

JSON: `experiments/archive/representation_and_objectives/data/ewok_interaction_margin_probe/ewok_interaction_margin_probe.json`

CSV: `experiments/archive/representation_and_objectives/data/ewok_interaction_margin_probe/ewok_interaction_margin_records.csv`
