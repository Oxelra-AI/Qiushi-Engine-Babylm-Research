# identity edge state transport gate token-level repaired RTD-vs-WWM context probe

Evidence JSON: `experiments/archive/initial_model_studies/data/rtd_wwm_context_probe_tokenlevel.json`
Rows CSV: `experiments/archive/initial_model_studies/data/context_probe_tokenlevel_rows.csv`

Fast profile from completed counterfactual binding credit gate direct evaluation: RTD-WWM BLiMP +0.17, Supplement -2.80, EWoK +2.18, Entity -1.39, COMPS +0.13, Reading +0.045.

RTD shortcut metrics at 1M: replaced recall 0.2554, above-majority 0.0186, pred-original 0.9413, label-original 0.8486.

| metric | mean | median | pos frac | n |
|---|---:|---:|---:|---:|
| wwm_delta_deleted | +0.00002 | +0.00000 | 0.525 | 240 |
| wwm_delta_shuffled | -0.00001 | +0.00000 | 0.425 | 240 |
| wwm_delta_unrelated | -0.00001 | +0.00000 | 0.512 | 240 |
| wwm_spec_deleted_vs_unrelated | +0.00003 | +0.00000 | 0.433 | 240 |
| wwm_spec_shuffled_vs_unrelated | -0.00000 | +0.00000 | 0.417 | 240 |
| rtd_mlm_delta_deleted | +0.00183 | +0.00132 | 0.546 | 240 |
| rtd_mlm_delta_shuffled | -0.00002 | +0.00002 | 0.529 | 240 |
| rtd_mlm_delta_unrelated | +0.00286 | +0.00101 | 0.542 | 240 |
| rtd_mlm_spec_deleted_vs_unrelated | -0.00103 | +0.00080 | 0.533 | 240 |
| rtd_mlm_spec_shuffled_vs_unrelated | -0.00288 | -0.00105 | 0.471 | 240 |
| rtd_minus_wwm_delta_deleted | +0.00181 | +0.00142 | 0.542 | 240 |
| rtd_minus_wwm_delta_shuffled | -0.00001 | +0.00005 | 0.542 | 240 |
| rtd_minus_wwm_delta_unrelated | +0.00286 | +0.00089 | 0.542 | 240 |
| rtd_minus_wwm_spec_deleted_vs_unrelated | -0.00106 | +0.00091 | 0.533 | 240 |
| rtd_minus_wwm_spec_shuffled_vs_unrelated | -0.00287 | -0.00097 | 0.475 | 240 |
