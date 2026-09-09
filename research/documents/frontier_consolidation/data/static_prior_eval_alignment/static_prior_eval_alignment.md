# legal deficit decomposition and prior mismatch static-prior/evaluation-loss alignment

This is a post-hoc diagnostic of the already-built corpus-only priors. It does **not** tune the prior from evaluation data.

## Correlation of legal_step35 deficit with evaluation-token prior pressure

Positive Pearson/Spearman means higher-prior subtasks gained more or lost less vs the old non-submittable reference; negative means higher-prior subtasks lost more.

| column | n_uid | scheme | mean eval weight | Pearson(delta,weight) | Spearman | low-quartile Δ | high-quartile Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| BLiMP | 67 | relation_only_v1 | 0.9736 | 0.1622 | 0.2200 | -3.8206 | 1.9363 |
| BLiMP | 67 | relation_info_v1 | 0.9936 | 0.1675 | 0.1379 | -3.0381 | 1.9363 |
| Supplement | 5 | relation_only_v1 | 0.9796 | 0.2126 | -0.2000 | 1.3000 | 1.4200 |
| Supplement | 5 | relation_info_v1 | 0.9815 | -0.1944 | -0.1000 | 1.3000 | 1.4200 |
| EWoK | 11 | relation_only_v1 | 0.9930 | 0.2571 | 0.3000 | -7.3150 | 1.8300 |
| EWoK | 11 | relation_info_v1 | 1.0001 | 0.4591 | 0.4727 | -7.3150 | 1.8300 |
| ALL | 83 | relation_only_v1 | 0.9765 | 0.1445 | 0.2172 | -4.6880 | 0.2430 |
| ALL | 83 | relation_info_v1 | 0.9938 | 0.1816 | 0.1626 | -2.9545 | 2.1470 |

## Focus subtasks

| column | uid | Δ legal_step35-old | relation_only mean weight | frac >1.25 | relation_info mean weight |
|---|---|---:|---:|---:|---:|
| EWoK | physical-dynamics | -19.50 | 1.0056 | 0.0780 | 0.9964 |
| EWoK | social-properties | -13.75 | 0.9712 | 0.0255 | 0.9911 |
| EWoK | material-dynamics | -8.83 | 0.9494 | 0.0000 | 0.9753 |
| EWoK | quantitative-properties | -5.80 | 0.9686 | 0.0266 | 0.9789 |
| EWoK | physical-relations | -1.83 | 1.0204 | 0.1004 | 1.0052 |
| EWoK | spatial-relations | 3.06 | 1.0427 | 0.1130 | 1.0284 |
| EWoK | material-properties | 4.97 | 0.9780 | 0.0517 | 0.9818 |
| EWoK | social-interactions | 5.78 | 0.9827 | 0.0361 | 1.0129 |
| Supplement | qa_congruence_easy | -12.50 | 0.9753 | 0.0377 | 0.9824 |
| Supplement | qa_congruence_tricky | -3.64 | 0.9721 | 0.0171 | 0.9915 |
| Supplement | hypernym | 1.30 | 0.9704 | 0.0449 | 0.9628 |
| Supplement | subject_aux_inversion | 1.42 | 1.0083 | 0.0805 | 1.0018 |
| Supplement | turn_taking | 2.85 | 0.9718 | 0.0316 | 0.9691 |

## Interpretation for route selection

- The static-prior branch is mechanically valid, but this diagnostic asks whether it is scientifically matched to the legal deficit.
- If high `relation_only_v1` pressure correlates weakly or positively with Δ, the existing relation prior is not concentrated on the losing subtasks; choose it only if mature clean-vs-reinvest deltas independently show the amplified relation content is where reinvestment weakens.
- If high pressure correlates negatively and the mature deltas show the same relation/dynamic loss, then a single-variable `relation_only_v1` screen is better justified.

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_eval_alignment/static_prior_eval_alignment.json`
